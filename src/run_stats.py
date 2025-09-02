import os
import time
import shutil
import numpy as np
import torch
import gc
from collections import defaultdict
import random

from src.logger import logger

from src.run_qat import run_load_model
from src.utils import setup_dataloader
from src.make_ex_name import *
from src.models.model import create_model
from src.quantizer.uniform.lsq import LSQ_quantizer

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

class StatsCollector:
    """Stats collector that uses streaming statistics and sampling for memory efficiency"""
    def __init__(self, max_samples_per_tensor=10000, sample_ratio=0.1):
        self.static_stats = {}  # For weights, biases, buffers (computed once)
        self.streaming_stats = {}  # For running statistics (mean, std, min, max)
        self.max_samples_per_tensor = max_samples_per_tensor
        self.sample_ratio = sample_ratio  # Ratio of values to sample from each tensor
        self.sample_buffers = defaultdict(list)  # Temporary buffers for sampled values
        
    def reset(self):
        self.static_stats = {}
        self.streaming_stats = {}
        self.sample_buffers.clear()
    
    def register_static_tensor(self, name, tensor):
        """Register static tensors (weights, biases) - computed once"""
        if name not in self.static_stats and tensor is not None and tensor.numel() > 0:
            with torch.no_grad():
                flat_tensor = tensor.flatten()
                self.static_stats[name] = {
                    'mean': float(torch.mean(flat_tensor).cpu()),
                    'std': float(torch.std(flat_tensor).cpu()),
                    'min': float(torch.min(flat_tensor).cpu()),
                    'max': float(torch.max(flat_tensor).cpu()),
                    'shape': list(tensor.shape),
                    'numel': tensor.numel(),
                    'samples': self._sample_tensor(flat_tensor)  # Small sample for histograms
                }
    
    def _sample_tensor(self, flat_tensor, max_samples=1000):
        """Sample a subset of tensor values for histogram generation"""
        numel = flat_tensor.numel()
        if numel <= max_samples:
            return flat_tensor.cpu().numpy()
        else:
            # Random sampling
            indices = torch.randperm(numel)[:max_samples]
            return flat_tensor[indices].cpu().numpy()
    
    def update_streaming_stats(self, name, tensor):
        """Update streaming statistics for dynamic tensors"""
        if tensor is None or tensor.numel() == 0:
            return
            
        with torch.no_grad():
            flat_tensor = tensor.flatten()
            batch_mean = float(torch.mean(flat_tensor))
            batch_std = float(torch.std(flat_tensor))
            batch_min = float(torch.min(flat_tensor))
            batch_max = float(torch.max(flat_tensor))
            batch_size = flat_tensor.numel()
            
            if name not in self.streaming_stats:
                self.streaming_stats[name] = {
                    'count': 0,
                    'mean': 0.0,
                    'M2': 0.0,  # For Welford's algorithm
                    'min': float('inf'),
                    'max': float('-inf'),
                }
            
            stats = self.streaming_stats[name]
            
            # Update min/max
            stats['min'] = min(stats['min'], batch_min)
            stats['max'] = max(stats['max'], batch_max)
            
            # Welford's online algorithm for mean and variance
            old_count = stats['count']
            stats['count'] += batch_size
            delta = batch_mean - stats['mean']
            stats['mean'] += delta * batch_size / stats['count']
            delta2 = batch_mean - stats['mean']
            stats['M2'] += delta * delta2 * old_count * batch_size / stats['count']
            
            # Sample values for histogram (if buffer not full)
            if len(self.sample_buffers[name]) < self.max_samples_per_tensor:
                sample_size = min(
                    int(flat_tensor.numel() * self.sample_ratio),
                    self.max_samples_per_tensor - len(self.sample_buffers[name])
                )
                if sample_size > 0:
                    sampled = self._sample_tensor(flat_tensor, sample_size)
                    self.sample_buffers[name].extend(sampled)
    
    def get_final_stats(self):
        """Get final computed statistics"""
        final_stats = {}
        
        # Add static stats
        final_stats.update(self.static_stats)
        
        # Add streaming stats with computed std deviation
        for name, stats in self.streaming_stats.items():
            if stats['count'] > 1:
                variance = stats['M2'] / (stats['count'] - 1)
                std = np.sqrt(variance)
            else:
                std = 0.0
                
            final_stats[name] = {
                'mean': stats['mean'],
                'std': std,
                'min': stats['min'],
                'max': stats['max'],
                'count': stats['count'],
                'samples': np.array(self.sample_buffers[name]) if self.sample_buffers[name] else np.array([])
            }
        
        return final_stats

class ModuleStatsHook:
    """Hook that uses streaming statistics instead of accumulating all data"""
    def __init__(self, name, stats_collector):
        self.name = name
        self.stats_collector = stats_collector
    
    def __call__(self, module, input, output):
        if output is not None:
            # Use streaming statistics instead of accumulating all data
            self.stats_collector.update_streaming_stats(f"{self.name}.activation", output)

def attach_stats_hooks(model, stats_collector, module_types):
    hooks = []
    
    for name, module in model.named_modules():
        # Check by class name for the specific modules you want
        if module.__class__.__name__ in module_types:
            hook_fn = ModuleStatsHook(name, stats_collector)
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
            logger.debug(f"Attached hook to: {name}, type: {module.__class__.__name__}")
    
    return hooks

def collect_parameter_stats(net, stats_collector, log_tensor_suffixes=None):
    """Collect static statistics from model parameters and buffers"""
    # Collect parameter statistics (static - only once)
    for name, param in net.named_parameters():
        if param is not None:
            # Check if parameter name ends with any of the desired suffixes
            _, suffix = name.rsplit('.', 1) if '.' in name else ('', name)
            if suffix in log_tensor_suffixes:
                stats_collector.register_static_tensor(f"{name}", param.data)
    
    # Collect buffer statistics (static - only once)  
    for name, buf in net.named_buffers():
        if buf is not None:
            # Check if buffer name ends with any of the desired suffixes
            _, suffix = name.rsplit('.', 1) if '.' in name else ('', name)
            if suffix in log_tensor_suffixes:
                stats_collector.register_static_tensor(f"{name}", buf)

@torch.no_grad()
def inference(args, net, dataloader, num_batches=1, stats_collector=None):
    """Run inference for specified number of batches"""
    batch_count = 0
    
    for batch_data in dataloader['val']:
        if batch_count >= num_batches:
            break
        
        # Memory cleanup every few batches
        if batch_count > 0 and batch_count % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache() if args.device == 'cuda' else None
        
        if args.dataset_name in ['imagenet', 'imagenet-mini']:
            images, _ = batch_data
            images = images.to(args.device, non_blocking=True)
            _ = net(images)
        elif args.dataset_name == 'co3d':
            from src.models.depth.vggt.training.train_utils.general import copy_data_to_device
            batch_data = copy_data_to_device(batch_data, args.device, non_blocking=True)
            _ = net(images=batch_data["images"])
        
        batch_count += 1
        logger.debug(f"Processed batch {batch_count}/{num_batches}")
    
    return batch_count

def plot_histogram(data, title, bins=256):
    """Single histogram plot"""
    mean_val = np.mean(data)
    std_val = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=bins, edgecolor='black', color='skyblue', alpha=0.7)
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.4e}')
    ax.axvline(max_val, color='green', linestyle='dashed', linewidth=1.5, label=f'Max: {max_val:.4e}')
    ax.axvline(min_val, color='blue', linestyle='dashed', linewidth=1.5, label=f'Min: {min_val:.4e}')
    
    # Add std to legend without a line - create invisible line for legend entry
    ax.plot([], [], ' ', label=f'Std: {std_val:.4e}')
    
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    fig.tight_layout() 
    return fig

def plot_comparison_histogram(original_data, quantized_data, title, bins=256):
    """Comparison histogram with overlay"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot both histograms with transparency
    ax.hist(original_data, bins=bins, alpha=0.6, color='blue', label='Original', density=True)
    ax.hist(quantized_data, bins=bins, alpha=0.6, color='red', label='Quantized', density=True)
    
    # Statistics for original
    orig_mean = np.mean(original_data)
    orig_std = np.std(original_data)
    
    # Statistics for quantized
    quant_mean = np.mean(quantized_data)
    quant_std = np.std(quantized_data)
    
    # Add vertical lines for means
    ax.axvline(orig_mean, color='blue', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(quant_mean, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_title(f'{title} - Comparison')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add text with statistics
    stats_text = f'Original: μ={orig_mean:.4e}, σ={orig_std:.4e}\nQuantized: μ={quant_mean:.4e}, σ={quant_std:.4e}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    
    fig.tight_layout()
    return fig

def get_layer_and_suffix(name):
    parts = name.split('.')
    if len(parts) == 1:
        return 'root', parts[0]
    return '.'.join(parts[:-1]), parts[-1]

def histogram_from_stats(stats_collector, writer, step=0, model_type="", 
                        log_tensor_suffixes=None, image_suffixes=None, scalar_suffixes=None):
    """Generate histograms from collected statistics"""
    logger.debug(f"Logging {model_type} histograms to TensorBoard...")
    
    def log_tensor_stats(writer, tag_prefix, suffix, stats, step, model_type):
        if not stats or ('samples' not in stats and 'mean' not in stats):
            return

        tag_name = f"{tag_prefix}/{suffix}_{model_type}" if model_type else f"{tag_prefix}/{suffix}"
        
        # Log scale values as scalars (for quantization scales)
        if suffix in scalar_suffixes and 'samples' in stats and len(stats['samples']) > 0:
            scalar_value = float(stats['samples'][0] if hasattr(stats['samples'], '__len__') else stats['mean'])
            writer.add_scalar(tag_name, scalar_value, step)
        else:
            # Log histogram data to TensorBoard
            if 'samples' in stats and len(stats['samples']) > 0:
                writer.add_histogram(tag_name, stats['samples'], step)
                
                # Create histogram images for selected tensors
                if suffix in image_suffixes and len(stats['samples']) > 10:
                    fig = plot_histogram(stats['samples'], f"{tag_prefix}/{suffix}_{model_type}")
                    writer.add_figure(f"{tag_name}_image", fig, step)
                    plt.close(fig)
    
    all_stats = stats_collector.get_final_stats()
    logger.debug(f"Retrieved {len(all_stats)} stats for {model_type} model")
    
    for tensor_name, stats in all_stats.items():
        layer, suffix = get_layer_and_suffix(tensor_name)
        
        if suffix in log_tensor_suffixes:
            tag_prefix = f"{layer}"
            log_tensor_stats(writer, tag_prefix, suffix, stats, step, model_type)
            logger.debug(f"Logged stats for {tensor_name} under {tag_prefix}/{suffix}")
    
    # Force write to disk to prevent data loss
    writer.flush()
    logger.debug(f"Completed histogram generation for {model_type} model")

def compare_stats(original_stats, quantized_stats, writer, step=0, 
                 comparison_suffixes=None, image_suffixes=None):
    """Create comparison histograms between original and quantized models"""
    logger.debug("Creating comparison histograms...")
    
    # Get final stats from both collectors
    original_final = original_stats.get_final_stats()
    quantized_final = quantized_stats.get_final_stats()
    
    # Find common tensors between both models
    original_tensors = set(original_final.keys())
    quantized_tensors = set(quantized_final.keys())
    common_tensors = original_tensors.intersection(quantized_tensors)
    
    logger.info(f"Found {len(common_tensors)} common tensors for comparison")
    
    comparison_count = 0
    for tensor_name in common_tensors:
        layer, suffix = get_layer_and_suffix(tensor_name)
        
        if suffix in comparison_suffixes and suffix in image_suffixes:
            original_stats_data = original_final[tensor_name]
            quantized_stats_data = quantized_final[tensor_name]
            
            # Only create comparison if both have sample data
            if ('samples' in original_stats_data and len(original_stats_data['samples']) > 10 and
                'samples' in quantized_stats_data and len(quantized_stats_data['samples']) > 10):
                
                tag_name = f"{layer}/{suffix}_comparison"
                fig = plot_comparison_histogram(
                    original_stats_data['samples'], quantized_stats_data['samples'], 
                    f"{layer}/{suffix}_comparison"
                )
                writer.add_figure(tag_name, fig, step)
                plt.close(fig)
                
                comparison_count += 1
                logger.debug(f"Created comparison histogram for {tensor_name}")
                
                # Trigger garbage collection every 10 comparisons to keep memory usage low
                if comparison_count % 10 == 0:
                    gc.collect()
    
    logger.info(f"Created {comparison_count} comparison histograms")

def collect_model_stats(args, net, dataloader, model_type, num_inference_batches=10, module_types=None, log_tensor_suffixes=None):
    """Collect statistics for a single model using optimized approach"""
    logger.info(f"Collecting statistics for {model_type} model...")
    
    stats_collector = StatsCollector(max_samples_per_tensor=1000, sample_ratio=0.01)  # Reduced from 5000/0.05
    
    # Collect parameter and buffer statistics (static - only once)
    logger.debug("Collecting parameter and buffer statistics...")
    collect_parameter_stats(net, stats_collector, log_tensor_suffixes)
    
    # Attach hooks for activation statistics (dynamic - streaming)
    logger.debug("Attaching hooks for streaming activation statistics...")
    hooks = attach_stats_hooks(net, stats_collector, module_types)
    
    logger.debug(f"Model inference started for {num_inference_batches} batches...")
    batches_processed = inference(args, net, dataloader, num_batches=num_inference_batches, stats_collector=stats_collector)
    logger.info(f"Processed {batches_processed} batches for {model_type} activation statistics")
    
    for hook in hooks:
        hook.remove()
    
    return stats_collector

def run_stats(args):
    """Main stats collection function"""
    MODULE_TYPES = ['Conv2d', 'Linear', 'QConv2d', 'QLinear', 'LSQ_quantizer', 'ReLU', 'MinMax_quantizer']
    # Suffixes from static parameters (weights, scale), or dynamic tensors (activation)
    LOG_TENSOR_SUFFIXES = ['weight', 'bias', 'activation', 'scale', 'w_scale', 'x_scale']
    IMAGE_SUFFIXES = ['weight', 'activation']
    SCALAR_SUFFIXES = ['scale', 'w_scale', 'x_scale']
    COMPARISON_SUFFIXES = ['weight', 'activation']
    
    num_inference_batches = max(1, int(1 / args.batch_size))

    logger.debug("Run STATS arguments:\n", args)
    logger.info(f"Using GPU rank {args.gpu_rank} for stats")
    logger.info(f"Number of inference batches: {num_inference_batches}")

    # Setup dataloader once for reuse
    logger.info("Setting up dataloader...")
    pin_memory = True if args.device == 'cuda' else False
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory=pin_memory, DDP_mode=False, model=args.model)
    logger.info("Dataloader setup completed")

    # Setup TensorBoard writer
    args.ex_name = make_ex_name(args)
    save_path = os.path.join(args.save_path, "stats", args.dataset_name, args.ex_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        logger.info(f"Deleted existing stats directory: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(save_path)

    logger.info("=" * 50)
    logger.info("COLLECTING ORIGINAL MODEL STATISTICS")
    logger.info("=" * 50)
    
    original_net = create_model(args)
    original_net = original_net.to(args.device)
    original_net.eval()
    
    original_stats = collect_model_stats(args, original_net, dataloader, "original", num_inference_batches, MODULE_TYPES, LOG_TENSOR_SUFFIXES)
    
    # Force memory cleanup before histogram generation
    gc.collect()
    
    histogram_from_stats(original_stats, writer, step=0, model_type="original", 
                        log_tensor_suffixes=LOG_TENSOR_SUFFIXES, 
                        image_suffixes=IMAGE_SUFFIXES, 
                        scalar_suffixes=SCALAR_SUFFIXES)
    
    # Check disk space before continuing
    disk_usage = shutil.disk_usage(args.save_path)
    free_gb = disk_usage.free / (1024**3)
    logger.info(f"Disk space available: {free_gb:.2f} GB")
    
    # Free GPU memory
    del original_net
    torch.cuda.empty_cache() if args.device == 'cuda' else None
    gc.collect()
    
    logger.info("=" * 50)
    logger.info("COLLECTING QUANTIZED MODEL STATISTICS")
    logger.info("=" * 50)
    
    quantized_net = run_load_model(args)
    quantized_net.eval()
    
    quantized_stats = collect_model_stats(args, quantized_net, dataloader, "quantized", num_inference_batches, MODULE_TYPES, LOG_TENSOR_SUFFIXES)
    histogram_from_stats(quantized_stats, writer, step=0, model_type="quantized",
                        log_tensor_suffixes=LOG_TENSOR_SUFFIXES, 
                        image_suffixes=IMAGE_SUFFIXES, 
                        scalar_suffixes=SCALAR_SUFFIXES)
    
    # Create comparison histograms (this needs the sample data)
    logger.info("=" * 50)
    logger.info("CREATING COMPARISON HISTOGRAMS")
    logger.info("=" * 50)
    
    compare_stats(original_stats, quantized_stats, writer, step=0,
                 comparison_suffixes=COMPARISON_SUFFIXES, 
                 image_suffixes=IMAGE_SUFFIXES)
    
    # Free GPU memory
    del quantized_net
    torch.cuda.empty_cache() if args.device == 'cuda' else None
    gc.collect()
    
    writer.close()
    logger.info(f"Run STATS completed successfully. TensorBoard logs saved to {save_path}")
    
    # Log final statistics
    original_final = original_stats.get_final_stats()
    quantized_final = quantized_stats.get_final_stats()
    logger.info(f"Original - Total tensors: {len(original_final)}")
    logger.info(f"Quantized - Total tensors: {len(quantized_final)}")