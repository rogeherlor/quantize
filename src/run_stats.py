import os
import time
import shutil
import numpy as np
import torch

from src.logger import logger

from src.run_qat import run_load_model
from src.utils import setup_dataloader
from src.make_ex_name import *
from src.models.model import create_model
from src.quantizer.uniform.lsq import LSQ_quantizer

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

class StatsCollector:
    def __init__(self):
        self.static_stats = {}  # For weights, biases, buffers (registered once)
        self.dynamic_stats = {}  # For activations (accumulated across inferences)
    
    def reset(self):
        self.static_stats = {}
        self.dynamic_stats = {}
    
    def reset_dynamic_only(self):
        """Reset only dynamic stats while keeping static ones"""
        self.dynamic_stats = {}
    
    def register_static_tensor(self, name, tensor):
        """Register static tensors (weights, biases) - only once"""
        if name not in self.static_stats and tensor is not None and tensor.numel() > 0:
            data = tensor.detach().cpu().numpy().flatten()
            self.static_stats[name] = data
    
    def register_dynamic_tensor(self, name, tensor):
        """Register dynamic tensors (activations) - accumulate across inferences"""
        if tensor is not None and tensor.numel() > 0:
            data = tensor.detach().cpu().numpy().flatten()
            if name not in self.dynamic_stats:
                self.dynamic_stats[name] = []
            self.dynamic_stats[name].append(data)
    
    def finalize_dynamic_stats(self):
        """Concatenate all accumulated dynamic stats"""
        finalized = {}
        for name, data_list in self.dynamic_stats.items():
            if data_list:
                finalized[name] = np.concatenate(data_list)
        return finalized
    
    @property
    def stats(self):
        """Get all stats (static + finalized dynamic)"""
        all_stats = self.static_stats.copy()
        all_stats.update(self.finalize_dynamic_stats())
        return all_stats

class ModuleStatsHook:
    def __init__(self, name, stats_collector):
        self.name = name
        self.stats_collector = stats_collector
    
    def __call__(self, module, input, output):
        if output is not None:
            self.stats_collector.register_dynamic_tensor(f"{self.name}.activation", output)

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

def collect_parameter_stats(net, stats_collector):
    """Collect static statistics from model parameters and buffers"""
    # Collect parameter statistics (static - only once)
    for name, param in net.named_parameters():
        if param is not None:
            stats_collector.register_static_tensor(f"{name}", param.data)
    
    # Collect buffer statistics (static - only once)
    for name, buf in net.named_buffers():
        if buf is not None:
            stats_collector.register_static_tensor(f"{name}", buf)

@torch.no_grad()
def inference(args, net, num_batches=1):
    """Run inference for specified number of batches"""
    pin_memory = True if args.device == 'cuda' else False
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory=pin_memory, DDP_mode=False, model=args.model)
    
    batch_count = 0
    for batch_data in dataloader['val']:
        if batch_count >= num_batches:
            break
        
        if args.dataset_name in ['imagenet', 'imagenet-mini']:
            images, _ = batch_data
            images = images.to(args.device)
            net(images)
        elif args.dataset_name == 'co3d':
            from src.models.depth.vggt.training.train_utils.general import copy_data_to_device
            batch_data = copy_data_to_device(batch_data, args.device, non_blocking=True)
            net(images=batch_data["images"])
            
        batch_count += 1
        logger.debug(f"Processed batch {batch_count}/{num_batches}")
    
    return batch_count

def plot_histogram(data, title, bins=256):
    """Single histogram plot"""
    mean_val = np.mean(data)
    min_val = np.min(data)
    max_val = np.max(data)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=bins, edgecolor='black', color='skyblue', alpha=0.7)
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.4e}')
    ax.axvline(max_val, color='green', linestyle='dashed', linewidth=1.5, label=f'Max: {max_val:.4e}')
    ax.axvline(min_val, color='blue', linestyle='dashed', linewidth=1.5, label=f'Min: {min_val:.4e}')
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
    logger.debug(f"Logging {model_type} histograms from collected statistics to TensorBoard...")
    
    def log_tensor(writer, tag_prefix, suffix, data, step, model_type):
        if data is None or data.size == 0:
            return

        tag_name = f"{tag_prefix}/{suffix}_{model_type}" if model_type else f"{tag_prefix}/{suffix}"
        
        # Log scale values as scalars
        if suffix in scalar_suffixes:
            # For scale tensors, log mean value as scalar
            scalar_value = float(data.item() if hasattr(data, 'item') else data[0])
            writer.add_scalar(tag_name, scalar_value, step)
        else:
            # Always log histogram data to TensorBoard for non-scale tensors
            writer.add_histogram(tag_name, data, step)
            
            # Only create and log histogram images for non-scale tensors
            if suffix in image_suffixes:
                fig = plot_histogram(data, f"{tag_prefix}/{suffix}_{model_type}")
                writer.add_figure(f"{tag_name}_image", fig, step)
                plt.close(fig)
    
    all_stats = stats_collector.stats
    for tensor_name, data in all_stats.items():
        layer, suffix = get_layer_and_suffix(tensor_name)
        
        if suffix in log_tensor_suffixes:
            tag_prefix = f"{layer}"
            log_tensor(writer, tag_prefix, suffix, data, step, model_type)
            logger.debug(f"Logged {model_type} {'scalar' if suffix in scalar_suffixes else 'histogram'} for {tensor_name}")

def compare_stats(original_stats, quantized_stats, writer, step=0, 
                 comparison_suffixes=None, image_suffixes=None):
    """Create comparison histograms between original and quantized models"""
    logger.debug("Creating comparison histograms...")
    
    # Find common tensors between both models
    original_tensors = set(original_stats.stats.keys())
    quantized_tensors = set(quantized_stats.stats.keys())
    common_tensors = original_tensors.intersection(quantized_tensors)
    
    logger.info(f"Found {len(common_tensors)} common tensors for comparison")
    
    for tensor_name in common_tensors:
        layer, suffix = get_layer_and_suffix(tensor_name)
        
        if suffix in comparison_suffixes and suffix in image_suffixes:
            original_data = original_stats.stats[tensor_name]
            quantized_data = quantized_stats.stats[tensor_name]
            
            tag_name = f"{layer}/{suffix}_comparison"
            fig = plot_comparison_histogram(
                original_data, quantized_data, 
                f"{layer}/{suffix}_comparison"
            )
            writer.add_figure(tag_name, fig, step)
            plt.close(fig)
            
            logger.debug(f"Created comparison histogram for {tensor_name}")

def collect_model_stats(args, net, model_type, num_inference_batches=10, module_types=None):
    """Collect statistics for a single model"""
    logger.info(f"Collecting statistics for {model_type} model...")
    
    stats_collector = StatsCollector()
    
    # Collect parameter and buffer statistics (static - only once)
    logger.debug("Collecting parameter and buffer statistics...")
    collect_parameter_stats(net, stats_collector)
    
    # Attach hooks for activation statistics (dynamic - accumulated)
    logger.debug("Attaching hooks for activation statistics...")
    hooks = attach_stats_hooks(net, stats_collector, module_types)
    
    # Run multiple inferences to accumulate activation statistics
    logger.debug(f"Model inference started for {num_inference_batches} batches...")
    batches_processed = inference(args, net, num_batches=num_inference_batches)
    logger.info(f"Processed {batches_processed} batches for {model_type} activation statistics")
    
    for hook in hooks:
        hook.remove()
    
    return stats_collector

def run_stats(args):
    MODULE_TYPES = ['Conv2d', 'Linear', 'QConv2d', 'QLinear', 'LSQ_quantizer', 'ReLU', 'MinMax_quantizer']
    # Suffixes from static parameters (weights, scale), or dynamic tensors (activation)
    LOG_TENSOR_SUFFIXES = ['weight', 'bias', 'activation', 'scale', 'w_scale', 'x_scale']
    IMAGE_SUFFIXES = ['weight', 'activation']
    SCALAR_SUFFIXES = ['scale', 'w_scale', 'x_scale']
    COMPARISON_SUFFIXES = ['weight', 'activation']
    
    num_inference_batches = max(1, int(1 / args.batch_size))

    logger.debug("Run STATS arguments:\n", args)
    logger.info(f"Using GPU rank {args.gpu_rank} for stats")

    # Setup TensorBoard writer
    args.ex_name = make_ex_name(args)
    save_path = os.path.join(args.save_path, "stats", args.dataset_name, args.ex_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        logger.info(f"Deleted existing stats directory: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(save_path)

    # Collect statistics for original model
    logger.info("=" * 50)
    logger.info("COLLECTING ORIGINAL MODEL STATISTICS")
    logger.info("=" * 50)
    
    original_net = create_model(args)
    original_net = original_net.to(args.device)
    original_net.eval()
    
    original_stats = collect_model_stats(args, original_net, "original", num_inference_batches, MODULE_TYPES)
    histogram_from_stats(original_stats, writer, step=0, model_type="original", 
                        log_tensor_suffixes=LOG_TENSOR_SUFFIXES, 
                        image_suffixes=IMAGE_SUFFIXES, 
                        scalar_suffixes=SCALAR_SUFFIXES)
    
    # Free GPU memory
    del original_net
    torch.cuda.empty_cache() if args.device == 'cuda' else None
    
    # Collect statistics for quantized model
    logger.info("=" * 50)
    logger.info("COLLECTING QUANTIZED MODEL STATISTICS")
    logger.info("=" * 50)
    
    quantized_net = run_load_model(args)
    quantized_net.eval()
    
    quantized_stats = collect_model_stats(args, quantized_net, "quantized", num_inference_batches, MODULE_TYPES)
    histogram_from_stats(quantized_stats, writer, step=0, model_type="quantized",
                        log_tensor_suffixes=LOG_TENSOR_SUFFIXES, 
                        image_suffixes=IMAGE_SUFFIXES, 
                        scalar_suffixes=SCALAR_SUFFIXES)
    
    # Create comparison histograms
    logger.info("=" * 50)
    logger.info("CREATING COMPARISON HISTOGRAMS")
    logger.info("=" * 50)
    
    compare_stats(original_stats, quantized_stats, writer, step=0,
                 comparison_suffixes=COMPARISON_SUFFIXES, 
                 image_suffixes=IMAGE_SUFFIXES)
    
    # Free remaining GPU memory
    del quantized_net
    torch.cuda.empty_cache() if args.device == 'cuda' else None
    
    writer.close()
    logger.info(f"Run STATS completed successfully. TensorBoard logs saved to {save_path}")
    logger.info(f"Original - Static: {len(original_stats.static_stats)}, Dynamic: {len(original_stats.dynamic_stats)}")
    logger.info(f"Quantized - Static: {len(quantized_stats.static_stats)}, Dynamic: {len(quantized_stats.dynamic_stats)}")
