import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import gc
from collections import defaultdict
import random

from src.logger import logger

from src.run_qat import run_load_model
from src.utils import setup_dataloader, stepsize_init
from src.make_ex_name import *
from src.models.model import create_model
from src.quantizer.uniform.lsq import LSQ_quantizer
from src.utils_layer_selection import expand_layer_names, filter_stats_by_layers, filter_snapshots_by_layers

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class StatsCollector:
    """Stats collector that uses streaming statistics and sampling for memory efficiency"""
    def __init__(self, max_samples_per_tensor=10000, sample_ratio=0.1):
        self.static_stats = {}  # For weights, biases, buffers (computed once)
        self.streaming_stats = {}  # For running statistics (mean, std, min, max)
        self.max_samples_per_tensor = max_samples_per_tensor
        self.sample_ratio = sample_ratio  # Ratio of values to sample from each tensor
        self.sample_buffers = defaultdict(list)  # Temporary buffers for sampled values
        self.snapshots = {}  # Snapshots for structure visualization
        
    def reset(self):
        self.static_stats = {}
        self.streaming_stats = {}
        self.sample_buffers.clear()
        self.snapshots.clear()
    
    def register_static_tensor(self, name, tensor):
        """Register static tensors (weights, biases) - computed once"""
        if name not in self.static_stats and tensor is not None and tensor.numel() > 0:
            with torch.no_grad():
                # Capture snapshot for visualization (if not too large)
                if name not in self.snapshots and tensor.numel() < 10_000_000:
                    self.snapshots[name] = tensor.detach().cpu().numpy()
                
                flat_tensor = tensor.flatten()
                
                # Optimization: Compute all stats on GPU and transfer once
                mean_t = torch.mean(flat_tensor)
                std_t = torch.std(flat_tensor)
                min_t = torch.min(flat_tensor)
                max_t = torch.max(flat_tensor)
                
                stats = torch.stack([mean_t, std_t, min_t, max_t]).cpu().numpy()
                
                self.static_stats[name] = {
                    'mean': float(stats[0]),
                    'std': float(stats[1]),
                    'min': float(stats[2]),
                    'max': float(stats[3]),
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
            # Optimization: Use randint instead of randperm
            indices = torch.randint(0, numel, (max_samples,), device=flat_tensor.device)
            return flat_tensor[indices].cpu().numpy()
    
    def update_streaming_stats(self, name, tensor):
        """Update streaming statistics for dynamic tensors"""
        if tensor is None or tensor.numel() == 0:
            return
            
        with torch.no_grad():
            # Capture snapshot for visualization (first valid tensor encountered)
            if name not in self.snapshots:
                # If batched, take first element
                snapshot = tensor
                if tensor.dim() > 0 and tensor.shape[0] > 0:
                    snapshot = tensor[0] # Take first sample in batch
                
                if snapshot.numel() < 10_000_000:
                    self.snapshots[name] = snapshot.detach().cpu().numpy()

            flat_tensor = tensor.flatten()
            
            # Optimization: Compute all stats on GPU first to avoid multiple CPU-GPU syncs
            t_mean = torch.mean(flat_tensor)
            t_std = torch.std(flat_tensor)
            t_min = torch.min(flat_tensor)
            t_max = torch.max(flat_tensor)
            
            # Move to CPU in a single transfer
            stats_vec = torch.stack([t_mean, t_std, t_min, t_max]).cpu().numpy()
            
            batch_mean = float(stats_vec[0])
            batch_std = float(stats_vec[1])
            batch_min = float(stats_vec[2])
            batch_max = float(stats_vec[3])
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
        # input is a tuple of tensors in PyTorch hooks
        if input is not None and len(input) > 0:
            # Get the first input tensor (usually the main input)
            input_tensor = input[0]
            if isinstance(input_tensor, torch.Tensor):
                self.stats_collector.update_streaming_stats(f"{self.name}.activation_in", input_tensor)
        if output is not None and isinstance(output, torch.Tensor):
            self.stats_collector.update_streaming_stats(f"{self.name}.activation_out", output)

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

def plot_histogram(data, title, bins=256, clip_value=None):
    """Single histogram plot with optional gradient clipping"""
    original_data = data
    n_clipped = 0
    
    # Apply clipping if specified (for gradients)
    if clip_value is not None and '.grad' in title:
        n_clipped = np.sum((np.abs(data) > clip_value))
        data = np.clip(data, -clip_value, clip_value)
    
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
    
    # Add clipping info if applicable
    if n_clipped > 0:
        clip_pct = (n_clipped / len(original_data)) * 100
        ax.plot([], [], ' ', label=f'Clipped: {n_clipped} ({clip_pct:.2f}%)')
    
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
    
    orig_mean = np.mean(original_data)
    orig_std = np.std(original_data)
    
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
                        log_tensor_suffixes=None, image_suffixes=None, scalar_suffixes=None, gradient_clip_value=None):
    """Generate histograms from collected statistics"""
    logger.debug(f"Logging {model_type} histograms to TensorBoard...")
    
    def log_tensor_stats(writer, tag_prefix, suffix, stats, step, model_type, gradient_clip_value):
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
                # Apply gradient clipping for histogram visualization
                samples = stats['samples']
                if gradient_clip_value is not None and suffix == 'grad':
                    samples = np.clip(samples, -gradient_clip_value, gradient_clip_value)
                writer.add_histogram(tag_name, samples, step)
                
                # Create histogram images for selected tensors (including gradients)
                if (suffix in image_suffixes or suffix == 'grad') and len(stats['samples']) > 10:
                    fig = plot_histogram(stats['samples'], f"{tag_prefix}/{suffix}_{model_type}", clip_value=gradient_clip_value)
                    writer.add_figure(f"{tag_name}_image", fig, step)
                    plt.close(fig)
    
    all_stats = stats_collector.get_final_stats()
    logger.debug(f"Retrieved {len(all_stats)} stats for {model_type} model")
    
    for tensor_name, stats in all_stats.items():
        layer, suffix = get_layer_and_suffix(tensor_name)
        
        if suffix in log_tensor_suffixes or suffix == 'grad':
            tag_prefix = f"{layer}"
            log_tensor_stats(writer, tag_prefix, suffix, stats, step, model_type, gradient_clip_value)
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
    logger.info(f"Attached {len(hooks)} hooks for activation collection")
    
    logger.debug(f"Model inference started for {num_inference_batches} batches...")
    batches_processed = inference(args, net, dataloader, num_batches=num_inference_batches, stats_collector=stats_collector)
    logger.info(f"Processed {batches_processed} batches for {model_type} activation statistics")
    
    for hook in hooks:
        hook.remove()
    
    return stats_collector

def collect_gradient_stats(args, net, dataloader, stats_collector, num_inference_batches=10):
    """Collect gradient statistics across multiple batches using streaming statistics (same as activations)"""
    logger.info(f"Collecting gradient statistics across {num_inference_batches} batches...")
    
    net.train() # Enable gradients
    
    # Get iterator from train loader, else val loader
    loader = dataloader.get('train', dataloader.get('val'))
    
    # Load loss configuration once for CO3D
    if args.dataset_name == 'co3d':
        from hydra import compose, initialize_config_dir
        config_dir = os.path.join(os.path.dirname(__file__), 'models/depth/vggt/training/config')
        config_dir = os.path.abspath(config_dir)
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="default")
            loss_conf = cfg.loss
        from hydra.utils import instantiate
        from src.models.depth.vggt.training.train_utils.general import copy_data_to_device
        from src.models.depth.vggt.training.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
    
    # Collect gradients across multiple batches (same as activations)
    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= num_inference_batches:
            break
            
        net.zero_grad()
        
        # Move to device and forward/backward with actual loss
        if args.dataset_name in ['imagenet', 'imagenet-mini']:
            images, labels = batch_data
            images = images.to(args.device)
            labels = labels.to(args.device)
            output = net(images)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, labels)
        elif args.dataset_name == 'co3d':
            # Normalize batch data (same as in training)
            with torch.cuda.amp.autocast(enabled=False):
                normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
                    normalize_camera_extrinsics_and_points_batch(
                        extrinsics=batch_data["extrinsics"],
                        cam_points=batch_data["cam_points"],
                        world_points=batch_data["world_points"],
                        depths=batch_data["depths"],
                        point_masks=batch_data["point_masks"],
                    )
                batch_data["extrinsics"] = normalized_extrinsics
                batch_data["cam_points"] = normalized_cam_points
                batch_data["world_points"] = normalized_world_points
                batch_data["depths"] = normalized_depths
            
            batch_data = copy_data_to_device(batch_data, args.device, non_blocking=True)
            
            # Forward pass
            output = net(images=batch_data["images"])
            
            # Filter out outputs for heads that are disabled in loss config
            if loss_conf.point is None and "world_points" in output:
                del output["world_points"]
            if loss_conf.track is None and "track" in output:
                del output["track"]
            
            # Use the actual training loss function
            loss_fn = instantiate(loss_conf, _recursive_=False)
            loss_dict = loss_fn(output, batch_data)
            loss = loss_dict["objective"]
        
        loss.backward()
        
        logger.info(f"Batch {batch_idx + 1}/{num_inference_batches}: loss = {loss.item():.4f}")
        
        # Collect parameter gradients using streaming statistics (same approach as activations)
        for name, param in net.named_parameters():
            if param.grad is not None:
                stats_collector.update_streaming_stats(f"{name}.grad", param.grad)
    
    net.zero_grad()
    # Restore eval mode
    net.eval()
    logger.info(f"Completed gradient collection across {num_inference_batches} batches")

def plot_layer_box_plots(stats_collector, save_dir, prefix="", gradient_clip_value=None, layer_filter=None):
    """Generate Box and Whisker plots for weights, activations, and gradients"""
    stats = stats_collector.get_final_stats()
    
    # Filter stats if layer_filter is provided
    stats = filter_stats_by_layers(stats, layer_filter)
    
    # Define categories to plot
    categories = {
        'Weights': '.weight',
        'Activations': '.activation_out',
        'Gradients': '.grad'
    }
    
    for cat_name, suffix in categories.items():
        filtered_data = []
        labels = []
        
        # Keep keys in the order they appear (insertion order in dict = execution order)
        # This preserves the forward pass order from input to output
        sorted_keys = list(stats.keys())
        
        for key in sorted_keys:
            # Check if key matches the category
            if not key.endswith(suffix):
                continue
                
            s = stats[key]
            if 'samples' in s and len(s['samples']) > 0:
                samples = s['samples']
                # Apply gradient clipping for gradients
                if cat_name == 'Gradients' and gradient_clip_value is not None:
                    samples = np.clip(samples, -gradient_clip_value, gradient_clip_value)
                filtered_data.append(samples)
                # Clean label
                label = key.replace('.weight', '').replace('.activation_out', '').replace('.grad', '')
                labels.append(label)
        
        if not filtered_data:
            continue

        # Plot
        # Dynamic width based on number of layers
        width = max(12, len(filtered_data) * 0.25)
        fig, ax = plt.subplots(figsize=(width, 8))
        
        # Create boxplot
        ax.boxplot(filtered_data, labels=labels, showfliers=True, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                  medianprops=dict(color='red'))
        
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        title = f"Distribution of {cat_name} by Layer ({prefix})"
        if cat_name == 'Gradients' and gradient_clip_value is not None:
            title += f" [Clipped at ±{gradient_clip_value}]"
        ax.set_title(title)
        ax.set_ylabel('Value')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{prefix}_{cat_name.lower()}_boxplot.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {cat_name} boxplot to {save_path}")

def plot_3d_surface(stats_collector, save_dir, prefix="", gradient_clip_value=None, layer_filter=None):
    """Generate combined 3D surface plots with unified scale for comparison"""
    # Use snapshots if available, otherwise fallback to stats
    snapshots = getattr(stats_collector, 'snapshots', {})
    
    # Filter snapshots by layer specification
    snapshots = filter_snapshots_by_layers(snapshots, layer_filter)
    
    # Define categories to plot
    categories = {
        'Weights': '.weight',
        'Activations': '.activation_out',
        'Gradients': '.grad'
    }
    
    for cat_name, suffix in categories.items():
        logger.info(f"Generating combined 3D surface plot for {cat_name}...")
        
        # First pass: collect all valid layers and compute global z-range
        layer_data_list = []
        sorted_keys = sorted(list(snapshots.keys()))
        global_min = float('inf')
        global_max = float('-inf')
        
        for key in sorted_keys:
            if not key.endswith(suffix):
                continue
                
            data = snapshots[key]
            layer_name = key.replace(suffix, '')
            
            # Extract 2D data for plotting
            plot_data = None
            x_label = 'Dimension 1'
            y_label = 'Dimension 0'
            
            if data.ndim == 2:
                plot_data = data
                y_label = 'Dim 0 (Tokens/OutCh)'
                x_label = 'Dim 1 (EmbedDim/InCh)'
            elif data.ndim == 3:
                plot_data = data[0]
                y_label = 'Dim 1'
                x_label = 'Dim 2'
            elif data.ndim == 4:
                # For Conv: reshape [Out, In, H, W] -> [Out, In*H*W]
                out_ch, in_ch, h, w = data.shape
                plot_data = data.reshape(out_ch, in_ch * h * w)
                y_label = 'Out Channels'
                x_label = 'In*H*W'
            
            if plot_data is None or plot_data.size == 0:
                continue

            # Apply gradient clipping
            if cat_name == 'Gradients' and gradient_clip_value is not None:
                plot_data = np.clip(plot_data, -gradient_clip_value, gradient_clip_value)

            # Update global min/max for unified scale
            global_min = min(global_min, np.min(plot_data))
            global_max = max(global_max, np.max(plot_data))
            
            layer_data_list.append({
                'name': layer_name,
                'data': plot_data,
                'x_label': x_label,
                'y_label': y_label
            })
            
            # Limit to reasonable number of subplots
            if len(layer_data_list) >= 20:
                break
        
        if not layer_data_list:
            logger.info(f"No valid layers found for {cat_name}")
            continue
        
        # Create combined figure with subplots
        n_layers = len(layer_data_list)
        n_cols = min(4, n_layers)  # Max 4 columns
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        
        for idx, layer_info in enumerate(layer_data_list):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
            
            plot_data = layer_info['data']
            layer_name = layer_info['name']
            
            # Downsample if too large
            MAX_GRID = 50
            h, w = plot_data.shape
            stride_h = max(1, h // MAX_GRID)
            stride_w = max(1, w // MAX_GRID)
            plot_data_sub = plot_data[::stride_h, ::stride_w]
            
            # Coordinates
            x = np.arange(0, w, stride_w)[:plot_data_sub.shape[1]]
            y = np.arange(0, h, stride_h)[:plot_data_sub.shape[0]]
            X, Y = np.meshgrid(x, y)
            Z = plot_data_sub
            
            # Plot with unified color scale
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, 
                                  edgecolor='none', alpha=0.9,
                                  vmin=global_min, vmax=global_max)
            
            # Set unified z-axis limits
            ax.set_zlim(global_min, global_max)
            
            # Compact labels
            ax.set_xlabel(layer_info['x_label'], fontsize=7)
            ax.set_ylabel(layer_info['y_label'], fontsize=7)
            ax.set_zlabel('Value', fontsize=7)
            ax.tick_params(labelsize=6)
            
            # Shorter title
            title = layer_name.split('.')[-1] if '.' in layer_name else layer_name
            ax.set_title(title, fontsize=8, pad=5)
            
            # Adjust view
            ax.view_init(elev=25, azim=-60)
        
        # Add global title and colorbar
        main_title = f"{cat_name} - All Layers ({prefix})"
        if cat_name == 'Gradients' and gradient_clip_value is not None:
            main_title += f" [Clipped ±{gradient_clip_value}]"
        fig.suptitle(main_title, fontsize=14, fontweight='bold')
        
        # Add single colorbar for all subplots
        fig.colorbar(surf, ax=fig.get_axes(), shrink=0.6, aspect=20, pad=0.05)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        save_path = os.path.join(save_dir, f"{prefix}_{cat_name.lower()}_combined_3d.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated combined 3D plot for {cat_name} with {n_layers} layers (z-range: [{global_min:.4e}, {global_max:.4e}])")

def analyze_quantization_difficulty(stats_collector, save_dir, prefix="", layer_filter=None):
    """Analyze and rank layers by quantization difficulty"""
    stats = stats_collector.get_final_stats()
    
    # DEBUG: Log what we have before filtering
    logger.info(f"Stats before filtering: {len(stats)} total entries")
    activation_keys_before = [k for k in stats.keys() if '.activation_out' in k]
    logger.info(f"Activation keys before filtering: {len(activation_keys_before)}")
    if activation_keys_before:
        logger.info(f"Sample activation keys: {activation_keys_before[:5]}")
    
    # Filter stats if layer_filter is provided
    stats = filter_stats_by_layers(stats, layer_filter)
    
    # DEBUG: Log what we have after filtering
    logger.info(f"Stats after filtering: {len(stats)} total entries")
    activation_keys_after = [k for k in stats.keys() if '.activation_out' in k]
    logger.info(f"Activation keys after filtering: {len(activation_keys_after)}")
    if activation_keys_after:
        logger.info(f"Sample activation keys after filter: {activation_keys_after[:5]}")
    
    difficulties = []
    
    for key, stat in stats.items():
        # Focus on weights and activations
        if not (key.endswith('.weight') or key.endswith('.activation_out')):
            continue
        
        layer_name = key.rsplit('.', 1)[0]
        tensor_type = 'weight' if key.endswith('.weight') else 'activation'
        
        # Compute difficulty metrics
        std = stat.get('std', 0)
        mean = abs(stat.get('mean', 0))
        min_val = stat.get('min', 0)
        max_val = stat.get('max', 0)
        
        # Avoid division by zero
        range_val = max_val - min_val
        
        # Expert metrics for quantization difficulty:
        # 1. Dynamic range: large range needs more bits
        dynamic_range = np.log10(range_val + 1e-10)
        
        # 2. Coefficient of variation: high CV means uneven distribution
        cv = std / (abs(mean) + 1e-10)
        
        # 3. Outlier ratio: estimate from range vs std
        outlier_metric = range_val / (std + 1e-10)
        
        # Combined difficulty score (higher = harder to quantize)
        difficulty_score = dynamic_range * 0.3 + cv * 0.4 + np.log10(outlier_metric + 1) * 0.3
        
        difficulties.append({
            'layer': layer_name,
            'type': tensor_type,
            'difficulty': difficulty_score,
            'dynamic_range': dynamic_range,
            'cv': cv,
            'outlier_metric': outlier_metric,
            'std': std,
            'range': range_val
        })
    
    # Sort by difficulty
    difficulties.sort(key=lambda x: x['difficulty'], reverse=True)
    
    # Save report
    report_path = os.path.join(save_dir, f"{prefix}_quantization_difficulty_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"QUANTIZATION DIFFICULTY ANALYSIS - {prefix.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write("Layers ranked by difficulty (hardest first):\n")
        f.write("Higher scores indicate layers that are harder to quantize.\n\n")
        f.write("Difficulty Factors:\n")
        f.write("  - Dynamic Range: Large value ranges need more quantization levels\n")
        f.write("  - Coeff. of Variation (CV): High CV indicates uneven distributions\n")
        f.write("  - Outlier Metric: Presence of extreme values relative to typical values\n\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6}{'Layer':<50}{'Type':<12}{'Difficulty':<12}\n")
        f.write("-"*80 + "\n")
        
        for rank, d in enumerate(difficulties[:30], 1):  # Top 30
            f.write(f"{rank:<6}{d['layer']:<50}{d['type']:<12}{d['difficulty']:<12.3f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED METRICS FOR TOP 10 HARDEST LAYERS\n")
        f.write("="*80 + "\n")
        
        for rank, d in enumerate(difficulties[:10], 1):
            f.write(f"\n{rank}. {d['layer']} ({d['type']})\n")
            f.write(f"   Difficulty Score:  {d['difficulty']:.3f}\n")
            f.write(f"   Dynamic Range:     {d['dynamic_range']:.3f} (log scale)\n")
            f.write(f"   Coeff. Variation:  {d['cv']:.3f}\n")
            f.write(f"   Outlier Metric:    {d['outlier_metric']:.3f}\n")
            f.write(f"   Std Dev:           {d['std']:.4e}\n")
            f.write(f"   Value Range:       {d['range']:.4e}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        f.write("1. Focus on layers with high difficulty scores for mixed-precision\n")
        f.write("2. Layers with high CV may benefit from per-channel quantization\n")
        f.write("3. Layers with high outlier metrics may need outlier clipping\n")
        f.write("4. First/last layers often need higher precision (watch for them in top 10)\n")
        f.write("5. Compare original vs quantized to see which layers lose most accuracy\n")
    
    logger.info(f"Quantization difficulty analysis saved to {report_path}")
    return difficulties

def run_stats(args):
    """Main stats collection function"""
    MODULE_TYPES = ['Conv2d', 'Linear', 'QConv2d', 'QLinear', 'LSQ_quantizer', 'ReLU', 'MinMax_quantizer']
    # Suffixes from static parameters (weights, scale), or dynamic tensors (activation)
    LOG_TENSOR_SUFFIXES = ['weight', 'bias', 'activation_in', 'activation_out', 'scale', 'w_scale', 'x_scale']
    IMAGE_SUFFIXES = ['weight', 'activation_in', 'activation_out']
    SCALAR_SUFFIXES = ['scale', 'w_scale', 'x_scale']
    COMPARISON_SUFFIXES = ['weight', 'activation_in', 'activation_out']
    
    # Gradient clipping value for visualization (to handle outliers)
    GRADIENT_CLIP_VALUE = getattr(args, 'gradient_clip_value', 100)  # Default to 100
    logger.info(f"Using gradient clipping value: ±{GRADIENT_CLIP_VALUE} for visualization")
    
    # ==================== LAYER SELECTION CONFIGURATION ====================
    #   None                                           - Analyze all layers
    #   ["aggregator.frame_blocks.12", "camera_head"] - Multiple layer groups
    SELECTED_LAYERS_CONFIG = [
                "aggregator.frame_blocks.12", "aggregator.frame_blocks.13", 
                "aggregator.frame_blocks.14", "aggregator.frame_blocks.15",
                "aggregator.frame_blocks.16", "aggregator.frame_blocks.17",
                "aggregator.global_blocks.12", "aggregator.global_blocks.13",
                "aggregator.global_blocks.14", "aggregator.global_blocks.15",
                "aggregator.global_blocks.16", "aggregator.global_blocks.17"
            ]
    
    if SELECTED_LAYERS_CONFIG:
        logger.info(f"Layer selection enabled: {SELECTED_LAYERS_CONFIG}")
    else:
        logger.info("Analyzing ALL layers (no filtering)")
    # =======================================================================
    
    num_inference_batches = max(5, int(1 / args.batch_size))

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
    
    # Expand layer specifications to include all sub-layers
    original_layer_filter = expand_layer_names(original_net, SELECTED_LAYERS_CONFIG) if SELECTED_LAYERS_CONFIG else None
    
    original_stats = collect_model_stats(args, original_net, dataloader, "original", num_inference_batches, MODULE_TYPES, LOG_TENSOR_SUFFIXES)
    
    # Collect gradients for original model (same number of batches as activations)
    collect_gradient_stats(args, original_net, dataloader, original_stats, num_inference_batches)
    
    gc.collect()
    
    histogram_from_stats(original_stats, writer, step=0, model_type="original", 
                        log_tensor_suffixes=LOG_TENSOR_SUFFIXES, 
                        image_suffixes=IMAGE_SUFFIXES, 
                        scalar_suffixes=SCALAR_SUFFIXES,
                        gradient_clip_value=GRADIENT_CLIP_VALUE)
    
    # Plot box plots for original model
    plot_layer_box_plots(original_stats, save_path, prefix="original", gradient_clip_value=GRADIENT_CLIP_VALUE, layer_filter=original_layer_filter)
    
    # Plot 3D surface plots for original model
    plot_3d_surface(original_stats, save_path, prefix="original", gradient_clip_value=GRADIENT_CLIP_VALUE, layer_filter=original_layer_filter)
    
    # Analyze quantization difficulty for original model
    logger.info("Analyzing quantization difficulty for original model...")
    analyze_quantization_difficulty(original_stats, save_path, prefix="original", layer_filter=original_layer_filter)
    
    disk_usage = shutil.disk_usage(args.save_path)
    free_gb = disk_usage.free / (1024**3)
    logger.info(f"Disk space available: {free_gb:.2f} GB")
    
    del original_net
    torch.cuda.empty_cache() if args.device == 'cuda' else None
    gc.collect()
    
    logger.info("=" * 50)
    logger.info("COLLECTING QUANTIZED MODEL STATISTICS")
    logger.info("=" * 50)
    
    if args.dataset_name == 'co3d':
        from src.models.depth.qvggt import replace
        quantized_net = create_model(args)
        quantized_net = quantized_net.to(args.device)
        quantized_net = replace(args, quantized_net)
        stepsize_init(quantized_net, dataloader["train"], args.device, num_batches=args.init_num, dataset_name=args.dataset_name)
    else:
        quantized_net = run_load_model(args)
    
    torch.set_grad_enabled(True)
    quantized_net.eval()
    
    # Expand layer specifications for quantized model
    quantized_layer_filter = expand_layer_names(quantized_net, SELECTED_LAYERS_CONFIG) if SELECTED_LAYERS_CONFIG else None
    
    quantized_stats = collect_model_stats(args, quantized_net, dataloader, "quantized", num_inference_batches, MODULE_TYPES, LOG_TENSOR_SUFFIXES)
    
    # Collect gradients for quantized model (same number of batches as activations)
    collect_gradient_stats(args, quantized_net, dataloader, quantized_stats, num_inference_batches)
    
    histogram_from_stats(quantized_stats, writer, step=0, model_type="quantized",
                        log_tensor_suffixes=LOG_TENSOR_SUFFIXES, 
                        image_suffixes=IMAGE_SUFFIXES, 
                        scalar_suffixes=SCALAR_SUFFIXES,
                        gradient_clip_value=GRADIENT_CLIP_VALUE)
    
    # Plot box plots for quantized model
    plot_layer_box_plots(quantized_stats, save_path, prefix="quantized", gradient_clip_value=GRADIENT_CLIP_VALUE, layer_filter=quantized_layer_filter)
    
    # Plot 3D surface plots for quantized model
    plot_3d_surface(quantized_stats, save_path, prefix="quantized", gradient_clip_value=GRADIENT_CLIP_VALUE, layer_filter=quantized_layer_filter)
    
    # Analyze quantization difficulty for quantized model
    logger.info("Analyzing quantization difficulty for quantized model...")
    analyze_quantization_difficulty(quantized_stats, save_path, prefix="quantized", layer_filter=quantized_layer_filter)
    
    logger.info("=" * 50)
    logger.info("CREATING COMPARISON HISTOGRAMS")
    logger.info("=" * 50)
    
    compare_stats(original_stats, quantized_stats, writer, step=0,
                 comparison_suffixes=COMPARISON_SUFFIXES, 
                 image_suffixes=IMAGE_SUFFIXES)
    
    del quantized_net
    torch.cuda.empty_cache() if args.device == 'cuda' else None
    gc.collect()
    
    writer.close()
    logger.info(f"Run STATS completed successfully. TensorBoard logs saved to {save_path}")
    
    original_final = original_stats.get_final_stats()
    quantized_final = quantized_stats.get_final_stats()
    logger.info(f"Original - Total tensors: {len(original_final)}")
    logger.info(f"Quantized - Total tensors: {len(quantized_final)}")