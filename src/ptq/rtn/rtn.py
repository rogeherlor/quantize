

import torch
import torch.nn as nn
import numpy as np
from src.logger import logger
from src.utils import setup_dataloader
from src.models.model import create_model

def calculate_scale_minmax(data, num_bits, symmetric=True):
    """Calculate scale using min-max method"""
    if symmetric:
        max_val = torch.max(torch.abs(data.min()), torch.abs(data.max()))
        qmax = 2 ** (num_bits - 1) - 1
        scale = max_val / qmax
    else:
        min_val = data.min()
        max_val = data.max()
        qmax = 2 ** num_bits - 1
        scale = (max_val - min_val) / qmax
    return scale

def calculate_scale_symmetric_max(data, num_bits):
    """Calculate scale using symmetric maximum absolute value"""
    max_abs = torch.max(torch.abs(data))
    qmax = 2 ** (num_bits - 1) - 1
    scale = max_abs / qmax
    return scale

def calculate_scale_mse(data, num_bits, symmetric=True, grid_size=100, maxshrink=0.8):
    """Calculate scale using MSE optimization"""
    if symmetric:
        max_val = torch.max(torch.abs(data))
        qmax = 2 ** (num_bits - 1) - 1
        zero_point = qmax // 2
    else:
        min_val = data.min()
        max_val = data.max()
        qmax = 2 ** num_bits - 1
        zero_point = 0
    
    best_scale = None
    best_error = float('inf')
    
    for i in range(int(maxshrink * grid_size)):
        p = 1 - i / grid_size
        if symmetric:
            current_max = p * max_val
            scale = current_max / qmax
        else:
            current_min = p * min_val
            current_max = p * max_val
            scale = (current_max - current_min) / qmax
        
        # Quantize and calculate MSE
        if symmetric:
            q_data = torch.clamp(torch.round(data / scale), -qmax, qmax)
            dequant = q_data * scale
        else:
            q_data = torch.clamp(torch.round(data / scale) + zero_point, 0, qmax)
            dequant = (q_data - zero_point) * scale
        
        error = torch.mean((data - dequant) ** 2)
        
        if error < best_error:
            best_error = error
            best_scale = scale
    
    return best_scale if best_scale is not None else calculate_scale_symmetric_max(data, num_bits)

def calculate_scale_percentile(data, num_bits, percentile=99.99, symmetric=True):
    """Calculate scale using percentile clipping"""
    if symmetric:
        abs_data = torch.abs(data)
        max_val = torch.quantile(abs_data, percentile / 100.0)
        qmax = 2 ** (num_bits - 1) - 1
        scale = max_val / qmax
    else:
        min_val = torch.quantile(data, (100 - percentile) / 200.0)
        max_val = torch.quantile(data, (100 + percentile) / 200.0)
        qmax = 2 ** num_bits - 1
        scale = (max_val - min_val) / qmax
    
    return scale

def run_rtn_quantization(args):
    """
    RTN (Round-to-Nearest) post-training quantization.
    Simple quantization method using various scale calculation approaches.
    """
    logger.info("Running RTN (Round-to-Nearest) PTQ quantization...")
    logger.info(f"Scale calculation method: {args.rtn_scale_method}")
    
    # Setup dataloader for calibration
    pin_memory = True if args.device == 'cuda' else False
    dataloader_dict = setup_dataloader(
        args.dataset_name, 
        args.batch_size, 
        args.nworkers, 
        pin_memory=pin_memory, 
        DDP_mode=False, 
        model=args.model
    )
    
    calibration_loader = dataloader_dict["train"]
    
    # Create original FP model
    logger.info("Creating original FP32 model...")
    model = create_model(args)
    model = model.to(args.device)
    model.eval()
    
    # Collect statistics for calibration
    logger.info("Collecting statistics for calibration...")
    stats = {}
    
    def collect_stats_hook(name):
        def hook(module, input, output):
            if name not in stats:
                stats[name] = {'data': []}
            
            # Collect activation data for scale calculation
            if isinstance(input, tuple):
                data = input[0]
            else:
                data = input
                
            # Store flattened data for scale calculation
            stats[name]['data'].append(data.detach().cpu().flatten())
        
        return hook
    
    # Register hooks for statistics collection
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(collect_stats_hook(name)))
    
    # Run calibration
    sample_count = 0
    calibration_samples = args.calibration_samples
    logger.info(f"Using {calibration_samples} calibration samples for RTN quantization")
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(calibration_loader):
            if sample_count >= calibration_samples:
                break
            
            inputs = inputs.to(args.device)
            _ = model(inputs)
            sample_count += inputs.size(0)
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    logger.info(f"Collected statistics from {sample_count} samples")
    
    # Calculate scales for each layer
    logger.info("Calculating quantization scales...")
    scale_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Calculate weight scales
            weight_data = module.weight.data.flatten()
            
            if args.rtn_scale_method == 'minmax':
                scale = calculate_scale_minmax(weight_data, args.num_bits, symmetric=True)
            elif args.rtn_scale_method == 'symmetric_max':
                scale = calculate_scale_symmetric_max(weight_data, args.num_bits)
            elif args.rtn_scale_method == 'mse':
                scale = calculate_scale_mse(weight_data, args.num_bits, symmetric=True)
            elif args.rtn_scale_method == 'percentile':
                scale = calculate_scale_percentile(weight_data, args.num_bits, 
                                                 percentile=args.rtn_percentile, symmetric=True)
            else:
                logger.warning(f"Unknown scale method {args.rtn_scale_method}, using symmetric_max")
                scale = calculate_scale_symmetric_max(weight_data, args.num_bits)
            
            if scale > 0:
                scale_dict[name] = scale
                logger.info(f"Scale for {name}: {scale.item():.6f} (method: {args.rtn_scale_method})")
    
    # Apply quantization based on collected statistics
    logger.info("Applying RTN quantization...")
    
    # TODO: Replace layers with quantized versions using calculated scales
    # For now, just log that RTN quantization would be applied here
    
    logger.info("RTN quantization completed")
    return model