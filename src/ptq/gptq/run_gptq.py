import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
from functools import partial

from src.logger import logger
from src.utils import *
from src.models.model import create_model
from src.module_quantization import QLinear, QConv2d
from src.run_qat import run_load_model, run_one_epoch

from .gptq import GPTQ
from .quant import Quantizer

def find_layers(module, layers=None):
    """Find all Linear and Conv2d layers in a module."""
    if layers is None:
        layers = [nn.Linear, nn.Conv2d]
    
    if type(module) in layers:
        return {None: module}
    
    res = {}
    for name, child in module.named_children():
        res.update({name + '.' + k if k else name: v for k, v in find_layers(child, layers).items()})
    
    return res


def quantize_layer_with_gptq(layer, dataloader, args, num_samples=None):
    """
    Quantize a single layer using GPTQ algorithm integrated with your quantization framework.
    """
    if num_samples is None:
        num_samples = args.calibration_samples  # Clean dotdict access
    
    logger.info(f"Quantizing layer: {layer.__class__.__name__} with {num_samples} samples")
    
    # Create GPTQ instance for this layer
    gptq = GPTQ(layer)
    
    # Use the Quantizer from quant.py
    quantizer = Quantizer()
    quantizer.configure(
        bits=args.num_bits,
        perchannel=False,  # You can modify this based on your needs
        sym=True,          # Symmetric quantization
        mse=False,         # You can enable MSE optimization if needed
        maxshrink=0.8
    )
    
    # Set up the quantizer
    gptq.quantizer = quantizer
    
    # Collect statistics by running forward passes
    layer.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
                
            inputs = inputs.to(args.device)
            
            # Hook to capture input and output
            def hook_fn(module, input, output):
                gptq.add_batch(input[0], output)
            
            handle = layer.register_forward_hook(hook_fn)
            
            try:
                # Forward pass to collect statistics
                _ = layer(inputs)
                sample_count += inputs.size(0)
            finally:
                handle.remove()
    
    logger.info(f"Collected statistics from {sample_count} samples")
    
    # Run GPTQ quantization with clean dotdict access
    gptq.fasterquant(
        blocksize=args.gptq_blocksize,
        percdamp=args.gptq_percdamp,
        groupsize=args.gptq_groupsize,
        actorder=args.gptq_actorder,
        static_groups=args.gptq_static_groups
    )
    
    # Clean up
    gptq.free()
    
    logger.info("Layer quantization completed")


def quantize_model_gptq(model, dataloader, args):
    """
    Quantize an entire model using GPTQ algorithm and extract scales.
    """
    logger.info("Starting GPTQ model quantization...")
    
    # Find all quantizable layers
    layers_dict = find_layers(model)
    logger.info(f"Found {len(layers_dict)} layers to quantize")
    
    # Create GPTQ instances for each layer
    quantizers = {}
    for layer_name, layer in layers_dict.items():
        quantizer = Quantizer()
        quantizer.configure(
            bits=args.num_bits,
            perchannel=False,
            sym=True,
            mse=False,
            maxshrink=0.8
        )
        
        quantizers[layer_name] = GPTQ(layer)
        quantizers[layer_name].quantizer = quantizer
    
    # Register hooks to collect statistics
    def add_batch_hook(name):
        def hook(module, input, output):
            if len(input) > 0 and input[0] is not None:
                quantizers[name].add_batch(input[0], output)
        return hook
    
    hooks = []
    for layer_name in quantizers.keys():
        layer = layers_dict[layer_name]
        hook = layer.register_forward_hook(add_batch_hook(layer_name))
        hooks.append(hook)
    
    # Collect statistics
    model.eval()
    sample_count = 0
    calibration_samples = args.calibration_samples
    logger.info(f"Collecting statistics from {calibration_samples} samples...")
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if sample_count >= calibration_samples:
                break
            
            inputs = inputs.to(args.device)
            _ = model(inputs)
            sample_count += inputs.size(0)
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed {sample_count}/{calibration_samples} samples")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    logger.info(f"Collected statistics from {sample_count} samples")
    
    # Quantize each layer and extract scales
    scale_dict = {}
    for layer_name, gptq_instance in quantizers.items():
        logger.info(f"Quantizing layer: {layer_name}")
        
        # If no samples collected, use simple RTN (minmax)
        if gptq_instance.nsamples == 0:
            logger.warning(f"No samples for layer {layer_name}, using RTN quantization")
            W = gptq_instance.layer.weight.data
            if isinstance(gptq_instance.layer, nn.Conv2d):
                W = W.flatten(1)
            
            w_max = torch.max(torch.abs(W.min()), torch.abs(W.max()))
            qmax = 2 ** (args.num_bits - 1) - 1
            scale = w_max / qmax
            if scale > 0:
                scale_dict[layer_name] = scale
                logger.info(f"RTN scale for {layer_name}: {scale.item():.6f}")
            continue
        
        # Run GPTQ quantization
        gptq_instance.fasterquant(
            blocksize=args.gptq_blocksize,
            percdamp=args.gptq_percdamp,
            groupsize=args.gptq_groupsize,
            actorder=args.gptq_actorder,
            static_groups=args.gptq_static_groups
        )
        
        # Extract scale
        if hasattr(gptq_instance.quantizer, 'scale') and gptq_instance.quantizer.scale is not None:
            scale_value = gptq_instance.quantizer.scale.clone()
            if torch.all(scale_value > 0) and torch.isfinite(scale_value).all():
                if scale_value.numel() == 1:
                    scale_dict[layer_name] = scale_value
                    logger.info(f"Extracted scale for {layer_name}: {scale_value.item():.6f}")
                else:
                    # Take first element if all are the same, otherwise use mean
                    unique_vals = torch.unique(scale_value)
                    if len(unique_vals) == 1:
                        scale_dict[layer_name] = unique_vals[0:1]
                        logger.info(f"Extracted uniform scale for {layer_name}: {unique_vals[0].item():.6f}")
                    else:
                        mean_scale = scale_value.mean().unsqueeze(0)
                        scale_dict[layer_name] = mean_scale
                        logger.info(f"Extracted mean scale for {layer_name}: {mean_scale.item():.6f}")
            else:
                logger.warning(f"Invalid scale for {layer_name}: {scale_value}")
        else:
            logger.warning(f"No scale available for layer {layer_name}")
        
        gptq_instance.free()
    
    logger.info("GPTQ model quantization completed")
    return model, scale_dict, quantizers


def transfer_gptq_scales_to_lsq(quantized_net, scale_dict, quantizers, args):
    """
    Transfer GPTQ scales to LSQ quantizers.
    Simple 1:1 scale transfer since both use symmetric quantization.
    """
    logger.info("Transferring GPTQ scales to LSQ quantizers...")
    
    layer_count = 0
    transferred_count = 0
    
    logger.info(f"Available GPTQ scales for layers: {list(scale_dict.keys())}")
    
    for name, module in quantized_net.named_modules():
        if hasattr(module, 'w_quantizer') and module.w_quantizer is not None:
            layer_count += 1
            
            # Disable activation quantization for PTQ
            if hasattr(module, 'x_quantizer'):
                module.x_quantizer = None
            
            # Find matching GPTQ scale
            gptq_scale = None
            if name in scale_dict:
                gptq_scale = scale_dict[name]
            else:
                # Try substring matching
                for gptq_name, scale in scale_dict.items():
                    if gptq_name in name or name in gptq_name:
                        gptq_scale = scale
                        break
            
            if gptq_scale is not None and hasattr(module, 'w_scale'):
                # Transfer scale directly (GPTQ and LSQ both use symmetric quantization)
                if not isinstance(gptq_scale, torch.Tensor):
                    gptq_scale = torch.tensor(gptq_scale, device=module.w_scale.device)
                else:
                    gptq_scale = gptq_scale.to(module.w_scale.device)
                
                # Ensure single value for per-tensor quantization
                if gptq_scale.numel() > 1:
                    gptq_scale = gptq_scale.mean().unsqueeze(0)
                
                module.w_scale.data = gptq_scale.clone().detach().reshape_as(module.w_scale.data)
                module.w_Qparms['scale'] = module.w_scale
                
                # Set initialization flag
                if hasattr(module, 'init_state'):
                    module.init_state = torch.tensor(True)
                
                transferred_count += 1
                logger.info(f"Transferred scale {gptq_scale.item():.6f} to layer: {name}")
            else:
                logger.warning(f"No GPTQ scale found for layer: {name}")
    
    logger.info(f"Scale transfer completed: {transferred_count}/{layer_count} layers")
    return quantized_net



def run_gptq_quantization(args):
    logger.info("Starting GPTQ quantization pipeline...")
    
    pin_memory = True if args.device == 'cuda' else False
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory=pin_memory, DDP_mode=False, model=args.model)

    calibration_loader = dataloader["train"]

    logger.info("Creating original model...")
    original_net = create_model(args)
    original_net = original_net.to(args.device)
    original_net.eval()

    cudnn.benchmark = True if args.device == 'cuda' else False
    task_loss_fn = nn.CrossEntropyLoss()
    criterion_val = Multiple_Loss( {"task_loss": task_loss_fn})
    
    # Create a copy of the original model for GPTQ quantization
    logger.info("Creating copy of original model for GPTQ...")
    import copy
    gptq_model = copy.deepcopy(original_net)
    gptq_model.eval()

    logger.info("Creating quantized model structure with LSQ quantizers...")
    # Store original x_quantizer and temporarily set to None for PTQ
    original_x_quantizer = args.x_quantizer
    args.x_quantizer = None  # Disable activation quantization for PTQ
    
    quantized_net = run_load_model(args)
    quantized_net.eval()
    
    # Restore original x_quantizer in args
    args.x_quantizer = original_x_quantizer

    # Run GPTQ to get scales (this will modify gptq_model in-place)
    logger.info("Running GPTQ to extract optimal scales...")
    gptq_model, scale_dict, quantizers = quantize_model_gptq(gptq_model, calibration_loader, args)

    # Evaluate the GPTQ-quantized model
    logger.info("Evaluating GPTQ-quantized model...")
    val_accuracy_gptq, val_top5_accuracy_gptq, val_loss_gptq, best_acc_gptq, val_loss_dict_gptq = run_one_epoch(gptq_model, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False)
    logger.info(f'[GPTQ] val_Loss: {val_loss_dict_gptq["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy_gptq:.5f}, val_top5_Acc: {val_top5_accuracy_gptq:.5f}')

    # Transfer GPTQ scales to LSQ quantizers
    logger.info("Transferring GPTQ scales to LSQ quantizers...")
    quantized_net = transfer_gptq_scales_to_lsq(quantized_net, scale_dict, quantizers, args)

    # Evaluate the final LSQ quantized model
    logger.info("Evaluating final LSQ quantized model...")
    val_accuracy_lsq, val_top5_accuracy_lsq, val_loss_lsq, best_acc_lsq, val_loss_dict_lsq = run_one_epoch(quantized_net, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False)
    logger.info(f'[LSQ] val_Loss: {val_loss_dict_lsq["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy_lsq:.5f}, val_top5_Acc: {val_top5_accuracy_lsq:.5f}')

    # Evaluate original FP32 model
    logger.info("Evaluating original FP32 model...")
    val_accuracy, val_top5_accuracy,  val_loss, best_acc, val_loss_dict = run_one_epoch(original_net, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False)
    logger.info(f'[Original FP32] val_Loss: {val_loss_dict["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy:.5f}, val_top5_Acc: {val_top5_accuracy:.5f}')


    # Summary comparison
    logger.info("="*60)
    logger.info("QUANTIZATION RESULTS SUMMARY:")
    logger.info("="*60)
    logger.info(f"Original FP32:  Loss: {val_loss_dict['task_loss'].item():.5f}, Top1: {val_accuracy:.5f}, Top5: {val_top5_accuracy:.5f}")
    logger.info(f"GPTQ:           Loss: {val_loss_dict_gptq['task_loss'].item():.5f}, Top1: {val_accuracy_gptq:.5f}, Top5: {val_top5_accuracy_gptq:.5f}")
    logger.info(f"LSQ (final):    Loss: {val_loss_dict_lsq['task_loss'].item():.5f}, Top1: {val_accuracy_lsq:.5f}, Top5: {val_top5_accuracy_lsq:.5f}")
    logger.info("="*60)
    
    # Calculate accuracy drops
    gptq_drop = val_accuracy - val_accuracy_gptq
    lsq_drop = val_accuracy - val_accuracy_lsq
    logger.info(f"Accuracy drops: GPTQ: {gptq_drop:.5f}, LSQ: {lsq_drop:.5f}")
    
    return quantized_net