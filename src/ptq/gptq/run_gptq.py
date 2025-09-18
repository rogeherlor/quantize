import os
import gc

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
from src.models.depth.qvggt import run_evaluation_vggt

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

class StopForward(Exception):
    pass

def quantize_model_gptq(model, dataloader, args):
    layers_dict = find_layers(model)
    layer_names = list(layers_dict.keys())
    logger.info(f"Found {len(layer_names)} layers to quantize")

    if args.skip_layers:
        layer_names = [name for name in layer_names if not any(pat in name for pat in args.skip_layers)]
        logger.info(f"After filtering skip_layers, processing {len(layer_names)} layers")

    model.eval()
    model = model.cpu()
    
    scale_dict = {}
    
    # Simple approach: just use catcher for every layer
    # This is reliable and works for all architectures
    for i, layer_name in enumerate(layer_names):
        logger.info(f"Processing layer {i+1}/{len(layer_names)}: {layer_name}")
        
        # Collect inputs using catcher approach
        layer_inputs = _collect_layer_inputs(model, dataloader, layer_name, args)
        
        # Get the layer and move to GPU
        parent = model
        subnames = layer_name.split('.')
        for name in subnames[:-1]:
            parent = getattr(parent, name)
        layer = getattr(parent, subnames[-1])
        layer = layer.to(args.device)
        
        # Setup GPTQ if this layer should be quantized
        if _should_quantize_layer(layer_name, args):
            quantizer = Quantizer()
            quantizer.configure(
                bits=args.num_bits,
                perchannel=False,
                sym=True,
                mse=args.mse,
                maxshrink=0.8
            )
            
            gptq_instance = GPTQ(layer)
            gptq_instance.quantizer = quantizer
            
            def hook_fn(module, inp, out):
                gptq_instance.add_batch(inp[0], out)
            
            hook = layer.register_forward_hook(hook_fn)
            
            # Run forward pass through layer
            with torch.no_grad():
                for inp in layer_inputs:
                    inp = inp.to(args.device)
                    _ = layer(inp)
            
            hook.remove()
            
            # Run quantization
            if gptq_instance.nsamples > 0:
                logger.info(f"Running GPTQ quantization for {layer_name}")
                gptq_instance.fasterquant(
                    blocksize=args.gptq_blocksize,
                    percdamp=args.gptq_percdamp,
                    groupsize=args.gptq_groupsize,
                    actorder=args.gptq_actorder,
                    static_groups=args.gptq_static_groups
                )
                
                if hasattr(gptq_instance.quantizer, 'scale') and gptq_instance.quantizer.scale is not None:
                    scale_value = gptq_instance.quantizer.scale.clone().cpu()
                    if torch.all(scale_value > 0) and torch.isfinite(scale_value).all():
                        if scale_value.numel() == 1:
                            scale_dict[layer_name] = scale_value
                            logger.info(f"Extracted scale for {layer_name}: {scale_value.item():.6f}")
                        else:
                            unique_vals = torch.unique(scale_value)
                            if len(unique_vals) == 1:
                                scale_dict[layer_name] = unique_vals[0:1]
                                logger.info(f"Uniform scale for {layer_name}: {unique_vals[0].item():.6f}")
                            else:
                                mean_scale = scale_value.mean().unsqueeze(0)
                                scale_dict[layer_name] = mean_scale
                                logger.info(f"Mean scale for {layer_name}: {mean_scale.item():.6f}")
                    else:
                        logger.warning(f"Invalid scale for {layer_name}")
                else:
                    logger.warning(f"No scale available for {layer_name}")
                
                gptq_instance.free()
            else:
                logger.warning(f"No samples collected for {layer_name}, skipping quantization")
        
        # Cleanup
        layer = layer.cpu()
        setattr(parent, subnames[-1], layer)
        torch.cuda.empty_cache()
        gc.collect()

    model = model.to(args.device)
    logger.info(f"Quantization completed. Processed {len(scale_dict)}/{len(layer_names)} layers")
    return model, scale_dict

def _should_quantize_layer(layer_name, args):
    """Check if a layer should be quantized."""
    if args.skip_layers:
        return not any(pat in layer_name for pat in args.skip_layers)
    return True

def _collect_layer_inputs(model, dataloader, layer_name, args):
    """Collect inputs for a specific layer using catcher approach."""
    layer_inputs = []
    
    class Catcher(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
        def forward(self, x, *args, **kwargs):
            layer_inputs.append(x.detach().cpu())
            raise StopForward

    # Get layer path and install catcher
    parent = model
    subnames = layer_name.split('.')
    for name in subnames[:-1]:
        parent = getattr(parent, name)
    orig_layer = getattr(parent, subnames[-1])
    
    catcher = Catcher(orig_layer)
    setattr(parent, subnames[-1], catcher)
    model = model.to(args.device)

    # Collect inputs
    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= args.calibration_samples:
                break
                
            if args.dataset_name in ['imagenet', 'imagenet-mini']:
                inputs, _ = batch
                batch_input = inputs.to(args.device)
                sample_count += inputs.size(0)
                try:
                    model(batch_input)
                except StopForward:
                    pass
            elif args.dataset_name == 'co3d':
                from src.models.depth.vggt.training.train_utils.general import copy_data_to_device
                batch = copy_data_to_device(batch, args.device, non_blocking=True)
                batch_input = batch["images"]
                sample_count += batch["images"].size(0)
                try:
                    model(images=batch_input)
                except StopForward:
                    pass

    # Restore and cleanup
    setattr(parent, subnames[-1], orig_layer)
    model = model.cpu()
    
    logger.info(f"Collected {len(layer_inputs)} input batches for {layer_name}")
    return layer_inputs


def transfer_gptq_scales_to_lsq(quantized_net, scale_dict, args):
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
    logger.info("Starting GPTQ...")
    args.x_quantizer = None
    
    pin_memory = True if args.device == 'cuda' else False
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory=pin_memory, DDP_mode=False, model=args.model)

    calibration_loader = dataloader["train"]

    logger.info("Creating original model...")
    original_net = create_model(args)
    original_net = original_net.to(args.device)
    original_net.eval()

    cudnn.benchmark = True if args.device == 'cuda' else False

    logger.info("Running GPTQ to extract optimal scales...")
    gptq_model, scale_dict = quantize_model_gptq(original_net, calibration_loader, args)

    gptq_model_path = os.path.join(args.output_path, "gptq_model.pt")
    os.makedirs(os.path.dirname(gptq_model_path), exist_ok=True)
    torch.save(gptq_model.state_dict(), gptq_model_path)
    logger.info(f"Saved GPTQ model to {gptq_model_path}")

    if args.dataset_name in ['imagenet', 'imagenet-mini']:
        task_loss_fn = nn.CrossEntropyLoss()
        criterion_val = Multiple_Loss( {"task_loss": task_loss_fn})

        logger.info("Evaluating quantized model after GPTQ (before scale transfer)...")
        val_accuracy_gptq, val_top5_accuracy_gptq, val_loss_gptq, best_acc_gptq, val_loss_dict_gptq = run_one_epoch(gptq_model, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False)
        logger.info(f'[GPTQ] val_Loss: {val_loss_dict_gptq["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy_gptq:.5f}, val_top5_Acc: {val_top5_accuracy_gptq:.5f}')

    elif args.dataset_name == 'co3d':
        logger.info("Evaluating GPTQ model using CO3D evaluation script...")
        run_evaluation_vggt(gptq_model_path)
    
    del gptq_model
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    logger.info("Transferring GPTQ scales to LSQ quantizers...")
    quantized_net = run_load_model(args)
    quantized_net.eval()
    quantized_net = transfer_gptq_scales_to_lsq(quantized_net, scale_dict, args)

    lsq_model_path = os.path.join(args.output_path, "lsq_gptq_model.pt")
    torch.save(quantized_net.state_dict(), lsq_model_path)
    logger.info(f"Saved LSQ quantized model to {lsq_model_path}")

    if args.dataset_name in ['imagenet', 'imagenet-mini']:
        logger.info("Evaluating final LSQ quantized model...")
        val_accuracy_lsq, val_top5_accuracy_lsq, val_loss_lsq, best_acc_lsq, val_loss_dict_lsq = run_one_epoch(quantized_net, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False)
        logger.info(f'[LSQ] val_Loss: {val_loss_dict_lsq["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy_lsq:.5f}, val_top5_Acc: {val_top5_accuracy_lsq:.5f}')
        
        logger.info("Evaluating original FP32 model...")
        original_net = create_model(args)
        original_net = original_net.to(args.device)
        original_net.eval()
        val_accuracy, val_top5_accuracy,  val_loss, best_acc, val_loss_dict = run_one_epoch(original_net, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False)
        logger.info(f'[Original FP32] val_Loss: {val_loss_dict["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy:.5f}, val_top5_Acc: {val_top5_accuracy:.5f}')

        logger.info("="*60)
        logger.info("QUANTIZATION RESULTS SUMMARY:")
        logger.info("="*60)
        logger.info(f"Original FP32:  Loss: {val_loss_dict['task_loss'].item():.5f}, Top1: {val_accuracy:.5f}, Top5: {val_top5_accuracy:.5f}")
        logger.info(f"GPTQ:           Loss: {val_loss_dict_gptq['task_loss'].item():.5f}, Top1: {val_accuracy_gptq:.5f}, Top5: {val_top5_accuracy_gptq:.5f}")
        logger.info(f"LSQ (final):    Loss: {val_loss_dict_lsq['task_loss'].item():.5f}, Top1: {val_accuracy_lsq:.5f}, Top5: {val_top5_accuracy_lsq:.5f}")
        logger.info("="*60)
        
        gptq_drop = val_accuracy - val_accuracy_gptq
        lsq_drop = val_accuracy - val_accuracy_lsq
        logger.info(f"Accuracy drops: GPTQ: {gptq_drop:.5f}, LSQ: {lsq_drop:.5f}")

    elif args.dataset_name == 'co3d':
        logger.info("Evaluating final LSQ quantized model...")
        run_evaluation_vggt(lsq_model_path)
    
    return quantized_net