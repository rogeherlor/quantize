import os
import gc
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from functools import partial
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from src.logger import logger
from src.utils import *
from src.models.model import create_model
from src.run_qat import run_load_model, run_one_epoch
from src.models.depth.qvggt import run_evaluation_vggt, replace
from src.initializer.lsq_init import LSQ_initializer


def collect_activation_scales_smoothquant(model, dataloader, args):
    """Collect activation scales using abs().max()"""
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        if torch.is_tensor(tensor):
            hidden_dim = tensor.shape[-1]
            tensor_flat = tensor.view(-1, hidden_dim).abs().detach()
            max_vals = tensor_flat.max(dim=0)[0].float().cpu()
            
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], max_vals)
            else:
                act_scales[name] = max_vals

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(partial(stat_input_hook, name=name))
            )

    sample_count = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Collecting activation scales")
        for batch in pbar:
            if sample_count >= args.calibration_samples:
                break
            try:
                if args.dataset_name in ["imagenet", "imagenet-mini"]:
                    inputs, _ = batch
                    sample_count += inputs.size(0)
                    model(inputs.to(args.device))
                elif args.dataset_name == "co3d":
                    from src.models.depth.vggt.training.train_utils.general import copy_data_to_device
                    batch = copy_data_to_device(batch, args.device, non_blocking=True)
                    sample_count += batch["images"].size(0)
                    model(images=batch["images"])
                pbar.set_postfix({"samples": sample_count})
            except Exception as e:
                logger.warning(f"Error during forward pass: {e}")
                continue

    for h in hooks:
        h.remove()
    
    logger.info(f"Collected activation scales for {len(act_scales)} layers")
    return act_scales

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    """Apply SmoothQuant smoothing to layer norm and fully connected layers"""
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    if ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

@torch.no_grad()
def smooth_vit_blocks(model, act_scales, alpha=0.5):
    """Apply SmoothQuant smoothing to ViT blocks"""
    from timm.models.vision_transformer import Block
    
    for name, module in model.named_modules():
        if isinstance(module, Block):
            logger.debug(f"Smoothing ViT block: {name}")
            
            # Smooth attention layer
            attn_ln = module.norm1
            qkv = module.attn.qkv
            qkv_scale_key = name + ".attn.qkv"
            qkv_input_scales = act_scales[qkv_scale_key]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            # Smooth MLP layer
            ffn_ln = module.norm2
            fc1 = module.mlp.fc1
            fc1_scale_key = name + ".mlp.fc1"
            fc1_input_scales = act_scales[fc1_scale_key]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)

def apply_smoothquant(model, dataloader, args):
    """Apply SmoothQuant smoothing to model"""
    model.eval()

    logger.info("Collecting activation scales for smoothing with {} samples...".format(args.calibration_samples))
    act_scales = collect_activation_scales_smoothquant(model, dataloader, args)
    
    logger.info(f"Applying SmoothQuant smoothing with alpha={getattr(args, 'smoothquant_alpha', 0.5)}...")
    smooth_vit_blocks(model, act_scales, alpha=getattr(args, 'smoothquant_alpha', 0.5))
    
    return model

def initialize_lsq_scales(model, dataloader, args):
    """Initialize LSQ scales from calibration data"""
    model.eval()
    
    # Initialize weight scales from abs().max()
    logger.info("Initializing weight scales from abs().max()...")
    weight_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module, "w_scale") and hasattr(module, "w_Qp"):
            weight_max = module.weight.data.abs().max()
            qp = float(module.w_Qp)
            if weight_max > 0:
                w_scale = weight_max / qp
            else:
                w_scale = torch.tensor(1e-5)
            module.w_scale.data = w_scale.to(module.w_scale.device).to(module.w_scale.dtype)
            if hasattr(module, "w_Qparms"):
                module.w_Qparms['scale'] = module.w_scale
            weight_count += 1
    logger.info(f"Initialized {weight_count} weight scales")
    
    # Initialize activation scales with small positive values to avoid assertion errors
    logger.info("Pre-initializing activation scales with small positive values...")
    act_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "x_scale") and hasattr(module, "x_Qp"):
            # Set a small initial scale to pass the assertion during calibration
            module.x_scale.data = torch.tensor(1e-3).to(module.x_scale.device).to(module.x_scale.dtype)
            if hasattr(module, "x_Qparms"):
                module.x_Qparms['scale'] = module.x_scale
            act_count += 1
    logger.info(f"Pre-initialized {act_count} activation scales")
    
    # Collect activation scales via forward pass
    logger.info(f"Collecting activation scales for LSQ initialization with {args.calibration_samples} samples...")
    act_scales = {}
    
    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if torch.is_tensor(x):
            max_val = x.abs().max().float().cpu()
            if name in act_scales:
                act_scales[name] = max(act_scales[name], max_val.item())
            else:
                act_scales[name] = max_val.item()
    
    hooks = []
    hook_count = 0
    for name, m in model.named_modules():
        if hasattr(m, "x_scale") and hasattr(m, "x_Qp"):
            hooks.append(
                m.register_forward_hook(partial(stat_input_hook, name=name))
            )
            hook_count += 1
    
    logger.info(f"Registered {hook_count} hooks for activation collection")
    
    sample_count = 0
    batch_count = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Collecting activation scales for LSQ")
        for batch in pbar:
            if sample_count >= args.calibration_samples:
                break
            try:
                if args.dataset_name in ["imagenet", "imagenet-mini"]:
                    inputs, _ = batch
                    sample_count += inputs.size(0)
                    batch_count += 1
                    model(inputs.to(args.device))
                elif args.dataset_name == "co3d":
                    from src.models.depth.vggt.training.train_utils.general import copy_data_to_device
                    batch = copy_data_to_device(batch, args.device, non_blocking=True)
                    sample_count += batch["images"].size(0)
                    batch_count += 1
                    model(images=batch["images"])
                pbar.set_postfix({"samples": sample_count, "batches": batch_count, "scales": len(act_scales)})
            except Exception as e:
                logger.warning(f"Error during forward pass: {e}")
                continue
    
    for h in hooks:
        h.remove()
    
    logger.info(f"Collected {len(act_scales)} activation scales from {batch_count} batches ({sample_count} samples)")
    
    # Set activation scales from collected statistics
    logger.info("Setting activation scales from collected statistics...")
    act_count = 0
    missing_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "x_scale") and hasattr(module, "x_Qp"):
            if name in act_scales:
                act_max = act_scales[name]
                qp = float(module.x_Qp)
                if act_max > 0:
                    x_scale = act_max / qp
                else:
                    x_scale = 1e-5
                module.x_scale.data = torch.tensor(x_scale).to(module.x_scale.device).to(module.x_scale.dtype)
                if hasattr(module, "x_Qparms"):
                    module.x_Qparms['scale'] = module.x_scale
                act_count += 1
                
                if act_count <= 5:  # Log first 5 for debugging
                    logger.info(f"[{name}] w_scale={module.w_scale.data.item():.6f}, x_scale={module.x_scale.data.item():.6f}")
            else:
                missing_count += 1
                logger.debug(f"No activation statistics collected for {name}, keeping pre-initialized scale")
    
    logger.info(f"Set {act_count} activation scales from statistics")
    if missing_count > 0:
        logger.warning(f"{missing_count} layers had no activation statistics collected - using pre-initialized scales")
    
    # Now set init_state to True to prevent re-initialization during training/evaluation
    logger.info("Setting init_state to True for all quantized modules...")
    for name, module in model.named_modules():
        if hasattr(module, "init_state"):
            module.init_state = torch.tensor(True)
    logger.info("LSQ scale initialization completed")

def run_smoothquant_quantization(args):
    logger.info("Starting SmoothQuant with LSQ...")
    
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory=False, DDP_mode=False, model=args.model)
    calibration_loader = dataloader["train"]
    cudnn.benchmark = True if args.device == 'cuda' else False

    # Step 1: Create original FP32 model
    logger.info("\nStep 1: Creating original FP32 model...")
    original_fp_model = create_model(args)
    original_fp_model = original_fp_model.to(args.device)
    original_fp_model.eval()

    # Step 2: Create smoothed FP32 model
    logger.info("\nStep 2: Creating smoothed FP32 model with SmoothQuant...")
    smoothed_fp_model = create_model(args)
    smoothed_fp_model = smoothed_fp_model.to(args.device)
    smoothed_fp_model = apply_smoothquant(smoothed_fp_model, calibration_loader, args)
    
    # Save smoothed model
    smoothquant_model_path = os.path.join(args.output_path, "smoothquant_smoothed_fp_model.pt")
    os.makedirs(os.path.dirname(smoothquant_model_path), exist_ok=True)
    torch.save(smoothed_fp_model.state_dict(), smoothquant_model_path)
    logger.info(f"Saved smoothed FP32 model to {smoothquant_model_path}")

    # Step 3: Apply LSQ replacement to original model
    logger.info("\nStep 3: Applying LSQ replacement to original model...")
    original_lsq_model = replace(args, original_fp_model)
    original_lsq_model = original_lsq_model.to(args.device)
    initialize_lsq_scales(original_lsq_model, calibration_loader, args)

    # Step 4: Apply LSQ replacement to smoothed model
    logger.info("\nStep 4: Applying LSQ replacement to smoothed model...")
    smoothed_lsq_model = replace(args, smoothed_fp_model)
    smoothed_lsq_model = smoothed_lsq_model.to(args.device)
    initialize_lsq_scales(smoothed_lsq_model, calibration_loader, args)

    # Save LSQ models
    original_lsq_path = os.path.join(args.output_path, "lsq_original_model.pt")
    smoothed_lsq_path = os.path.join(args.output_path, "lsq_smoothquant_model.pt")
    torch.save(original_lsq_model.state_dict(), original_lsq_path)
    torch.save(smoothed_lsq_model.state_dict(), smoothed_lsq_path)
    logger.info(f"Saved original LSQ model to {original_lsq_path}")
    logger.info(f"Saved smoothed LSQ model to {smoothed_lsq_path}")

    # Evaluation
    if args.dataset_name in ['imagenet', 'imagenet-mini']:
        task_loss_fn = nn.CrossEntropyLoss()
        criterion_val = Multiple_Loss({"task_loss": task_loss_fn})
        
        logger.info("\n1. Evaluating original LSQ model...")
        val_accuracy_original, val_top5_accuracy_original, val_loss_original, _, val_loss_dict_original = run_one_epoch(
            original_lsq_model, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False
        )
        logger.info(f'[Original LSQ] val_Loss: {val_loss_dict_original["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy_original:.5f}, val_top5_Acc: {val_top5_accuracy_original:.5f}')
        
        logger.info("\n2. Evaluating smoothed LSQ model...")
        val_accuracy_smoothed, val_top5_accuracy_smoothed, val_loss_smoothed, _, val_loss_dict_smoothed = run_one_epoch(
            smoothed_lsq_model, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False
        )
        logger.info(f'[Smoothed LSQ] val_Loss: {val_loss_dict_smoothed["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy_smoothed:.5f}, val_top5_Acc: {val_top5_accuracy_smoothed:.5f}')

        logger.info("\n" + "="*70)
        logger.info("SMOOTHQUANT QUANTIZATION RESULTS SUMMARY")
        logger.info("="*70)
        logger.info(f"{'Model':<25} {'Loss':>10} {'Top-1 Acc':>12} {'Top-5 Acc':>12} {'Acc Diff':>10}")
        logger.info("-"*70)
        logger.info(f"{'Original LSQ':<25} {val_loss_dict_original['task_loss'].item():>10.5f} {val_accuracy_original:>12.5f} {val_top5_accuracy_original:>12.5f} {'-':>10}")
        acc_diff = val_accuracy_smoothed - val_accuracy_original
        logger.info(f"{'Smoothed LSQ':<25} {val_loss_dict_smoothed['task_loss'].item():>10.5f} {val_accuracy_smoothed:>12.5f} {val_top5_accuracy_smoothed:>12.5f} {acc_diff:>+10.5f}")
        logger.info("="*70)
        logger.info(f"\nAccuracy difference (Smoothed - Original): {acc_diff:+.5f} ({acc_diff*100:+.2f}%)")
        logger.info("="*70 + "\n")

    elif args.dataset_name == 'co3d':
        logger.info("\nEvaluating original LSQ model...")
        run_evaluation_vggt(original_lsq_model)
        logger.info("\nEvaluating smoothed LSQ model...")
        run_evaluation_vggt(smoothed_lsq_model)
    
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    
    return smoothed_lsq_model
