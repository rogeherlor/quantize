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
            
            # 🔍 DEBUG: Log first collection for each layer
            if name not in act_scales:
                logger.debug(f"  First collection [{name}]: mean={max_vals.mean():.6f}, max={max_vals.max():.6f}, min={max_vals.min():.6f}")
            
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
    
    # 🔍 DEBUG: Activation scale statistics across all layers
    all_act_means = [s.mean().item() for s in act_scales.values()]
    all_act_maxs = [s.max().item() for s in act_scales.values()]
    logger.info(f"  Activation stats across all layers:")
    logger.info(f"    Mean of means: {np.mean(all_act_means):.6f}")
    logger.info(f"    Mean of maxs: {np.mean(all_act_maxs):.6f}")
    logger.info(f"    Std of means: {np.std(all_act_means):.6f}")
    
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
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-9)

    # 🔍 DEBUG: Check scale statistics
    act_mean = act_scales.mean().item()
    act_max = act_scales.max().item()
    act_min = act_scales.min().item()
    weight_mean = weight_scales.mean().item()
    weight_max = weight_scales.max().item()
    weight_min = weight_scales.min().item()
    ratio_mean = act_mean / weight_mean
    ratio_max = act_max / weight_max
    
    logger.debug(f"    Act scales: mean={act_mean:.6f}, max={act_max:.6f}, min={act_min:.6f}")
    logger.debug(f"    Weight scales: mean={weight_mean:.6f}, max={weight_max:.6f}, min={weight_min:.6f}")
    logger.debug(f"    Ratio (act/weight): mean={ratio_mean:.2f}x, max={ratio_max:.2f}x")

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-9)
        .to(device)
        .to(dtype)
    )
    
    # 🔍 DEBUG: Check final smoothing scales
    scales_mean = scales.mean().item()
    scales_max = scales.max().item()
    scales_min = scales.min().item()
    logger.debug(f"    Smoothing scales (alpha={alpha}): mean={scales_mean:.6f}, max={scales_max:.6f}, min={scales_min:.6f}")

    ln.weight.div_(scales)
    if ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

@torch.no_grad()
def smooth_vit_blocks(model, act_scales, alpha=0.5, top_k=5):
    """Apply SmoothQuant smoothing to ViT blocks
    
    Args:
        model: Model to smooth
        act_scales: Dict of activation scales per layer
        alpha: Smoothing factor
        top_k: Number of layers with highest activation scales to smooth (None = smooth all)
    """
    from timm.models.vision_transformer import Block
    
    # If top_k is specified, find the layers with highest activation scales
    if top_k is not None:
        # Calculate mean activation scale for each block's QKV and FC1 layers
        block_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, Block):
                # Check QKV layer
                qkv_scale_key = name + ".attn.qkv"
                fc1_scale_key = name + ".mlp.fc1"
                
                if qkv_scale_key in act_scales and fc1_scale_key in act_scales:
                    qkv_mean = act_scales[qkv_scale_key].mean().item()
                    fc1_mean = act_scales[fc1_scale_key].mean().item()
                    # Use the maximum of the two as the score for this block
                    block_scores[name] = max(qkv_mean, fc1_mean)
        
        # Sort blocks by score (highest first) and select top_k
        sorted_blocks = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)
        layers_to_smooth = set([name for name, score in sorted_blocks[:top_k]])
        
        logger.info(f"\nSelecting top {top_k} layers with highest activation scales to smooth:")
        for i, (name, score) in enumerate(sorted_blocks[:top_k]):
            logger.info(f"  {i+1}. {name}: max_act_scale={score:.6f}")
        logger.info("")
    else:
        layers_to_smooth = None
        logger.info("Smoothing all layers")
    
    smoothed_count = 0
    for name, module in model.named_modules():
        if isinstance(module, Block):
            # Skip if not in top_k (when top_k is specified)
            if layers_to_smooth is not None and name not in layers_to_smooth:
                continue
                
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
            
            smoothed_count += 1
    
    logger.info(f"Smoothed {smoothed_count} ViT blocks")

def apply_smoothquant(model, dataloader, args):
    """Apply SmoothQuant smoothing to model"""
    model.eval()

    logger.info("Collecting activation scales for smoothing with {} samples...".format(args.calibration_samples))
    act_scales = collect_activation_scales_smoothquant(model, dataloader, args)
    
    # Get top_k parameter (default to None for smoothing all layers)
    top_k = getattr(args, 'smoothquant_top_k', None)
    alpha = getattr(args, 'smoothquant_alpha', 0.5)
    
    if top_k is not None:
        logger.info(f"Applying SmoothQuant smoothing to top {top_k} layers with alpha={alpha}...")
    else:
        logger.info(f"Applying SmoothQuant smoothing to all layers with alpha={alpha}...")
    
    smooth_vit_blocks(model, act_scales, alpha=alpha, top_k=top_k)
    
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
            # Ensure same shape as initialized parameter [1]
            module.w_scale.data = torch.tensor([w_scale.item()], dtype=module.w_scale.dtype, device=module.w_scale.device)
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
            # Ensure shape [1] to match parameter initialization
            module.x_scale.data = torch.tensor([1e-3], dtype=module.x_scale.dtype, device=module.x_scale.device)
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
                # Ensure shape [1] to match parameter initialization
                module.x_scale.data = torch.tensor([x_scale], dtype=module.x_scale.dtype, device=module.x_scale.device)
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

def _run_smoothquant_single_alpha(args, alpha_value, dataloader, original_fp_model=None, original_results=None):
    """Run SmoothQuant quantization for a single alpha value
    
    Args:
        args: Arguments
        alpha_value: The alpha value for this run
        dataloader: Data loader dict
        original_fp_model: Original FP32 model (reused across alphas)
        original_results: Dict with original model results (to avoid re-evaluation)
    """
    import copy
    
    calibration_loader = dataloader["train"]
    
    # Create and evaluate original FP32 model if not provided (for first run)
    if original_fp_model is None:
        logger.info("\nStep 1: Creating original FP32 model...")
        original_fp_model = create_model(args)
        original_fp_model = original_fp_model.to(args.device)
        original_fp_model.eval()
    
    # Step 2: Create smoothed FP32 model
    logger.info(f"\nStep 2: Creating smoothed FP32 model with SmoothQuant (alpha={alpha_value})...")
    smoothed_fp_model = create_model(args)
    smoothed_fp_model = smoothed_fp_model.to(args.device)
    
    # Temporarily set alpha for this run
    original_alpha = args.smoothquant_alpha
    args.smoothquant_alpha = alpha_value
    smoothed_fp_model = apply_smoothquant(smoothed_fp_model, calibration_loader, args)
    args.smoothquant_alpha = original_alpha

    # Save smoothed model
    smoothquant_model_path = os.path.join(args.output_path, f"smoothquant_smoothed_fp_model_alpha_{alpha_value}.pt")
    os.makedirs(os.path.dirname(smoothquant_model_path), exist_ok=True)
    torch.save(smoothed_fp_model.state_dict(), smoothquant_model_path)
    logger.info(f"Saved smoothed FP32 model to {smoothquant_model_path}")

    # Step 3: Apply LSQ replacement to original model (only if not already done)
    if original_results is None:
        logger.info("\nStep 3: Applying LSQ replacement to original model...")
        original_lsq_model = replace(args, original_fp_model)
        original_lsq_model = original_lsq_model.to(args.device)
        initialize_lsq_scales(original_lsq_model, calibration_loader, args)
    else:
        logger.info("\nStep 3: Skipping original model quantization (already done)...")
        original_lsq_model = None

    # Step 4: Apply LSQ replacement to smoothed model
    logger.info("\nStep 4: Applying LSQ replacement to smoothed model...")
    smoothed_lsq_model = replace(args, smoothed_fp_model)
    smoothed_lsq_model = smoothed_lsq_model.to(args.device)
    initialize_lsq_scales(smoothed_lsq_model, calibration_loader, args)

    # Save LSQ models
    if original_lsq_model is not None:
        original_lsq_path = os.path.join(args.output_path, "lsq_original_model.pt")
        torch.save(original_lsq_model.state_dict(), original_lsq_path)
        logger.info(f"Saved original LSQ model to {original_lsq_path}")
    
    smoothed_lsq_path = os.path.join(args.output_path, f"lsq_smoothquant_model_alpha_{alpha_value}.pt")
    torch.save(smoothed_lsq_model.state_dict(), smoothed_lsq_path)
    logger.info(f"Saved smoothed LSQ model to {smoothed_lsq_path}")

    # Evaluation
    result_dict = {'alpha': alpha_value}
    if args.dataset_name in ['imagenet', 'imagenet-mini']:
        task_loss_fn = nn.CrossEntropyLoss()
        criterion_val = Multiple_Loss({"task_loss": task_loss_fn})
        
        # Evaluate original model only if not already done
        if original_results is None:
            logger.info("\n1. Evaluating original LSQ model...")
            val_accuracy_original, val_top5_accuracy_original, val_loss_original, _, val_loss_dict_original = run_one_epoch(
                original_lsq_model, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False
            )
            logger.info(f'[Original LSQ] val_Loss: {val_loss_dict_original["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy_original:.5f}, val_top5_Acc: {val_top5_accuracy_original:.5f}')
        else:
            logger.info("\n1. Using cached original LSQ model results...")
            val_accuracy_original = original_results['val_accuracy_original']
            val_top5_accuracy_original = original_results['val_top5_accuracy_original']
            val_loss_dict_original = original_results['val_loss_dict_original']
            logger.info(f'[Original LSQ] val_Loss: {val_loss_dict_original["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy_original:.5f}, val_top5_Acc: {val_top5_accuracy_original:.5f}')
        
        logger.info(f"\n2. Evaluating smoothed LSQ model (alpha={alpha_value})...")
        val_accuracy_smoothed, val_top5_accuracy_smoothed, val_loss_smoothed, _, val_loss_dict_smoothed = run_one_epoch(
            smoothed_lsq_model, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False
        )
        logger.info(f'[Smoothed LSQ] val_Loss: {val_loss_dict_smoothed["task_loss"].item():.5f}, val_top1_Acc: {val_accuracy_smoothed:.5f}, val_top5_Acc: {val_top5_accuracy_smoothed:.5f}')

        logger.info("\n" + "="*70)
        logger.info(f"SMOOTHQUANT QUANTIZATION RESULTS (alpha={alpha_value})")
        logger.info("="*70)
        logger.info(f"{'Model':<25} {'Loss':>10} {'Top-1 Acc':>12} {'Top-5 Acc':>12} {'Acc Diff':>10}")
        logger.info("-"*70)
        logger.info(f"{'Original LSQ':<25} {val_loss_dict_original['task_loss'].item():>10.5f} {val_accuracy_original:>12.5f} {val_top5_accuracy_original:>12.5f} {'-':>10}")
        acc_diff = val_accuracy_smoothed - val_accuracy_original
        logger.info(f"{'Smoothed LSQ':<25} {val_loss_dict_smoothed['task_loss'].item():>10.5f} {val_accuracy_smoothed:>12.5f} {val_top5_accuracy_smoothed:>12.5f} {acc_diff:>+10.5f}")
        logger.info("="*70)
        logger.info(f"\nAccuracy difference (Smoothed - Original): {acc_diff:+.5f} ({acc_diff*100:+.2f}%)")
        logger.info("="*70 + "\n")
        
        # Store results for sweep comparison
        result_dict.update({
            'original_top1_acc': val_accuracy_original * 100,
            'original_top5_acc': val_top5_accuracy_original * 100,
            'smoothed_top1_acc': val_accuracy_smoothed * 100,
            'smoothed_top5_acc': val_top5_accuracy_smoothed * 100,
            'original_loss': val_loss_dict_original['task_loss'].item(),
            'smoothed_loss': val_loss_dict_smoothed['task_loss'].item(),
            'acc_diff': acc_diff * 100
        })

    elif args.dataset_name == 'co3d':
        logger.info("\nEvaluating original LSQ model...")
        run_evaluation_vggt(original_lsq_model)
        logger.info("\nEvaluating smoothed LSQ model...")
        run_evaluation_vggt(smoothed_lsq_model)
    
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    
    # Prepare original results for reuse
    if original_results is None:
        original_results = {
            'val_accuracy_original': val_accuracy_original,
            'val_top5_accuracy_original': val_top5_accuracy_original,
            'val_loss_dict_original': val_loss_dict_original
        }
    
    return smoothed_lsq_model, original_fp_model, original_results, result_dict


def run_smoothquant_quantization(args):
    """
    Run SmoothQuant quantization. Handles both single alpha value and alpha sweep.
    If args.smoothquant_alpha is a list, runs quantization for each alpha value.
    """
    logger.info("Starting SmoothQuant with LSQ...")
    
    # Setup dataloader once
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory=False, DDP_mode=False, model=args.model)
    cudnn.benchmark = True if args.device == 'cuda' else False
    
    # Check if alpha is a list for sweep
    alpha_list = args.smoothquant_alpha if isinstance(args.smoothquant_alpha, list) else [args.smoothquant_alpha]
    is_sweep = isinstance(args.smoothquant_alpha, list)
    
    if is_sweep:
        logger.info(f"\n{'='*90}")
        logger.info(f"ALPHA SWEEP: Testing {len(alpha_list)} alpha values: {alpha_list}")
        logger.info(f"{'='*90}\n")
    
    results_summary = []
    original_fp_model = None  # Reuse for all alphas
    original_results = None   # Cache original model evaluation results
    
    for idx, alpha_value in enumerate(alpha_list):
        if is_sweep:
            logger.info(f"\n{'#'*90}")
            logger.info(f"## Experiment {idx+1}/{len(alpha_list)}: Running SmoothQuant with alpha = {alpha_value}")
            logger.info(f"{'#'*90}\n")
        
        # Run quantization for this alpha
        smoothed_model, original_fp_model, original_results, result_dict = _run_smoothquant_single_alpha(
            args, alpha_value, dataloader, original_fp_model, original_results
        )
        results_summary.append(result_dict)
        
        if is_sweep:
            logger.info(f"\n✓ Completed alpha = {alpha_value}\n")
    
    # Print final summary if sweep
    if is_sweep and len(results_summary) > 0 and 'smoothed_top1_acc' in results_summary[0]:
        logger.info(f"\n{'='*90}")
        logger.info(f"ALPHA SWEEP SUMMARY - {len(results_summary)} experiments")
        logger.info(f"{'='*90}")
        logger.info(f"{'Alpha':<10} {'Original Top-1':>15} {'Smoothed Top-1':>15} {'Difference':>12} {'Status':>12}")
        logger.info(f"{'-'*90}")
        
        best_alpha = None
        best_smoothed_acc = -float('inf')
        
        for result in results_summary:
            alpha = result['alpha']
            orig_acc = result.get('original_top1_acc', 0)
            smooth_acc = result.get('smoothed_top1_acc', 0)
            diff = smooth_acc - orig_acc
            
            status = ""
            if smooth_acc > best_smoothed_acc:
                best_smoothed_acc = smooth_acc
                best_alpha = alpha
                status = "✓ Best"
            
            logger.info(f"{alpha:<10.2f} {orig_acc:>14.4f}% {smooth_acc:>14.4f}% {diff:>+11.4f}% {status:>12}")
        
        logger.info(f"{'-'*90}")
        logger.info(f"Best alpha: {best_alpha} with {best_smoothed_acc:.4f}% top-1 accuracy")
        logger.info(f"{'='*90}\n")
    
    return smoothed_model
