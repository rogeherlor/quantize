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

@torch.no_grad()
def smooth_vggt_blocks(model, act_scales, alpha=0.5, quant_blocks=None):
    """Apply SmoothQuant smoothing to VGGT blocks (memory-efficient block-by-block)
    
    Args:
        model: Model to smooth
        act_scales: Dict of activation scales per layer
        alpha: Smoothing factor
        quant_blocks: List of block names to quantize (from args.quant_blocks)
    """
    from src.models.depth.vggt.vggt.layers.block import Block
    
    if quant_blocks is None:
        logger.warning("No quant_blocks specified for VGGT smoothing")
        return
    
    smoothed_count = 0
    
    # Expand blocks if they are ModuleList/Sequential (reuse GPTQ's logic)
    expanded_blocks = []
    for block_name in quant_blocks:
        try:
            parent, leaf = _resolve_parent_and_leaf(model, block_name)
            block_module = getattr(parent, leaf) if hasattr(parent, leaf) else parent[int(leaf)]
            
            if isinstance(block_module, (nn.ModuleList, nn.Sequential)):
                # Process each submodule in the list
                for i in range(len(block_module)):
                    expanded_blocks.append(f"{block_name}.{i}")
            else:
                expanded_blocks.append(block_name)
        except Exception as e:
            logger.warning(f"Could not resolve block {block_name}: {e}")
            continue
    
    logger.info(f"Processing {len(expanded_blocks)} VGGT blocks for smoothing")
    
    # Process each block
    for block_name in expanded_blocks:
        try:
            parent, leaf = _resolve_parent_and_leaf(model, block_name)
            block_module = getattr(parent, leaf) if hasattr(parent, leaf) else parent[int(leaf)]
            
            if isinstance(block_module, Block):
                # Smooth attention: norm1 -> attn.qkv
                qkv_name = f"{block_name}.attn.qkv"
                if qkv_name in act_scales and hasattr(block_module, 'norm1') and hasattr(block_module.attn, 'qkv'):
                    logger.debug(f"Smoothing {block_name}.norm1 -> {block_name}.attn.qkv")
                    smooth_ln_fcs(block_module.norm1, [block_module.attn.qkv], act_scales[qkv_name], alpha)
                    smoothed_count += 1
                
                # Smooth MLP: norm2 -> mlp.fc1
                fc1_name = f"{block_name}.mlp.fc1"
                if fc1_name in act_scales and hasattr(block_module, 'norm2') and hasattr(block_module.mlp, 'fc1'):
                    logger.debug(f"Smoothing {block_name}.norm2 -> {block_name}.mlp.fc1")
                    smooth_ln_fcs(block_module.norm2, [block_module.mlp.fc1], act_scales[fc1_name], alpha)
                    smoothed_count += 1
            else:
                # Not a Block, skip
                logger.debug(f"Skipping {block_name} - not a Block instance")
                
        except Exception as e:
            logger.warning(f"Error processing block {block_name}: {e}")
            continue
    
    logger.info(f"Smoothed {smoothed_count} layer pairs in VGGT blocks")

def _resolve_parent_and_leaf(model, dotted_path):
    """
    Given "foo.bar.3.baz", return (parent_module, leaf_name_or_index).
    Handles ModuleList / Sequential indices.
    (Borrowed from GPTQ implementation)
    """
    obj = model
    parts = dotted_path.split(".")
    for p in parts[:-1]:
        if hasattr(obj, p):
            obj = getattr(obj, p)
        elif p.isdigit() and isinstance(obj, (nn.ModuleList, nn.Sequential)):
            obj = obj[int(p)]
        else:
            raise AttributeError(f"Cannot resolve '{p}' in '{dotted_path}'")
    return obj, parts[-1]

def apply_smoothquant(model, dataloader, args):
    """Apply SmoothQuant smoothing to model"""
    model.eval()

    logger.info("Collecting activation scales for smoothing with {} samples...".format(args.calibration_samples))
    act_scales = collect_activation_scales_smoothquant(model, dataloader, args)

    
    # Get top_k parameter (default to None for smoothing all layers)
    top_k = getattr(args, 'smoothquant_top_k', None)
    alpha = getattr(args, 'smoothquant_alpha', 0.5)
    
    if args.dataset_name == 'co3d':
        # VGGT smoothing
        quant_blocks = getattr(args, 'quant_blocks', None)
        logger.info(f"Applying SmoothQuant smoothing to VGGT with alpha={alpha}...")
        smooth_vggt_blocks(model, act_scales, alpha=alpha, quant_blocks=quant_blocks)
    else:
        # ViT smoothing (ImageNet)
        if top_k is not None:
            logger.info(f"Applying SmoothQuant smoothing to top {top_k} layers with alpha={alpha}...")
        else:
            logger.info(f"Applying SmoothQuant smoothing to all layers with alpha={alpha}...")
        
        smooth_vit_blocks(model, act_scales, alpha=alpha, top_k=top_k)
    
    return model

def disable_quantizers_for_layers(net, layer_patterns):
    """
    Disable w_quantizer and x_quantizer for layers matching specific patterns.
    
    Args:
        net: The neural network model
        layer_patterns: List of string patterns to match in layer names (e.g., ['.attn.proj', '.mlp.fc2'])
    """
    disabled_layers = []
    disabled_count = 0
    
    for name, module in net.named_modules():
        # Check if this layer name matches any of the patterns
        if any(pattern in name for pattern in layer_patterns):
            disabled_this_layer = False
            if hasattr(module, 'w_quantizer') and module.w_quantizer is not None:
                module.w_quantizer = None
                disabled_count += 1
                disabled_this_layer = True
                logger.info(f"  ✓ Disabled w_quantizer for: {name}")
            if hasattr(module, 'x_quantizer') and module.x_quantizer is not None:
                module.x_quantizer = None
                disabled_count += 1
                disabled_this_layer = True
                logger.info(f"  ✓ Disabled x_quantizer for: {name}")
            
            if disabled_this_layer:
                disabled_layers.append(name)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"QUANTIZER DISABLING SUMMARY:")
    logger.info(f"  Total layers affected: {len(disabled_layers)}")
    logger.info(f"  Total quantizers disabled: {disabled_count}")
    logger.info(f"  Patterns used: {layer_patterns}")
    logger.info(f"{'='*60}\n")
    
    return net

def verify_quantizer_status(net, layer_patterns=None):
    """
    Verify and print the quantizer status for all layers or specific patterns.
    
    Args:
        net: The neural network model
        layer_patterns: Optional list of patterns to filter layers
    """
    logger.info("\n" + "="*60)
    logger.info("QUANTIZER STATUS VERIFICATION:")
    logger.info("="*60)
    
    quantized_count = 0
    fp_count = 0
    
    for name, module in net.named_modules():
        if hasattr(module, 'w_quantizer') or hasattr(module, 'x_quantizer'):
            # If patterns specified, only show matching layers
            if layer_patterns and not any(pattern in name for pattern in layer_patterns):
                continue
            
            w_status = "✓ Quantized" if (hasattr(module, 'w_quantizer') and module.w_quantizer is not None) else "✗ FP32"
            x_status = "✓ Quantized" if (hasattr(module, 'x_quantizer') and module.x_quantizer is not None) else "✗ FP32"
            
            if w_status == "✗ FP32" or x_status == "✗ FP32":
                fp_count += 1
                logger.info(f"  {name}:")
                logger.info(f"    w_quantizer: {w_status}")
                logger.info(f"    x_quantizer: {x_status}")
            else:
                quantized_count += 1
    
    logger.info(f"\nSummary:")
    logger.info(f"  Layers with FP32 (no quantization): {fp_count}")
    logger.info(f"  Layers fully quantized: {quantized_count}")
    logger.info("="*60 + "\n")
    
    return fp_count, quantized_count

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
    # This creates FIVE naive variants for ablation study:
    # 3a. Naive quantization (all layers)
    # 3b. Naive quantization with proj/fc2 disabled
    # 3c. Naive quantization with ONLY proj/fc2 quantized (qkv/fc1 disabled)
    # 3d. Naive quantization with ONLY attention layers (proj disabled)
    # 3e. Naive quantization with ONLY MLP layers (fc2 disabled)
    naive_lsq_all = None
    naive_lsq_no_proj_fc2 = None
    naive_lsq_only_proj_fc2 = None
    
    if original_results is None:
        # 3a. Naive quantization - ALL layers quantized
        logger.info("\nStep 3a: Creating NAIVE quantization model (all layers quantized)...")
        naive_lsq_all = replace(args, create_model(args).to(args.device))
        initialize_lsq_scales(naive_lsq_all, calibration_loader, args)
        
        # 3b. Naive quantization - with proj/fc2 DISABLED
        logger.info("\nStep 3b: Creating NAIVE quantization model (NO proj/fc2)...")
        naive_lsq_no_proj_fc2 = replace(args, create_model(args).to(args.device))
        logger.info("\n==> Disabling quantizers for .attn.proj and .mlp.fc2 layers")
        naive_lsq_no_proj_fc2 = disable_quantizers_for_layers(naive_lsq_no_proj_fc2, ['.attn.proj', '.mlp.fc2'])
        verify_quantizer_status(naive_lsq_no_proj_fc2, layer_patterns=['.attn.proj', '.mlp.fc2'])
        initialize_lsq_scales(naive_lsq_no_proj_fc2, calibration_loader, args)
        
        # 3c. Naive quantization - ONLY proj/fc2 quantized (qkv/fc1 disabled)
        logger.info("\nStep 3c: Creating NAIVE quantization model (ONLY proj/fc2)...")
        naive_lsq_only_proj_fc2 = replace(args, create_model(args).to(args.device))
        logger.info("\n==> Disabling quantizers for .attn.qkv and .mlp.fc1 layers")
        naive_lsq_only_proj_fc2 = disable_quantizers_for_layers(naive_lsq_only_proj_fc2, ['.attn.qkv', '.mlp.fc1'])
        verify_quantizer_status(naive_lsq_only_proj_fc2, layer_patterns=['.attn.qkv', '.mlp.fc1'])
        initialize_lsq_scales(naive_lsq_only_proj_fc2, calibration_loader, args)
    else:
        logger.info("\nStep 3: Skipping naive quantization models (already done)...")

    # Step 4: Apply LSQ replacement to smoothed models
    # IMPORTANT: We need to save the FP32 smoothed weights before any LSQ replacement,
    # since replace() modifies the model in-place
    logger.info("\nStep 4: Saving smoothed FP32 weights for reuse...")
    smoothed_fp_weights = smoothed_fp_model.state_dict()
    
    # Step 4a: SmoothQuant with ALL layers quantized
    logger.info("\nStep 4a: Creating SMOOTHQUANT model (all layers quantized)...")
    smoothed_lsq_all = replace(args, smoothed_fp_model)
    smoothed_lsq_all = smoothed_lsq_all.to(args.device)
    initialize_lsq_scales(smoothed_lsq_all, calibration_loader, args)

    # Step 4b: SmoothQuant with proj/fc2 DISABLED
    logger.info("\nStep 4b: Creating SMOOTHQUANT model (proj/fc2 disabled)...")
    # Create a fresh FP32 model and load the smoothed weights
    smoothed_fp_model_copy = create_model(args).to(args.device)
    smoothed_fp_model_copy.load_state_dict(smoothed_fp_weights)
    
    # Now apply LSQ replacement
    smoothed_lsq_selective = replace(args, smoothed_fp_model_copy)
    smoothed_lsq_selective = smoothed_lsq_selective.to(args.device)
    logger.info("\n==> Disabling quantizers for .attn.proj and .mlp.fc2 layers (smoothquant selective)")
    smoothed_lsq_selective = disable_quantizers_for_layers(smoothed_lsq_selective, ['.attn.proj', '.mlp.fc2'])
    verify_quantizer_status(smoothed_lsq_selective, layer_patterns=['.attn.proj', '.mlp.fc2'])
    initialize_lsq_scales(smoothed_lsq_selective, calibration_loader, args)

    # Save LSQ models
    if naive_lsq_all is not None:
        naive_all_path = os.path.join(args.output_path, "lsq_naive_all_layers.pt")
        torch.save(naive_lsq_all.state_dict(), naive_all_path)
        logger.info(f"Saved naive LSQ model (all layers) to {naive_all_path}")
    
    if naive_lsq_no_proj_fc2 is not None:
        naive_no_proj_fc2_path = os.path.join(args.output_path, "lsq_naive_no_proj_fc2.pt")
        torch.save(naive_lsq_no_proj_fc2.state_dict(), naive_no_proj_fc2_path)
        logger.info(f"Saved naive LSQ model (no proj/fc2) to {naive_no_proj_fc2_path}")
    
    if naive_lsq_only_proj_fc2 is not None:
        naive_only_proj_fc2_path = os.path.join(args.output_path, "lsq_naive_only_proj_fc2.pt")
        torch.save(naive_lsq_only_proj_fc2.state_dict(), naive_only_proj_fc2_path)
        logger.info(f"Saved naive LSQ model (only proj/fc2) to {naive_only_proj_fc2_path}")
    
    smoothed_all_path = os.path.join(args.output_path, f"lsq_smoothquant_all_layers_alpha_{alpha_value}.pt")
    torch.save(smoothed_lsq_all.state_dict(), smoothed_all_path)
    logger.info(f"Saved smoothed LSQ model (all layers) to {smoothed_all_path}")
    
    smoothed_selective_path = os.path.join(args.output_path, f"lsq_smoothquant_selective_alpha_{alpha_value}.pt")
    torch.save(smoothed_lsq_selective.state_dict(), smoothed_selective_path)
    logger.info(f"Saved smoothed LSQ model (selective) to {smoothed_selective_path}")

    # Evaluation - ABLATION STUDY
    result_dict = {'alpha': alpha_value}
    if args.dataset_name in ['imagenet', 'imagenet-mini']:
        task_loss_fn = nn.CrossEntropyLoss()
        criterion_val = Multiple_Loss({"task_loss": task_loss_fn})
        
        # Evaluate all models only if not already done
        if original_results is None:
            logger.info("\n" + "="*70)
            logger.info("ABLATION STUDY: Evaluating all quantization variants")
            logger.info("="*70)
            
            # 1. Naive quantization - ALL layers
            logger.info("\n1. Evaluating NAIVE quantization (all layers)...")
            val_acc_naive_all, val_top5_naive_all, _, _, val_loss_naive_all = run_one_epoch(
                naive_lsq_all, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False
            )
            logger.info(f'[Naive All] val_Loss: {val_loss_naive_all["task_loss"].item():.5f}, val_top1_Acc: {val_acc_naive_all:.5f}, val_top5_Acc: {val_top5_naive_all:.5f}')
            
            # 2. Naive quantization - NO proj/fc2
            logger.info("\n2. Evaluating NAIVE quantization (NO proj/fc2)...")
            val_acc_naive_no_proj_fc2, val_top5_naive_no_proj_fc2, _, _, val_loss_naive_no_proj_fc2 = run_one_epoch(
                naive_lsq_no_proj_fc2, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False
            )
            logger.info(f'[Naive No Proj/FC2] val_Loss: {val_loss_naive_no_proj_fc2["task_loss"].item():.5f}, val_top1_Acc: {val_acc_naive_no_proj_fc2:.5f}, val_top5_Acc: {val_top5_naive_no_proj_fc2:.5f}')
            
            # 3. Naive quantization - ONLY proj/fc2
            logger.info("\n3. Evaluating NAIVE quantization (ONLY proj/fc2)...")
            val_acc_naive_only_proj_fc2, val_top5_naive_only_proj_fc2, _, _, val_loss_naive_only_proj_fc2 = run_one_epoch(
                naive_lsq_only_proj_fc2, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False
            )
            logger.info(f'[Naive Only Proj/FC2] val_Loss: {val_loss_naive_only_proj_fc2["task_loss"].item():.5f}, val_top1_Acc: {val_acc_naive_only_proj_fc2:.5f}, val_top5_Acc: {val_top5_naive_only_proj_fc2:.5f}')
        else:
            logger.info("\n" + "="*70)
            logger.info("ABLATION STUDY: Using cached naive quantization results")
            logger.info("="*70)
            val_acc_naive_all = original_results['val_acc_naive_all']
            val_top5_naive_all = original_results['val_top5_naive_all']
            val_loss_naive_all = original_results['val_loss_naive_all']
            val_acc_naive_no_proj_fc2 = original_results['val_acc_naive_no_proj_fc2']
            val_top5_naive_no_proj_fc2 = original_results['val_top5_naive_no_proj_fc2']
            val_loss_naive_no_proj_fc2 = original_results['val_loss_naive_no_proj_fc2']
            val_acc_naive_only_proj_fc2 = original_results['val_acc_naive_only_proj_fc2']
            val_top5_naive_only_proj_fc2 = original_results['val_top5_naive_only_proj_fc2']
            val_loss_naive_only_proj_fc2 = original_results['val_loss_naive_only_proj_fc2']
            logger.info(f'[Naive All] val_Loss: {val_loss_naive_all["task_loss"].item():.5f}, val_top1_Acc: {val_acc_naive_all:.5f}, val_top5_Acc: {val_top5_naive_all:.5f}')
            logger.info(f'[Naive No Proj/FC2] val_Loss: {val_loss_naive_no_proj_fc2["task_loss"].item():.5f}, val_top1_Acc: {val_acc_naive_no_proj_fc2:.5f}, val_top5_Acc: {val_top5_naive_no_proj_fc2:.5f}')
            logger.info(f'[Naive Only Proj/FC2] val_Loss: {val_loss_naive_only_proj_fc2["task_loss"].item():.5f}, val_top1_Acc: {val_acc_naive_only_proj_fc2:.5f}, val_top5_Acc: {val_top5_naive_only_proj_fc2:.5f}')
        
        # 3. SmoothQuant - ALL layers
        logger.info(f"\n3. Evaluating SMOOTHQUANT (all layers, alpha={alpha_value})...")
        val_acc_smooth_all, val_top5_smooth_all, _, _, val_loss_smooth_all = run_one_epoch(
            smoothed_lsq_all, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False
        )
        logger.info(f'[SmoothQuant All] val_Loss: {val_loss_smooth_all["task_loss"].item():.5f}, val_top1_Acc: {val_acc_smooth_all:.5f}, val_top5_Acc: {val_top5_smooth_all:.5f}')
        
        # 4. SmoothQuant - proj/fc2 DISABLED
        logger.info(f"\n4. Evaluating SMOOTHQUANT (proj/fc2 disabled, alpha={alpha_value})...")
        val_acc_smooth_sel, val_top5_smooth_sel, _, _, val_loss_smooth_sel = run_one_epoch(
            smoothed_lsq_selective, dataloader, None, criterion_val, 0, "val", 0, args, ddp_initialized=False
        )
        logger.info(f'[SmoothQuant Selective] val_Loss: {val_loss_smooth_sel["task_loss"].item():.5f}, val_top1_Acc: {val_acc_smooth_sel:.5f}, val_top5_Acc: {val_top5_smooth_sel:.5f}')

        # Calculate diffs for storage
        diff_naive_no_proj_fc2 = val_acc_naive_no_proj_fc2 - val_acc_naive_all
        diff_naive_only_proj_fc2 = val_acc_naive_only_proj_fc2 - val_acc_naive_all
        diff_smooth_all = val_acc_smooth_all - val_acc_naive_all
        diff_smooth_sel = val_acc_smooth_sel - val_acc_naive_all
        
        # Store results for sweep comparison
        result_dict.update({
            'naive_all_top1': val_acc_naive_all * 100,
            'naive_all_top5': val_top5_naive_all * 100,
            'naive_all_loss': val_loss_naive_all['task_loss'].item(),
            
            'naive_no_proj_fc2_top1': val_acc_naive_no_proj_fc2 * 100,
            'naive_no_proj_fc2_top5': val_top5_naive_no_proj_fc2 * 100,
            'naive_no_proj_fc2_loss': val_loss_naive_no_proj_fc2['task_loss'].item(),
            
            'naive_only_proj_fc2_top1': val_acc_naive_only_proj_fc2 * 100,
            'naive_only_proj_fc2_top5': val_top5_naive_only_proj_fc2 * 100,
            'naive_only_proj_fc2_loss': val_loss_naive_only_proj_fc2['task_loss'].item(),
            
            'smooth_all_top1': val_acc_smooth_all * 100,
            'smooth_all_top5': val_top5_smooth_all * 100,
            'smooth_all_loss': val_loss_smooth_all['task_loss'].item(),
            
            'smooth_sel_top1': val_acc_smooth_sel * 100,
            'smooth_sel_top5': val_top5_smooth_sel * 100,
            'smooth_sel_loss': val_loss_smooth_sel['task_loss'].item(),
            
            'diff_naive_no_proj_fc2': diff_naive_no_proj_fc2 * 100,
            'diff_naive_only_proj_fc2': diff_naive_only_proj_fc2 * 100,
            'diff_smooth_all': diff_smooth_all * 100,
            'diff_smooth_sel': diff_smooth_sel * 100,
        })

    elif args.dataset_name == 'co3d':
        if naive_lsq_all is not None:
            logger.info("\n1. Evaluating naive LSQ model (all layers)...")
            run_evaluation_vggt(naive_lsq_all)
            logger.info("\n2. Evaluating naive LSQ model (NO proj/fc2)...")
            run_evaluation_vggt(naive_lsq_no_proj_fc2)
            logger.info("\n3. Evaluating naive LSQ model (ONLY proj/fc2)...")
            run_evaluation_vggt(naive_lsq_only_proj_fc2)
        logger.info("\n4. Evaluating smoothed LSQ model (all layers)...")
        run_evaluation_vggt(smoothed_lsq_all)
        logger.info("\n5. Evaluating smoothed LSQ model (NO proj/fc2)...")
        run_evaluation_vggt(smoothed_lsq_selective)
    
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    
    # Prepare original results for reuse (cache naive quantization results)
    if original_results is None:
        original_results = {
            'val_acc_naive_all': val_acc_naive_all,
            'val_top5_naive_all': val_top5_naive_all,
            'val_loss_naive_all': val_loss_naive_all,
            'val_acc_naive_no_proj_fc2': val_acc_naive_no_proj_fc2,
            'val_top5_naive_no_proj_fc2': val_top5_naive_no_proj_fc2,
            'val_loss_naive_no_proj_fc2': val_loss_naive_no_proj_fc2,
            'val_acc_naive_only_proj_fc2': val_acc_naive_only_proj_fc2,
            'val_top5_naive_only_proj_fc2': val_top5_naive_only_proj_fc2,
            'val_loss_naive_only_proj_fc2': val_loss_naive_only_proj_fc2,
        }
    
    return smoothed_lsq_selective, original_fp_model, original_results, result_dict


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
    
    # Print all results at the end
    if len(results_summary) > 0 and 'smooth_sel_top1' in results_summary[0]:
        logger.info(f"\n{'='*140}")
        logger.info(f"FINAL ABLATION STUDY RESULTS - ALL ALPHAS")
        logger.info(f"{'='*140}\n")
        
        # Print detailed results for each alpha
        for result in results_summary:
            alpha = result['alpha']
            logger.info(f"{'='*100}")
            logger.info(f"ALPHA = {alpha}")
            logger.info(f"{'='*100}")
            logger.info(f"{'Model':<45} {'Loss':>10} {'Top-1 Acc':>12} {'Top-5 Acc':>12} {'Acc Diff':>12}")
            logger.info(f"{'-'*100}")
            
            logger.info(f"{'1. Naive (all layers)':<45} {result['naive_all_loss']:>10.5f} {result['naive_all_top1']:>11.4f}% {result['naive_all_top5']:>11.4f}% {'baseline':>12}")
            logger.info(f"{'2. Naive (NO proj/fc2, keep qkv/fc1)':<45} {result['naive_no_proj_fc2_loss']:>10.5f} {result['naive_no_proj_fc2_top1']:>11.4f}% {result['naive_no_proj_fc2_top5']:>11.4f}% {result['diff_naive_no_proj_fc2']:>+11.4f}%")
            logger.info(f"{'3. Naive (ONLY proj/fc2, NO qkv/fc1)':<45} {result['naive_only_proj_fc2_loss']:>10.5f} {result['naive_only_proj_fc2_top1']:>11.4f}% {result['naive_only_proj_fc2_top5']:>11.4f}% {result['diff_naive_only_proj_fc2']:>+11.4f}%")
            logger.info(f"{'4. SmoothQuant (all layers)':<45} {result['smooth_all_loss']:>10.5f} {result['smooth_all_top1']:>11.4f}% {result['smooth_all_top5']:>11.4f}% {result['diff_smooth_all']:>+11.4f}%")
            logger.info(f"{'5. SmoothQuant (NO proj/fc2)':<45} {result['smooth_sel_loss']:>10.5f} {result['smooth_sel_top1']:>11.4f}% {result['smooth_sel_top5']:>11.4f}% {result['diff_smooth_sel']:>+11.4f}%")
            
            logger.info(f"{'='*100}")
            logger.info(f"Key Insights:")
            logger.info(f"  • Removing proj/fc2 from naive: {result['diff_naive_no_proj_fc2']:+.4f}%")
            logger.info(f"  • Quantizing ONLY proj/fc2: {result['diff_naive_only_proj_fc2']:+.4f}%")
            logger.info(f"  • SmoothQuant all vs naive all: {result['diff_smooth_all']:+.4f}%")
            logger.info(f"  • SmoothQuant selective vs naive all: {result['diff_smooth_sel']:+.4f}%")
            
            # Calculate layer sensitivity
            qkv_fc1_impact = result['naive_no_proj_fc2_top1'] - result['naive_only_proj_fc2_top1']
            proj_fc2_impact = result['diff_naive_only_proj_fc2']
            smooth_sel_vs_naive_sel = result['smooth_sel_top1'] - result['naive_no_proj_fc2_top1']
            
            logger.info(f"\nLayer Sensitivity Analysis:")
            logger.info(f"  • qkv/fc1 quantization impact: {qkv_fc1_impact:+.4f}%")
            logger.info(f"  • proj/fc2 quantization impact: {proj_fc2_impact:+.4f}%")
            logger.info(f"  • SmoothQuant selective vs naive selective: {smooth_sel_vs_naive_sel:+.4f}%")
            logger.info(f"{'='*100}\n")
        
        # Print compact comparison table
        if is_sweep and len(results_summary) > 1:
            logger.info(f"\n{'='*140}")
            logger.info(f"ALPHA COMPARISON SUMMARY")
            logger.info(f"{'='*140}")
            logger.info(f"{'Alpha':<8} {'Naive All':>12} {'No Proj/FC2':>12} {'Only Proj/FC2':>13} {'Smooth All':>12} {'Smooth Sel':>12} {'Best':>18}")
            logger.info(f"{'-'*140}")
            
            best_alpha = None
            best_acc = -float('inf')
            best_method = None
            
            for result in results_summary:
                alpha = result['alpha']
                naive_all = result.get('naive_all_top1', 0)
                naive_no = result.get('naive_no_proj_fc2_top1', 0)
                naive_only = result.get('naive_only_proj_fc2_top1', 0)
                smooth_all = result.get('smooth_all_top1', 0)
                smooth_sel = result.get('smooth_sel_top1', 0)
                
                # Find best for this alpha
                methods = [
                    ('Naive All', naive_all),
                    ('No Proj/FC2', naive_no),
                    ('Only Proj/FC2', naive_only),
                    ('Smooth All', smooth_all),
                    ('Smooth Sel', smooth_sel)
                ]
                local_best_method, local_best_acc = max(methods, key=lambda x: x[1])
                
                # Track global best
                if local_best_acc > best_acc:
                    best_acc = local_best_acc
                    best_alpha = alpha
                    best_method = local_best_method
                
                best_marker = f"✓ {local_best_method}"
                logger.info(f"{alpha:<8.2f} {naive_all:>11.4f}% {naive_no:>11.4f}% {naive_only:>12.4f}% {smooth_all:>11.4f}% {smooth_sel:>11.4f}% {best_marker:>18}")
            
            logger.info(f"{'-'*140}")
            logger.info(f"Global Best: {best_method} @ alpha={best_alpha} with {best_acc:.4f}% top-1 accuracy")
            logger.info(f"{'='*140}\n")
    
    return smoothed_model
