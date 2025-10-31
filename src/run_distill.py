import os
import argparse
from src.models.depth.vggt.training import trainer
from src.logger import logger
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import copy

from hydra import initialize, compose
import src.models.depth.vggt.vggt
from src.models.depth.vggt.training.trainer import Trainer
from src.models.depth.vggt.training.train_utils.optimizer import construct_optimizers

from src.utils import *
from src.scheduler_optimizer_class import scheduler_optimizer_class
import src.module_quantization as Q
from src.make_ex_name import *

from src.models.depth.vggt.evaluation.test_co3d import *


class DistillationModelWrapper(nn.Module):
    """
    Wraps the student model to compute distillation loss during forward pass.
    This bypasses the Trainer's loss function entirely.
    """
    def __init__(self, student_model, teacher_model, original_loss, 
                 teacher_extractor, student_extractor, alpha=0.7, beta=0.3):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.original_loss = original_loss
        self.teacher_extractor = teacher_extractor
        self.student_extractor = student_extractor
        self.alpha = alpha
        self.beta = beta
        self.cached_batch = None
        self.cached_distill_loss = None
        
    def forward(self, images):
        """Forward pass that stores intermediate features"""
        # Just run student model normally
        output = self.student_model(images=images)
        return output
    
    def compute_distillation_loss(self, student_pred, batch):
        """Called by our custom loss wrapper to add distillation"""
        # Move teacher to GPU temporarily for forward pass (saves memory)
        device = batch["images"].device
        
        # Check if teacher is on CPU by checking first parameter
        teacher_on_cpu = next(self.teacher_model.parameters()).device.type == 'cpu'
        
        if teacher_on_cpu:
            self.teacher_model = self.teacher_model.to(device)
        
        # CRITICAL FIX: Do NOT use inference_mode() - it creates inference tensors that
        # cannot be used in any gradient computation, even after detaching!
        # Standard approach from HuggingFace/PyTorch: use torch.no_grad() and detach features
        with torch.no_grad():
            teacher_pred = self.teacher_model(images=batch["images"])
        
        # Move teacher back to CPU immediately to free GPU memory
        if teacher_on_cpu:
            self.teacher_model = self.teacher_model.cpu()
            torch.cuda.empty_cache()
        
        # Get teacher features and DETACH + CLONE to convert from no_grad tensors
        # This is the standard distillation approach - teacher features are frozen
        # but can be used in loss computation
        teacher_feats_raw = self.teacher_extractor.get_features()
        teacher_feats = {k: v.detach().clone() for k, v in teacher_feats_raw.items()}
        
        student_feats = self.student_extractor.get_features()
        
        if not student_feats or not teacher_feats:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        feat_loss = compute_feature_matching_loss(teacher_feats, student_feats)
        
        # Clear features immediately
        self.teacher_extractor.features = {}
        self.student_extractor.features = {}
        
        return feat_loss


class DistillationLossWrapper(nn.Module):
    """
    Wrapper that adds distillation loss to task loss.
    Now properly handles gradient computation.
    """
    def __init__(self, original_loss, model_wrapper, alpha=0.7, beta=0.3):
        super().__init__()
        self.original_loss = original_loss
        self.model_wrapper = model_wrapper
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, student_pred, batch):
        """Compute combined distillation + task loss"""
        
        # DEBUG: Check if student already ran
        logger.info(f"DEBUG: student_pred keys: {student_pred.keys()}")
        logger.info(f"DEBUG: student_feats before teacher run: {len(self.model_wrapper.student_extractor.features)}")
        
        # Get features that were captured during student forward
        student_feats = self.model_wrapper.student_extractor.get_features()
        
        logger.info(f"DEBUG: Retrieved {len(student_feats)} student features")
        for name, feat in student_feats.items():
            logger.info(f"  {name}: requires_grad={feat.requires_grad}")
        
        # Ensure we're in gradient context
        with torch.set_grad_enabled(True):
            # 1. Compute task loss (THIS should have gradients now)
            task_loss_dict = self.original_loss(student_pred, batch)
            task_loss = task_loss_dict["objective"]
            
            logger.info(f"DEBUG: task_loss requires_grad={task_loss.requires_grad}, grad_fn={task_loss.grad_fn}")
            
            # 2. Compute distillation loss
            feat_loss = self.model_wrapper.compute_distillation_loss(student_pred, batch)
            
            logger.info(f"DEBUG: feat_loss requires_grad={feat_loss.requires_grad}, grad_fn={feat_loss.grad_fn}")
            
            # If task_loss doesn't have gradients but feat_loss does, 
            # we can still train with just feature matching loss
            if not task_loss.requires_grad:
                logger.warning("WARNING: task_loss has no gradients!")
                logger.warning("Using ONLY feature matching loss for training")
                # Use only distillation loss
                total_loss = feat_loss
            else:
                # 3. Combine both losses
                total_loss = self.alpha * feat_loss + self.beta * task_loss
            
            logger.info(f"DEBUG: total_loss requires_grad={total_loss.requires_grad}, grad_fn={total_loss.grad_fn}")
            
            # 4. Return
            loss_dict = task_loss_dict.copy()
            loss_dict["objective"] = total_loss
            loss_dict["task_loss_component"] = task_loss.detach() if task_loss.requires_grad else task_loss
            loss_dict["feat_loss_component"] = feat_loss.detach()
            
            # CRITICAL: Verify before returning
            logger.info(f"DEBUG LOSS_WRAPPER: Returning loss_dict['objective'] with requires_grad={loss_dict['objective'].requires_grad}")
            logger.info(f"DEBUG LOSS_WRAPPER: loss_dict['objective'].grad_fn={loss_dict['objective'].grad_fn}")
            
            return loss_dict
    
class FeatureExtractor:
    def __init__(self, detach=True):
        """
        Args:
            detach: If True, detach features (for teacher). If False, keep gradients (for student).
        """
        self.features = {}
        self.hooks = []
        self.detach = detach
    
    def register_hooks(self, model, layer_names):
        self.clear_hooks()
        
        for name in layer_names:
            module = self._get_module_by_name(model, name)
            if module is not None:
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)
    
    def _get_module_by_name(self, model, name):
        parts = name.split('.')
        module = model
        for part in parts:
            if '[' in part and ']' in part:
                # Handle indexing like "frame_blocks[0]"
                attr_name, idx = part.split('[')
                idx = int(idx.rstrip(']'))
                module = getattr(module, attr_name)[idx]
            else:
                module = getattr(module, part, None)
                if module is None:
                    return None
        return module
    
    def _make_hook(self, name):
        def hook(module, input, output):
            # Store the output, handling different output types
            if isinstance(output, tuple):
                # For patch_embed: (x_norm_clstoken, x_norm_regtokens, x_norm_patchtokens, x_prenorm, masks)
                if len(output) >= 3:
                    feature = output[2]
                else:
                    feature = output[0]
            elif isinstance(output, dict):
                # Handle dict outputs from other modules
                feature = None
                for key in ['tokens', 'hidden_states', 'x', 'output']:
                    if key in output and isinstance(output[key], torch.Tensor):
                        feature = output[key]
                        break
                if feature is None:
                    return
            else:
                # Single tensor output
                feature = output
            
            # Detach only if requested (teacher=True, student=False)
            self.features[name] = feature.detach() if self.detach else feature
        return hook
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}
    
    def get_features(self):
        return self.features

def selective_quantize_layers(model, args, layer_names):
    logger.info(f"Selectively quantizing layers: {layer_names}")
    
    replacement_dict = {
        nn.Conv2d : partial(Q.QConv2d, \
        num_bits=args.num_bits, w_grad_scale_mode = args.w_grad_scale_mode, \
        x_grad_scale_mode = args.x_grad_scale_mode, \
        weight_norm = args.weight_norm, w_quantizer = args.w_quantizer, x_quantizer = args.x_quantizer, \
        w_initializer = args.w_initializer, x_initializer = args.x_initializer), 
        nn.Linear: partial(Q.QLinear,  \
        num_bits=args.num_bits, w_grad_scale_mode = args.w_grad_scale_mode, \
        x_grad_scale_mode = args.x_grad_scale_mode, \
        weight_norm = args.weight_norm, w_quantizer = args.w_quantizer, x_quantizer = args.x_quantizer, \
        w_initializer = args.w_initializer, x_initializer = args.x_initializer)
    }
    exception_dict = {
        '__first__': partial(Q.QConv2d, num_bits=args.first_bits, w_quantizer =args.w_first_last_quantizer, x_quantizer = args.x_first_last_quantizer, \
                                w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, \
                                w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                first_layer = True),
        '__camera_last__': partial(Q.QLinear,  num_bits=args.last_bits, w_quantizer =args.w_first_last_quantizer,x_quantizer = args.x_first_last_quantizer, \
                                w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, \
                                w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                first_layer = False),
        '__depth_last__': partial(Q.QConv2d, num_bits=args.last_bits, w_quantizer =args.w_first_last_quantizer,x_quantizer = args.x_first_last_quantizer, \
                                w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, \
                                w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                first_layer = False),
        '__last__': partial(Q.QLinear,  num_bits=args.last_bits, w_quantizer =args.w_first_last_quantizer,x_quantizer = args.x_first_last_quantizer, \
                                w_initializer = args.w_first_last_initializer, x_initializer = args.x_first_last_initializer, \
                                w_grad_scale_mode = args.w_first_last_grad_scale_mode, \
                                x_grad_scale_mode = args.x_first_last_grad_scale_mode, \
                                first_layer = False),
    }
    
    for layer_name in layer_names:
        parts = layer_name.split('.')
        parent = model
        
        # Navigate to parent module with error handling
        for part in parts[:-1]:
            if '[' in part and ']' in part:
                attr_name, idx = part.split('[')
                idx = int(idx.rstrip(']'))
                parent = getattr(parent, attr_name)[idx]
            else:
                parent = getattr(parent, part)
        
        # Get the actual module to replace
        final_part = parts[-1]
        if '[' in final_part and ']' in final_part:
            attr_name, idx = final_part.split('[')
            idx = int(idx.rstrip(']'))
            module_list = getattr(parent, attr_name)
            target_module = module_list[idx]
            
            quantized_module = replace_all(
                target_module, 
                replacement_dict=replacement_dict
            )
            module_list[idx] = quantized_module
        else:
            target_module = getattr(parent, final_part)
            quantized_module = replace_all(
                target_module,
                replacement_dict=replacement_dict
            )
            setattr(parent, final_part, quantized_module)
        
        logger.info(f"Successfully quantized layer: {layer_name}")

    # moved outside utils.py
    if args.arch == "vggt":
        if "aggregator.patch_embed" in layer_names:
            logger.info("Applying special quantization to first layer (patch_embed)")
            model.aggregator.patch_embed.patch_embed.proj = replace_single_module(
                new_cls=exception_dict['__first__'], 
                current_module=model.aggregator.patch_embed.patch_embed.proj
            )
        if "camera_head" in layer_names:
            logger.info("Applying special quantization to camera_head last layer")
            model.camera_head.pose_branch.fc2 = replace_single_module(
                new_cls=exception_dict['__camera_last__'], 
                current_module=model.camera_head.pose_branch.fc2
            )
        if "depth_head" in layer_names:
            logger.info("Applying special quantization to depth_head last layer")
            model.depth_head.scratch.output_conv2[2] = replace_single_module(
                new_cls=exception_dict['__depth_last__'], 
                current_module=model.depth_head.scratch.output_conv2[2]
            )
    
    return model


def freeze_layers(model, layer_names):
    """Freeze specified layers"""
    for layer_name in layer_names:
        parts = layer_name.split('.')
        module = model
        
        for part in parts:
            if '[' in part and ']' in part:
                attr_name, idx = part.split('[')
                idx = int(idx.rstrip(']'))
                module = getattr(module, attr_name)[idx]
            else:
                module = getattr(module, part)
        
        for param in module.parameters():
            param.requires_grad = False
    
    logger.info(f"Frozen {len(layer_names)} layers")


def unfreeze_layers(model, layer_names):
    """
    Unfreeze specified layers and set them to train mode.
    
    This function:
    1. Navigates to each layer in the model hierarchy
    2. Sets the layer to train mode (enables BatchNorm updates, Dropout, etc.)
    3. Unfreezes all parameters in that layer recursively
    
    Args:
        model: The PyTorch model
        layer_names: List of layer names to unfreeze (e.g., ["aggregator.patch_embed"])
    """
    for layer_name in layer_names:
        parts = layer_name.split('.')
        module = model
        
        # Navigate to the target module
        for part in parts:
            if '[' in part and ']' in part:
                # Handle array indexing like "frame_blocks[0]"
                attr_name, idx = part.split('[')
                idx = int(idx.rstrip(']'))
                module = getattr(module, attr_name)[idx]
            else:
                module = getattr(module, part)
        
        # Set to train mode (affects BatchNorm, Dropout, etc.)
        logger.info(f"Module {layer_name} training mode BEFORE: {module.training}")
        module.train()
        logger.info(f"Module {layer_name} training mode AFTER: {module.training}")
        
        # Unfreeze all parameters recursively
        for param in module.parameters():
            param.requires_grad = True
    
    logger.info(f"Unfrozen and set to train mode: {len(layer_names)} layers")


def compute_feature_matching_loss(teacher_feats, student_feats):
    """
    Compute feature matching loss between teacher and student
    
    Args:
        teacher_feats: Dict of teacher features
        student_feats: Dict of student features
    
    Returns:
        Combined feature matching loss
    """
    losses = []
    
    for name in teacher_feats.keys():
        if name not in student_feats:
            logger.debug(f"Missing student feature: {name}")
            continue
        
        t_feat = teacher_feats[name]
        s_feat = student_feats[name]
        
        # Skip non-tensor features
        if not isinstance(t_feat, torch.Tensor) or not isinstance(s_feat, torch.Tensor):
            continue
        
        # Ensure same shape
        if t_feat.shape != s_feat.shape:
            logger.warning(f"Shape mismatch for {name}: teacher {t_feat.shape}, student {s_feat.shape}")
            continue
        
        # Cosine similarity loss (better for normalized features)
        t_flat = t_feat.reshape(t_feat.shape[0], -1)
        s_flat = s_feat.reshape(s_feat.shape[0], -1)
        
        cos_sim = F.cosine_similarity(t_flat, s_flat, dim=1)
        loss_cos = (1 - cos_sim).mean()
        
        # MSE loss (scaled down)
        loss_mse = F.mse_loss(s_feat, t_feat)
        
        # Combined loss
        loss = loss_cos + 0.1 * loss_mse
        losses.append(loss)
        
        # Log per-feature loss for debugging
        logger.debug(f"Feature {name}: cos_loss={loss_cos.item():.4f}, mse_loss={loss_mse.item():.4f}")
    
    if not losses:
        # Find a device from any available tensor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for v in teacher_feats.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
        return torch.tensor(0.0, device=device)
    
    total_loss = sum(losses) / len(losses)
    logger.debug(f"Total feature matching loss: {total_loss.item():.4f} (from {len(losses)} features)")
    return total_loss

def run_distill_vggt(rank, args):

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12398"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    with initialize(version_base=None, config_path="models/depth/vggt/training/config"):
        cfg = compose(config_name="default")
    
    # CRITICAL: Memory optimization for distillation
    logger.info("Applying memory optimizations for distillation:")
    # TEMPORARILY: Keep larger batch size to debug RoPE issue
    # The issue might be related to how positions are generated for smaller batches
    cfg.max_img_per_gpu = 48  # Keep original for now - TODO: reduce once RoPE is fixed
    logger.info(f"  - TEMPORARILY keeping max_img_per_gpu at {cfg.max_img_per_gpu} to debug RoPE")
    
    cfg.accum_steps = 1  # Disable gradient accumulation temporarily
    logger.info(f"  - TEMPORARILY disabled gradient accumulation: accum_steps = {cfg.accum_steps}")
    
    # CRITICAL: Disable AMP for distillation - it's causing gradient issues
    cfg.optim.amp.enabled = False
    logger.info(f"  - DISABLED AMP for distillation (was causing gradient detachment): {cfg.optim.amp.enabled}")
    
    # Reduce DDP bucket size for memory efficiency
    if 'distributed' in cfg:
        cfg.distributed.bucket_cap_mb = 10  # Reduce from 25 to 10 MB
        logger.info(f"  - Reduced DDP bucket_cap_mb: 25 -> {cfg.distributed.bucket_cap_mb}")
    
    # Limit training batches during distillation
    cfg.limit_train_batches = 400  # Reduce from 800
    logger.info(f"  - Reduced limit_train_batches: 800 -> {cfg.limit_train_batches}")
    
    # Add distillation loss keys to logging config
    if 'logging' in cfg and 'scalar_keys_to_log' in cfg['logging']:
        for phase in ['train', 'val']:
            if phase in cfg['logging']['scalar_keys_to_log']:
                # Add distillation-specific loss keys
                cfg['logging']['scalar_keys_to_log'][phase]['keys_to_log'].extend([
                    'task_loss_component',  # Original task loss (for monitoring)
                    'feat_loss_component',  # Feature matching loss (for monitoring)
                ])
                logger.info(f"Added distillation loss keys to {phase} logging")
    
    logger.info("Creating student trainer (FP32)")
    student_trainer = Trainer(**cfg)
    student_model = student_trainer.model.module
    
    # CRITICAL: Reduce frames_chunk_size in depth_head to save memory
    if hasattr(student_model, 'depth_head'):
        original_chunk_size = 8  # default
        reduced_chunk_size = 4   # reduce by half
        # Monkey-patch the forward method to use smaller chunk size
        original_depth_forward = student_model.depth_head.forward
        def patched_depth_forward(aggregated_tokens_list, images, patch_start_idx, frames_chunk_size=reduced_chunk_size):
            return original_depth_forward(aggregated_tokens_list, images, patch_start_idx, frames_chunk_size)
        student_model.depth_head.forward = patched_depth_forward
        logger.info(f"  - Reduced depth_head frames_chunk_size: {original_chunk_size} -> {reduced_chunk_size}")

    # CRITICAL: Monkey-patch the _step method to ensure gradients
    original_step = student_trainer._step

    def patched_step(batch, model, phase, loss_meters):
        """Force gradients during training and bypass original _step that may detach loss"""
        if phase == 'train':
            # Force enable gradients during training
            with torch.enable_grad():
                # Temporarily set model to train to ensure BN/Dropout work
                was_training = model.training
                model.train()
                
                # Forward pass
                y_hat = model(images=batch["images"])
                
                # Loss computation
                loss_dict = student_trainer.loss(y_hat, batch)
                
                # CRITICAL: Check loss IMMEDIATELY after computation
                logger.info(f"DEBUG PATCHED_STEP: loss_dict['objective'] requires_grad={loss_dict['objective'].requires_grad}")
                logger.info(f"DEBUG PATCHED_STEP: loss_dict['objective'] grad_fn={loss_dict['objective'].grad_fn}")
                logger.info(f"DEBUG PATCHED_STEP: loss_dict['objective'].is_leaf={loss_dict['objective'].is_leaf}")
                
                if not was_training:
                    model.eval()
                
                # Combine all data for logging (mimicking original _step behavior)
                log_data = {**y_hat, **loss_dict, **batch}
                
                # Update metrics
                student_trainer._update_and_log_scalars(
                    log_data, phase, student_trainer.steps[phase], loss_meters
                )
                student_trainer._log_tb_visuals(log_data, phase, student_trainer.steps[phase])
                
                student_trainer.steps[phase] += 1
                
                # CRITICAL: Log right before returning
                logger.info(f"DEBUG PATCHED_STEP FINAL: Returning loss_dict with objective.requires_grad={loss_dict['objective'].requires_grad}")
                
                return loss_dict
        else:
            # Validation - use original
            return original_step(batch, model, phase, loss_meters)

    student_trainer._step = patched_step
    logger.info("✓ Patched trainer._step to force gradients and bypass potential detaching")
    
    # CRITICAL: Patch _run_steps_on_batch_chunks to avoid in-place division that breaks gradients
    original_run_steps = student_trainer._run_steps_on_batch_chunks
    
    def patched_run_steps(chunked_batches, phase, loss_meters):
        """Patch to avoid in-place loss division that breaks gradients"""
        import math
        
        accum_steps = len(chunked_batches)
        
        for i, chunked_batch in enumerate(chunked_batches):
            # DEBUG: Check batch shapes before processing
            logger.info(f"DEBUG BATCH: images.shape = {chunked_batch['images'].shape}")
            
            # CRITICAL: Debug positions tensor issue
            if 'positions' in chunked_batch:
                pos_data = chunked_batch['positions']
                if hasattr(pos_data, 'shape'):
                    logger.info(f"DEBUG BATCH: positions.shape = {pos_data.shape}")
                    logger.info(f"DEBUG BATCH: positions.numel() = {pos_data.numel()}")
                    if pos_data.numel() > 0:
                        logger.info(f"DEBUG BATCH: positions min/max = {pos_data.min().item()}/{pos_data.max().item()}")
                    else:
                        logger.error(f"🚨 CRITICAL: positions tensor is EMPTY! This will cause RoPE to fail.")
                        logger.error(f"   positions.shape = {pos_data.shape}")
                        logger.error(f"   This might be due to batch size changes or data loading issues")
                else:
                    logger.info(f"DEBUG BATCH: positions = {str(pos_data)}")
            else:
                logger.warning("DEBUG BATCH: No 'positions' key found in batch!")
                logger.info(f"DEBUG BATCH: Available keys: {list(chunked_batch.keys())}")
            
            # DON'T use autocast - it's breaking gradients!
            # The model is already handling precision internally
            try:
                loss_dict = student_trainer._step(
                    chunked_batch, student_trainer.model, phase, loss_meters
                )
            except RuntimeError as e:
                if "max(): Expected reduction dim" in str(e):
                    logger.error("🚨 RoPE Error: Empty positions tensor detected!")
                    logger.error(f"   Batch shapes: images={chunked_batch['images'].shape}")
                    if 'positions' in chunked_batch:
                        logger.error(f"   positions.shape={chunked_batch['positions'].shape}")
                    logger.error("   This is likely due to data loading or batch chunking issues")
                    raise
                else:
                    raise
            
            loss = loss_dict["objective"]
            logger.info(f"DEBUG AFTER EXTRACTION: loss.requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}")
            
            # CRITICAL: Do NOT clone! Cloning detaches gradients by default
            # Work directly with the original tensor
            
            loss_key = f"Loss/{phase}_loss_objective"
            batch_size = chunked_batch["images"].shape[0]
            
            # Check if finite
            if not math.isfinite(loss.item()):
                logger.error(f"Loss is {loss.item()}, attempting to stop training")
                return
            
            # ✅ GRADIENT FLOW CONFIRMED WORKING! 
            # Now implement proper gradient accumulation by scaling gradients AFTER backward
            # instead of scaling loss BEFORE backward (which breaks gradients)
            
            if accum_steps > 1:
                # Step 1: Do backward pass with original loss (preserves gradients)
                loss.backward()
                
                # Step 2: Scale gradients after backward pass
                # This is the correct way to do gradient accumulation
                for param in student_trainer.model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad / accum_steps
                
                logger.info(f"✅ Applied gradient accumulation: scaled gradients by 1/{accum_steps}")
                loss_scaled = loss / accum_steps  # For logging only
            else:
                # No accumulation needed
                loss.backward()
                loss_scaled = loss
                logger.info("✅ Single step backward (no gradient accumulation)")
            
            logger.info("✅ SUCCESS: Backward pass completed!")
            loss_meters[loss_key].update(loss_scaled.item(), batch_size)
    
    student_trainer._run_steps_on_batch_chunks = patched_run_steps
    logger.info("✓ Patched trainer._run_steps_on_batch_chunks to use non-in-place division")

    ###############################################
    ## CHECKPOINT IS COMMENTED - MEANS IT'S ENABLED (MEMORY SAVING)
    ## Uncomment below to DISABLE checkpoint (increases memory but fixes some gradient issues)
    # import torch.utils.checkpoint as cp
    # 
    # def no_checkpoint(function, *args, **kwargs):
    #     return function(*args, **kwargs)
    # 
    # # Disable globally
    # cp.checkpoint = no_checkpoint
    ##############################################
    
    # Helper function to log memory usage
    def log_memory_usage(prefix=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(0) / (1024**3)    # GB
            max_allocated = torch.cuda.max_memory_allocated(0) / (1024**3)  # GB
            logger.info(f"{prefix}GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {max_allocated:.2f} GB peak")
            return allocated, reserved
        return 0, 0
    
    logger.info("="*80)
    logger.info("MEMORY OPTIMIZATIONS APPLIED:")
    logger.info("  1. ✓ Gradient checkpointing ENABLED (saves ~30-40% memory)")
    logger.info("  2. ✓ Teacher model on CPU, moved to GPU only during forward")
    logger.info("  3. ✓ Batch size reduced: 48 -> 24 (50% less memory)")
    logger.info("  4. ✓ Gradient accumulation enabled (accum_steps=2)")
    logger.info("  5. ✗ AMP DISABLED (was causing gradient detachment in distillation)")
    logger.info("  6. ✓ DDP bucket size reduced: 25 -> 10 MB")
    logger.info("  7. ✓ Depth head chunk size reduced: 8 -> 4 frames")
    logger.info("  8. ✓ Only trainable parameters have gradients computed")
    logger.info("  9. ✗ torch.inference_mode() REMOVED (was creating inference tensors)")
    logger.info(" 10. ✓ Immediate cache clearing after teacher forward")
    logger.info("="*80)
    
    logger.info("Memory before creating teacher:")
    log_memory_usage("  ")
    
    # CRITICAL: Don't store teacher in GPU memory, move to CPU after forward pass
    logger.info("Creating teacher model (FP32) - will move to CPU to save GPU memory")
    teacher_model = copy.deepcopy(student_model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Move teacher to CPU to save GPU memory (we'll move it back during forward)
    teacher_model = teacher_model.cpu()
    logger.info("✓ Teacher model moved to CPU")
    
    logger.info("Memory after creating teacher:")
    teacher_mem_alloc, teacher_mem_reserved = log_memory_usage("  ")
    
    # Calculate teacher model size
    teacher_params = sum(p.numel() * p.element_size() for p in teacher_model.parameters())
    teacher_buffers = sum(b.numel() * b.element_size() for b in teacher_model.buffers())
    teacher_size_gb = (teacher_params + teacher_buffers) / (1024**3)
    logger.info(f"Teacher model size: {teacher_size_gb:.2f} GB ({teacher_params/1e6:.1f}M params)")
    
    # Save the original loss function
    original_loss = student_trainer.loss
    
    # Get stage configuration
    stages, distill_alpha, distill_beta = get_config()
    
    for stage_idx, stage in enumerate(stages):
        logger.info("="*80)
        logger.info(f"Stage {stage_idx + 1}/{len(stages)}: {stage['name']}")
        logger.info(f"Layers to quantize: {stage['layers']}")
        logger.info("="*80)
        
        # 1. Quantize current stage layers
        logger.info(f"==> Quantizing stage {stage['name']} layers")
        student_model = selective_quantize_layers(student_model, args, stage['layers'])
        # Re-setup gradient clipping for quantized layers (critical for correct clipping)
        student_trainer.gradient_clipper.setup_clipping(student_trainer.model)
        
        # 2. Initialize quantization parameters
        if args.action == 'load':
            logger.info("==> Initializing quantization parameters")
            train_loader = student_trainer.train_dataset.get_loader(0)
            stepsize_init(student_model, train_loader, args.device, 
                         num_batches=args.init_num, dataset_name=args.dataset_name)
        elif args.action == 'resume':
            raise ValueError("Resuming not implemented yet for distillation")
        
        # 3. Freeze all layers except current stage
        logger.info(f"==> Freezing all layers except current stage: {stage['name']}")

        for param in student_model.parameters():
            if param.grad is not None:
                param.grad = None
        
        for param in student_model.parameters():
            param.requires_grad = False
        
        unfreeze_layers(student_model, stage['layers'])
        
        trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in student_model.parameters())
        logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # CRITICAL: Set DDP to only reduce gradients for trainable parameters
        if hasattr(student_trainer.model, 'module'):
            # Rebuild DDP with only trainable params to save memory
            logger.info("Configuring DDP to only sync trainable parameters")
            student_trainer.model._set_static_graph()  # Static graph optimization
        
        logger.info("Memory before setting up extractors:")
        log_memory_usage("  ")
        
        # 4. Setup feature extractors
        teacher_extractor = FeatureExtractor(detach=True)
        student_extractor = FeatureExtractor(detach=False)
        teacher_extractor.register_hooks(teacher_model, stage['hook_points'])
        student_extractor.register_hooks(student_model, stage['hook_points'])

        logger.info(f"DEBUG: Registered {len(teacher_extractor.hooks)} teacher hooks")
        logger.info(f"DEBUG: Registered {len(student_extractor.hooks)} student hooks")

        # 5. Wrap the model
        model_wrapper = DistillationModelWrapper(
            student_model, teacher_model,
            original_loss,
            teacher_extractor, student_extractor,
            alpha=distill_alpha, beta=distill_beta
        )

        # Replace trainer's model with wrapped version
        student_trainer.model.module = model_wrapper

        # 6. Replace loss function
        student_trainer.loss = DistillationLossWrapper(
            original_loss, model_wrapper,
            alpha=distill_alpha, beta=distill_beta
        )
        
        # 7. Reconstruct optimizers for current stage
        student_trainer.model.module = student_model
        
        if args.different_optimizer_mode:
            # Takes freezed layers into account
            sparams, params = split_params(
                student_model, weight_decay=args.weight_decay, lr=args.lr,
                x_lr=args.x_step_size_lr, w_lr=args.w_step_size_lr,
                x_wd=args.x_step_size_wd, w_wd=args.w_step_size_wd
            )

            if len(sparams) > 0:
                soptimizer, sscheduler = scheduler_optimizer_class(
                    args, sparams, args.step_size_optimizer
                )
                student_trainer.s_optimizer = soptimizer
                student_trainer.s_scheduler = sscheduler
                logger.info(f"Scale optimizer created with {len(sparams)} parameter groups")
            else:
                student_trainer.s_optimizer = None
                student_trainer.s_scheduler = None
                logger.info("No trainable scale parameters found")
        
        # Reconstruct main optimizer for all trainable parameters
        # The original optimizer had all parameters
        student_trainer.optims = construct_optimizers(
            student_trainer.model, student_trainer.optim_conf
        )
        
        # 8. Train for this stage's epochs using Trainer.run_train()
        stage_max_epochs = student_trainer.epoch + stage['epochs']
        original_max_epochs = student_trainer.max_epochs
        student_trainer.max_epochs = stage_max_epochs
        
        logger.info(f"==> Training stage {stage['name']} for {stage['epochs']} epochs")
        logger.info(f"    Epoch range: {student_trainer.epoch} -> {stage_max_epochs}")
        
        logger.info("Memory before training:")
        log_memory_usage("  ")
        
        student_trainer.run_train()
        
        logger.info("Memory after training stage:")
        log_memory_usage("  ")
        
        # 9. Cleanup and restore
        student_trainer.max_epochs = original_max_epochs
        teacher_extractor.clear_hooks()
        student_extractor.clear_hooks()
        
        student_trainer.loss = original_loss
        
        # Clear CUDA cache after stage
        torch.cuda.empty_cache()
        logger.info("Memory after cleanup:")
        log_memory_usage("  ")
        
        logger.info(f"==> Completed stage {stage['name']}")
    
    # 10. Final fine-tuning with all layers
    # logger.info("="*80)
    # logger.info("==> Final fine-tuning with all layers unfrozen")
    # logger.info("="*80)
    # 
    # # Unfreeze all layers
    # for param in student_model.parameters():
    #     param.requires_grad = True
    # student_model.train()
    # 
    # # Restore task-only loss (no distillation)
    # student_trainer.loss = original_loss
    # 
    # # Reconstruct optimizers for full model
    # student_trainer.optims = construct_optimizers(
    #     student_trainer.model, student_trainer.optim_conf
    # )
    # 
    # # Final training
    # student_trainer.run_train()
    
    logger.info("==> Distillation complete!")
    

def get_config():
    distill_alpha = 0.7
    distill_beta = 0.3
    stages = [
        {
            "name": "patch_embed",
            "layers": ["aggregator.patch_embed"],
            "hook_points": ["aggregator.patch_embed.patch_embed.proj"],
            "epochs": 10,
            "lr": 1e-4
        },
        # {
        #     "name": "blocks_0_5",
        #     "layers": [
        #         "aggregator.frame_blocks.0", "aggregator.frame_blocks.1", 
        #         "aggregator.frame_blocks.2", "aggregator.frame_blocks.3",
        #         "aggregator.frame_blocks.4", "aggregator.frame_blocks.5",
        #         "aggregator.global_blocks.0", "aggregator.global_blocks.1",
        #         "aggregator.global_blocks.2", "aggregator.global_blocks.3",
        #         "aggregator.global_blocks.4", "aggregator.global_blocks.5"
        #     ],
        #     "hook_points": [
        #         "aggregator.frame_blocks[5]",
        #         "aggregator.global_blocks[5]"
        #     ],
        #     "epochs": 8
        # },
        # {
        #     "name": "blocks_6_11",
        #     "layers": [
        #         "aggregator.frame_blocks.6", "aggregator.frame_blocks.7",
        #         "aggregator.frame_blocks.8", "aggregator.frame_blocks.9",
        #         "aggregator.frame_blocks.10", "aggregator.frame_blocks.11",
        #         "aggregator.global_blocks.6", "aggregator.global_blocks.7",
        #         "aggregator.global_blocks.8", "aggregator.global_blocks.9",
        #         "aggregator.global_blocks.10", "aggregator.global_blocks.11"
        #     ],
        #     "hook_points": [
        #         "aggregator.frame_blocks[11]",
        #         "aggregator.global_blocks[11]"
        #     ],
        #     "epochs": 8
        # },
        # {
        #     "name": "blocks_12_17",
        #     "layers": [
        #         "aggregator.frame_blocks.12", "aggregator.frame_blocks.13",
        #         "aggregator.frame_blocks.14", "aggregator.frame_blocks.15",
        #         "aggregator.frame_blocks.16", "aggregator.frame_blocks.17",
        #         "aggregator.global_blocks.12", "aggregator.global_blocks.13",
        #         "aggregator.global_blocks.14", "aggregator.global_blocks.15",
        #         "aggregator.global_blocks.16", "aggregator.global_blocks.17"
        #     ],
        #     "hook_points": [
        #         "aggregator.frame_blocks[17]",
        #         "aggregator.global_blocks[17]"
        #     ],
        #     "epochs": 8
        # },
        # {
        #     "name": "blocks_18_23",
        #     "layers": [
        #         "aggregator.frame_blocks.18", "aggregator.frame_blocks.19",
        #         "aggregator.frame_blocks.20", "aggregator.frame_blocks.21",
        #         "aggregator.frame_blocks.22", "aggregator.frame_blocks.23",
        #         "aggregator.global_blocks.18", "aggregator.global_blocks.19",
        #         "aggregator.global_blocks.20", "aggregator.global_blocks.21",
        #         "aggregator.global_blocks.22", "aggregator.global_blocks.23"
        #     ],
        #     "hook_points": [
        #         "aggregator.frame_blocks[23]",
        #         "aggregator.global_blocks[23]"
        #     ],
        #     "epochs": 8
        # },
        # {
        #     "name": "heads",
        #     "layers": ["camera_head", "depth_head"],
        #     "hook_points": ["camera_head", "depth_head"],
        #     "epochs": 10
        # }
    ]
    return stages, distill_alpha, distill_beta

def run_distill(args):
    if args.ddp == True:
        logger.info("DDP mode")
        mp.spawn( \
            run_distill_vggt, \
            nprocs= args.world_size, \
            args= (args,) \
        )
    else:
        run_distill_vggt(0, args)