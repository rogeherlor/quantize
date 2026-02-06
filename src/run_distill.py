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

from hydra import initialize, compose
import src.models.depth.vggt.vggt
from src.models.depth.vggt.training.trainer import Trainer
from src.models.depth.vggt.training.train_utils.optimizer import construct_optimizers

from src.utils import *
from src.scheduler_optimizer_class import scheduler_optimizer_class
import src.module_quantization as Q
from src.make_ex_name import *

from src.models.depth.vggt.evaluation.test_co3d import *

from src.models.depth.qvggt import run_evaluation_vggt, load_ckp_vggt

class DistillationLossWrapper(nn.Module):
    """
    Simple distillation loss: task_loss + feature_matching_loss
    Standard practice - teacher stays on GPU, uses @torch.no_grad()
    """
    def __init__(self, task_loss, teacher_model, teacher_extractor, student_extractor, alpha=0.7, beta=0.3):
        super().__init__()
        self.task_loss = task_loss
        self.teacher_model = teacher_model
        self.teacher_extractor = teacher_extractor
        self.student_extractor = student_extractor
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, student_pred, batch):
        """Compute combined distillation + original loss"""
        
        # 1. Compute original loss from original loss function
        task_loss_dict = self.task_loss(student_pred, batch)
        task_loss_val = task_loss_dict["objective"]
        
        # 2. Get student features (already captured during forward)
        student_feats = self.student_extractor.get_features()
        
        # 3. Run teacher and get features (no gradients needed)
        with torch.no_grad():
            teacher_pred = self.teacher_model(images=batch["images"])
            
            # Compute teacher loss for monitoring
            teacher_loss_dict = self.task_loss(teacher_pred, batch)
            teacher_loss_val = teacher_loss_dict["objective"]
        
        teacher_feats = self.teacher_extractor.get_features()
        
        # 4. Compute feature matching loss (per-layer for better monitoring)
        if student_feats and teacher_feats:
            feat_losses = compute_feature_matching_loss_per_layer(teacher_feats, student_feats)
            # Average per-layer losses or fallback to zero connected to graph
            feat_loss = sum(feat_losses.values()) / len(feat_losses) if feat_losses else task_loss_val * 0.0
        else:
            feat_losses = {}
            # CRITICAL: Use task_loss_val * 0.0 instead of torch.tensor() to maintain computation graph
            feat_loss = task_loss_val * 0.0
        
        # 5. Combine losses
        total_loss = self.alpha * feat_loss + self.beta * task_loss_val
        
        # 6. Clear features for next iteration
        self.teacher_extractor.features = {}
        self.student_extractor.features = {}
        
        # 7. Return loss dict with components for logging
        # CRITICAL: Keep objective with gradients! Only detach logging components
        loss_dict = task_loss_dict.copy()
        loss_dict["objective"] = total_loss  # ← Must have gradients for backward()
        
        loss_dict["task_loss_component"] = task_loss_val.detach()  # ← Detach for logging only
        loss_dict["feat_loss_component"] = feat_loss.detach()  # ← Detach for logging only
        loss_dict["teacher_loss_component"] = teacher_loss_val.detach()  # ← Detach for logging only
        
        # Log per-layer feature losses for detailed monitoring
        for layer_name, layer_loss in feat_losses.items():
            # Sanitize layer name for logging (remove special chars)
            clean_name = layer_name.replace('[', '_').replace(']', '').replace('.', '_')
            loss_dict[f"feat_loss_{clean_name}"] = layer_loss.detach()
        
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
            elif isinstance(output, list):
                # Handle list outputs (e.g. camera_head returns list of pose encodings)
                if len(output) > 0:
                    feature = output[-1]
                else:
                    return
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

def compute_feature_matching_loss_per_layer(teacher_feats, student_feats):
    """
    Compute feature matching loss between teacher and student (per-layer).
    Returns a dict of losses for detailed monitoring.
    
    Args:
        teacher_feats: Dict of teacher features {layer_name: tensor}
        student_feats: Dict of student features {layer_name: tensor}
    
    Returns:
        Dict of per-layer losses {layer_name: loss_tensor}
    """
    layer_losses = {}
    
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
        
        # Combined loss for this layer
        layer_loss = loss_cos ## loss_cos + 0.1 * loss_mse
        layer_losses[name] = layer_loss
        
        # Log per-layer loss for debugging
        ## logger.debug(f"Layer {name}: cos_loss={loss_cos.item():.4f}, mse_loss={loss_mse.item():.4f}, total={layer_loss.item():.4f}")
    
    return layer_losses

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

    logger.info("Applying memory optimizations for distillation:")
    cfg.limit_train_batches = 400
    cfg.limit_val_batches = 100
    logger.info(f"  - Reduced limit_train_batches: 800 -> {cfg.limit_train_batches}")
    logger.info(f"  - Reduced limit_val_batches: 400 -> {cfg.limit_val_batches}")
    
    # default.yaml logging - add distillation loss to tensorboard
    if 'logging' in cfg and 'scalar_keys_to_log' in cfg['logging']:
        for phase in ['train', 'val']:
            if phase in cfg['logging']['scalar_keys_to_log']:
                cfg['logging']['scalar_keys_to_log'][phase]['keys_to_log'].extend([
                    'task_loss_component',  # Original loss
                    'feat_loss_component',  # Average feature matching loss
                    'teacher_loss_component',  # Teacher original loss
                ])
                logger.info(f"Added distillation loss keys to {phase} logging")
                logger.info("  Note: Per-layer feature losses (feat_loss_<layer>) are logged automatically")
    
    logger.info("Creating student trainer (FP32)")
    student_trainer = Trainer(**cfg)
    student_model = student_trainer.model.module

    student_trainer.model.module.train()
    torch.set_grad_enabled(True)
    
    logger.info("Creating teacher model (FP32) - loading from state_dict instead of deepcopy")
    from src.models.depth.vggt.vggt.models.vggt import VGGT
    img_size = cfg.get('img_size', 518)
    patch_size = cfg.get('patch_size', 14)
    embed_dim = student_model.aggregator.camera_token.shape[-1]
    
    teacher_model = VGGT(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        enable_camera=(student_model.camera_head is not None),
        enable_point=(student_model.point_head is not None),
        enable_depth=(student_model.depth_head is not None),
        enable_track=(student_model.track_head is not None)
    ).to(student_trainer.device)
    
    teacher_model.load_state_dict(student_model.state_dict())
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    run_evaluation_vggt(teacher_model)
    
    # Distill Single combined loss
    task_loss = student_trainer.loss
    stages = get_config()
    
    for stage_idx, stage in enumerate(stages):
        logger.info("="*50)
        logger.info(f"Stage {stage_idx + 1}/{len(stages)}: {stage['name']}")
        logger.info(f"Layers to quantize: {stage['layers']}")
        logger.info("="*50)
        
        # 1. Quantize current stage layers
        logger.info(f"==> Quantizing stage {stage['name']} layers")
        student_model = selective_quantize_layers(student_model, args, stage['layers'])
        student_trainer.gradient_clipper.setup_clipping(student_trainer.model)
        
        # 2. Initialize quantization parameters. Careful, grad is disabled here.
        # if args.action == 'load':
        stepsize_init(student_trainer.model.module, student_trainer.train_dataset.get_loader(0), args.device, num_batches=args.init_num, dataset_name=args.dataset_name)
        
        # 3. Expert approach: Enable gradient flow but only update quantized layers
        logger.info(f"==> Setting up gradient flow for stage: {stage['name']}")
        # Clear any existing gradients
        for param in student_model.parameters():
            if param.grad is not None:
                param.grad = None
        # Keep ALL parameters with requires_grad=True for gradient flow
        for param in student_model.parameters():
            param.requires_grad = True

        logger.info(f"Layer status for stage {stage['name']}:")
        
        # Log grad status for each layer
        logger.info(student_trainer.model.module)
        param_to_class = {}
        for m_name, m in student_model.named_modules():
            for p_name, _ in m.named_parameters(recurse=False):
                full_name = f"{m_name}.{p_name}" if m_name else p_name
                param_to_class[full_name] = m.__class__.__name__
        for name, param in student_model.named_parameters():
            is_updated = any(layer_name in name for layer_name in stage['layers'])
            status = "Trainable" if is_updated else "Frozen (Gradient Flow Only)"
            class_name = param_to_class.get(name, "Unknown")
            logger.info(f"  {name} [{class_name}]: {status} | requires_grad={param.requires_grad}")
        
        # 4. Setup hooks for feature extraction
        teacher_extractor = FeatureExtractor(detach=True)
        student_extractor = FeatureExtractor(detach=False)
        teacher_extractor.register_hooks(teacher_model, stage['hook_points'])
        student_extractor.register_hooks(student_model, stage['hook_points'])

        logger.info(f"Registered {len(teacher_extractor.hooks)} teacher hooks")
        logger.info(f"Registered {len(student_extractor.hooks)} student hooks")

        # 5. Replace loss function with distillation loss
        stage_alpha = stage.get('alpha', 0.7)
        stage_beta = stage.get('beta', 0.3)
        logger.info(f"Using distillation weights: alpha={stage_alpha}, beta={stage_beta}")

        student_trainer.loss = DistillationLossWrapper(
            task_loss, teacher_model,
            teacher_extractor, student_extractor,
            alpha=stage_alpha, beta=stage_beta
        )
        
        # 6. Reconstruct optimizers for current stage
        if args.different_optimizer_mode:
            sparams, params = split_params(\
                student_trainer.model.module, weight_decay=args.weight_decay, lr = args.lr, x_lr= args.x_step_size_lr, \
                    w_lr= args.w_step_size_lr, x_wd = args.x_step_size_wd, w_wd = args.w_step_size_wd)
            soptimizer, sscheduler = scheduler_optimizer_class(args, sparams, args.step_size_optimizer)
            student_trainer.s_optimizer = soptimizer
            student_trainer.s_scheduler = sscheduler
        
        # Reconstruct optimizer to ONLY update current stage parameters
        # Filter parameters: only those in current stage layers
        stage_param_names = set()
        for layer_name in stage['layers']:
            for name, param in student_model.named_parameters():
                if layer_name in name:
                    stage_param_names.add(name)
        
        stage_params = [p for n, p in student_model.named_parameters() if n in stage_param_names]
        stage_param_count = sum(p.numel() for p in stage_params)
        logger.info(f"Optimizer will update {len(stage_params)} parameter tensors ({stage_param_count:,} values) from stage '{stage['name']}'")
        
        # Create optimizer using the original configuration but for stage parameters only
        # This respects the optimizer type (AdamW, etc.) and other settings from config
        from hydra.utils import instantiate
        optimizer_conf = student_trainer.optim_conf.optimizer.copy()
        # Override LR if specified in stage config
        if 'lr' in stage:
            optimizer_conf['lr'] = stage['lr']
            
        logger.info(f"Creating optimizer {optimizer_conf['_target_']} with lr={optimizer_conf['lr']}")
        optimizer = instantiate(optimizer_conf, params=stage_params)
        
        # Wrap in the expected format
        from collections import namedtuple
        OptimWrapper = namedtuple('OptimWrapper', ['optimizer', 'schedulers', 'step_schedulers', 'zero_grad'])
        student_trainer.optims = [
            OptimWrapper(
                optimizer=optimizer,
                schedulers=[{}],  # No scheduler for now
                step_schedulers=lambda x: None,  # No-op
                # CRITICAL: Zero grad for the WHOLE model, not just optimizer params.
                # Otherwise, gradients for non-updated layers will accumulate and break gradient clipping.
                zero_grad=lambda **kwargs: student_model.zero_grad(**kwargs)
            )
        ]
        
        # 7. Load checkpoint for 'resume' action AFTER optimizer is reconstructed (so param groups match)
        # Skip scheduler loading since each stage creates its own fresh scheduler
        if args.init_from and args.action == 'resume':
            load_ckp_vggt(args, student_trainer, load_scheduler=False)
            
        logger.info(f"==> Validating stage {stage['name']} validation before training")
        student_trainer.run_val()
        student_trainer.model.module.eval()
        run_evaluation_vggt(student_trainer.model.module)
        student_trainer.model.module.train()
        
        # 8. Train for this stage's epochs using Trainer.run_train()
        stage_max_epochs = student_trainer.epoch + stage['epochs']
        original_max_epochs = student_trainer.max_epochs
        student_trainer.max_epochs = stage_max_epochs
        
        logger.info(f"==> Training stage {stage['name']} for {stage['epochs']} epochs")
        logger.info(f"    Epoch range: {student_trainer.epoch} -> {stage_max_epochs}")
        
        torch.set_grad_enabled(True)
        student_trainer.run_train()

        # Force validation at the end of the stage because run_train skips the last epoch validation
        logger.info(f"==> Validating stage {stage['name']} final epoch")
        student_trainer.run_val()
        student_trainer.model.module.eval()
        run_evaluation_vggt(student_trainer.model.module)
        student_trainer.model.module.train()
        
        # 9. Cleanup and restore
        student_trainer.max_epochs = original_max_epochs
        teacher_extractor.clear_hooks()
        student_extractor.clear_hooks()
        
        student_trainer.loss = task_loss
        
        torch.cuda.empty_cache()
        
        logger.info(f"==> Completed stage {stage['name']}")
    
    # 10. Final fine-tuning with all layers
    logger.info("==> Distillation complete!")
    

def get_config():
    # Distillation loss weights (all layers trainable, no freezing)
    # Feature loss: matches intermediate representations from teacher
    # Task loss: ensures final output solves the actual task correctly
    
    stages = [
        #{
        #    "name": "patch_embed",
        #    "layers": ["aggregator.patch_embed"], # all blocks inside patch_embed
        #    "hook_points": ["aggregator.patch_embed.patch_embed.proj"],
        #    "epochs": 10,
        #    "lr": 1e-6
        #},
        #{
        #    "name": "blocks_0_5",
        #    "layers": [
        #        "aggregator.frame_blocks.0", "aggregator.frame_blocks.1", 
        #        "aggregator.frame_blocks.2", "aggregator.frame_blocks.3",
        #        "aggregator.frame_blocks.4", "aggregator.frame_blocks.5",
        #        "aggregator.global_blocks.0", "aggregator.global_blocks.1",
        #        "aggregator.global_blocks.2", "aggregator.global_blocks.3",
        #        "aggregator.global_blocks.4", "aggregator.global_blocks.5"
        #    ],
        #    "hook_points": [
        #        "aggregator.frame_blocks[5]",
        #        "aggregator.global_blocks[5]"
        #    ],
        #    "epochs": 10,
        #    "lr": 1e-6,
        #    "alpha": 0.7,
        #    "beta": 0.3
        #},
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
        #     "epochs": 0,
        #     "lr": 1e-3,
        #     "alpha": 1.0,
        #     "beta": 0.0
        # },
        {
            "name": "blocks_12_17",
            "layers": [
                "aggregator.frame_blocks.12", "aggregator.frame_blocks.13", 
                "aggregator.frame_blocks.14", "aggregator.frame_blocks.15",
                "aggregator.frame_blocks.16", "aggregator.frame_blocks.17",
                "aggregator.global_blocks.12", "aggregator.global_blocks.13",
                "aggregator.global_blocks.14", "aggregator.global_blocks.15",
                "aggregator.global_blocks.16", "aggregator.global_blocks.17"
            ],
            "hook_points": [
                "aggregator.frame_blocks[12]",
                "aggregator.global_blocks[12]",
                "aggregator.frame_blocks[13]",
                "aggregator.global_blocks[13]",
                "aggregator.frame_blocks[14]",
                "aggregator.global_blocks[14]",
                "aggregator.frame_blocks[15]",
                "aggregator.global_blocks[15]",
                "aggregator.frame_blocks[16]",
                "aggregator.global_blocks[16]",
                "aggregator.frame_blocks[17]",
                "aggregator.global_blocks[17]"
            ],
            "epochs": 15,
            "lr": 1e-4,
            "alpha": 1.0,
            "beta": 0.0
        },
        {
            "name": "blocks_12_17",
            "layers": [
                "aggregator.frame_blocks.12", "aggregator.frame_blocks.13", 
                "aggregator.frame_blocks.14", "aggregator.frame_blocks.15",
                "aggregator.frame_blocks.16", "aggregator.frame_blocks.17",
                "aggregator.global_blocks.12", "aggregator.global_blocks.13",
                "aggregator.global_blocks.14", "aggregator.global_blocks.15",
                "aggregator.global_blocks.16", "aggregator.global_blocks.17"
            ],
            "hook_points": [
                "aggregator.frame_blocks[12]",
                "aggregator.global_blocks[12]",
                "aggregator.frame_blocks[13]",
                "aggregator.global_blocks[13]",
                "aggregator.frame_blocks[14]",
                "aggregator.global_blocks[14]",
                "aggregator.frame_blocks[15]",
                "aggregator.global_blocks[15]",
                "aggregator.frame_blocks[16]",
                "aggregator.global_blocks[16]",
                "aggregator.frame_blocks[17]",
                "aggregator.global_blocks[17]"
            ],
            "epochs": 15,
            "lr": 1e-3,
            "alpha": 0.0,
            "beta": 1.0
        },
        {
            "name": "blocks_18_23",
            "layers": [
                "aggregator.frame_blocks.18", "aggregator.frame_blocks.19", 
                "aggregator.frame_blocks.20", "aggregator.frame_blocks.21",
                "aggregator.frame_blocks.22", "aggregator.frame_blocks.23",
                "aggregator.global_blocks.18", "aggregator.global_blocks.19",
                "aggregator.global_blocks.20", "aggregator.global_blocks.21",
                "aggregator.global_blocks.22", "aggregator.global_blocks.23"
            ],
            "hook_points": [
                "aggregator.frame_blocks[23]",
                "aggregator.global_blocks[23]"
            ],
            "epochs": 10,
            "lr": 1e-5,
            "alpha": 1.0,
            "beta": 0.0
        },
        {
            "name": "blocks_18_23",
            "layers": [
                "aggregator.frame_blocks.18", "aggregator.frame_blocks.19", 
                "aggregator.frame_blocks.20", "aggregator.frame_blocks.21",
                "aggregator.frame_blocks.22", "aggregator.frame_blocks.23",
                "aggregator.global_blocks.18", "aggregator.global_blocks.19",
                "aggregator.global_blocks.20", "aggregator.global_blocks.21",
                "aggregator.global_blocks.22", "aggregator.global_blocks.23"
            ],
            "hook_points": [
                "aggregator.frame_blocks[23]",
                "aggregator.global_blocks[23]"
            ],
            "epochs": 10,
            "lr": 1e-5,
            "alpha": 0.0,
            "beta": 1.0
        },
        #{
        #   "name": "block_all_attn",
        #   "layers": [
        #       "aggregator.frame_blocks.0.attn",
        #   ],
        #   "hook_points": [
        #       "aggregator.frame_blocks[0].mlp.fc2"
        #   ],
        #   "epochs": 4,
        #   "lr": 1e-6,
        #    "alpha": 0.7,
        #    "beta": 0.3
        #},
        {
            "name": "head2",
            "layers": ["depth_head"],
            "hook_points": ["depth_head"],
            "epochs": 10,
            "lr": 1e-5,
            "alpha": 0.7,
            "beta": 0.3
        },
        {
            "name": "head1",
            "layers": ["camera_head"],
            "hook_points": ["camera_head"],
            "epochs": 10,
            "lr": 1e-5,
            "alpha": 0.7,
            "beta": 0.3
        }
    ]
    return stages

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