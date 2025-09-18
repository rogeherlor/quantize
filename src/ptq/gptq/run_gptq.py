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
        child_layers = find_layers(child, layers)
        for k, v in child_layers.items():
            full_name = name + '.' + k if k else name
            res[full_name] = v
    return res

class StopForward(Exception):
    pass


def quantize_model_gptq(model, dataloader, args):
    """
    GPTQ quantization using user-specified block list.
    Each block in args.quant_blocks is processed with a Catcher and all quantizable layers inside are quantized together.
    If a block is a ModuleList/Sequential, each submodule is processed as a separate block, one at a time (only one on GPU).
    """
    logger.info("Starting GPTQ block-list quantization...")
    model.eval()

    if not hasattr(args, 'quant_blocks') or not args.quant_blocks:
        raise ValueError("You must specify args.quant_blocks as a list of block names (dotted paths) to quantize.")

    logger.info(f"User-specified quantization blocks: {args.quant_blocks}")
    scale_dict = {}

    # Expand blocks if they are ModuleList/Sequential
    expanded_blocks = []
    for block_name in args.quant_blocks:
        parent, leaf = _resolve_parent_and_leaf(model, block_name)
        block_module = getattr(parent, leaf) if hasattr(parent, leaf) else parent[int(leaf)]
        if isinstance(block_module, (nn.ModuleList, nn.Sequential)):
            for i in range(len(block_module)):
                expanded_blocks.append(f"{block_name}.{i}")
        else:
            expanded_blocks.append(block_name)

    for block_idx, block_name in enumerate(expanded_blocks):
        logger.info(f"Processing block {block_idx+1}/{len(expanded_blocks)}: {block_name}")
        parent, leaf = _resolve_parent_and_leaf(model, block_name)
        block_module = getattr(parent, leaf) if hasattr(parent, leaf) else parent[int(leaf)]
        block_inputs = _collect_block_inputs_with_catcher(model, dataloader, block_name, args)
        block_scale_dict = _process_block_with_gptq(block_module, block_inputs, block_name, args)
        scale_dict.update(block_scale_dict)
        if args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"Block-list quantization completed. Processed {len(scale_dict)} layers")
    return model, scale_dict


def _resolve_parent_and_leaf(model, dotted_path):
    """
    Given "foo.bar.3.baz", return (parent_module, leaf_name_or_index).
    Handles ModuleList / Sequential indices.
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


def _set_child(parent, leaf, value):
    if hasattr(parent, leaf):
        setattr(parent, leaf, value)
    elif leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = value
    else:
        raise AttributeError(f"Cannot set '{leaf}' on {parent}")


def _collect_block_inputs_with_catcher(model, dataloader, block_name, args):
    """
    Collect (*args, **kwargs) for a specific block using Catcher.
    Handles ModuleList/Sequential by attaching Catcher to each submodule.
    Returns a dict: {subblock_name: [inputs,...]} if block is a list, else just [inputs,...]
    """
    block_inputs = {}

    class Catcher(nn.Module):
        def __init__(self, module, inputs_list):
            super().__init__()
            self.module = module
            self.inputs_list = inputs_list

        def forward(self, *args, **kwargs):
            def _cpu_detach(obj):
                if torch.is_tensor(obj):
                    return obj.detach().cpu()
                if isinstance(obj, (list, tuple)):
                    return type(obj)(_cpu_detach(x) for x in obj)
                if isinstance(obj, dict):
                    return {k: _cpu_detach(v) for k, v in obj.items()}
                return obj
            self.inputs_list.append((_cpu_detach(args), _cpu_detach(kwargs)))
            raise StopForward
        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__(name)
            return getattr(self.module, name)
        def __getitem__(self, idx):
            return self.module[idx]
        def __dir__(self):
            return list(set(super().__dir__()) | set(dir(self.module)))

    parent, leaf = _resolve_parent_and_leaf(model, block_name)
    orig_module = getattr(parent, leaf) if hasattr(parent, leaf) else parent[int(leaf)]

    # Handle ModuleList/Sequential
    if isinstance(orig_module, (nn.ModuleList, nn.Sequential)):
        catchers = []
        orig_submodules = []
        for i, submodule in enumerate(orig_module):
            subblock_name = f"{block_name}.{i}"
            block_inputs[subblock_name] = []
            catcher = Catcher(submodule, block_inputs[subblock_name])
            orig_submodules.append(submodule)
            orig_module[i] = catcher
            catchers.append((i, catcher))
    else:
        block_inputs = []
        catcher = Catcher(orig_module, block_inputs)
        _set_child(parent, leaf, catcher)

    model = model.to(args.device)
    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
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
            except StopForward:
                pass

    # Restore original modules
    if isinstance(orig_module, (nn.ModuleList, nn.Sequential)):
        for i, submodule in enumerate(orig_submodules):
            orig_module[i] = submodule
    else:
        _set_child(parent, leaf, orig_module)
    model = model.cpu()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    if isinstance(orig_module, (nn.ModuleList, nn.Sequential)):
        logger.info(f"Collected input batches for submodules: {list(block_inputs.keys())}")
        return block_inputs
    else:
        logger.info(f"Collected {len(block_inputs)} input batches for block {block_name}")
        return block_inputs


def _process_block_with_gptq(block_module, block_inputs, block_name, args):
    """
    Process a single block with GPTQ quantization.
    Handles ModuleList/Sequential by processing each submodule separately if needed.
    """
    scale_dict = {}
    if isinstance(block_inputs, dict):
        # block_module is ModuleList/Sequential
        for i, sub_inputs in block_inputs.items():
            submodule = block_module[int(i.split('.')[-1])]
            sub_name = i
            sub_scale_dict = _process_block_with_gptq(submodule, sub_inputs, sub_name, args)
            scale_dict.update(sub_scale_dict)
        return scale_dict
    block_module = block_module.to(args.device)
    full_layers = find_layers(block_module)
    if not full_layers:
        logger.info(f"No quantizable layers in block {block_name}")
        block_module = block_module.cpu()
        return {}
    logger.info(f"Found {len(full_layers)} quantizable layers in {block_name}")
    gptq_instances = {}
    for name, layer in full_layers.items():
        quantizer = Quantizer()
        quantizer.configure(
            bits=getattr(args, "num_bits", 8),
            perchannel=False,
            sym=True,
            mse=getattr(args, "mse", False),
            maxshrink=0.8,
        )
        gptq = GPTQ(layer)
        gptq.quantizer = quantizer
        gptq_instances[name] = gptq
    def add_batch(name):
        def hook_fn(module, inp, out):
            x = inp[0] if isinstance(inp, (list, tuple)) else inp
            if torch.is_tensor(x):
                gptq_instances[name].add_batch(x.data, out.data)
        return hook_fn
    handles = [layer.register_forward_hook(add_batch(name))
               for name, layer in full_layers.items()]
    def _to_device(obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_device(x, device) for x in obj)
        if isinstance(obj, dict):
            return {k: _to_device(v, device) for k, v in obj.items()}
        return obj
    with torch.no_grad():
        for args_in, kwargs_in in block_inputs:
            args_in = _to_device(args_in, args.device)
            kwargs_in = _to_device(kwargs_in, args.device)
            block_module(*args_in, **kwargs_in)
    for h in handles:
        h.remove()
    for name, gptq in gptq_instances.items():
        if gptq.nsamples > 0:
            gptq.fasterquant(
                blocksize=getattr(args, "gptq_blocksize", 128),
                percdamp=getattr(args, "gptq_percdamp", 0.01),
                groupsize=getattr(args, "gptq_groupsize", -1),
                actorder=getattr(args, "gptq_actorder", False),
                static_groups=getattr(args, "gptq_static_groups", False),
            )
            full_name = f"{block_name}.{name}" if name else block_name
            scale = getattr(gptq.quantizer, "scale", None)
            if scale is not None:
                scale_value = scale.clone().cpu()
                if torch.all(scale_value > 0) and torch.isfinite(scale_value).all():
                    scale_dict[full_name] = (
                        scale_value.mean().unsqueeze(0)
                        if scale_value.numel() > 1 else scale_value
                    )
                    logger.info(f"Extracted scale for {full_name}")
            gptq.free()
    block_module = block_module.cpu()
    return scale_dict

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
    
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory=False, DDP_mode=False, model=args.model)

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