import os
import torch
import torch.nn as nn

from src.utils import *
from src.logger import logger
from src.make_ex_name import make_ex_name

from src.models.model import create_model
from src.ptq.gptq.run_gptq import run_gptq_quantization
from src.ptq.rtn.rtn import run_rtn_quantization

PTQ_CONFIG = {
    'gptq': dotdict({
        'quant_blocks': ['aggregator.patch_embed', 
        *[f'aggregator.frame_blocks.{i}.attn.qkv' for i in range(24)], 
        *[f'aggregator.frame_blocks.{i}.attn.proj' for i in range(24)],
        *[f'aggregator.frame_blocks.{i}.mlp' for i in range(24)], 
        *[f'aggregator.global_blocks.{i}.attn.qkv' for i in range(24)],
        *[f'aggregator.global_blocks.{i}.attn.proj' for i in range(24)],
        *[f'aggregator.global_blocks.{i}.mlp' for i in range(24)], 
        'camera_head.trunk.0', 'camera_head.trunk.1', 'camera_head.trunk.2', 'camera_head.trunk.3', 'camera_head.embed_pose', 'camera_head.poseLN_modulation', 'camera_head.pose_branch'],
        'gptq_blocksize': 128,
        'mse': True,
        'gptq_percdamp': 0.1,  # 0.01,
        'gptq_groupsize': -1,
        'gptq_actorder': False,
        'gptq_static_groups': False,
        'calibration_samples': 4096,  # 10000,
        'replace_with_qmodules': True,
        'output_path': './data3/rogelio/model_zoo/vggt/'
    }),
    'rtn': dotdict({
        'calibration_samples': 1000,
        'rtn_scale_method': 'symmetric_max',  # minmax, symmetric_max, mse, percentile
        'rtn_percentile': 99.99,  # Used when rtn_scale_method='percentile'
        'rtn_symmetric': True
    })
}

def save_quantized_model(model, args):
    save_dir = f"{args.save_path}/ptq/{args.dataset_name}/{args.model}"
    ex_name = make_ex_name(args)
    save_path = f"{save_dir}/{ex_name}"
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': args
    }
    
    torch.save(checkpoint, f"{save_path}/ptq_model.pth")
    logger.info(f"Quantized model saved to {save_path}/ptq_model.pth")

def run_ptq(args):

    logger.debug("Run PTQ arguments:\n", args)
    logger.info(f"Using GPU rank {args.gpu_rank} for PTQ")
    
    ptq_algorithm = args.ptq_algorithm
    logger.info(f"Using PTQ algorithm: {ptq_algorithm}")
    ptq_config = PTQ_CONFIG.get(ptq_algorithm)

    for key, value in ptq_config.items():
        setattr(args, key, value)
    
    if ptq_algorithm == 'gptq':
        logger.info(f"  - Calibration samples: {args.calibration_samples}")
        logger.info(f"  - GPTQ block size: {args.gptq_blocksize}")
        logger.info(f"  - GPTQ damping: {args.gptq_percdamp}")
        logger.info(f"  - GPTQ group size: {args.gptq_groupsize}")
        logger.info(f"  - GPTQ activation order: {args.gptq_actorder}")
        logger.info(f"  - Replace with Q-modules: {args.replace_with_qmodules}")
        qmodel = run_gptq_quantization(args)
    elif ptq_algorithm == 'rtn':
        qmodel = run_rtn_quantization(args)

    logger.info("Saving quantized model...")
    save_quantized_model(qmodel, args)