import os
import time

from src.logger import logger

from src.run_qat import run_load_model

def histogram(net):
    logger.info("Generating histogram of weights and activations...")
    for name, param in net.named_parameters():
        logger.debug(f"Layer name: {name}")
        if name.endswith('weight') or name.endswith('bias') or name.endswith('w_scale') or name.endswith('x_scale'):
            logger.debug(f"{name}: {param.data.shape}")
            if param.requires_grad:
                logger.debug(f"  - requires_grad: {param.requires_grad}")
            else:
                logger.debug(f"  - requires_grad: False")
            logger.debug(f"  - min: {param.data.min().item()}, max: {param.data.max().item()}")
            logger.debug(f"  - mean: {param.data.mean().item()}, std: {param.data.std().item()}")
            logger.debug(f"  - histogram: {param.data.histogram()}")

def run_stats(args):
    logger.debug("Run STATS arguments:\n", args)
    net = run_load_model(args)
    net.eval()
    logger.debug("Model:\n {net}")
    histogram(net)