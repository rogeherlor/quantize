import os
import time

from src.logger import logger

from src.run_qat import run_load_model
from src.utils import setup_dataloader

def one_inference(args, net):
    pin_memory = True if args.device == 'cuda' else False
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory = pin_memory, DDP_mode = False, model = args.model)
    dummy_input, _ = next(iter(dataloader['val']))
    dummy_input = dummy_input.to(args.device)
    output = net(dummy_input)
    return output

def activation_stats(net, output):
    pass

def histogram(net):
    logger.info("Generating histogram of weights and activations...")
    param_dict = dict(net.named_parameters())
    BLUE = "\033[94m"
    ORANGE = "\033[38;5;208m"
    RESET = "\033[0m"
    for name, param in net.named_parameters():
        if name.endswith('weight'):
            logger.debug(f"{BLUE}{name} (weight): {param.data.shape}")
            logger.debug(f"  - min: {param.data.min().item()}, max: {param.data.max().item()}")
            logger.debug(f"  - mean: {param.data.mean().item()}, std: {param.data.std().item()}")
            import numpy as np
            np.set_printoptions(precision=20, suppress=False)
            logger.debug(f"  - first values: {param.data.flatten()[:20]}...")
            scale_name = name[:-6] + 'w_scale'
            scale = param_dict.get(scale_name, None)
            if scale is not None:
                try:
                    divided = (param.data.flatten()[:20] / scale.data).cpu()
                    logger.debug(f"  - w_scale: {scale.data}")
                    logger.debug(f"  - first values / w_scale: {divided}...{RESET}")
                except Exception as e:
                    logger.debug(f"  - Could not divide by w_scale: {e}{RESET}")
        elif name.endswith('bias'):
            logger.debug(f"{ORANGE}{name} (bias): {param.data.shape}")
            logger.debug(f"  - min: {param.data.min().item()}, max: {param.data.max().item()}")
            logger.debug(f"  - mean: {param.data.mean().item()}, std: {param.data.std().item()}")
            logger.debug(f"  - first values: {param.data.flatten()[:20]}...{RESET}")

def run_stats(args):
    logger.debug("Run STATS arguments:\n", args)
    net = run_load_model(args)

    net.eval()
    output = one_inference(args, net)

    histogram(net)