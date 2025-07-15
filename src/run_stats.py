import os
import time
import numpy as np

from xml.parsers.expat import model

from src.logger import logger

from src.run_qat import run_load_model
from src.utils import setup_dataloader, log_detailed_params
from src.make_ex_name import *

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

def one_inference(args, net):
    pin_memory = True if args.device == 'cuda' else False
    dataloader = setup_dataloader(args.dataset_name, args.batch_size, args.nworkers, pin_memory = pin_memory, DDP_mode = False, model = args.model)
    dummy_input, _ = next(iter(dataloader['val']))
    dummy_input = dummy_input.to(args.device)
    output = net(dummy_input)
    return output

def inspect(net):
    logger.info("Generating histogram of weights and activations...")
    param_dict = dict(net.named_parameters())
    BLUE = "\033[94m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    GREY = "\033[90m"
    RESET = "\033[0m"
    for name, param in net.named_parameters():
        if name.endswith('weight'):
            logger.debug(f"{BLUE}{name} (weight): {param.data.shape}")
            logger.debug(f"  - min: {param.data.min().item()}, max: {param.data.max().item()}")
            logger.debug(f"  - mean: {param.data.mean().item()}, std: {param.data.std().item()}")
            logger.debug(f"  - first values: {param.data.flatten()[:20]}...{RESET}")
            scale_name = name[:-6] + 'w_scale'
            scale = param_dict.get(scale_name, None)
            if scale is not None:
                try:
                    w_scale_np = scale.data.cpu().numpy()
                    logger.debug(f"  - w_scale: {np.array2string(w_scale_np, precision=20, separator=', ')}")
                except Exception as e:
                    logger.debug(f"  - Could not divide by w_scale: {e}{RESET}")
        elif name.endswith('bias'):
            logger.debug(f"{ORANGE}{name} (bias): {param.data.shape}")
            logger.debug(f"  - min: {param.data.min().item()}, max: {param.data.max().item()}")
            logger.debug(f"  - mean: {param.data.mean().item()}, std: {param.data.std().item()}")
            logger.debug(f"  - first values: {param.data.flatten()[:20]}...{RESET}")
        else:
            logger.debug(f"{GREY}Skipped {name}{RESET}")
    
    for name, buf in net.named_buffers():
        if name.endswith('yq') or name.endswith('y'):
            logger.debug(f"{GREEN}{name}: {buf.shape}")
            logger.debug(f"  - min: {buf.min().item()}, max: {buf.max().item()}")
            logger.debug(f"  - mean: {buf.mean().item()}, std: {buf.std().item()}")
            logger.debug(f"  - first values: {buf.flatten()[:20]}...{RESET}")
        elif name.endswith('running_mean'): # or name.endswith('running_var'):
            logger.debug(f"{ORANGE}{name}: {buf.shape}")
            logger.debug(f"  - min: {buf.min().item()}, max: {buf.max().item()}")
            logger.debug(f"  - mean: {buf.mean().item()}, std: {buf.std().item()}")
            logger.debug(f"  - first values: {buf.flatten()[:20]}...{RESET}")
        else:
            logger.debug(f"{GREY}Skipped {name}{RESET}")

def plot_histogram(data, title, bins=256):
    mean_val = np.mean(data)
    min_val = np.min(data)
    max_val = np.max(data)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=bins, edgecolor='black', color='skyblue', alpha=0.7)
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val}')
    ax.axvline(max_val, color='green', linestyle='dashed', linewidth=1.5, label=f'Max: {max_val}')
    ax.axvline(min_val, color='blue', linestyle='dashed', linewidth=1.5, label=f'Min: {min_val}')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig

def histogram(net, writer, step=0):
    logger.info("Generating histogram of weights, biases, scales, and activations...")
    param_dict = dict(net.named_parameters())
    buffer_dict = dict(net.named_buffers())

    def get_layer_and_suffix(name):
        parts = name.split('.')
        if len(parts) == 1:
            return 'root', parts[0]
        return '.'.join(parts[:-1]), parts[-1]

    logger.debug("Logging histograms and scatter plots to TensorBoard...")
    def log_tensor(writer, tag_prefix, suffix, data, step):
        if data is None or data.size == 0:
            return
        # writer.add_histogram(f"{tag_prefix}/{suffix}", data, step)
        fig = plot_histogram(data, f"{tag_prefix}/{suffix}")
        writer.add_figure(f"{tag_prefix}/{suffix}_image", fig, step)

    logger.debug("Logging parameters...")
    for name, param in param_dict.items():
        layer, suffix = get_layer_and_suffix(name)
        tag_prefix = f"{layer}"
        if suffix in ['weight', 'bias', 'w_scale']:
            data = param.data.detach().cpu().numpy().flatten()
            log_tensor(writer, tag_prefix, suffix, data, step)

    logger.debug("Logging buffers...")
    y_buffers = {}
    for name, buf in buffer_dict.items():
        layer, suffix = get_layer_and_suffix(name)
        tag_prefix = f"{layer}"

        if suffix in ['x', 'scale', 'y', 'yq']:
            data = buf.detach().cpu().numpy().flatten()
            log_tensor(writer, tag_prefix, suffix, data, step)

def run_stats(args):
    logger.debug("Run STATS arguments:\n", args)
    net = run_load_model(args)

    net.eval()
    _ = one_inference(args, net)

    args.ex_name = make_ex_name(args)
    save_path = os.path.join(args.save_path, "stats", args.dataset_name, args.ex_name)
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(save_path)

    inspect(net)
    histogram(net, writer, step=0)
    logger.info(f"Run STATS completed successfully. TensorBoard logs saved to {save_path}")
