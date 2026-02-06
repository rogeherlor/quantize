import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.quantizer.uniform import *
from src.quantizer.nonuniform import *

def LSQC_initializer(x, Qparms, Qn, Qp, quantizer, *args):
    dev = x.device
    Qp_on_device = torch.tensor(Qp, dtype=torch.float).to(dev)
    Qn_on_device = torch.tensor(Qn, dtype=torch.float).to(dev)

    if "init_scale" in Qparms.keys():
        scale = Qparms["init_scale"]
    else:
        scale = torch.zeros(quantizer.scale_shape, dtype=torch.float).to(dev)

    if len(quantizer.scale_shape) == 2:  # Linear layer: (out_features, 1) or (1, in_features)
        if quantizer.mode == "weight":
            # Per output channel: (out_features, 1)
            scale = 2 * x.detach().abs().mean(dim=1, keepdim=True) / math.sqrt(Qp_on_device)
        elif quantizer.mode == "activation":
            # Per input channel: (1, in_features)
            scale = 2 * x.detach().abs().mean(dim=0, keepdim=True).unsqueeze(0) / math.sqrt(Qp_on_device)
    
    elif len(quantizer.scale_shape) == 3:  # Transformer tokens: (1, seq_len, 1) or (1, num_special+1, 1)
        if quantizer.mode == "activation":
            # Per-token scales for activations: (1, seq_len, 1)
            # x shape: (batch, seq_len, channels)
            scale = 2 * x.detach().abs().mean(dim=(0, 2), keepdim=True) / math.sqrt(Qp_on_device)
        else:
            # Fallback for weights (shouldn't happen with 3D)
            scale = torch.max(scale, 2 * x.detach().abs().mean() / math.sqrt(Qp_on_device))
    
    elif len(quantizer.scale_shape) == 4:  # Conv2d layer: (out_channels, 1, 1, 1) or (1, in_channels, 1, 1)
        if quantizer.mode == "weight":
            # Per output channel: (out_channels, 1, 1, 1)
            # weight shape: (out_channels, in_channels, H, W)
            scale = 2 * x.detach().abs().mean(dim=(1, 2, 3), keepdim=True) / math.sqrt(Qp_on_device)
        elif quantizer.mode == "activation":
            # Per input channel: (1, in_channels, 1, 1)
            # input shape: (batch, in_channels, H, W)
            scale = 2 * x.detach().abs().mean(dim=(0, 2, 3), keepdim=True).unsqueeze(0) / math.sqrt(Qp_on_device)
    
    else:
        scale = torch.max(scale, 2 * x.detach().abs().mean() / math.sqrt(Qp_on_device))
    
    Qparms["init_scale"] = scale