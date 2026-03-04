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
    
    if "init_offset" in Qparms.keys():
        offset = Qparms["init_offset"]
    else:
        offset = torch.zeros(quantizer.scale_shape, dtype=torch.float).to(dev)

    if len(quantizer.scale_shape) == 2:  # Linear layer: (out_features, 1) or (1, in_features)
        if quantizer.mode == "weight":
            # Per output channel: (out_features, 1)
            x_per_channel = x.detach()
            scale = 2 * x_per_channel.abs().mean(dim=1, keepdim=True) / math.sqrt(Qp_on_device)
            # Compute optimal offset: offset = -round(mean(x) / scale)
            offset = -(x_per_channel.mean(dim=1, keepdim=True) / scale).round()
        elif quantizer.mode == "activation":
            # Per input channel: (1, in_features)
            x_per_channel = x.detach()
            scale = 2 * x_per_channel.abs().mean(dim=0, keepdim=True).unsqueeze(0) / math.sqrt(Qp_on_device)
            offset = -(x_per_channel.mean(dim=0, keepdim=True).unsqueeze(0) / scale).round()
    
    elif len(quantizer.scale_shape) == 3:  # Transformer tokens: (1, seq_len, 1) or (1, 2×(num_special+1), 1)
        if quantizer.mode == "activation":
            # Per-token scales and offsets for activations with separate first/rest frame handling
            # x shape: (batch, seq_len, channels) where seq_len may be P or S×P
            # Target shape: [1, 2×(num_special+1), 1]
            
            if hasattr(quantizer, 'num_special_tokens') and quantizer.num_special_tokens > 0:
                # Separate statistics for first frame vs. rest frames
                import src.quantizer.uniform.lsqc as lsqc_module
                per_frame_dim = lsqc_module._vggt_tokens_per_frame
                num_frames = lsqc_module._vggt_num_frames
                
                if per_frame_dim is not None and num_frames is not None and x.shape[1] >= per_frame_dim:
                    # Reshape to (batch, num_frames, tokens_per_frame, channels)
                    batch_size = x.shape[0] // num_frames if x.shape[1] == per_frame_dim else x.shape[0]
                    
                    if x.shape[1] == per_frame_dim:
                        # Frame attention: (B×S, P, C)
                        x_reshaped = x.view(batch_size, num_frames, per_frame_dim, x.shape[-1])
                    else:
                        # Global attention: (B, S×P, C)
                        x_reshaped = x.view(batch_size, num_frames, per_frame_dim, x.shape[-1])
                    
                    # Split into first frame and rest frames
                    first_frame = x_reshaped[:, 0:1, :, :]  # (B, 1, P, C)
                    rest_frames = x_reshaped[:, 1:, :, :]   # (B, S-1, P, C)
                    
                    # Compute scales per token for first frame
                    first_scales = 2 * first_frame.detach().abs().mean(dim=(0, 1, 3)) / math.sqrt(Qp_on_device)  # (P,)
                    first_means = first_frame.detach().mean(dim=(0, 1, 3))  # (P,)
                    
                    # Compute scales per token for rest frames (average across all rest frames)
                    rest_scales = 2 * rest_frames.detach().abs().mean(dim=(0, 1, 3)) / math.sqrt(Qp_on_device)  # (P,)
                    rest_means = rest_frames.detach().mean(dim=(0, 1, 3))  # (P,)
                    
                    # Split into special tokens and patches for each frame type
                    num_special = quantizer.num_special_tokens
                    first_special_scales = first_scales[:num_special].unsqueeze(0).unsqueeze(-1)  # (1, 5, 1)
                    first_patch_scale = first_scales[num_special:].mean().view(1, 1, 1)  # (1, 1, 1)
                    rest_special_scales = rest_scales[:num_special].unsqueeze(0).unsqueeze(-1)  # (1, 5, 1)
                    rest_patch_scale = rest_scales[num_special:].mean().view(1, 1, 1)  # (1, 1, 1)
                    
                    # Compute offsets: offset = -round(mean / scale)
                    first_special_means = first_means[:num_special].unsqueeze(0).unsqueeze(-1)
                    first_patch_mean = first_means[num_special:].mean().view(1, 1, 1)
                    rest_special_means = rest_means[:num_special].unsqueeze(0).unsqueeze(-1)
                    rest_patch_mean = rest_means[num_special:].mean().view(1, 1, 1)
                    
                    first_special_offsets = -(first_special_means / first_special_scales).round()
                    first_patch_offset = -(first_patch_mean / first_patch_scale).round()
                    rest_special_offsets = -(rest_special_means / rest_special_scales).round()
                    rest_patch_offset = -(rest_patch_mean / rest_patch_scale).round()
                    
                    # Concatenate: [first_special, first_patch, rest_special, rest_patch]
                    scale = torch.cat([
                        first_special_scales, first_patch_scale,
                        rest_special_scales, rest_patch_scale
                    ], dim=1)  # (1, 12, 1)
                    offset = torch.cat([
                        first_special_offsets, first_patch_offset,
                        rest_special_offsets, rest_patch_offset
                    ], dim=1)  # (1, 12, 1)
                else:
                    # Fallback: compute per-token means
                    x_per_token = x.detach()
                    scale = 2 * x_per_token.abs().mean(dim=(0, 2), keepdim=True) / math.sqrt(Qp_on_device)
                    offset = -(x_per_token.mean(dim=(0, 2), keepdim=True) / scale).round()
            else:
                # Standard per-token scales and offsets
                x_per_token = x.detach()
                scale = 2 * x_per_token.abs().mean(dim=(0, 2), keepdim=True) / math.sqrt(Qp_on_device)
                offset = -(x_per_token.mean(dim=(0, 2), keepdim=True) / scale).round()
        else:
            # Fallback for weights (shouldn't happen with 3D)
            x_det = x.detach()
            scale = torch.max(scale, 2 * x_det.abs().mean() / math.sqrt(Qp_on_device))
            offset = -(x_det.mean() / scale).round()
    
    elif len(quantizer.scale_shape) == 4:  # Conv2d layer: (out_channels, 1, 1, 1) or (1, in_channels, 1, 1)
        if quantizer.mode == "weight":
            # Per output channel: (out_channels, 1, 1, 1)
            # weight shape: (out_channels, in_channels, H, W)
            x_per_channel = x.detach()
            scale = 2 * x_per_channel.abs().mean(dim=(1, 2, 3), keepdim=True) / math.sqrt(Qp_on_device)
            offset = -(x_per_channel.mean(dim=(1, 2, 3), keepdim=True) / scale).round()
        elif quantizer.mode == "activation":
            # Per input channel: (1, in_channels, 1, 1)
            # input shape: (batch, in_channels, H, W)
            x_per_channel = x.detach()
            scale = 2 * x_per_channel.abs().mean(dim=(0, 2, 3), keepdim=True).unsqueeze(0) / math.sqrt(Qp_on_device)
            offset = -(x_per_channel.mean(dim=(0, 2, 3), keepdim=True).unsqueeze(0) / scale).round()
    
    else:
        x_det = x.detach()
        scale = torch.max(scale, 2 * x_det.abs().mean() / math.sqrt(Qp_on_device))
        offset = -(x_det.mean() / scale).round()
    
    Qparms["init_scale"] = scale
    Qparms["init_offset"] = offset