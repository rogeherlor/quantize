import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Global variables for VGGT context (set by Aggregator during forward pass)
_vggt_num_frames = None
_vggt_tokens_per_frame = None

#----------------------------------------------------------
# LSQC - Learned Step-Size Quantization with Per-Channel Scales
#
# Learnable quantization scales:
#   Weight scales: [d_out, 1] for Linear, [d_out, 1, 1, 1] for Conv2d
#     - One learnable scale per output channel (fixed)
#
#   Activation scales (lazy initialization):
#     - Base scale structure: [1, num_special + 1, 1]
#       * num_special individual learnable scales (e.g., 5 for 1 camera + 4 registers)
#       * 1 shared learnable scale for ALL patch tokens (resolution-independent)
#     - Expansion: Shared patch scale broadcast to match actual number of patches
#       * Works with any image resolution
#       * Frame attention: [special_scales, patch_scale × N_patches]
#       * Global attention: Same pattern replicated per frame
#     - Each special token has its own scale, all patches share one scale
#     - Gradients from all patch tokens flow back to single shared patch scale
#     - Compatible with INT8 matmul (scales applied after matmul)
#
# Use case (VGGT):
#   - Initialize with any resolution: learns 5 special + 1 patch scale
#   - Works with variable image sizes during training/inference
#   - Hardware-efficient: Y = (X_int8 @ W_int8^T) * scale_x * scale_w
#
# References:
#   - LSQ: Learned Step Size Quantization (Esser et al., 2019)
#   - VGGT: Visual Geometry Grounded Transformer
#----------------------------------------------------------
class LSQC_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, scale_shape=None, num_special_tokens=5, **kwargs):
        super(LSQC_quantizer, self).__init__()
        self.mode = mode
        self.scale_shape = scale_shape
        self.lazy_init = (scale_shape is None and mode == "activation")  # Lazy init for activations
        self.base_dim = None  # Base dimension for replication (set during first forward)
        self.num_special_tokens = num_special_tokens  # Number of special tokens (camera + register)
        
        if isinstance(module, nn.Conv2d):
            self.dim_reduction = 1
        else:
            self.dim_reduction = -1

        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            if module.first_layer:
                module.x_Qn = 2 ** (num_bits-1)
                module.x_Qp = 2 ** (num_bits-1) - 1
            else:
                module.x_Qn = 2 ** (num_bits-1)
                module.x_Qp = 2 ** (num_bits-1) - 1
            
            # Lazy initialization - scale created in first forward pass
            if not self.lazy_init:
                self.scale_shape = self.scale_shape if self.scale_shape is not None else (1,)
                module.x_scale = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
                module.x_Qparms['scale'] = module.x_scale
            # else: scale will be created in forward based on input shape
            
        elif mode == "weight":
            module.w_Qn = 2 ** (num_bits-1)
            module.w_Qp = 2 ** (num_bits-1) - 1
            self.scale_shape = self.scale_shape if self.scale_shape is not None else (1,)
            module.w_scale = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        # Get learnable base scale
        base_scale = Qparms['scale']
        
        # For lazy-initialized activations with special token handling:
        # Structure: [num_special_tokens individual scales] + [1 shared scale for all patches]
        # Example: [s_cam, s_reg1, s_reg2, s_reg3, s_reg4, s_patch_shared]
        if self.lazy_init and self.num_special_tokens > 0 and len(x.shape) >= 3:
            current_dim = x.shape[-2]
            
            # base_scale should always be [1, num_special + 1, 1]
            # Split into special token scales and shared patch scale
            if base_scale.shape[1] == self.num_special_tokens + 1:
                # Correct format: [num_special + 1]
                special_scales = base_scale[:, :self.num_special_tokens, :]  # [1, 5, 1]
                patch_scale = base_scale[:, self.num_special_tokens:, :]      # [1, 1, 1]
            else:
                # Legacy format or initialization: base_scale shape [1, base_dim, 1]
                # Need to reshape to new format
                special_scales = base_scale[:, :self.num_special_tokens, :]
                # Average all patch scales into one shared scale
                patch_scale = base_scale[:, self.num_special_tokens:, :].mean(dim=1, keepdim=True)
            
            # Try to get P (tokens per frame) from global VGGT context
            per_frame_dim = _vggt_tokens_per_frame
            
            if per_frame_dim is not None and current_dim > per_frame_dim:
                # Global attention: current_dim = S × P, replicate pattern
                num_patches_per_frame = per_frame_dim - self.num_special_tokens
                patch_scales_per_frame = patch_scale.expand(1, num_patches_per_frame, -1)
                frame_pattern = torch.cat([special_scales, patch_scales_per_frame], dim=1)
                
                num_frames = current_dim // per_frame_dim
                scale = frame_pattern.repeat(1, num_frames, 1)
            else:
                # Frame attention: build pattern for current_dim
                num_patches_current = current_dim - self.num_special_tokens
                patch_scales_expanded = patch_scale.expand(1, num_patches_current, -1)
                scale = torch.cat([special_scales, patch_scales_expanded], dim=1)
        else:
            scale = base_scale
        
        yq = _LSQC_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode)
        y = yq * scale
        return y

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        # Initialize learned quantization scales from calibration
        if "init_scale" in Qparms and "scale" in Qparms:
            init_scale = Qparms["init_scale"]
            target_scale = Qparms["scale"]
            
            if init_scale.numel() == 1:
                # Scalar initialization - fill all scales
                target_scale.data.fill_(init_scale.item())
            elif init_scale.shape == target_scale.shape:
                # Same shape - direct copy
                target_scale.data.copy_(init_scale)
            else:
                # Shape mismatch - need to handle conversion
                if self.lazy_init and self.num_special_tokens > 0:
                    # Convert from full per-token scales to special + shared format
                    # init_scale: [1, base_dim, 1] -> target: [1, num_special + 1, 1]
                    if init_scale.shape[1] >= self.num_special_tokens + 1:
                        # Copy special token scales
                        target_scale.data[:, :self.num_special_tokens, :] = \
                            init_scale[:, :self.num_special_tokens, :]
                        # Average patch scales into shared patch scale
                        target_scale.data[:, self.num_special_tokens:, :] = \
                            init_scale[:, self.num_special_tokens:, :].mean(dim=1, keepdim=True)
                    else:
                        # Fallback: use mean value
                        target_scale.data.fill_(init_scale.mean().item())
                else:
                    # Standard case: broadcast if possible
                    try:
                        target_scale.data.copy_(init_scale.expand_as(target_scale))
                    except:
                        # Fallback: use scalar value
                        target_scale.data.fill_(init_scale.mean().item())

def _LSQC_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode):

    # Qn_on_device = torch.tensor(Qn, dtype=torch.float, device=x.device)
    # Qp_on_device = torch.tensor(Qp, dtype=torch.float, device=x.device)
    qn_t = float(Qn)
    qp_t = float(Qp)

    # print(scale)
    # if scale <= 0:
    #     scale_grad = scale.grad if scale.grad is not None else "No gradient"
    #     print(f"Scale assertion failed!")
    #     print(f"  Scale value: {scale.item() if scale.numel() == 1 else scale}")
    #     print(f"  Scale gradient: {scale_grad}")
    #     print(f"  Qn: {qn_t}, Qp: {qp_t}")
    # assert scale > 0, 'scale = {}, {}, {}'.format(scale, qn_t, qp_t)

    # gradient scaling
    if num_elements > 0:
        if grad_scale_mode == "10_fac":
            grad_scale = torch.tensor(10.0, device=x.device)
        elif grad_scale_mode == "LSQ_grad_scale":
            grad_scale = 1.0 / torch.sqrt(num_elements * qp_t)
        else:
            grad_scale = torch.tensor(1.0, device=x.device)

        bw_scale = scale * grad_scale
        scale = (scale - bw_scale).detach() + bw_scale
    
    x  = x / scale
    x = torch.clamp(x, min=-float(qn_t), max=float(qp_t))
    # xq = torch.round(x)
    y  = (torch.round(x) - x).detach() + x
    
    # y  = scale * y

    return  y

