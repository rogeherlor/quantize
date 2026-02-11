import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Global variables for VGGT context (set by Aggregator during forward pass)
_vggt_num_frames = None
_vggt_tokens_per_frame = None
_vggt_batch_size = None

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
        
        # Apply gradient scaling to base scale BEFORE expansion (only during training)
        # This ensures gradient flow is correct to the learnable parameter
        if num_elements > 0 and self.training:
            if grad_scale_mode == "10_fac":
                grad_scale = 10.0
            elif grad_scale_mode == "LSQ_grad_scale":
                Qp_val = float(Qp)
                grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * Qp_val, device=base_scale.device))
            else:
                grad_scale = 1.0
            
            # Apply gradient scaling trick to base parameter
            grad_scale_tensor = torch.tensor(grad_scale, device=base_scale.device, dtype=base_scale.dtype)
            bw_scale = base_scale * grad_scale_tensor
            base_scale = (base_scale - bw_scale).detach() + bw_scale
        
        # Debug: print ACTUAL learnable parameter (should be 6 values: 5 special + 1 patch)
        # if self.mode == "activation":
        #     actual_param = Qparms['scale']  # The actual nn.Parameter
        #     print(f"BASE PARAM {actual_param.shape}: {actual_param.data.squeeze()[:12]}")
        
        # For lazy-initialized activations with special token handling:
        # Structure: [frame1: 5 special + 1 patch] + [frames2+: 5 special + 1 patch]
        # Total: 12 learnable parameters
        if self.lazy_init and self.num_special_tokens > 0 and len(x.shape) >= 3:
            current_dim = x.shape[-2]
            
            # base_scale should be [1, 12, 1]: [frame1_scales (6)] + [rest_frames_scales (6)]
            per_frame_scales = self.num_special_tokens + 1  # 6 scales per frame type
            
            if base_scale.shape[1] == 2 * per_frame_scales:
                # Correct format: [1, 12, 1]
                frame1_scales = base_scale[:, :per_frame_scales, :]      # [1, 6, 1] for frame 1
                rest_frames_scales = base_scale[:, per_frame_scales:, :] # [1, 6, 1] for frames 2+
            else:
                # Legacy: use same scales for all frames
                if base_scale.shape[1] >= per_frame_scales:
                    frame1_scales = base_scale[:, :per_frame_scales, :]
                    rest_frames_scales = frame1_scales.clone()
                else:
                    # Fallback: replicate what we have
                    frame1_scales = base_scale
                    rest_frames_scales = base_scale
            
            # Split each frame type into special + patch
            frame1_special = frame1_scales[:, :self.num_special_tokens, :]       # [1, 5, 1]
            frame1_patch = frame1_scales[:, self.num_special_tokens:, :]          # [1, 1, 1]
            
            rest_special = rest_frames_scales[:, :self.num_special_tokens, :]    # [1, 5, 1]
            rest_patch = rest_frames_scales[:, self.num_special_tokens:, :]       # [1, 1, 1]
            
            # Try to get P (tokens per frame) from global VGGT context
            per_frame_dim = _vggt_tokens_per_frame
            num_frames = _vggt_num_frames
            B = _vggt_batch_size
            
            if per_frame_dim is not None and current_dim > per_frame_dim:
                # Global attention: current_dim = S × P
                num_patches_per_frame = per_frame_dim - self.num_special_tokens
                
                # Build frame 1 pattern
                frame1_patch_expanded = frame1_patch.repeat(1, num_patches_per_frame, 1)
                frame1_pattern = torch.cat([frame1_special, frame1_patch_expanded], dim=1)
                
                # Build rest frames pattern
                rest_patch_expanded = rest_patch.repeat(1, num_patches_per_frame, 1)
                rest_pattern = torch.cat([rest_special, rest_patch_expanded], dim=1)
                
                # Concatenate: frame1 + (rest_pattern × (num_frames - 1))
                num_frames_calc = current_dim // per_frame_dim
                if num_frames_calc > 1:
                    rest_repeated = rest_pattern.repeat(1, num_frames_calc - 1, 1)
                    scale = torch.cat([frame1_pattern, rest_repeated], dim=1)
                else:
                    scale = frame1_pattern
                
                # Debug: verify expansion
                # frame1_sample = scale[0, :min(15, per_frame_dim), 0].detach().cpu().numpy()
                # if num_frames_calc > 1:
                    # frame2_sample = scale[0, per_frame_dim:per_frame_dim+min(15, per_frame_dim), 0].detach().cpu().numpy()
                    # print(f"  Frame1 (first 15): {frame1_sample}")
                    # print(f"  Frame2 (first 15): {frame2_sample}")
                # else:
                    # print(f"  Frame1 (first 15): {frame1_sample}")
            elif per_frame_dim is not None and current_dim == per_frame_dim and num_frames is not None and B is not None:
                # Frame attention: shape is (B*S, P, C)
                # Due to slice_expand_and_flatten, frames are interleaved:
                # [batch0_frame0, batch0_frame1, ..., batch0_frame(S-1), batch1_frame0, ...]
                # So frame 0 elements are at positions [0, S, 2S, 3S, ..., (B-1)*S]
                num_patches_current = current_dim - self.num_special_tokens
                
                # Build scale patterns for both frame types
                frame1_patch_expanded = frame1_patch.repeat(1, num_patches_current, 1)
                frame1_pattern = torch.cat([frame1_special, frame1_patch_expanded], dim=1)  # [1, P, 1]
                
                rest_patch_expanded = rest_patch.repeat(1, num_patches_current, 1)
                rest_pattern = torch.cat([rest_special, rest_patch_expanded], dim=1)  # [1, P, 1]
                
                # Create scale tensor with alternating pattern: [frame0, frame1+, ..., frame1+] repeated B times
                batch_size_total = x.shape[0]  # B*S
                scale = rest_pattern.repeat(batch_size_total, 1, 1)  # [B*S, P, 1] - all rest by default
                
                # Override every S-th position (0, S, 2S, ...) with frame0 scales
                for batch_idx in range(B):
                    frame0_pos = batch_idx * num_frames  # Position of frame 0 for this batch
                    scale[frame0_pos:frame0_pos+1, :, :] = frame1_pattern
                
                # Debug: verify correct assignment
                # print(f"  Frame attention - B={B}, S={num_frames}, total_batch={batch_size_total}")
                # print(f"  Frame0 positions: {[i*num_frames for i in range(B)]}")
                # print(f"  Frame0 (pos 0) scales: {scale[0, :6, 0].detach().cpu().numpy()}")
                # if num_frames > 1:
                    # print(f"  Frame1 (pos 1) scales: {scale[1, :6, 0].detach().cpu().numpy()}")
            else:
                # Fallback: use frame1 scales for all (legacy behavior)
                num_patches_current = current_dim - self.num_special_tokens
                patch_scales_expanded = frame1_patch.repeat(1, num_patches_current, 1)
                scale = torch.cat([frame1_special, patch_scales_expanded], dim=1)
        else:
            scale = base_scale

        yq = _LSQC_quantizer(x, scale, Qn, Qp)
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

def _LSQC_quantizer(x, scale, Qn, Qp):
    qn_t = float(Qn)
    qp_t = float(Qp)
    
    # Quantize: divide by scale, clamp, round
    x = x / scale
    x = torch.clamp(x, min=-qn_t, max=qp_t)
    y = (torch.round(x) - x).detach() + x  # STE: straight-through estimator
    
    return y

