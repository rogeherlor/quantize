import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Global variables for VGGT context (set by Aggregator during forward pass)
_vggt_num_frames = None
_vggt_tokens_per_frame = None
_vggt_batch_size = None

#----------------------------------------------------------
# LSQC - LSQ+ with Per-Channel Scales and Zero-Points (Offsets)
#
# Learnable quantization parameters (scale + offset):
#   Weight params: [d_out, 1] for Linear, [d_out, 1, 1, 1] for Conv2d
#     - One learnable scale + offset per output channel (fixed)
#
#   Activation params (lazy initialization):
#     - Base structure: [1, num_special + 1, 1]
#       * num_special individual learnable params (e.g., 5 for 1 camera + 4 registers)
#       * 1 shared learnable param for ALL patch tokens (resolution-independent)
#     - Expansion: Shared patch params broadcast to match actual number of patches
#       * Works with any image resolution
#       * Frame attention: [special_params, patch_param × N_patches]
#       * Global attention: Same pattern replicated per frame
#     - Each special token has its own scale/offset, all patches share one scale/offset
#     - Gradients from all patch tokens flow back to single shared patch params
#
# Quantization formula (LSQ+):
#   q = clamp(round(x/s + z), -Qn, Qp)  [forward with offset]
#   y = (q - z) * s                      [dequantization]
#
# Use case (VGGT):
#   - Initialize with any resolution: learns 5 special + 1 patch (scale + offset)
#   - Works with variable image sizes during training/inference
#   - Asymmetric quantization for better range utilization
#
# References:
#   - LSQ+: Learned Step Size Quantization (Bhalgat et al., 2020)
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
            num_bits = 4
            if module.first_layer:
                module.x_Qn = 2 ** (num_bits-1)
                module.x_Qp = 2 ** (num_bits-1) - 1
            else:
                module.x_Qn = 2 ** (num_bits-1)
                module.x_Qp = 2 ** (num_bits-1) - 1
            
            # Lazy initialization - scale and offset created in first forward pass
            if not self.lazy_init:
                self.scale_shape = self.scale_shape if self.scale_shape is not None else (1,)
                module.x_scale = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
                module.x_offset = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
                module.x_Qparms['scale'] = module.x_scale
                module.x_Qparms['offset'] = module.x_offset
            # else: scale and offset will be created in forward based on input shape
            
        elif mode == "weight":
            num_bits = 4
            module.w_Qn = 2 ** (num_bits-1)
            module.w_Qp = 2 ** (num_bits-1) - 1
            self.scale_shape = self.scale_shape if self.scale_shape is not None else (1,)
            module.w_scale = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
            module.w_offset = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale
            module.w_Qparms['offset'] = module.w_offset

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        # Get learnable base parameters (scale and offset)
        base_scale = Qparms['scale']
        base_offset = Qparms['offset']
        
        # Apply gradient scaling to base parameters BEFORE expansion (only during training)
        # This ensures gradient flow is correct to the learnable parameters
        grad_scale_mode = "LSQ_grad_scale"
        if num_elements > 0 and self.training:
            if grad_scale_mode == "10_fac":
                grad_scale = 10.0
            elif grad_scale_mode == "LSQ_grad_scale":
                Qp_val = float(Qp)
                grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * Qp_val, device=base_scale.device))
            else:
                grad_scale = 1.0
            
            # Apply gradient scaling trick to base parameters
            grad_scale_tensor = torch.tensor(grad_scale, device=base_scale.device, dtype=base_scale.dtype)
            bw_scale = base_scale * grad_scale_tensor
            base_scale = (base_scale - bw_scale).detach() + bw_scale
            bw_offset = base_offset * grad_scale_tensor
            base_offset = (base_offset - bw_offset).detach() + bw_offset
        
        # Debug: print ACTUAL learnable parameters (should be 12 values each: [5 special + 1 patch] × 2 frame types)
        # if self.mode == "activation":
        #     actual_scale = Qparms['scale']  # The actual nn.Parameter
        #     actual_offset = Qparms['offset']
        #     print(f"BASE SCALE {actual_scale.shape}: {actual_scale.data.squeeze()[:12]}")
        #     print(f"BASE OFFSET {actual_offset.shape}: {actual_offset.data.squeeze()[:12]}")
        
        # For lazy-initialized activations with special token handling:
        # Structure: [frame1: 5 special + 1 patch] + [frames2+: 5 special + 1 patch]
        # Total: 12 learnable parameters per tensor (scale and offset)
        if self.lazy_init and self.num_special_tokens > 0 and len(x.shape) >= 3:
            current_dim = x.shape[-2]
            
            # base_scale and base_offset should be [1, 12, 1]: [frame1_params (6)] + [rest_frames_params (6)]
            per_frame_params = self.num_special_tokens + 1  # 6 params per frame type
            
            # Process scales
            if base_scale.shape[1] == 2 * per_frame_params:
                # Correct format: [1, 12, 1]
                frame1_scales = base_scale[:, :per_frame_params, :]      # [1, 6, 1] for frame 1
                rest_frames_scales = base_scale[:, per_frame_params:, :] # [1, 6, 1] for frames 2+
            else:
                # Legacy: use same scales for all frames
                if base_scale.shape[1] >= per_frame_params:
                    frame1_scales = base_scale[:, :per_frame_params, :]
                    rest_frames_scales = frame1_scales.clone()
                else:
                    # Fallback: replicate what we have
                    frame1_scales = base_scale
                    rest_frames_scales = base_scale
            
            # Process offsets (same structure as scales)
            if base_offset.shape[1] == 2 * per_frame_params:
                frame1_offsets = base_offset[:, :per_frame_params, :]
                rest_frames_offsets = base_offset[:, per_frame_params:, :]
            else:
                if base_offset.shape[1] >= per_frame_params:
                    frame1_offsets = base_offset[:, :per_frame_params, :]
                    rest_frames_offsets = frame1_offsets.clone()
                else:
                    frame1_offsets = base_offset
                    rest_frames_offsets = base_offset
            
            # Split each frame type into special + patch (scales)
            frame1_special_s = frame1_scales[:, :self.num_special_tokens, :]       # [1, 5, 1]
            frame1_patch_s = frame1_scales[:, self.num_special_tokens:, :]          # [1, 1, 1]
            
            rest_special_s = rest_frames_scales[:, :self.num_special_tokens, :]    # [1, 5, 1]
            rest_patch_s = rest_frames_scales[:, self.num_special_tokens:, :]       # [1, 1, 1]
            
            # Split each frame type into special + patch (offsets)
            frame1_special_o = frame1_offsets[:, :self.num_special_tokens, :]       # [1, 5, 1]
            frame1_patch_o = frame1_offsets[:, self.num_special_tokens:, :]          # [1, 1, 1]
            
            rest_special_o = rest_frames_offsets[:, :self.num_special_tokens, :]    # [1, 5, 1]
            rest_patch_o = rest_frames_offsets[:, self.num_special_tokens:, :]       # [1, 1, 1]
            
            # Try to get P (tokens per frame) from global VGGT context
            per_frame_dim = _vggt_tokens_per_frame
            num_frames = _vggt_num_frames
            B = _vggt_batch_size
            
            if per_frame_dim is not None and current_dim > per_frame_dim:
                # Global attention: current_dim = S × P
                num_patches_per_frame = per_frame_dim - self.num_special_tokens
                
                # Build frame 1 pattern (scale and offset)
                frame1_patch_expanded_s = frame1_patch_s.repeat(1, num_patches_per_frame, 1)
                frame1_pattern_s = torch.cat([frame1_special_s, frame1_patch_expanded_s], dim=1)
                frame1_patch_expanded_o = frame1_patch_o.repeat(1, num_patches_per_frame, 1)
                frame1_pattern_o = torch.cat([frame1_special_o, frame1_patch_expanded_o], dim=1)
                
                # Build rest frames pattern (scale and offset)
                rest_patch_expanded_s = rest_patch_s.repeat(1, num_patches_per_frame, 1)
                rest_pattern_s = torch.cat([rest_special_s, rest_patch_expanded_s], dim=1)
                rest_patch_expanded_o = rest_patch_o.repeat(1, num_patches_per_frame, 1)
                rest_pattern_o = torch.cat([rest_special_o, rest_patch_expanded_o], dim=1)
                
                # Concatenate: frame1 + (rest_pattern × (num_frames - 1))
                num_frames_calc = current_dim // per_frame_dim
                if num_frames_calc > 1:
                    rest_repeated_s = rest_pattern_s.repeat(1, num_frames_calc - 1, 1)
                    scale = torch.cat([frame1_pattern_s, rest_repeated_s], dim=1)
                    rest_repeated_o = rest_pattern_o.repeat(1, num_frames_calc - 1, 1)
                    offset = torch.cat([frame1_pattern_o, rest_repeated_o], dim=1)
                else:
                    scale = frame1_pattern_s
                    offset = frame1_pattern_o
                
                # Debug: verify expansion
                # frame1_sample_s = scale[0, :min(15, per_frame_dim), 0].detach().cpu().numpy()
                # frame1_sample_o = offset[0, :min(15, per_frame_dim), 0].detach().cpu().numpy()
                # if num_frames_calc > 1:
                    # frame2_sample_s = scale[0, per_frame_dim:per_frame_dim+min(15, per_frame_dim), 0].detach().cpu().numpy()
                    # frame2_sample_o = offset[0, per_frame_dim:per_frame_dim+min(15, per_frame_dim), 0].detach().cpu().numpy()
                    # print(f"  Frame1 scale (first 15): {frame1_sample_s}")
                    # print(f"  Frame1 offset (first 15): {frame1_sample_o}")
                    # print(f"  Frame2 scale (first 15): {frame2_sample_s}")
                    # print(f"  Frame2 offset (first 15): {frame2_sample_o}")
                # else:
                    # print(f"  Frame1 scale (first 15): {frame1_sample_s}")
                    # print(f"  Frame1 offset (first 15): {frame1_sample_o}")
            elif per_frame_dim is not None and current_dim == per_frame_dim and num_frames is not None and B is not None:
                # Frame attention: shape is (B*S, P, C)
                # Due to slice_expand_and_flatten, frames are interleaved:
                # [batch0_frame0, batch0_frame1, ..., batch0_frame(S-1), batch1_frame0, ...]
                # So frame 0 elements are at positions [0, S, 2S, 3S, ..., (B-1)*S]
                num_patches_current = current_dim - self.num_special_tokens
                
                # Build patterns for both frame types (scale and offset)
                frame1_patch_expanded_s = frame1_patch_s.repeat(1, num_patches_current, 1)
                frame1_pattern_s = torch.cat([frame1_special_s, frame1_patch_expanded_s], dim=1)  # [1, P, 1]
                frame1_patch_expanded_o = frame1_patch_o.repeat(1, num_patches_current, 1)
                frame1_pattern_o = torch.cat([frame1_special_o, frame1_patch_expanded_o], dim=1)  # [1, P, 1]
                
                rest_patch_expanded_s = rest_patch_s.repeat(1, num_patches_current, 1)
                rest_pattern_s = torch.cat([rest_special_s, rest_patch_expanded_s], dim=1)  # [1, P, 1]
                rest_patch_expanded_o = rest_patch_o.repeat(1, num_patches_current, 1)
                rest_pattern_o = torch.cat([rest_special_o, rest_patch_expanded_o], dim=1)  # [1, P, 1]
                
                # Create tensors with alternating pattern: [frame0, frame1+, ..., frame1+] repeated B times
                batch_size_total = x.shape[0]  # B*S
                scale = rest_pattern_s.repeat(batch_size_total, 1, 1)  # [B*S, P, 1] - all rest by default
                offset = rest_pattern_o.repeat(batch_size_total, 1, 1)  # [B*S, P, 1]
                
                # Override every S-th position (0, S, 2S, ...) with frame0 params
                for batch_idx in range(B):
                    frame0_pos = batch_idx * num_frames  # Position of frame 0 for this batch
                    scale[frame0_pos:frame0_pos+1, :, :] = frame1_pattern_s
                    offset[frame0_pos:frame0_pos+1, :, :] = frame1_pattern_o
                
                # Debug: verify correct assignment
                # print(f"  Frame attention - B={B}, S={num_frames}, total_batch={batch_size_total}")
                # print(f"  Frame0 positions: {[i*num_frames for i in range(B)]}")
                # print(f"  Frame0 (pos 0) scale: {scale[0, :6, 0].detach().cpu().numpy()}")
                # print(f"  Frame0 (pos 0) offset: {offset[0, :6, 0].detach().cpu().numpy()}")
                # if num_frames > 1:
                    # print(f"  Frame1 (pos 1) scale: {scale[1, :6, 0].detach().cpu().numpy()}")
                    # print(f"  Frame1 (pos 1) offset: {offset[1, :6, 0].detach().cpu().numpy()}")
            else:
                # Fallback: use frame1 params for all (legacy behavior)
                num_patches_current = current_dim - self.num_special_tokens
                patch_scales_expanded = frame1_patch_s.repeat(1, num_patches_current, 1)
                scale = torch.cat([frame1_special_s, patch_scales_expanded], dim=1)
                patch_offsets_expanded = frame1_patch_o.repeat(1, num_patches_current, 1)
                offset = torch.cat([frame1_special_o, patch_offsets_expanded], dim=1)
        else:
            scale = base_scale
            offset = base_offset

        yq = _LSQC_quantizer(x, scale, offset, Qn, Qp)
        y = (yq - offset) * scale
        return y

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        # Initialize learned quantization parameters (scale and offset) from calibration
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
                    # init_scale: [1, base_dim, 1] or [1, 2*base_dim, 1] -> target: [1, 2*(num_special+1), 1]
                    per_frame_params = self.num_special_tokens + 1  # 6
                    if init_scale.shape[1] == 2 * per_frame_params:
                        # Already in correct 2-frame format [1, 12, 1] - direct copy
                        target_scale.data.copy_(init_scale)
                    elif init_scale.shape[1] >= self.num_special_tokens + 1:
                        # Single frame format - copy to both frame types
                        target_scale.data[:, :per_frame_params, :] = init_scale[:, :per_frame_params, :]
                        target_scale.data[:, per_frame_params:, :] = init_scale[:, :per_frame_params, :]
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
        
        # Initialize offsets from calibration
        if "init_offset" in Qparms and "offset" in Qparms:
            init_offset = Qparms["init_offset"]
            target_offset = Qparms["offset"]
            
            if init_offset.numel() == 1:
                target_offset.data.fill_(init_offset.item())
            elif init_offset.shape == target_offset.shape:
                target_offset.data.copy_(init_offset)
            else:
                if self.lazy_init and self.num_special_tokens > 0:
                    per_frame_params = self.num_special_tokens + 1  # 6
                    if init_offset.shape[1] == 2 * per_frame_params:
                        # Already in correct 2-frame format [1, 12, 1] - direct copy
                        target_offset.data.copy_(init_offset)
                    elif init_offset.shape[1] >= self.num_special_tokens + 1:
                        # Single frame format - copy to both frame types
                        target_offset.data[:, :per_frame_params, :] = init_offset[:, :per_frame_params, :]
                        target_offset.data[:, per_frame_params:, :] = init_offset[:, :per_frame_params, :]
                    else:
                        target_offset.data.fill_(init_offset.mean().item())
                else:
                    try:
                        target_offset.data.copy_(init_offset.expand_as(target_offset))
                    except:
                        target_offset.data.fill_(init_offset.mean().item())

def _LSQC_quantizer(x, scale, offset, Qn, Qp):
    qn_t = float(Qn)
    qp_t = float(Qp)
    
    # LSQ+ quantization: divide by scale, add offset, clamp, round
    x = x / scale + offset
    x = torch.clamp(x, min=-qn_t, max=qp_t)
    y = (torch.round(x) - x).detach() + x  # STE: straight-through estimator
    
    return y

