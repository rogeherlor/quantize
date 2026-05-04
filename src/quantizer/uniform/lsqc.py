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
#     - Base structure: [1, 2*(num_special+1), 1]
#       * num_special individual learnable params (e.g., 5 for 1 camera + 4 registers)
#       * 1 shared learnable param for ALL patch tokens (resolution-independent)
#       * Both frame-type params stored: [frame1_params (6)] + [rest_frames_params (6)]
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
            num_bits = 8
            module.x_Qn = 2 ** (num_bits-1)
            module.x_Qp = 2 ** (num_bits-1) - 1
            # Lazy initialization - scale and offset created in first forward pass
            # (scale and offset will be created in _forward_common based on input shape)

        elif mode == "weight":
            num_bits = 4
            module.w_Qn = 2 ** (num_bits-1)
            module.w_Qp = 2 ** (num_bits-1) - 1
            module.w_scale = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
            module.w_offset = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale
            module.w_Qparms['offset'] = module.w_offset

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        base_scale = Qparms['scale']
        base_offset = Qparms['offset']

        grad_scale_mode = "LSQ_grad_scale"
        if num_elements > 0 and self.training:
            Qp_val = float(Qp)
            grad_scale = 1.0 / torch.sqrt(torch.tensor(num_elements * Qp_val, device=base_scale.device))
            grad_scale_tensor = torch.tensor(grad_scale, device=base_scale.device, dtype=base_scale.dtype)
            bw_scale = base_scale * grad_scale_tensor
            base_scale = (base_scale - bw_scale).detach() + bw_scale
            bw_offset = base_offset * grad_scale_tensor
            base_offset = (base_offset - bw_offset).detach() + bw_offset

        if self.lazy_init and self.num_special_tokens > 0 and len(x.shape) >= 3:
            current_dim = x.shape[-2]
            per_frame_params = self.num_special_tokens + 1  # e.g. 6

            # base_scale / base_offset must be [1, 2*per_frame_params, 1]
            frame1_scales      = base_scale[:, :per_frame_params, :]
            rest_frames_scales = base_scale[:, per_frame_params:, :]
            frame1_offsets      = base_offset[:, :per_frame_params, :]
            rest_frames_offsets = base_offset[:, per_frame_params:, :]

            frame1_special_s = frame1_scales[:, :self.num_special_tokens, :]
            frame1_patch_s   = frame1_scales[:, self.num_special_tokens:, :]
            rest_special_s   = rest_frames_scales[:, :self.num_special_tokens, :]
            rest_patch_s     = rest_frames_scales[:, self.num_special_tokens:, :]

            frame1_special_o = frame1_offsets[:, :self.num_special_tokens, :]
            frame1_patch_o   = frame1_offsets[:, self.num_special_tokens:, :]
            rest_special_o   = rest_frames_offsets[:, :self.num_special_tokens, :]
            rest_patch_o     = rest_frames_offsets[:, self.num_special_tokens:, :]

            per_frame_dim = _vggt_tokens_per_frame
            num_frames    = _vggt_num_frames
            B             = _vggt_batch_size

            if per_frame_dim is not None and current_dim > per_frame_dim:
                # Global attention: current_dim = num_frames × per_frame_dim
                num_patches_per_frame = per_frame_dim - self.num_special_tokens

                frame1_pattern_s = torch.cat([frame1_special_s, frame1_patch_s.repeat(1, num_patches_per_frame, 1)], dim=1)
                frame1_pattern_o = torch.cat([frame1_special_o, frame1_patch_o.repeat(1, num_patches_per_frame, 1)], dim=1)
                rest_pattern_s   = torch.cat([rest_special_s,   rest_patch_s.repeat(1, num_patches_per_frame, 1)],   dim=1)
                rest_pattern_o   = torch.cat([rest_special_o,   rest_patch_o.repeat(1, num_patches_per_frame, 1)],   dim=1)

                num_frames_calc = current_dim // per_frame_dim
                if num_frames_calc > 1:
                    scale  = torch.cat([frame1_pattern_s, rest_pattern_s.repeat(1, num_frames_calc - 1, 1)], dim=1)
                    offset = torch.cat([frame1_pattern_o, rest_pattern_o.repeat(1, num_frames_calc - 1, 1)], dim=1)
                else:
                    scale  = frame1_pattern_s
                    offset = frame1_pattern_o

            elif per_frame_dim is not None and current_dim == per_frame_dim and num_frames is not None and B is not None:
                # Frame attention: shape is (B*num_frames, per_frame_dim, C)
                num_patches_current = current_dim - self.num_special_tokens

                frame1_pattern_s = torch.cat([frame1_special_s, frame1_patch_s.repeat(1, num_patches_current, 1)], dim=1)
                frame1_pattern_o = torch.cat([frame1_special_o, frame1_patch_o.repeat(1, num_patches_current, 1)], dim=1)
                rest_pattern_s   = torch.cat([rest_special_s,   rest_patch_s.repeat(1, num_patches_current, 1)],   dim=1)
                rest_pattern_o   = torch.cat([rest_special_o,   rest_patch_o.repeat(1, num_patches_current, 1)],   dim=1)

                batch_size_total = x.shape[0]  # B*num_frames
                positions = torch.arange(batch_size_total, device=x.device)
                is_frame0 = (positions % num_frames == 0).view(batch_size_total, 1, 1)

                scale  = torch.where(is_frame0, frame1_pattern_s.expand(batch_size_total, -1, -1),
                                                rest_pattern_s.expand(batch_size_total, -1, -1))
                offset = torch.where(is_frame0, frame1_pattern_o.expand(batch_size_total, -1, -1),
                                                rest_pattern_o.expand(batch_size_total, -1, -1))
            else:
                raise RuntimeError(
                    f"LSQC: cannot determine attention type. "
                    f"per_frame_dim={per_frame_dim}, current_dim={current_dim}, "
                    f"num_frames={num_frames}, B={B}. "
                    f"Ensure _vggt_tokens_per_frame and _vggt_num_frames are set before forward."
                )
        else:
            scale  = base_scale
            offset = base_offset

        yq = _LSQC_quantizer(x, scale, offset, Qn, Qp)
        y = (yq - offset) * scale
        return y

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        if "init_scale" in Qparms and "scale" in Qparms:
            init_scale = Qparms["init_scale"]
            target_scale = Qparms["scale"]
            if init_scale.shape == target_scale.shape:
                target_scale.data.copy_(init_scale)
            else:
                # lazy_init: init_scale is [1, 2*per_frame_params, 1], must match target exactly
                target_scale.data.copy_(init_scale)

        if "init_offset" in Qparms and "offset" in Qparms:
            init_offset = Qparms["init_offset"]
            target_offset = Qparms["offset"]
            if init_offset.shape == target_offset.shape:
                target_offset.data.copy_(init_offset)
            else:
                target_offset.data.copy_(init_offset)


def _LSQC_quantizer(x, scale, offset, Qn, Qp):
    qn_t = float(Qn)
    qp_t = float(Qp)
    x = x / scale + offset
    x = torch.clamp(x, min=-qn_t, max=qp_t)
    y = (torch.round(x) - x).detach() + x  # STE
    return y
