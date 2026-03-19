import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use the VGGT context globals from lsqc
from src.quantizer.uniform.lsqc import (
    _vggt_num_frames,
    _vggt_tokens_per_frame,
    _vggt_batch_size,
    _LSQC_quantizer,
)
import src.quantizer.uniform.lsqc as _lsqc_module

# ============================================================
# LSQSC — LSQ with SmoothQuant migration + LSQC token-type scales
# ============================================================
#
# Strategy: d_in smooth migration  +  per-token-type scale/offset
#
#   This is a combination of LSQS and LSQC, addressing two orthogonal sources
#   of quantization difficulty:
#
#   1. d_in outliers (LSQS component):
#        x_smooth = x / s              (s ∈ R^{d_in}, suppresses outlier channels)
#        w_eff    = w * s              (weight compensation, applied in _forward_common)
#        At deployment: fold s into weights → zero d_in runtime cost
#
#   2. Inter-token variation (LSQC component):
#        Separate learnable scale + offset per token type (camera, register, patch).
#        VGGT has 5 special tokens + shared patch scale, two frame types → 12 params.
#        Token scales/offsets are required at inference (token dimension is efficient).
#
#   Combined forward:
#        x_smooth = x / exp(smooth_log_scale)           d_in migration
#        q  = clamp(round(x_smooth / s_tok + z_tok))    per-token-type LSQ+
#        y  = (q - z_tok) * s_tok
#
# Hardware efficiency:
#   Training  : elementwise divide on activations, multiply on weights
#   Inference : fold smooth into weights (free); token scales are efficient (O(tokens))
#
# Note: Given LSQC alone has not outperformed plain LSQ in experiments, LSQSC is
#       included for completeness and to test whether the combination helps.
#       Recommended to first validate LSQS alone before running LSQSC.
#
# Weight migration is applied in module_quantization._forward_common, same hook
# as LSQS_quantizer (the name check covers both 'LSQS_quantizer' and 'LSQSC_quantizer').
# ============================================================


class LSQSC_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, scale_shape=None, num_special_tokens=5, **kwargs):
        super(LSQSC_quantizer, self).__init__()
        self.mode = mode
        self.scale_shape = scale_shape
        self.lazy_init = (scale_shape is None and mode == "activation")
        self.base_dim = None
        self.num_special_tokens = num_special_tokens

        if isinstance(module, nn.Conv2d):
            self.dim_reduction = 1
        else:
            self.dim_reduction = -1

        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            num_bits = 4
            if module.first_layer:
                module.x_Qn = 2 ** (num_bits - 1)
                module.x_Qp = 2 ** (num_bits - 1) - 1
            else:
                module.x_Qn = 2 ** (num_bits - 1)
                module.x_Qp = 2 ** (num_bits - 1) - 1

            d_in = module.in_features
            # Per-input-channel smooth factor (log space, same as LSQS)
            module.smooth_log_scale = nn.Parameter(torch.zeros(d_in, dtype=torch.float32))
            module.x_Qparms['smooth_log_scale'] = module.smooth_log_scale

            # LSQC-style token scale + offset (lazy init for activations)
            if not self.lazy_init:
                self.scale_shape = self.scale_shape if self.scale_shape is not None else (1,)
                module.x_scale = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
                module.x_offset = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
                module.x_Qparms['scale'] = module.x_scale
                module.x_Qparms['offset'] = module.x_offset
            # else: scale and offset will be created in the first forward pass
            # (handled by module_quantization._forward_common lazy init block)

        elif mode == "weight":
            num_bits = 4
            module.w_Qn = 2 ** (num_bits - 1)
            module.w_Qp = 2 ** (num_bits - 1) - 1
            self.scale_shape = self.scale_shape if self.scale_shape is not None else (1,)
            module.w_scale = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
            module.w_offset = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale
            module.w_Qparms['offset'] = module.w_offset

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        # --- Step 1: apply d_in smooth migration ---
        if self.mode == "activation" and 'smooth_log_scale' in Qparms:
            smooth = torch.exp(Qparms['smooth_log_scale'])  # (d_in,)
            x = x / smooth  # broadcast: (..., d_in) / (d_in,)

        # --- Step 2: LSQC-style per-token-type quantization ---
        base_scale = Qparms['scale']
        base_offset = Qparms['offset']

        # Gradient scaling (same as lsqc.py)
        grad_scale_mode_inner = "LSQ_grad_scale"
        if num_elements > 0 and self.training:
            Qp_val = float(Qp)
            gs = 1.0 / torch.sqrt(torch.tensor(num_elements * Qp_val, device=base_scale.device))
            gs_t = torch.tensor(gs, device=base_scale.device, dtype=base_scale.dtype)
            bw_s = base_scale * gs_t
            base_scale = (base_scale - bw_s).detach() + bw_s
            bw_o = base_offset * gs_t
            base_offset = (base_offset - bw_o).detach() + bw_o

        # Token expansion (identical logic to lsqc.py)
        if self.lazy_init and self.num_special_tokens > 0 and len(x.shape) >= 3:
            scale, offset = self._expand_token_params(x, base_scale, base_offset)
        else:
            scale = base_scale
            offset = base_offset

        yq = _LSQC_quantizer(x, scale, offset, Qn, Qp)
        y = (yq - offset) * scale
        return y

    def _expand_token_params(self, x, base_scale, base_offset):
        """Expand compact 12-parameter token scales to match current input shape.

        Identical logic to LSQC_quantizer.forward token expansion.
        """
        current_dim = x.shape[-2]
        per_frame_params = self.num_special_tokens + 1  # 6

        def _split_frame(params):
            if params.shape[1] == 2 * per_frame_params:
                f1 = params[:, :per_frame_params, :]
                rest = params[:, per_frame_params:, :]
            elif params.shape[1] >= per_frame_params:
                f1 = params[:, :per_frame_params, :]
                rest = f1.clone()
            else:
                f1 = params
                rest = params
            return f1, rest

        f1_s, rest_s = _split_frame(base_scale)
        f1_o, rest_o = _split_frame(base_offset)

        f1_sp_s = f1_s[:, :self.num_special_tokens, :]
        f1_pt_s = f1_s[:, self.num_special_tokens:, :]
        rest_sp_s = rest_s[:, :self.num_special_tokens, :]
        rest_pt_s = rest_s[:, self.num_special_tokens:, :]

        f1_sp_o = f1_o[:, :self.num_special_tokens, :]
        f1_pt_o = f1_o[:, self.num_special_tokens:, :]
        rest_sp_o = rest_o[:, :self.num_special_tokens, :]
        rest_pt_o = rest_o[:, self.num_special_tokens:, :]

        # Read VGGT context from lsqc module's globals
        per_frame_dim = _lsqc_module._vggt_tokens_per_frame
        num_frames = _lsqc_module._vggt_num_frames
        B = _lsqc_module._vggt_batch_size

        if per_frame_dim is not None and current_dim > per_frame_dim:
            # Global attention: current_dim = S × P
            n_patches = per_frame_dim - self.num_special_tokens
            f1_pat_s = torch.cat([f1_sp_s, f1_pt_s.repeat(1, n_patches, 1)], dim=1)
            f1_pat_o = torch.cat([f1_sp_o, f1_pt_o.repeat(1, n_patches, 1)], dim=1)
            rest_pat_s = torch.cat([rest_sp_s, rest_pt_s.repeat(1, n_patches, 1)], dim=1)
            rest_pat_o = torch.cat([rest_sp_o, rest_pt_o.repeat(1, n_patches, 1)], dim=1)

            n_frames_calc = current_dim // per_frame_dim
            if n_frames_calc > 1:
                scale = torch.cat([f1_pat_s, rest_pat_s.repeat(1, n_frames_calc - 1, 1)], dim=1)
                offset = torch.cat([f1_pat_o, rest_pat_o.repeat(1, n_frames_calc - 1, 1)], dim=1)
            else:
                scale = f1_pat_s
                offset = f1_pat_o

        elif per_frame_dim is not None and current_dim == per_frame_dim and num_frames is not None and B is not None:
            # Frame attention: (B*S, P, C)
            n_patches_cur = current_dim - self.num_special_tokens
            f1_pat_s = torch.cat([f1_sp_s, f1_pt_s.repeat(1, n_patches_cur, 1)], dim=1)
            f1_pat_o = torch.cat([f1_sp_o, f1_pt_o.repeat(1, n_patches_cur, 1)], dim=1)
            rest_pat_s = torch.cat([rest_sp_s, rest_pt_s.repeat(1, n_patches_cur, 1)], dim=1)
            rest_pat_o = torch.cat([rest_sp_o, rest_pt_o.repeat(1, n_patches_cur, 1)], dim=1)

            batch_total = x.shape[0]
            scale = rest_pat_s.repeat(batch_total, 1, 1)
            offset = rest_pat_o.repeat(batch_total, 1, 1)
            for batch_idx in range(B):
                frame0_pos = batch_idx * num_frames
                scale[frame0_pos:frame0_pos + 1, :, :] = f1_pat_s
                offset[frame0_pos:frame0_pos + 1, :, :] = f1_pat_o
        else:
            # Fallback
            n_patches_cur = current_dim - self.num_special_tokens
            scale = torch.cat([f1_sp_s, f1_pt_s.repeat(1, n_patches_cur, 1)], dim=1)
            offset = torch.cat([f1_sp_o, f1_pt_o.repeat(1, n_patches_cur, 1)], dim=1)

        return scale, offset

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        # Initialize token scale (same logic as LSQC_quantizer.scale_to_Qparms)
        if 'init_scale' in Qparms and 'scale' in Qparms:
            init_scale = Qparms['init_scale']
            target_scale = Qparms['scale']
            if init_scale.numel() == 1:
                target_scale.data.fill_(init_scale.item())
            elif init_scale.shape == target_scale.shape:
                target_scale.data.copy_(init_scale)
            else:
                per_frame_params = self.num_special_tokens + 1
                if self.lazy_init and self.num_special_tokens > 0:
                    if init_scale.shape[1] == 2 * per_frame_params:
                        target_scale.data.copy_(init_scale)
                    elif init_scale.shape[1] >= per_frame_params:
                        target_scale.data[:, :per_frame_params, :] = init_scale[:, :per_frame_params, :]
                        target_scale.data[:, per_frame_params:, :] = init_scale[:, :per_frame_params, :]
                    else:
                        target_scale.data.fill_(init_scale.mean().item())
                else:
                    try:
                        target_scale.data.copy_(init_scale.expand_as(target_scale))
                    except Exception:
                        target_scale.data.fill_(init_scale.mean().item())

        if 'init_offset' in Qparms and 'offset' in Qparms:
            init_offset = Qparms['init_offset']
            target_offset = Qparms['offset']
            if init_offset.numel() == 1:
                target_offset.data.fill_(init_offset.item())
            elif init_offset.shape == target_offset.shape:
                target_offset.data.copy_(init_offset)
            else:
                per_frame_params = self.num_special_tokens + 1
                if self.lazy_init and self.num_special_tokens > 0:
                    if init_offset.shape[1] == 2 * per_frame_params:
                        target_offset.data.copy_(init_offset)
                    elif init_offset.shape[1] >= per_frame_params:
                        target_offset.data[:, :per_frame_params, :] = init_offset[:, :per_frame_params, :]
                        target_offset.data[:, per_frame_params:, :] = init_offset[:, :per_frame_params, :]
                    else:
                        target_offset.data.fill_(init_offset.mean().item())
                else:
                    try:
                        target_offset.data.copy_(init_offset.expand_as(target_offset))
                    except Exception:
                        target_offset.data.fill_(init_offset.mean().item())
        # smooth_log_scale is initialized directly by LSQS_initializer
