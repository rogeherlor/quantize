import torch
import torch.nn as nn

# ============================================================
# LSQHS — LSQ with SmoothQuant migration + Hadamard Rotation
# ============================================================
#
# Strategy: Per-channel smooth migration BEFORE full Hadamard rotation
#
#   Activations have structured outliers in specific d_in channels.
#   Step 1 — suppress them with per-channel smooth factors s ∈ R^{d_in}:
#
#     x_eff = x / s        (activation channels suppressed)
#     w_eff = w * s        (weight columns compensated)
#
#   Step 2 — apply Hadamard rotation to the now-more-uniform distribution:
#
#     x_rot = x_eff @ H   =  (x / s) @ H
#     w_rot = w_eff @ H   =  (w * s) @ H
#
#   Rotations cancel in the matmul output:
#
#     Q(x_rot) @ Q(w_rot)^T  ≈  (x/s)·H·H^T·(ws)^T  =  x·w^T  ✓
#
# Why smooth BEFORE Hadamard (not after):
#   Hadamard spreads all channels uniformly. If outlier channels enter the
#   rotation unattenuated, they raise the variance of every output channel,
#   reducing the effective number of bits across the board. Smooth first →
#   fewer outlier bits enter the rotation → Hadamard spreads a tighter
#   distribution → better utilisation of the quantization range.
#
# Parameter summary:
#   x_scale          : scalar LSQ step-size (learned)
#   smooth_log_scale : per-d_in log smooth factors (learned) ∈ R^{d_in}
#
# Implementation note:
#   smooth (x/s and w*s) is applied in _forward_common BEFORE the Hadamard
#   block, so this quantizer's forward() is identical to LSQH — standard LSQ
#   on the already-smoothed-and-rotated input. No smooth division here.
#
#   _forward_common detects 'LSQHS_quantizer' via:
#     _FULL_HADAMARD_QUANTIZERS  — triggers H init + rotation
#     _PRE_HADAMARD_SMOOTH_QUANTIZERS — triggers x/s and w*s before rotation
#
# Initialization:
#   Uses LSQS_initializer (SmoothQuant formula) on the pre-Hadamard x:
#     s_i = max(|x_i|)^alpha / max(|w_i|)^(1-alpha)    alpha=0.5
#   scale initialised from NMSE on smoothed activations.
# ============================================================

from .lsqh import _LSQ_quantizer


class LSQHS_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(LSQHS_quantizer, self).__init__()
        self.mode = mode
        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            num_bits = 4
            module.x_Qn = 2 ** (num_bits - 1)
            module.x_Qp = 2 ** (num_bits - 1) - 1

            d_in = module.in_features
            # Global LSQ step-size (same as LSQH)
            module.x_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            # Per-input-channel smooth factors in log space (same as LSQS)
            module.smooth_log_scale = nn.Parameter(torch.zeros(d_in, dtype=torch.float32))
            module.x_Qparms['scale'] = module.x_scale
            module.x_Qparms['smooth_log_scale'] = module.smooth_log_scale

        elif mode == "weight":
            num_bits = 4
            module.w_Qn = 2 ** (num_bits - 1)
            module.w_Qp = 2 ** (num_bits - 1) - 1
            module.w_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        # x has already been smoothed (x/s) and rotated (@ H) by _forward_common.
        # Standard LSQ quantization in the transformed space.
        scale = Qparms['scale']
        yq = _LSQ_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode)
        return yq * scale

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        if 'init_scale' in Qparms and 'scale' in Qparms:
            Qparms['scale'].data = torch.full(
                Qparms['scale'].size(),
                Qparms['init_scale'].clone().detach().item(),
                device=Qparms['scale'].device,
            )
        # smooth_log_scale is initialized directly by LSQS_initializer;
        # no action needed here.
