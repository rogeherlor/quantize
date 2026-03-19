import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# LSQA2 — LSQ with per-input-channel power-of-2 scale shifts
# ============================================================
#
# Strategy: Per-d_in channel quantization with hardware-friendly power-of-2 shifts
#
#   Activations have structured outliers in specific d_in channels.
#   LSQA2 assigns each input channel its own scale, parameterized as:
#
#     scale_i = scale_base * 2^(shift_i)
#
#   where:
#     scale_base: one global learnable scale (shared across all channels)
#     shift_i   : per-channel integer exponent (fixed after initialization,
#                 stored as float but rounded to integer during use)
#
#   This gives each channel a power-of-2 relative scale, which is hardware
#   friendly in the sense that 2^shift can be implemented as a bit-shift.
#   However, the d_in scale CANNOT be folded into weights the way LSQS can,
#   so it is still required at inference time.
#
# Hardware efficiency:
#   Moderate: scales are powers of 2 (bit-shifts, not multiplies), but
#   there is still one scale per d_in channel at inference — less efficient
#   than LSQS (zero overhead) or LSQBH (block rotation).
#   This quantizer is mainly useful for research/comparison purposes.
#
# Initialization:
#   Requires per-channel init_scale (shape d_in) to compute meaningful shifts.
#   With standard NMSE_initializer (scalar), shifts default to zero (→ standard LSQ).
#   For best results, provide a per-channel initializer that estimates max|x_i|.
#
# Comparison:
#   LSQ:   one global scale, no d_in adaptation
#   LSQA2: one base scale + fixed power-of-2 shifts per d_in channel
#   LSQS:  one base scale + fully learnable smooth factor per d_in channel (foldable)
# ============================================================


class LSQA2_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(LSQA2_quantizer, self).__init__()
        self.mode = mode
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

            if hasattr(module, 'in_features'):
                d_in = module.in_features
                # Learnable base scale (single scalar, receives gradients)
                module.x_scale_base = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
                # Per-channel power-of-2 shifts (fixed after init, no gradients)
                module.x_scale_shifts = nn.Parameter(
                    torch.zeros(d_in, dtype=torch.float32), requires_grad=False
                )
                module.x_Qparms['scale_base'] = module.x_scale_base
                module.x_Qparms['scale_shifts'] = module.x_scale_shifts
                module.x_Qparms['d_in'] = d_in
            else:
                # Fallback: single scale (no in_features, e.g. non-linear layers)
                module.x_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
                module.x_Qparms['scale'] = module.x_scale

        elif mode == "weight":
            num_bits = 4
            module.w_Qn = 2 ** (num_bits - 1)
            module.w_Qp = 2 ** (num_bits - 1) - 1
            module.w_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        if 'scale_base' in Qparms:
            return _LSQ_quantizer_per_channel(
                x, Qparms['scale_base'], Qparms['scale_shifts'],
                Qn, Qp, num_elements, grad_scale_mode
            )
        else:
            scale = Qparms['scale']
            yq = _LSQ_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode)
            return yq * scale

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        if 'scale_base' in Qparms:
            if 'init_scale' in Qparms:
                init_scales = Qparms['init_scale']
                if init_scales.numel() > 1:
                    base_scale, shifts = _compute_base_and_shifts(init_scales)
                    Qparms['scale_base'].data = torch.tensor(
                        [base_scale], device=Qparms['scale_base'].device
                    )
                    Qparms['scale_shifts'].data = shifts.to(Qparms['scale_shifts'].device)
                else:
                    # Scalar init → base scale only, zero shifts (equivalent to LSQ)
                    Qparms['scale_base'].data = torch.full(
                        Qparms['scale_base'].size(),
                        init_scales.item(),
                        device=Qparms['scale_base'].device,
                    )
                    Qparms['scale_shifts'].data.zero_()
        else:
            if 'init_scale' in Qparms:
                Qparms['scale'].data = torch.full(
                    Qparms['scale'].size(),
                    Qparms['init_scale'].clone().detach().item(),
                    device=Qparms['scale'].device,
                )


# -------------------------------------------------------
# Per-channel LSQ with power-of-2 shifts
# -------------------------------------------------------

def _LSQ_quantizer_per_channel(x, scale_base, scale_shifts, Qn, Qp, num_elements, grad_scale_mode):
    qn_t = float(Qn)
    qp_t = float(Qp)

    assert scale_base > 0, f'scale_base = {scale_base}, must be positive'

    shifts_rounded = torch.round(scale_shifts)
    scales = scale_base * torch.pow(2.0, shifts_rounded)  # (d_in,)
    scale_shape = [1] * (x.dim() - 1) + [scales.shape[0]]
    scales = scales.view(*scale_shape)

    if num_elements > 0:
        if grad_scale_mode == "10_fac":
            grad_scale = torch.tensor(10.0, device=x.device)
        elif grad_scale_mode == "LSQ_grad_scale":
            # Per-channel quantization: normalize by elements per channel, not total.
            # scale_shifts has shape (d_in,); dividing total by d_in gives the correct N.
            n_channels = scale_shifts.shape[0]
            num_elements_per_ch = max(1, num_elements // n_channels)
            grad_scale = 1.0 / torch.sqrt(num_elements_per_ch * qp_t)
        else:
            grad_scale = torch.tensor(1.0, device=x.device)

        bw_scale_base = scale_base * grad_scale
        scale_base_scaled = (scale_base - bw_scale_base).detach() + bw_scale_base
        shifts_rounded = torch.round(scale_shifts)
        scales = scale_base_scaled * torch.pow(2.0, shifts_rounded.detach())
        scales = scales.view(*scale_shape)

    x_scaled = x / scales
    x_clipped = torch.clamp(x_scaled, min=-qn_t, max=qp_t)
    y = (torch.round(x_clipped) - x_clipped).detach() + x_clipped
    return y * scales


def _compute_base_and_shifts(init_scales):
    base_scale = torch.median(init_scales).item()
    relative_scales = init_scales / base_scale
    log2_relative = torch.log2(torch.clamp(relative_scales, min=1e-8))
    shifts = torch.round(log2_relative)
    return base_scale, shifts


# -------------------------------------------------------
# Standard LSQ quantizer function (fallback)
# -------------------------------------------------------

def _LSQ_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode):
    qn_t = float(Qn)
    qp_t = float(Qp)

    if scale <= 0:
        scale_grad = scale.grad if scale.grad is not None else "No gradient"
        print(f"Scale assertion failed!")
        print(f"  Scale value: {scale.item() if scale.numel() == 1 else scale}")
        print(f"  Scale gradient: {scale_grad}")
        print(f"  Qn: {qn_t}, Qp: {qp_t}")
    assert scale > 0, 'scale = {}, {}, {}'.format(scale, qn_t, qp_t)

    if num_elements > 0:
        if grad_scale_mode == "10_fac":
            grad_scale = torch.tensor(10.0, device=x.device)
        elif grad_scale_mode == "LSQ_grad_scale":
            grad_scale = 1.0 / torch.sqrt(num_elements * qp_t)
        else:
            grad_scale = torch.tensor(1.0, device=x.device)

        bw_scale = scale * grad_scale
        scale = (scale - bw_scale).detach() + bw_scale

    x = x / scale
    x = torch.clamp(x, min=-qn_t, max=qp_t)
    y = (torch.round(x) - x).detach() + x
    return y
