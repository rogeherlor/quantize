import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# LSQS — LSQ with SmoothQuant-style Migration
# ============================================================
#
# Strategy: d_in outlier migration
#
#   Activations have structured outliers in specific input channels (d_in dimension).
#   A single per-tensor scale cannot suppress these outliers without wasting bits
#   on the remaining channels.
#
#   Solution: learn a per-input-channel scale factor s ∈ R^{d_in} that migrates
#   quantization difficulty from activations to weights:
#
#     x_eff = x / s        suppress outlier channels in activation
#     w_eff = w * s        compensate in weight (per-column multiply)
#
#     Q(x_eff) @ Q(w_eff)^T ≈ x @ w^T    (mathematically equivalent)
#
#   s is stored as log(s) for unconstrained gradient flow. At deployment, s is
#   folded into the weight matrix (W_deploy[:, i] = W_quant[:, i] * s[i]), so
#   there is ZERO d_in overhead at inference — same cost as standard per-tensor W4.
#
# Hardware efficiency:
#   Training  : elementwise divide on activations, elementwise multiply on weights
#   Inference : fold s into weights statically → standard GEMM, no d_in scale
#
# Initialization:
#   Uses LSQS_initializer (SmoothQuant formula):
#     s_i = max(|x_i|)^alpha / max(|w_i|)^(1-alpha)    alpha=0.5
#
# Weight migration is applied in module_quantization._forward_common (not here),
# in the same location as the existing Hadamard rotation block.
# ============================================================


class LSQS_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(LSQS_quantizer, self).__init__()
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

            d_in = module.in_features
            # One global LSQ step-size (learned, same as standard LSQ)
            module.x_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            # Per-input-channel smooth factor stored in log space for positivity
            module.smooth_log_scale = nn.Parameter(torch.zeros(d_in, dtype=torch.float32))
            module.x_Qparms['scale'] = module.x_scale
            module.x_Qparms['smooth_log_scale'] = module.smooth_log_scale

        elif mode == "weight":
            num_bits = 4
            module.w_Qn = 2 ** (num_bits - 1)
            module.w_Qp = 2 ** (num_bits - 1) - 1
            # Standard single-scale weight quantization (same as lsq.py)
            module.w_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        scale = Qparms['scale']

        if self.mode == "activation" and 'smooth_log_scale' in Qparms:
            smooth = torch.exp(Qparms['smooth_log_scale'])  # (d_in,), always positive
            # Divide activation channels by smooth factors to suppress outliers.
            # Weight compensation (w * smooth) is applied in _forward_common.
            x = x / smooth  # broadcast: (..., d_in) / (d_in,)

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


# -------------------------------------------------------
# Standard LSQ quantizer function (shared with lsq.py)
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
