import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# LSQH — LSQ with full Hadamard Rotation
# ============================================================
#
# Strategy: Full (d_in × d_in) Hadamard rotation on input channels
#
#   Activations have structured outliers in a few d_in channels. A random
#   orthogonal Hadamard matrix H spreads these outliers uniformly across ALL
#   channels before quantization, so a single per-tensor scale covers the
#   transformed distribution much more efficiently.
#
#   The same rotation is applied to both activations and weights so the
#   rotations cancel exactly in the matrix multiply output:
#
#     x_eff = x @ H          (d_in × d_in Hadamard applied to activations)
#     w_eff = w @ H          (same H applied to weight columns)
#
#     Q(x_eff) @ Q(w_eff)^T ≈ x @ w^T    (H @ H^T = I, rotations cancel)
#
# Hardware efficiency:
#   The Hadamard rotation is required at inference on every forward pass.
#   Cost: O(d_in log d_in) with the fast Hadamard transform, or O(d_in²)
#   with a dense matrix multiply. For VGGT d_in=1024, this is 1024×1024
#   matmul per layer — significant overhead compared to LSQS/LSQBH.
#
# Comparison with LSQBH:
#   LSQH:  full d_in rotation, better outlier dispersion, O(d_in²) cost
#   LSQBH: block rotation (G << d_in), partial dispersion, O(d_in log G) cost
#
# Initialization:
#   Standard NMSE_initializer. The Hadamard rotation spreads variance
#   uniformly, so NMSE on the rotated activations is appropriate.
#
# Implementation note:
#   Rotation is NOT performed inside this quantizer's forward(). Instead,
#   _forward_common detects LSQH_quantizer by class name and applies H to both
#   x and w (same block that handles LSQBH and LSQS — no flag needed on the module).
#   The rotated weight is cached at eval time since it doesn't change.
# ============================================================


class LSQH_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(LSQH_quantizer, self).__init__()
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
            # Single global scale; the Hadamard rotation makes the distribution
            # uniform enough that per-tensor quantization is sufficient.
            module.x_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            module.x_Qparms['scale'] = module.x_scale
            # Detection: _forward_common checks x_quantizer.__class__.__name__ == 'LSQH_quantizer'
            # to trigger full Hadamard rotation on both x and w. No flag needed.

        elif mode == "weight":
            num_bits = 4
            module.w_Qn = 2 ** (num_bits - 1)
            module.w_Qp = 2 ** (num_bits - 1) - 1
            module.w_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        # x has already been rotated by _forward_common before this call.
        # Standard LSQ quantization in rotated space.
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


# -------------------------------------------------------
# Standard LSQ quantizer function
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
