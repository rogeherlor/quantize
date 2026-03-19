import torch
import torch.nn as nn

# ============================================================
# LSQHB — LSQ with Hadamard + Binary (1-bit with learnable scale)
# ============================================================
#
# Strategy: After Hadamard rotation (applied by _forward_common),
# binarize activations to {-1, +1} and recover with a learnable scale.
#
# Binary scale relationship:
#   x @ w  ≈  (alpha_x * alpha_w) * sign(x_rot) @ sign(w_rot)
#
# sign(x_rot) @ sign(w_rot) is an inner product of binary vectors,
# computable via XNOR + popcount — no floating-point multiplies.
# For d=1024 and N tokens: N² dot products → N² × 16 popcount ops.
#
# Inspiration: BRIEF image descriptors, XNOR-Net, BitNet b1.58
#
# For attention (Q@K^T):
#   Q_b = sign(Q @ H),  K_b = sign(K @ H)
#   attn ≈ (alpha_Q * alpha_K) * hammingInnerProduct(Q_b, K_b)
#
# Initialization:
#   Uses LSQHB_initializer: alpha = mean(|x_rot|), which is the
#   optimal MMSE scale for 1-bit quantization of a zero-mean distribution.
#
# Implementation note:
#   Rotation is applied by _forward_common (same as LSQH) because
#   LSQHB_quantizer is in _FULL_HADAMARD_QUANTIZERS. This quantizer
#   then binarizes the already-rotated data.
# ============================================================


class LSQHB_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super().__init__()
        self.mode = mode
        self._setup(module, mode)

    def _setup(self, module, mode):
        if mode == "activation":
            # Qn/Qp = 1 signals binary; kept for API compatibility
            module.x_Qn = 1
            module.x_Qp = 1
            module.x_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            module.x_Qparms['scale'] = module.x_scale

        elif mode == "weight":
            module.w_Qn = 1
            module.w_Qp = 1
            module.w_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        alpha = Qparms['scale'].abs()   # ensure positive
        x_b = _binary_ste(x, alpha)
        return alpha * x_b

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        if 'init_scale' in Qparms and 'scale' in Qparms:
            Qparms['scale'].data = torch.full(
                Qparms['scale'].size(),
                Qparms['init_scale'].clone().detach().item(),
                device=Qparms['scale'].device,
            )


def _binary_ste(x, alpha):
    """
    Binarization with straight-through estimator.

    Forward:  x_b = sign(x) ∈ {-1, +1}  (0 maps to +1)
    Backward: gradient passes through for |x| <= alpha, zero outside.
              (Clipping STE — same policy as XNOR-Net)
    """
    # Clamp to [-alpha, +alpha] for STE window
    x_c = x.clamp(-alpha.detach(), alpha.detach())
    # {-1, +1} with 0 → +1
    x_sign = (x_c >= 0).to(x.dtype) * 2.0 - 1.0
    # Forward = x_sign, backward = x_c gradient
    return (x_sign - x_c).detach() + x_c


def LSQHB_initializer(x, Qparms, Qn, Qp, quantizer, mode):
    """
    Initialize binary quantizer scale as mean(|x|).

    For a zero-mean Gaussian N(0, σ²), E[|x|] = σ√(2/π) ≈ 0.798σ,
    which is the optimal MMSE scale for 1-bit quantization.
    Using the sample mean of absolute values is more robust than
    estimating σ, especially after Hadamard rotation.
    """
    with torch.no_grad():
        alpha = x.detach().abs().mean().item()
        alpha = max(alpha, 1e-6)
    Qparms['init_scale'] = torch.tensor(alpha, dtype=torch.float32)
