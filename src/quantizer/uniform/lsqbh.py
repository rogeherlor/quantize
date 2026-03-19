import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ptq.quarot.quarot_utils import random_hadamard_matrix

# ============================================================
# LSQBH — LSQ with Block Hadamard Rotation
# ============================================================
#
# Strategy: Block-diagonal Hadamard rotation on d_in
#
#   Full Hadamard rotation (lsqh) works by applying an (d_in × d_in) orthogonal
#   matrix that disperses outliers across all input channels. The cost is O(d_in²)
#   or O(d_in log d_in) with the fast Hadamard transform, but still requires a
#   full rotation at inference.
#
#   Block Hadamard partitions d_in into K = d_in/G independent blocks of size G
#   (G must be a power of 2, e.g. 64 or 128) and applies a separate Hadamard
#   matrix H_k to each block:
#
#     x_eff[..., k*G:(k+1)*G] = x[..., k*G:(k+1)*G] @ H_k
#     w_eff[row, k*G:(k+1)*G] = w[row, k*G:(k+1)*G] @ H_k
#
#     Q(x_eff) @ Q(w_eff)^T ≈ x @ w^T    (H_k @ H_k^T = I, rotations cancel)
#
#   For VGGT d_in=1024 with G=64: 16 independent blocks, each a 64×64 Hadamard.
#   Inference cost: O(d_in log G) = 1024 × 6 ≈ 6K ops vs O(d_in²) = 1M for full.
#
# Hardware efficiency:
#   The block rotation is required at inference but is much cheaper than the full
#   (d_in × d_in) rotation used in lsqh. Can be implemented as a bank of small
#   matrix-multiplies or with the fast Hadamard transform on each block.
#
# Initialization:
#   Standard NMSE_initializer. The Hadamard rotation spreads activation variance
#   uniformly, so no special initialization for the smooth factors is needed.
#
# Block rotation is applied in module_quantization._forward_common (not here),
# stored as module.H_blocks (K, G, G) lazily initialized on first forward.
# Both x and w receive the same rotation (same H_blocks shared from the module).
# ============================================================


class LSQBH_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, block_size=64, **kwargs):
        super(LSQBH_quantizer, self).__init__()
        self.mode = mode
        self.block_size = block_size  # G — must be a power of 2 and divide d_in
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
            # Single global scale — block rotation handles outlier dispersion
            module.x_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            module.x_Qparms['scale'] = module.x_scale

        elif mode == "weight":
            num_bits = 4
            module.w_Qn = 2 ** (num_bits - 1)
            module.w_Qp = 2 ** (num_bits - 1) - 1
            module.w_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        # Block rotation is applied in _forward_common before this is called,
        # so x is already in block-rotated space here.
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


def apply_block_hadamard(x, H_blocks, block_size):
    """Apply block Hadamard rotation to the last dimension of x.

    Args:
        x: tensor of shape (..., d_in)
        H_blocks: tensor of shape (K, G, G) — K independent Hadamard blocks
        block_size: G (must divide d_in)

    Returns:
        Rotated tensor of same shape as x.
    """
    *leading, d_in = x.shape
    K = d_in // block_size
    # Reshape to (..., K, G)
    x_blocks = x.reshape(*leading, K, block_size)
    # Batch matmul: (..., K, G) @ (K, G, G) -> (..., K, G)
    # H_blocks[k] is applied to x_blocks[..., k, :]
    x_rot = torch.einsum('...kg,kgj->...kj', x_blocks, H_blocks)
    return x_rot.reshape(*leading, d_in)


def apply_block_hadamard_weight(w, H_blocks, block_size):
    """Apply block Hadamard rotation to the d_in (column) dimension of a weight matrix.

    Args:
        w: tensor of shape (d_out, d_in)
        H_blocks: tensor of shape (K, G, G)
        block_size: G

    Returns:
        Rotated weight of same shape as w.
    """
    d_out, d_in = w.shape
    K = d_in // block_size
    # Reshape to (d_out, K, G)
    w_blocks = w.reshape(d_out, K, block_size)
    # (d_out, K, G) @ (K, G, G) -> (d_out, K, G)
    w_rot = torch.einsum('dkg,kgj->dkj', w_blocks, H_blocks)
    return w_rot.reshape(d_out, d_in)


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
