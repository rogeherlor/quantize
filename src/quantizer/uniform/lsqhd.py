import torch
import torch.nn as nn

from .lsqh import LSQH_quantizer

# ============================================================
# LSQHD — LSQ with random Diagonal + Hadamard (QuIP#-style)
# ============================================================
#
# Strategy: Apply a random diagonal sign matrix D = diag(d) where
# d ∈ {-1, +1}^d_in before the Hadamard rotation.
#
#   x_eff = (x * d) @ H
#   w_eff = (w * d) @ H
#
# Rotations cancel exactly in the matmul output:
#   Q(x_eff) @ Q(w_eff)^T ≈ x @ diag(d²) @ w^T = x @ w^T
# because d² = 1 elementwise.
#
# Why this helps (QuIP# insight):
#   The Hadamard H alone disperses structured channel outliers, but the
#   distribution still has some residual non-uniformity from the fixed
#   H structure. Premultiplying by a random diagonal D makes D·H a
#   uniformly random signed Hadamard, which achieves the maximum possible
#   incoherence bound — strictly tighter than H alone.
#
# Overhead vs LSQH:
#   One elementwise multiply on x (d_in elements) per forward pass.
#   Essentially free compared to the O(d²) Hadamard matmul.
#
# Implementation note:
#   d_vec is stored as a buffer on the module (not a Parameter — it is
#   fixed at initialization). _forward_common detects 'LSQHD_quantizer'
#   and applies d_vec *before* the Hadamard block.
#   The quantizer forward() itself is identical to LSQH (standard LSQ
#   in the rotated+sign-flipped space).
# ============================================================


class LSQHD_quantizer(LSQH_quantizer):
    """
    Hadamard + random diagonal sign flip (QuIP#-style incoherence).

    Inherits all quantization logic from LSQH_quantizer.
    The only difference is that _forward_common applies d_vec before H.
    """
    # Class name 'LSQHD_quantizer' is detected by _forward_common to
    # trigger both diagonal and Hadamard preprocessing.
    pass
