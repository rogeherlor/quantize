import math
import torch
import torch.nn as nn

# ============================================================
# LSQT — LSQ with TurboQuant Codebook Quantization
# ============================================================
#
# Strategy: Full Hadamard rotation (reused from LSQH infrastructure)
#           + non-uniform optimal scalar quantization via precomputed
#           Lloyd-Max codebook centroids for N(0,1).
#
# Background (TurboQuant, arXiv 2504.19874):
#   After a random orthogonal rotation (Hadamard), each coordinate of
#   a high-dimensional vector is approximately iid ~ N(0, 1/d).
#   Instead of mapping to a uniform integer grid (standard LSQ),
#   TurboQuant uses the optimal Lloyd-Max quantizer for the Gaussian
#   distribution.  This achieves near-optimal MSE distortion rate,
#   within a small constant factor of the information-theoretic lower
#   bound (Shannon distortion-rate function).
#
# Key difference from LSQH:
#   LSQH:  x/scale -> clamp -> round        (uniform grid)
#   LSQT:  x/scale -> nearest centroid       (optimal non-uniform grid)
#
# Scale parameterization:
#   Scale is stored as log(scale) to guarantee positivity without
#   constraining gradients.  scale = exp(raw_param) is always > 0,
#   and the gradient of exp() is always positive (no sign inversion),
#   so the optimizer can always drive the scale toward a good value.
#
# Initializer:
#   Use LSQT_initializer (defined below) which sets init_scale = std(x).
#   This ensures x/scale ~ N(0,1) matching the codebook's design
#   distribution.  NMSE_initializer's delta factors are calibrated for
#   uniform grids and will under-normalize (init_scale too small),
#   causing saturation at the codebook boundaries.
#
# Rotation is NOT performed inside this quantizer's forward(). Instead,
# _forward_common detects LSQT_quantizer via _FULL_HADAMARD_QUANTIZERS
# and applies H to both x and w before calling this quantizer.
# ============================================================

# Lloyd-Max optimal centroids for the standard normal distribution N(0,1).
# Precomputed via iterative Lloyd-Max (Max-Lloyd) algorithm.
# Symmetric: centroids are symmetric about zero.
LLOYD_MAX_CODEBOOKS = {
    1: [-0.7979, 0.7979],
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [-2.1520, -1.3439, -0.7560, -0.2451,
         0.2451,  0.7560,  1.3439,  2.1520],
    4: [-2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
         0.1284,  0.3881,  0.6568,  0.9424,  1.2562,  1.6180,  2.0690,  2.7326],
}


class LSQT_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(LSQT_quantizer, self).__init__()
        self.mode = mode
        self.num_bits = num_bits

        codebook = LLOYD_MAX_CODEBOOKS.get(num_bits)
        if codebook is None:
            raise ValueError(
                f"LSQT_quantizer: no Lloyd-Max codebook for num_bits={num_bits}. "
                f"Supported: {sorted(LLOYD_MAX_CODEBOOKS.keys())}"
            )
        self.register_buffer('codebook', torch.tensor(codebook, dtype=torch.float32))

        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            num_bits = self.num_bits
            module.x_Qn = 2 ** (num_bits - 1)
            module.x_Qp = 2 ** (num_bits - 1) - 1
            # Store log(scale) so that exp() in forward always yields a positive scale.
            # Initialized to log(1.0) = 0.0; overwritten by scale_to_Qparms after calibration.
            module.x_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            module.x_Qparms['scale'] = module.x_scale

        elif mode == "weight":
            num_bits = self.num_bits
            module.w_Qn = 2 ** (num_bits - 1)
            module.w_Qp = 2 ** (num_bits - 1) - 1
            module.w_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        # Recover positive scale via exp().  The stored parameter is log(scale),
        # so exp() is always > 0 and the gradient (exp itself) is always positive —
        # no sign inversion, no clamp needed, no assertion required.
        log_scale = Qparms['scale']
        scale = torch.exp(log_scale)

        # Gradient scaling on scale (identical logic to LSQ/LSQH, applied to
        # the exp()-transformed value so the STE operates in scale-space).
        qp_t = float(Qp)
        if num_elements > 0:
            if grad_scale_mode == "10_fac":
                grad_scale = torch.tensor(10.0, device=x.device)
            elif grad_scale_mode == "LSQ_grad_scale":
                grad_scale = 1.0 / torch.sqrt(num_elements * qp_t)
            else:
                grad_scale = torch.tensor(1.0, device=x.device)

            bw_scale = scale * grad_scale
            scale = (scale - bw_scale).detach() + bw_scale

        # Normalize
        x_norm = x / scale

        # Nearest centroid lookup via binary search on the sorted codebook.
        # searchsorted avoids materializing the (..., K) distance tensor.
        cb = self.codebook.to(x_norm.dtype)
        flat = x_norm.contiguous().reshape(-1)
        pos = torch.searchsorted(cb, flat)          # index into cb, in [0, K]
        pos_hi = pos.clamp(0, len(cb) - 1)
        pos_lo = (pos - 1).clamp(0)
        dist_lo = (flat - cb[pos_lo]).abs()
        dist_hi = (flat - cb[pos_hi]).abs()
        idx = torch.where(dist_lo <= dist_hi, pos_lo, pos_hi)
        x_quant = cb[idx].reshape(x_norm.shape)

        # STE: gradient flows through x_norm as if quantization were identity
        y = (x_quant - x_norm).detach() + x_norm

        return y * scale

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        # init_scale holds the actual desired scale (e.g. std(x) from LSQT_initializer).
        # Store its log so the parameter lives in log-space.
        if 'init_scale' in Qparms and 'scale' in Qparms:
            init_val = Qparms['init_scale'].clone().detach().item()
            init_val = max(init_val, 1e-6)  # guard against zero std
            Qparms['scale'].data.fill_(math.log(init_val))


# ============================================================
# LSQT_initializer
# ============================================================
# Sets init_scale = std(x) so that x/scale ~ N(0,1) at initialization,
# matching the distribution that the Lloyd-Max codebook was designed for.
# Unlike NMSE_initializer whose delta factors calibrate uniform step sizes,
# the codebook normalizes the full distribution, so delta = 1.0 is correct.
# ============================================================

def LSQT_initializer(x, Qparms, Qn, Qp, quantizer, mode):
    x_std = x.detach().float().std().clamp(min=1e-6)
    # Always overwrite: std(x) is the natural scale for codebook N(0,1) lookup.
    Qparms["init_scale"] = x_std
