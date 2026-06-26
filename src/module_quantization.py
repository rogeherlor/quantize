
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.quantizer.uniform import *
from src.quantizer.nonuniform import *
from src.initializer import *

from torch.utils.checkpoint import checkpoint

from src.ptq.quarot.quarot_utils import random_hadamard_matrix

# Quantizer classes that require full (d_in × d_in) Hadamard rotation on both x and w.
# Rotation is applied in _forward_common; these quantizers operate in rotated space.
_FULL_HADAMARD_QUANTIZERS = frozenset({'LSQH_quantizer', 'LSQHB_quantizer', 'LSQHD_quantizer', 'LSQHS_quantizer', 'LSQT_quantizer'})

# Quantizer classes that additionally apply a random diagonal sign flip D = diag(d ∈ {±1}^d_in)
# *before* the Hadamard: x_eff = (x * d) @ H.  D·H is strictly more incoherent than H alone.
_DIAGONAL_HADAMARD_QUANTIZERS = frozenset({'LSQHD_quantizer'})

# Quantizer classes that require SmoothQuant-style weight column migration (w * smooth).
# x / smooth is handled inside the quantizer forward; w * smooth is applied here.
_SMOOTH_QUANTIZERS = frozenset({'LSQS_quantizer', 'LSQSC_quantizer'})

# Quantizer classes that apply smooth (x/s, w*s) BEFORE the Hadamard rotation.
# Both x and w migration are handled in _forward_common, not inside the quantizer forward.
_PRE_HADAMARD_SMOOTH_QUANTIZERS = frozenset({'LSQHS_quantizer'})

#----------------------------------------------------------
# Fully connected layer
#----------------------------------------------------------
class QLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True,
                 num_bits=None, w_grad_scale_mode="LSQ_grad_scale",
                 x_grad_scale_mode="LSQ_grad_scale",
                 weight_norm=False, w_quantizer=None, x_quantizer=None,
                 w_initializer=None, x_initializer=None, first_layer=False):

        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.num_bits = num_bits
        self.x_grad_scale_mode = x_grad_scale_mode
        self.w_grad_scale_mode = w_grad_scale_mode
        self.weight_norm = weight_norm
        self.first_layer = first_layer
        self.x_Qparms = {}
        self.w_Qparms = {}

        # Full Hadamard rotation matrix — lazily initialized on first forward.
        # Used when x_quantizer is LSQH_quantizer / LSQHB_quantizer / LSQHD_quantizer.
        self.H = None
        self.rotated_weight = None  # eval-time cache for rotated weight

        # Random diagonal sign vector for LSQHD (QuIP#-style).
        # d_vec ∈ {-1, +1}^d_in; applied elementwise before Hadamard rotation.
        self.d_vec = None

        # Block Hadamard rotation matrices — lazily initialized on first forward.
        # Used when x_quantizer is LSQBH_quantizer.
        self.H_blocks = None

        if w_quantizer is not None and w_quantizer.__name__ in ("LSQC_quantizer", "LSQSC_quantizer"):
            self.w_scale_shape = (out_features, 1)
            self.w_quantizer = w_quantizer(self, num_bits, "weight",
                                           scale_shape=self.w_scale_shape)
        elif w_quantizer is not None:
            self.w_scale_shape = (1,)
            self.w_quantizer = w_quantizer(self, num_bits, "weight")
        else:
            self.w_quantizer = None

        if x_quantizer is not None and x_quantizer.__name__ in ("LSQC_quantizer", "LSQSC_quantizer"):
            self.x_scale_shape = None  # lazy init
            self.x_quantizer = x_quantizer(self, num_bits, "activation",
                                           scale_shape=None, num_special_tokens=5)
        elif x_quantizer is not None:
            self.x_scale_shape = (1,)
            self.x_quantizer = x_quantizer(self, num_bits, "activation")
        else:
            self.x_quantizer = None

        self.w_initializer = w_initializer
        self.x_initializer = x_initializer
        self.register_buffer('init_state', torch.tensor(False))

    def forward(self, input):
        # Lazily initialize full Hadamard matrix when LSQH/LSQHB/LSQHD is the activation quantizer.
        if self.x_quantizer is not None and \
                self.x_quantizer.__class__.__name__ in _FULL_HADAMARD_QUANTIZERS and \
                self.H is None:
            self.H = random_hadamard_matrix(self.in_features, device=input.device).to(input.dtype)

        # Lazily initialize diagonal sign vector for LSQHD (QuIP#-style).
        if self.x_quantizer is not None and \
                self.x_quantizer.__class__.__name__ in _DIAGONAL_HADAMARD_QUANTIZERS and \
                self.d_vec is None:
            d = torch.randint(0, 2, (self.in_features,), device=input.device).float() * 2 - 1
            self.d_vec = d.to(input.dtype)

        # Invalidate rotated weight cache every training step.
        if self.training:
            self.rotated_weight = None

        if self.training:
            return checkpoint(_forward_common, self, input, use_reentrant=False, preserve_rng_state=False)
        else:
            return _forward_common(self, input)


#----------------------------------------------------------
# Convolutional layer
#----------------------------------------------------------
class QConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_bits=None,
                 w_grad_scale_mode="LSQ_grad_scale",
                 x_grad_scale_mode="LSQ_grad_scale",
                 weight_norm=None, w_quantizer=None, x_quantizer=None,
                 w_initializer=None, x_initializer=None, first_layer=False):

        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)

        self.num_bits = num_bits
        self.x_grad_scale_mode = x_grad_scale_mode
        self.w_grad_scale_mode = w_grad_scale_mode
        self.weight_norm = weight_norm
        self.first_layer = first_layer
        self.x_Qparms = {}
        self.w_Qparms = {}

        self.H = None
        self.rotated_weight = None
        self.d_vec = None
        self.H_blocks = None

        if w_quantizer is not None and w_quantizer.__name__ == "LSQC_quantizer":
            self.w_scale_shape = (out_channels, 1, 1, 1)
            self.w_quantizer = w_quantizer(self, num_bits, "weight",
                                           scale_shape=self.w_scale_shape)
        elif w_quantizer is not None:
            self.w_scale_shape = (1,)
            self.w_quantizer = w_quantizer(self, num_bits, "weight")
        else:
            self.w_quantizer = None

        if x_quantizer is not None and x_quantizer.__name__ == "LSQC_quantizer":
            self.x_scale_shape = None
            self.x_quantizer = x_quantizer(self, num_bits, "activation",
                                           scale_shape=None, num_special_tokens=0)
        elif x_quantizer is not None:
            self.x_scale_shape = (1,)
            self.x_quantizer = x_quantizer(self, num_bits, "activation")
        else:
            self.x_quantizer = None

        self.w_initializer = w_initializer
        self.x_initializer = x_initializer
        self.register_buffer('init_state', torch.tensor(False))

    def forward(self, input):
        if self.x_quantizer is not None and \
                self.x_quantizer.__class__.__name__ in _FULL_HADAMARD_QUANTIZERS and \
                self.H is None:
            self.H = random_hadamard_matrix(self.in_channels, device=input.device).to(input.dtype)

        if self.x_quantizer is not None and \
                self.x_quantizer.__class__.__name__ in _DIAGONAL_HADAMARD_QUANTIZERS and \
                self.d_vec is None:
            d = torch.randint(0, 2, (self.in_channels,), device=input.device).float() * 2 - 1
            self.d_vec = d.to(input.dtype)

        if self.training:
            self.rotated_weight = None

        if self.training:
            return checkpoint(_forward_common, self, input, use_reentrant=False, preserve_rng_state=False)
        else:
            return _forward_common(self, input)


def _forward_common(module, input):
    # The quantizers carry fp32 scales, so dividing an fp16 activation by the
    # scale can promote it to fp32 mid-forward. To keep a quantized layer
    # dtype-transparent to its (often plain-fp16) neighbours, remember the input
    # dtype here and cast the output back to it before returning. No-op when the
    # forward is already single-dtype (fp32 server runs, autocast training).
    _orig_input_dtype = input.dtype

    if module.weight_norm in ["WN", "LWN"]:
        mean = module.weight.mean()
        std = module.weight.std()
        weight = module.weight.add(-mean).div(std)
    else:
        weight = module.weight

    # Save raw input before any rotations for LSQHS initializer.
    # LSQHS needs per-channel stats from the original (pre-smooth, pre-rotation) x.
    input_for_init = input

    # ----------------------------------------------------------------
    # Random diagonal sign flip (LSQHD_quantizer — QuIP#-style)
    # Apply d_vec ∈ {±1}^d_in elementwise before the Hadamard rotation.
    # (x * d) @ H @ H^T @ (w * d)^T = x @ diag(d²) @ w^T = x @ w^T
    # ----------------------------------------------------------------
    if module.x_quantizer is not None and \
            module.x_quantizer.__class__.__name__ in _DIAGONAL_HADAMARD_QUANTIZERS and \
            module.d_vec is not None:
        if module.__class__.__name__ == "QLinear":
            input = input * module.d_vec
            weight = weight * module.d_vec.unsqueeze(0)   # (1, d_in) broadcast over (out, d_in)
        elif module.__class__.__name__ == "QConv2d":
            # d_vec applied to channel dim (C) for both input and weight
            input = input * module.d_vec.view(1, -1, 1, 1)
            weight = weight * module.d_vec.view(1, -1, 1, 1)

    # ----------------------------------------------------------------
    # Pre-Hadamard smooth (LSQHS_quantizer)
    # Suppress per-channel outliers BEFORE rotation so Hadamard spreads
    # a tighter distribution.  Both x/s and w*s applied here so that
    # Q((x/s)H) @ Q((ws)H)^T ≈ (x/s)·H·H^T·(ws)^T = x·w^T.
    # ----------------------------------------------------------------
    if module.x_quantizer is not None and \
            module.x_quantizer.__class__.__name__ in _PRE_HADAMARD_SMOOTH_QUANTIZERS and \
            'smooth_log_scale' in module.x_Qparms and \
            module.__class__.__name__ == 'QLinear':
        smooth = torch.exp(module.x_Qparms['smooth_log_scale'])  # (d_in,)
        input = input / smooth                  # (..., d_in) / (d_in,)
        weight = weight * smooth.view(1, -1)    # (d_out, d_in) * (1, d_in)

    # ----------------------------------------------------------------
    # Full Hadamard rotation (LSQH / LSQHB / LSQHD / LSQHS quantizers)
    # Both x and w rotated: Q(xH) @ Q(wH)^T  ≈  x @ w^T  (H H^T = I)
    # Rotated weight cached at eval time since it doesn't change.
    # ----------------------------------------------------------------
    if module.x_quantizer is not None and \
            module.x_quantizer.__class__.__name__ in _FULL_HADAMARD_QUANTIZERS and \
            module.H is not None:
        if module.__class__.__name__ == "QLinear":
            input = torch.matmul(input, module.H)
        elif module.__class__.__name__ == "QConv2d":
            N, C, Hh, W = input.shape
            input = input.reshape(N, C, -1).permute(0, 2, 1)
            input = torch.matmul(input, module.H)
            input = input.permute(0, 2, 1).reshape(N, C, Hh, W)

        if module.rotated_weight is not None:
            weight = module.rotated_weight
        else:
            if module.__class__.__name__ == "QLinear":
                weight = torch.matmul(weight, module.H)
            elif module.__class__.__name__ == "QConv2d":
                out_c, in_c, kH, kW = weight.shape
                weight = weight.reshape(out_c, in_c, -1).permute(0, 2, 1)
                weight = torch.matmul(weight, module.H)
                weight = weight.permute(0, 2, 1).reshape(out_c, in_c, kH, kW)
            if not module.training:
                module.rotated_weight = weight

    # ----------------------------------------------------------------
    # Block Hadamard rotation (LSQBH_quantizer)
    # d_in split into K blocks of size G; each block rotated independently.
    # H_blocks lazily initialized and shared between x and w.
    # ----------------------------------------------------------------
    if module.x_quantizer is not None and \
            module.x_quantizer.__class__.__name__ == 'LSQBH_quantizer' and \
            module.__class__.__name__ == 'QLinear':
        from src.quantizer.uniform.lsqbh import apply_block_hadamard, apply_block_hadamard_weight
        G = module.x_quantizer.block_size
        if module.H_blocks is None:
            d_in = input.shape[-1]
            K = d_in // G
            H_single = random_hadamard_matrix(G, device=input.device).to(input.dtype)
            module.H_blocks = H_single.unsqueeze(0).expand(K, -1, -1).contiguous()
        input = apply_block_hadamard(input, module.H_blocks, G)
        weight = apply_block_hadamard_weight(weight, module.H_blocks, G)

    # ----------------------------------------------------------------
    # Initialization (runs once, on the first forward pass)
    # ----------------------------------------------------------------
    if module.init_state == False:
        if module.x_quantizer is not None and module.x_initializer is not None:
            Qparms_to_dev(input, module.x_Qparms)

            # Lazy creation of token-type scale/offset for LSQC / LSQSC
            if hasattr(module.x_quantizer, 'lazy_init') and module.x_quantizer.lazy_init:
                if len(input.shape) >= 3:
                    base_dim = input.shape[-2]
                    module.x_quantizer.base_dim = base_dim
                    num_scales = 2 * (module.x_quantizer.num_special_tokens + 1)
                    scale_shape = [1, num_scales] + [1] * (len(input.shape) - 2)
                else:
                    module.x_quantizer.base_dim = 1
                    scale_shape = [1, 1]
                module.x_scale = nn.Parameter(torch.zeros(scale_shape, dtype=torch.float32, device=input.device))
                module.x_offset = nn.Parameter(torch.zeros(scale_shape, dtype=torch.float32, device=input.device))
                module.x_Qparms['scale'] = module.x_scale
                module.x_Qparms['offset'] = module.x_offset
                module.x_quantizer.scale_shape = tuple(scale_shape)

            # For LSQS / LSQSC: provide detached weight reference so LSQS_initializer
            # can compute SmoothQuant migration factors from both x and w.
            # For LSQHS: same weight_ref needed, and we pass pre-rotation input_for_init
            # so per-channel stats reflect original channel structure (not rotated channels).
            if module.x_quantizer.__class__.__name__ in _SMOOTH_QUANTIZERS | _PRE_HADAMARD_SMOOTH_QUANTIZERS:
                module.x_Qparms['weight_ref'] = module.weight.detach()

            init_x = input_for_init if module.x_quantizer.__class__.__name__ in _PRE_HADAMARD_SMOOTH_QUANTIZERS else input

            if module.first_layer:
                module.x_initializer(init_x, module.x_Qparms, module.x_Qn, module.x_Qp, module.x_quantizer, "symmetric")
            else:
                module.x_initializer(init_x, module.x_Qparms, module.x_Qn, module.x_Qp, module.x_quantizer, "asymmetric")
            module.x_quantizer.scale_to_Qparms(module.x_Qparms, module.x_Qn, module.x_Qp)
            module.x_Qparms.pop('weight_ref', None)

        if module.w_quantizer is not None and module.w_initializer is not None:
            Qparms_to_dev(weight, module.w_Qparms)
            module.w_initializer(weight, module.w_Qparms, module.w_Qn, module.w_Qp, module.w_quantizer, "symmetric")
            module.w_quantizer.scale_to_Qparms(module.w_Qparms, module.w_Qn, module.w_Qp)

    # ----------------------------------------------------------------
    # SmoothQuant weight migration (LSQS / LSQSC)
    # x / s is applied inside the quantizer forward.
    # w * s is applied here so (x/s) @ (ws)^T = x @ w^T.
    # At deployment: fold s into weights for zero inference overhead.
    # ----------------------------------------------------------------
    if module.x_quantizer is not None and \
            module.x_quantizer.__class__.__name__ in _SMOOTH_QUANTIZERS and \
            'smooth_log_scale' in module.x_Qparms and \
            module.__class__.__name__ == 'QLinear':
        smooth = torch.exp(module.x_Qparms['smooth_log_scale'])  # (d_in,)
        weight = weight * smooth.view(1, -1)

    # ----------------------------------------------------------------
    # Quantization
    # ----------------------------------------------------------------
    if module.x_quantizer is not None:
        x_numel = input.numel()
        if hasattr(module, 'x_scale_shape') and 'scale' in module.x_Qparms and module.x_Qparms['scale'].numel() > 1:
            x_numel = x_numel // module.x_Qparms['scale'].numel()
        input = module.x_quantizer(input, module.x_Qparms, module.x_Qn, module.x_Qp, x_numel, module.x_grad_scale_mode)

    if module.w_quantizer is not None:
        w_numel = weight.numel()
        if hasattr(module, 'w_scale_shape') and module.w_Qparms['scale'].numel() > 1:
            w_numel = w_numel // module.w_Qparms['scale'].numel()
        weight = module.w_quantizer(weight, module.w_Qparms, module.w_Qn, module.w_Qp, w_numel, module.w_grad_scale_mode)

    if module.weight_norm in ["WN", "LWN"]:
        weight = weight.mul(std)

    # F.linear / F.conv2d require matching operand dtypes; align weight/bias to
    # the (possibly promoted) activation dtype before the op.
    bias = module.bias
    if weight.dtype != input.dtype:
        weight = weight.to(input.dtype)
    if bias is not None and bias.dtype != input.dtype:
        bias = bias.to(input.dtype)

    if module.__class__.__name__ == "QLinear":
        out = F.linear(input, weight, bias)
    elif module.__class__.__name__ == "QConv2d":
        out = F.conv2d(input, weight, bias, module.stride, module.padding, module.dilation, module.groups)

    # Restore the original activation dtype so the layer is transparent to the
    # rest of the network (e.g. an fp16 model whose scales promoted it to fp32).
    if out.dtype != _orig_input_dtype:
        out = out.to(_orig_input_dtype)

    return out


def Qparms_to_dev(x, Qparms):
    dev = x.device
    for iname in Qparms.keys():
        val = Qparms[iname]
        if isinstance(val, torch.Tensor):
            if val.device != dev:
                val.data = val.data.to(dev, non_blocking=True)
