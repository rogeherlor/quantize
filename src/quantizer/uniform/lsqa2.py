import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#----------------------------------------------------------
# LSQ with Per-Channel Quantization
#----------------------------------------------------------
class LSQ_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, use_per_channel=True, **kwargs):
        super(LSQ_quantizer, self).__init__()
        self.use_per_channel = use_per_channel
        self.mode = mode
        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            num_bits = 4
            if module.first_layer:
                module.x_Qn = 2 ** (num_bits-1)
                module.x_Qp = 2 ** (num_bits-1) - 1
            else:
                #module.x_Qn = 0
                #module.x_Qp = 2 ** (num_bits  ) - 1
                module.x_Qn = 2 ** (num_bits-1)
                module.x_Qp = 2 ** (num_bits-1) - 1
            
            # Per-channel quantization for inner dimension (d_in)
            if self.use_per_channel and hasattr(module, 'in_features'):
                d_in = module.in_features
                # Base scale (learnable)
                module.x_scale_base = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
                # Power-of-2 shifts per channel (fixed after init, stored as integers)
                module.x_scale_shifts = nn.Parameter(torch.zeros(d_in, dtype=torch.float32), requires_grad=False)
                module.x_Qparms['scale_base'] = module.x_scale_base
                module.x_Qparms['scale_shifts'] = module.x_scale_shifts
                module.x_Qparms['d_in'] = d_in
            else:
                # Standard single scale
                module.x_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
                module.x_Qparms['scale'] = module.x_scale
                
        elif mode == "weight":
            num_bits = 4
            module.w_Qn = 2 ** (num_bits-1)
            module.w_Qp = 2 ** (num_bits-1) - 1
            module.w_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32)) # Cast at use
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        # Check if using per-channel quantization
        if 'scale_base' in Qparms and self.use_per_channel:
            scale_base = Qparms['scale_base']
            scale_shifts = Qparms['scale_shifts']
            d_in = Qparms['d_in']
            
            # Per-channel quantization
            y = _LSQ_quantizer_per_channel(x, scale_base, scale_shifts, Qn, Qp, num_elements, grad_scale_mode)
            return y
        else:
            # Standard single scale quantization
            scale = Qparms['scale']
            yq = _LSQ_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode)
            y = yq * scale
            return y

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        if 'scale_base' in Qparms and self.use_per_channel:
            # Initialize base scale and compute shifts from init_scale
            if 'init_scale' in Qparms:
                init_scales = Qparms['init_scale']  # Shape: (d_in,) or scalar
                if init_scales.numel() > 1:
                    # Per-channel init scales provided - compute base and shifts
                    base_scale, shifts = compute_base_and_shifts(init_scales)
                    Qparms['scale_base'].data = torch.tensor([base_scale], device=Qparms['scale_base'].device)
                    Qparms['scale_shifts'].data = shifts.to(Qparms['scale_shifts'].device)
                else:
                    # Single init scale - use it as base, zero shifts
                    Qparms['scale_base'].data = torch.full(Qparms['scale_base'].size(), init_scales.item(), device=Qparms['scale_base'].device)
                    Qparms['scale_shifts'].data.zero_()
        else:
            # Standard single scale initialization
            Qparms["scale"].data = torch.full(Qparms["scale"].size(), Qparms["init_scale"].clone().detach(), device=Qparms["scale"].device)

#----------------------------------------------------------
# Per-Channel LSQ quantizer with power-of-2 shifts
#----------------------------------------------------------

def _LSQ_quantizer_per_channel(x, scale_base, scale_shifts, Qn, Qp, num_elements, grad_scale_mode):
    """
    Per-channel LSQ quantization with power-of-2 scaling.
    
    Args:
        x: Input tensor of shape (..., d_in)
        scale_base: Base scale (learnable scalar)
        scale_shifts: Power-of-2 shifts per channel, shape (d_in,)
                     Actual scale[i] = scale_base * 2^(scale_shifts[i])
        Qn, Qp: Quantization range
        num_elements: Number of elements for gradient scaling
        grad_scale_mode: Gradient scaling mode
    
    Returns:
        Quantized tensor of same shape as input
    """
    qn_t = float(Qn)
    qp_t = float(Qp)
    
    assert scale_base > 0, f'scale_base = {scale_base}, must be positive'
    
    # Compute per-channel scales: scale_i = scale_base * 2^(shift_i)
    # Round shifts to integers for hardware-efficient power-of-2 multipliers
    shifts_rounded = torch.round(scale_shifts)
    scales = scale_base * torch.pow(2.0, shifts_rounded)  # Shape: (d_in,)
    
    # Reshape scales for broadcasting: (..., 1, d_in) -> (..., d_in)
    # Add dimensions to match input
    scale_shape = [1] * (x.dim() - 1) + [scales.shape[0]]
    scales = scales.view(*scale_shape)
    
    # Gradient scaling for base scale only (shifts are fixed)
    if num_elements > 0:
        if grad_scale_mode == "10_fac":
            grad_scale = torch.tensor(10.0, device=x.device)
        elif grad_scale_mode == "LSQ_grad_scale":
            grad_scale = 1.0 / torch.sqrt(num_elements * qp_t)
        else:
            grad_scale = torch.tensor(1.0, device=x.device)
        
        # Apply gradient scaling through straight-through estimator
        bw_scale_base = scale_base * grad_scale
        scale_base_scaled = (scale_base - bw_scale_base).detach() + bw_scale_base
        
        # Recompute scales with gradient-scaled base
        shifts_rounded = torch.round(scale_shifts)
        scales = scale_base_scaled * torch.pow(2.0, shifts_rounded.detach())
        scales = scales.view(*scale_shape)
    
    # Quantize
    x_scaled = x / scales
    x_clipped = torch.clamp(x_scaled, min=-qn_t, max=qp_t)
    y = (torch.round(x_clipped) - x_clipped).detach() + x_clipped
    
    # Scale back
    y = y * scales
    
    return y


def compute_base_and_shifts(init_scales):
    """
    Compute base scale and power-of-2 shifts from initial per-channel scales.
    
    Args:
        init_scales: Tensor of initial scales, shape (d_in,)
    
    Returns:
        base_scale: Scalar base scale (median of init_scales)
        shifts: Integer shifts per channel, shape (d_in,)
    """
    # Use median as base scale (more robust than mean)
    base_scale = torch.median(init_scales).item()
    
    # Compute relative scales
    relative_scales = init_scales / base_scale
    
    # Find nearest power-of-2 for each relative scale
    # shift = round(log2(relative_scale))
    log2_relative = torch.log2(torch.clamp(relative_scales, min=1e-8))
    shifts = torch.round(log2_relative)
    
    return base_scale, shifts


#----------------------------------------------------------
# Original LSQ quantizer function
#----------------------------------------------------------

def _LSQ_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode):

    # Qn_on_device = torch.tensor(Qn, dtype=torch.float, device=x.device)
    # Qp_on_device = torch.tensor(Qp, dtype=torch.float, device=x.device)
    qn_t = float(Qn)
    qp_t = float(Qp)

    # print(scale)
    if scale <= 0:
        scale_grad = scale.grad if scale.grad is not None else "No gradient"
        print(f"Scale assertion failed!")
        print(f"  Scale value: {scale.item() if scale.numel() == 1 else scale}")
        print(f"  Scale gradient: {scale_grad}")
        print(f"  Qn: {qn_t}, Qp: {qp_t}")
    assert scale > 0, 'scale = {}, {}, {}'.format(scale, qn_t, qp_t)

    # gradient scaling
    if num_elements > 0:
        if grad_scale_mode == "10_fac":
            grad_scale = torch.tensor(10.0, device=x.device)
        elif grad_scale_mode == "LSQ_grad_scale":
            grad_scale = 1.0 / torch.sqrt(num_elements * qp_t)
        else:
            grad_scale = torch.tensor(1.0, device=x.device)

        bw_scale = scale * grad_scale
        scale = (scale - bw_scale).detach() + bw_scale
    
    x  = x / scale
    x = torch.clamp(x, min=-float(qn_t), max=float(qp_t))
    # xq = torch.round(x)
    y  = (torch.round(x) - x).detach() + x
    
    # y  = scale * y

    return  y

