import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#----------------------------------------------------------
# LSQ Channel - Enhanced for W8A8 QAT
#
# Key Improvements for Better W8A8 Performance:
#
# 1. LEARNED PER-CHANNEL ACTIVATION SCALES [1, d_in]
#    - Replaced dynamic per-token scales with learned per-channel scales
#    - Provides stable gradients during QAT training
#    - Hardware-friendly: aligns with weight scales [d_out, 1]
#    - Supports 2D [batch, d_in], 3D [batch, seq_len, d_in], and 4D [batch, C, H, W]
#
# 2. SMOOTHQUANT CHANNEL BALANCING (applied in module_quantization.py)
#    - Learnable channel-wise scale [1, d_in] 
#    - Redistributes quantization difficulty: weight *= s, activation /= s
#    - Critical for handling outliers in activation channels
#    - Orthogonal to quantization scales - works on different axis
#
# 3. LEARNABLE CLIPPING (LAC/LWC)
#    - Optional learnable clipping factors per channel
#    - Handles outliers without wasting quantization range
#    - Sigmoid-gated for stable training
#
# 4. HARDWARE-EFFICIENT SCALE AXES
#    - Weight scales: [d_out, 1] - one scale per output channel
#    - Activation scales: [1, d_in] - one scale per input channel
#    - Enables efficient INT8 matrix multiplication
#
# Usage:
#   - enable_learnable_clip=True: Activates LAC/LWC for outlier handling
#   - SmoothQuant balancing is automatic when using LSQC in module_quantization.py
#
# References:
#   - LSQ: Learned Step Size Quantization (Esser et al., 2019)
#   - SmoothQuant: Xiao et al., 2022
#   - QuaRot: Ashkboos et al., 2024 (LAC/LWC concepts)
#----------------------------------------------------------
class LSQC_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, scale_shape=None, 
                 enable_smooth_quant=False, enable_learnable_clip=False, **kwargs):
        super(LSQC_quantizer, self).__init__()
        self.mode = mode
        self.scale_shape = scale_shape if scale_shape is not None else (1,)
        self.enable_smooth_quant = enable_smooth_quant
        self.enable_learnable_clip = enable_learnable_clip
        
        if isinstance(module, nn.Conv2d):
            self.dim_reduction = 1
        else:
            self.dim_reduction = -1

        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            # Learned per-channel quantization scales for stable training
            if module.first_layer:
                module.x_Qn = 2 ** (num_bits-1)
                module.x_Qp = 2 ** (num_bits-1) - 1
            else:
                module.x_Qn = 2 ** (num_bits-1)
                module.x_Qp = 2 ** (num_bits-1) - 1
            
            # Learned quantization scale parameter
            module.x_scale = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
            module.x_Qparms['scale'] = module.x_scale
            
            # Optional: Learnable clipping factors for activation outliers
            if self.enable_learnable_clip:
                init_value = 4.0
                module.x_clip_max = nn.Parameter(torch.ones(self.scale_shape) * init_value)
                module.x_clip_min = nn.Parameter(torch.ones(self.scale_shape) * init_value)
                module.x_Qparms['clip_max'] = module.x_clip_max
                module.x_Qparms['clip_min'] = module.x_clip_min
            
        elif mode == "weight":
            module.w_Qn = 2 ** (num_bits-1)
            module.w_Qp = 2 ** (num_bits-1) - 1
            module.w_scale = nn.Parameter(torch.zeros(self.scale_shape, dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale
            
            # Optional: Learnable clipping factors for weight outliers
            if self.enable_learnable_clip:
                init_value = 4.0
                module.w_clip_max = nn.Parameter(torch.ones(self.scale_shape) * init_value)
                module.w_clip_min = nn.Parameter(torch.ones(self.scale_shape) * init_value)
                module.w_Qparms['clip_max'] = module.w_clip_max
                module.w_Qparms['clip_min'] = module.w_clip_min

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        if self.mode == "activation":
            # Learned per-channel quantization (stable for QAT training)
            scale = Qparms['scale']
            
            # Optional: Apply learnable clipping to handle outliers
            if self.enable_learnable_clip and 'clip_max' in Qparms:
                clip_max = Qparms['clip_max']
                clip_min = Qparms['clip_min']
                sigmoid = torch.nn.functional.sigmoid
                
                # Get per-channel max/min based on input shape
                if len(x.shape) == 2:  # Linear: [batch, d_in]
                    xmax = x.amax(0, keepdim=True)  # [1, d_in]
                    xmin = x.amin(0, keepdim=True)  # [1, d_in]
                elif len(x.shape) == 3:  # Transformer Linear: [batch, seq_len, d_in]
                    xmax = x.amax((0, 1), keepdim=True)  # [1, 1, d_in]
                    xmin = x.amin((0, 1), keepdim=True)  # [1, 1, d_in]
                else:  # Conv2d: [batch, channels, H, W]
                    xmax = x.amax((0, 2, 3), keepdim=True)  # [1, channels, 1, 1]
                    xmin = x.amin((0, 2, 3), keepdim=True)  # [1, channels, 1, 1]
                
                # Apply learnable clipping
                xmax_clipped = xmax * sigmoid(clip_max)
                xmin_clipped = xmin * sigmoid(clip_min)
                x = torch.clamp(x, min=xmin_clipped, max=xmax_clipped)
            
            # Quantize with learned scale
            yq = _LSQC_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode)
            y = yq * scale
            return y
        
        # Weight quantization: per-channel with learned scales
        scale = Qparms['scale']
        
        # Optional: Apply learnable clipping to handle outliers
        if self.enable_learnable_clip and 'clip_max' in Qparms:
            clip_max = Qparms['clip_max']
            clip_min = Qparms['clip_min']
            sigmoid = torch.nn.functional.sigmoid
            
            # Get per-channel max/min for weights
            if len(x.shape) == 2:  # Linear weights: [d_out, d_in]
                wmax = x.amax(1, keepdim=True)  # [d_out, 1]
                wmin = x.amin(1, keepdim=True)  # [d_out, 1]
            else:  # Conv2d weights: [d_out, d_in, k, k]
                wmax = x.amax((1, 2, 3), keepdim=True)  # [d_out, 1, 1, 1]
                wmin = x.amin((1, 2, 3), keepdim=True)  # [d_out, 1, 1, 1]
            
            # Apply learnable clipping
            wmax_clipped = wmax * sigmoid(clip_max)
            wmin_clipped = wmin * sigmoid(clip_min)
            x = torch.clamp(x, min=wmin_clipped, max=wmax_clipped)
        
        yq = _LSQC_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode)
        y = yq * scale
        return y

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        # Initialize learned quantization scales from calibration
        if "init_scale" in Qparms and "scale" in Qparms:
            if Qparms["init_scale"].numel() == 1:
                Qparms["scale"].data.fill_(Qparms["init_scale"].item())
            elif Qparms["init_scale"].shape == Qparms["scale"].shape:
                Qparms["scale"].data.copy_(Qparms["init_scale"])
            else:
                # Handle shape mismatch - broadcast if possible
                try:
                    Qparms["scale"].data.copy_(Qparms["init_scale"].expand_as(Qparms["scale"]))
                except:
                    # Fallback: use scalar value
                    Qparms["scale"].data.fill_(Qparms["init_scale"].mean().item())

def _LSQC_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode):

    # Qn_on_device = torch.tensor(Qn, dtype=torch.float, device=x.device)
    # Qp_on_device = torch.tensor(Qp, dtype=torch.float, device=x.device)
    qn_t = float(Qn)
    qp_t = float(Qp)

    # print(scale)
    # if scale <= 0:
    #     scale_grad = scale.grad if scale.grad is not None else "No gradient"
    #     print(f"Scale assertion failed!")
    #     print(f"  Scale value: {scale.item() if scale.numel() == 1 else scale}")
    #     print(f"  Scale gradient: {scale_grad}")
    #     print(f"  Qn: {qn_t}, Qp: {qp_t}")
    # assert scale > 0, 'scale = {}, {}, {}'.format(scale, qn_t, qp_t)

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

