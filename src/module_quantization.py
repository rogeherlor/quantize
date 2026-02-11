
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.quantizer.uniform import *
from src.quantizer.nonuniform import *
from src.initializer import *

from torch.utils.checkpoint import checkpoint

#----------------------------------------------------------
# Fully connected layer 
#----------------------------------------------------------
class QLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, \
        num_bits=None, w_grad_scale_mode = "LSQ_grad_scale", \
        x_grad_scale_mode = "LSQ_grad_scale",\
        weight_norm = False, w_quantizer = None, x_quantizer = None, \
        w_initializer = None, x_initializer = None, first_layer = False):

        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.num_bits   = num_bits
        self.x_grad_scale_mode = x_grad_scale_mode
        self.w_grad_scale_mode = w_grad_scale_mode
        self.weight_norm = weight_norm
        self.first_layer = first_layer
        self.x_Qparms={}
        self.w_Qparms={}

        if w_quantizer is not None and w_quantizer.__name__ == "LSQC_quantizer":
            self.w_scale_shape = (out_features, 1)
            self.w_quantizer = w_quantizer(self, num_bits, "weight", 
                                          scale_shape=self.w_scale_shape)
        elif w_quantizer is not None:
            self.w_scale_shape = (1,)
            self.w_quantizer = w_quantizer(self, num_bits, "weight")
        else:
            self.w_quantizer = None

        if x_quantizer is not None and x_quantizer.__name__ == "LSQC_quantizer":
            # Per-token learnable scales - will be initialized in first forward pass
            # Shape will be [1, num_special + 1, 1] for special tokens + shared patch scale
            self.x_scale_shape = None  # Lazy initialization
            self.x_quantizer = x_quantizer(self, num_bits, "activation",
                                          scale_shape=None, num_special_tokens=5)
        elif x_quantizer is not None:
            self.x_scale_shape = (1,)
            self.x_quantizer = x_quantizer(self, num_bits, "activation")
        else:
            self.x_quantizer = None

        self.w_initializer= w_initializer
        self.x_initializer= x_initializer
        self.register_buffer('init_state', torch.tensor(False))

    def forward(self, input):
        # A6000 (48 GB)
        if self.training:
            return checkpoint(_forward_common, self, input, use_reentrant=False, preserve_rng_state=False)
        else:
            # with torch.no_grad():
            return _forward_common(self, input)


#----------------------------------------------------------
# convolutional layer
#----------------------------------------------------------
class QConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, \
        stride=1, padding=0, dilation=1, groups=1, bias=True, \
        num_bits=None, \
        w_grad_scale_mode = "LSQ_grad_scale", \
        x_grad_scale_mode = "LSQ_grad_scale",  \
        weight_norm = None, w_quantizer = None, x_quantizer = None, \
        w_initializer = None, x_initializer = None, first_layer = False):

        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.num_bits   = num_bits
        self.x_grad_scale_mode = x_grad_scale_mode
        self.w_grad_scale_mode = w_grad_scale_mode
        self.weight_norm = weight_norm
        self.first_layer = first_layer
        self.x_Qparms={}
        self.w_Qparms={}

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
            # Per-spatial learnable scales - will be initialized in first forward pass
            # For Conv2d, no special token handling needed
            self.x_scale_shape = None  # Lazy initialization
            self.x_quantizer = x_quantizer(self, num_bits, "activation",
                                          scale_shape=None, num_special_tokens=0)
        elif x_quantizer is not None:
            self.x_scale_shape = (1,)
            self.x_quantizer = x_quantizer(self, num_bits, "activation")
        else:
            self.x_quantizer = None

        self.w_initializer= w_initializer
        self.x_initializer= x_initializer
        self.register_buffer('init_state', torch.tensor(False))

    def forward(self, input):
        # A6000 (48 GB)
        if self.training:
            return checkpoint(_forward_common, self, input, use_reentrant=False, preserve_rng_state=False)
        else:
            # with torch.no_grad():
            return _forward_common(self, input)

def _forward_common(module, input):
    if module.weight_norm in ["WN", "LWN"]:
        mean = module.weight.mean()
        std = module.weight.std()
        weight = module.weight.add(-mean).div(std)
    else :
        weight = module.weight
    if module.init_state == False:
        if module.x_quantizer != None and module.x_initializer != None:
            Qparms_to_dev(input, module.x_Qparms)
            
            # Lazy initialization of activation scale shape based on input dim -2
            if hasattr(module.x_quantizer, 'lazy_init') and module.x_quantizer.lazy_init:
                if len(input.shape) >= 3:
                    base_dim = input.shape[-2]
                    # Store base dimension for replication (per-frame token count in VGGT)
                    # In VGGT: base_dim = P tokens per frame (camera + register + patches)
                    # Frame attention: (B*S, P, C) - uses base_dim directly
                    # Global attention: (B, S*P, C) - base_dim replicated S times
                    module.x_quantizer.base_dim = base_dim
                    # Create base scale vector: [1, 2 * (num_special + 1), 1]
                    # First 6: frame1 (5 special + 1 patch)
                    # Next 6: frames 2+ (5 special + 1 patch)
                    num_scales = 2 * (module.x_quantizer.num_special_tokens + 1)  # 2 * 6 = 12
                    scale_shape = [1, num_scales] + [1] * (len(input.shape) - 2)
                else:
                    module.x_quantizer.base_dim = 1
                    scale_shape = [1, 1]
                module.x_scale = nn.Parameter(torch.zeros(scale_shape, dtype=torch.float32, device=input.device))
                module.x_Qparms['scale'] = module.x_scale
                module.x_quantizer.scale_shape = tuple(scale_shape)
            
            if module.first_layer:                
                module.x_initializer(input, module.x_Qparms, module.x_Qn, module.x_Qp, module.x_quantizer, "symmetric")
            else:
                module.x_initializer(input, module.x_Qparms, module.x_Qn, module.x_Qp, module.x_quantizer, "asymmetric")
            module.x_quantizer.scale_to_Qparms(module.x_Qparms, module.x_Qn, module.x_Qp)

        if module.w_quantizer != None and module.w_initializer != None:                
            Qparms_to_dev(weight, module.w_Qparms)
            module.w_initializer(weight, module.w_Qparms, module.w_Qn, module.w_Qp, module.w_quantizer, "symmetric")            
            module.w_quantizer.scale_to_Qparms(module.w_Qparms, module.w_Qn, module.w_Qp)

    if module.x_quantizer != None:
        x_numel = input.numel()
        # For LSQC activations with per-channel scales
        if hasattr(module, 'x_scale_shape') and 'scale' in module.x_Qparms and module.x_Qparms['scale'].numel() > 1:
             x_numel = x_numel // module.x_Qparms['scale'].numel()
        input  = module.x_quantizer(input, module.x_Qparms, module.x_Qn, module.x_Qp, x_numel, module.x_grad_scale_mode)
    if module.w_quantizer != None:
        w_numel = weight.numel()
        if hasattr(module, 'w_scale_shape') and module.w_Qparms['scale'].numel() > 1:
             w_numel = w_numel // module.w_Qparms['scale'].numel()
        weight = module.w_quantizer(weight, module.w_Qparms, module.w_Qn, module.w_Qp, w_numel, module.w_grad_scale_mode)

    if module.weight_norm in ["WN", "LWN"]:
        weight = weight.mul(std)

    # compute forward pass
    if   module.__class__.__name__ in ["QLinear"]:
        out = F.linear(input, weight, module.bias)
    elif module.__class__.__name__ in ["QConv2d"]:
        out = F.conv2d(input, weight, module.bias, module.stride, module.padding, module.dilation, module.groups)

    return out

def Qparms_to_dev(x, Qparms):
    dev = x.device
    for iname in Qparms.keys():
        if Qparms[iname].device != dev:
            Qparms[iname].data = Qparms[iname].data.to(dev, non_blocking=True)
