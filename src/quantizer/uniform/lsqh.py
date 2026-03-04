import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import QuaRot's Hadamard matrix generation
from src.ptq.quarot.quarot_utils import random_hadamard_matrix

#----------------------------------------------------------
# LSQ with Hadamard Rotation Support
#----------------------------------------------------------
class LSQ_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, use_hadamard=True, **kwargs):
        super(LSQ_quantizer, self).__init__()
        self.use_hadamard = use_hadamard
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
            module.x_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32)) # Cast at use
            module.x_Qparms['scale'] = module.x_scale
                
        elif mode == "weight":
            num_bits = 4
            module.w_Qn = 2 ** (num_bits-1)
            module.w_Qp = 2 ** (num_bits-1) - 1
            module.w_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32)) # Cast at use
            module.w_Qparms['scale'] = module.w_scale
        
        # Initialize rotation matrix storage - will be created lazily on first use
        self.rotation_matrix = None

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode, rotation_matrix=None):
        scale = Qparms['scale']
        
        # Apply Hadamard rotation if enabled
        if self.use_hadamard:
            # Initialize rotation matrix once on first forward pass
            if self.rotation_matrix is None:
                feature_dim = x.shape[-1]
                # Use QuaRot's pre-computed structured Hadamard matrix
                self.rotation_matrix = random_hadamard_matrix(feature_dim, x.device).to(x.dtype)
            
            # Apply rotation: x_rotated = x @ rotation_matrix
            x_rotated = torch.matmul(x, self.rotation_matrix)
            yq = _LSQ_quantizer(x_rotated, scale, Qn, Qp, num_elements, grad_scale_mode)
            y = yq * scale
            # Apply inverse rotation: y_unrotated = y @ rotation_matrix.T
            y_unrotated = torch.matmul(y, self.rotation_matrix.T)
            return y_unrotated
        else:
            yq = _LSQ_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode)
            y = yq * scale
            return y

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        Qparms["scale"].data = torch.full(Qparms["scale"].size(), Qparms["init_scale"].clone().detach(), device=Qparms["scale"].device)

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

