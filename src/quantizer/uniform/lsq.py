import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#----------------------------------------------------------
# LSQ
#----------------------------------------------------------
class LSQ_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(LSQ_quantizer, self).__init__()
        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            if module.first_layer:
                module.x_Qn = 2 ** (num_bits-1)
                module.x_Qp = 2 ** (num_bits-1) - 1
            else:
                module.x_Qn = 0
                module.x_Qp = 2 ** (num_bits  ) - 1
            module.x_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32)) # Cast at use
            module.x_Qparms['scale'] = module.x_scale
        elif mode == "weight":
            module.w_Qn = 2 ** (num_bits-1)
            module.w_Qp = 2 ** (num_bits-1) - 1
            module.w_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32)) # Cast at use
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        scale = Qparms['scale']
        yq = _LSQ_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode)
        y = yq * scale
        return y

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        Qparms["scale"].data = torch.full(Qparms["scale"].size(), Qparms["init_scale"].clone().detach(), device=Qparms["scale"].device)

def _LSQ_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode):

    # Qn_on_device = torch.tensor(Qn, dtype=torch.float, device=x.device)
    # Qp_on_device = torch.tensor(Qp, dtype=torch.float, device=x.device)
    qn_t = float(Qn)
    qp_t = float(Qp)

    # print(scale)
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

