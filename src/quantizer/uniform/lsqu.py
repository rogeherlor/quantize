import torch
import torch.nn as nn

from src.quantizer.uniform.lsq import _LSQ_quantizer

#----------------------------------------------------------
# LSQU — uniform symmetric LSQ that HONORS num_bits
#----------------------------------------------------------
# LSQ_quantizer hardcodes num_bits=4 inside params_set, so it can only ever
# produce 4-bit ranges regardless of the requested bit-width. That is fine for
# the QAT experiments, but it makes a faithful real-INT8 (W8Linear) deployment
# impossible: the learned scale spans only [-8, 7] while the int8 container is
# [-128, 127].
#
# LSQU is a drop-in replacement that uses the requested num_bits, so num_bits=8
# yields a genuine 8-bit symmetric range (Qn=128, Qp=127) and packs faithfully
# into W8Linear (real torch._int_mm execution). The quantization math, learned
# scale parameter, and initializer contract are identical to LSQ_quantizer.
class LSQU_quantizer(nn.Module):
    def __init__(self, module, num_bits, mode, **kwargs):
        super(LSQU_quantizer, self).__init__()
        self.params_set(module, num_bits, mode)

    def params_set(self, module, num_bits, mode):
        if mode == "activation":
            module.x_Qn = 2 ** (num_bits - 1)
            module.x_Qp = 2 ** (num_bits - 1) - 1
            module.x_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            module.x_Qparms['scale'] = module.x_scale
        elif mode == "weight":
            module.w_Qn = 2 ** (num_bits - 1)
            module.w_Qp = 2 ** (num_bits - 1) - 1
            module.w_scale = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            module.w_Qparms['scale'] = module.w_scale

    def forward(self, x, Qparms, Qn, Qp, num_elements, grad_scale_mode):
        scale = Qparms['scale']
        yq = _LSQ_quantizer(x, scale, Qn, Qp, num_elements, grad_scale_mode)
        return yq * scale

    def scale_to_Qparms(self, Qparms, Qn, Qp):
        Qparms["scale"].data = torch.full(
            Qparms["scale"].size(),
            Qparms["init_scale"].clone().detach(),
            device=Qparms["scale"].device,
        )
