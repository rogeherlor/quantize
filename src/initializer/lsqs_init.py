import torch

# NMSE delta constants (symmetric mode)
_DELTA_NORMAL_SYM = {
    1: 0.9956866859435065,
    3: 0.5860194414434872,
    7: 0.33520061219993685,
    15: 0.18813879027991698,
    31: 0.10406300944201481,
    63: 0.05686767238235839,
    127: 0.03076238758025524,
}


class LSQS_initializer:
    """Initializer for LSQS / LSQSC quantizers.

    Activation path (smooth_log_scale in Qparms):
        Computes per-input-channel smooth factors via the SmoothQuant formula:
            s_i = max(|x_i|)^alpha / max(|w_i|)^(1-alpha)    alpha = 0.5
        then initializes the LSQ step-size on the smoothed activations via NMSE.

    Weight path (no smooth_log_scale in Qparms):
        Initializes the LSQ step-size on the weight tensor via NMSE.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def __call__(self, x, Qparms, Qn, Qp, quantizer, mode, **kwargs):
        dev = x.device
        delta = _DELTA_NORMAL_SYM.get(int(Qp), 0.1)

        if 'smooth_log_scale' not in Qparms:
            # Weight path: plain NMSE scale initialization.
            scale_tmp = x.detach().std() * delta
            if scale_tmp.item() > 0:
                Qparms['init_scale'] = scale_tmp.to(dev)
            return

        # Activation path: SmoothQuant migration + NMSE scale init.
        if 'weight_ref' in Qparms:
            w = Qparms['weight_ref']  # (d_out, d_in), detached

            per_ch_max_x = x.detach().abs().amax(dim=list(range(x.dim() - 1)))  # (d_in,)
            per_col_max_w = w.abs().amax(dim=0)                                  # (d_in,)

            eps = 1e-8
            s_init = per_ch_max_x.pow(self.alpha) / (per_col_max_w.pow(1.0 - self.alpha) + eps)
            s_init = s_init.clamp(min=eps)

            Qparms['smooth_log_scale'].data = torch.log(s_init).to(dev)
            x_smooth = x.detach() / s_init.view(*([1] * (x.dim() - 1)), -1)
        else:
            # No weight reference — skip smooth migration (s=1, log_scale stays 0).
            x_smooth = x.detach()

        scale_tmp = x_smooth.std() * delta
        current = Qparms.get('init_scale', torch.tensor(0.0, device=dev))
        current_val = current.item() if isinstance(current, torch.Tensor) else float(current)

        if scale_tmp.item() > current_val:
            Qparms['init_scale'] = scale_tmp.to(dev)
