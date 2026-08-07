import torch


class PerNeuronScaledSGD(torch.optim.Optimizer):
    """Per-neuron scaled SGD optimizer for TwoLayerMLP.

    Exploits model homogeneity to scale per-neuron learning rates:

        eta_i = lr * ||theta_i||^(1 - degree)

    where theta_i = (W_in[i, :], W_out[:, i]) comprises all parameters
    associated with neuron i, and ``degree`` is the degree of homogeneity.

    Parameters
    ----------
    model : TwoLayerMLP
        The model whose parameters will be optimized.
    lr : float
        Base learning rate.
    degree : int or None
        Degree of homogeneity.  If None, defaults to ``model.k``.
    """

    def __init__(self, model, lr=1.0, degree=None):
        if degree is None:
            degree = model.k

        params = list(model.parameters())
        super().__init__(
            [{"params": params, "model": model}],
            dict(lr=lr, degree=degree),
        )

    @torch.no_grad()
    def step(self, closure=None):
        group = self.param_groups[0]
        model = group["model"]
        lr = group["lr"]
        degree = group["degree"]

        W_in = model.W_in
        W_out = model.W_out
        g_in = W_in.grad
        g_out = W_out.grad

        if g_in is None or g_out is None:
            return

        u2 = (W_in**2).sum(dim=1)  # (hidden_dim,)
        w2 = (W_out**2).sum(dim=0)  # (hidden_dim,)
        theta_norm = torch.sqrt(u2 + w2 + 1e-12)

        scale = theta_norm.pow(1 - degree)

        g_in.mul_(scale.view(-1, 1))
        g_out.mul_(scale.view(1, -1))

        W_in.add_(g_in, alpha=-lr)
        W_out.add_(g_out, alpha=-lr)


class HybridRNNOptimizer(torch.optim.Optimizer):
    """Hybrid optimizer for QuadraticRNN.

    Combines per-neuron scaled SGD for the MLP-like components
    (``W_in``, ``W_drive``, ``W_out``) with Adam for the recurrent
    component (``W_mix``).

    The per-neuron scaling is::

        eta_i = lr * ||theta_i||^scaling_factor

    where theta_i = (W_in[i,:], W_drive[i,:], W_out[:,i]).

    Parameters
    ----------
    model : QuadraticRNN
        The model whose parameters will be optimized.
    lr : float
        Learning rate for scaled SGD (W_in, W_drive, W_out).
    scaling_factor : float
        Exponent for per-neuron norm scaling.
    adam_lr : float
        Learning rate for Adam (W_mix).
    adam_betas : tuple[float, float]
        Adam beta parameters.
    adam_eps : float
        Adam epsilon for numerical stability.
    """

    def __init__(
        self,
        model,
        lr=1e-2,
        scaling_factor=-1,
        adam_lr=1e-3,
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
    ):
        scaled_params = [model.W_in, model.W_drive, model.W_out]
        adam_params = [model.W_mix]

        defaults = dict(
            model=model,
            lr=lr,
            scaling_factor=scaling_factor,
            adam_lr=adam_lr,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
        )

        super().__init__(
            [
                {"params": scaled_params, "type": "scaled_sgd"},
                {"params": adam_params, "type": "adam"},
            ],
            defaults,
        )

        self.state["step"] = 0
        for param in adam_params:
            self.state[param] = {
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
            }

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            if group["type"] == "scaled_sgd":
                model = self.defaults["model"]
                lr = self.defaults["lr"]
                scaling_factor = self.defaults["scaling_factor"]

                W_in, W_drive, W_out = model.W_in, model.W_drive, model.W_out
                g_in, g_drive, g_out = W_in.grad, W_drive.grad, W_out.grad

                if g_in is None or g_drive is None or g_out is None:
                    continue

                u2 = (W_in**2).sum(dim=1)  # (hidden_dim,)
                v2 = (W_drive**2).sum(dim=1)  # (hidden_dim,)
                w2 = (W_out**2).sum(dim=0)  # (hidden_dim,)
                theta_norm = torch.sqrt(u2 + v2 + w2 + 1e-12)

                scale = theta_norm.pow(scaling_factor)

                g_in_scaled = g_in * scale.view(-1, 1)
                g_drive_scaled = g_drive * scale.view(-1, 1)
                g_out_scaled = g_out * scale.view(1, -1)

                W_in.add_(g_in_scaled, alpha=-lr)
                W_drive.add_(g_drive_scaled, alpha=-lr)
                W_out.add_(g_out_scaled, alpha=-lr)

            elif group["type"] == "adam":
                adam_lr = self.defaults["adam_lr"]
                beta1, beta2 = self.defaults["adam_betas"]
                eps = self.defaults["adam_eps"]

                self.state["step"] += 1
                step = self.state["step"]

                for param in group["params"]:
                    if param.grad is None:
                        continue

                    grad = param.grad
                    state = self.state[param]

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step

                    step_size = adam_lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)

                    param.addcdiv_(exp_avg, denom, value=-step_size)
