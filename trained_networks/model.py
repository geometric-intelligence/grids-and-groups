import numpy as np
import torch
from torch import nn


class TwoLayerMLP(nn.Module):
    """Two-layer MLP for k-ary group composition.

    Architecture::

        x_flat = x_seq.reshape(batch, k * group_size)
        h = x_flat @ W_in.T                           # (batch, hidden_dim)
        h = nonlinearity(h)                            # element-wise
        y = h @ W_out.T * output_scale                 # (batch, group_size)

    Attributes
    ----------
    k : int
        Number of input group elements.
    group_size : int
        Dimension of each group-element vector.
    hidden_dim : int
        Hidden-layer width.
    """

    def __init__(
        self,
        group_size,
        hidden_dim=None,
        k=2,
        nonlinearity="square",
        init_scale=1.0,
        output_scale=1.0,
    ):
        super().__init__()
        self.group_size = group_size
        self.hidden_dim = hidden_dim if hidden_dim is not None else 50 * group_size
        self.k = k
        self.nonlinearity = nonlinearity
        self.init_scale = init_scale
        self.output_scale = output_scale

        self.W_in = nn.Parameter(
            init_scale * torch.randn(self.hidden_dim, k * group_size) / np.sqrt(k * group_size)
        )
        self.W_out = nn.Parameter(
            init_scale * torch.randn(group_size, self.hidden_dim) / np.sqrt(self.hidden_dim)
        )

    def forward(self, x_seq):
        """Forward pass.

        Parameters
        ----------
        x_seq : torch.Tensor, shape (batch, k, group_size)

        Returns
        -------
        torch.Tensor, shape (batch, group_size)
        """
        batch_size = x_seq.shape[0]
        k_actual = x_seq.shape[1]
        assert k_actual == self.k, f"Expected k={self.k} inputs, got {k_actual}"

        x_flat = x_seq.reshape(batch_size, self.k * self.group_size)
        h = x_flat @ self.W_in.T

        if self.nonlinearity == "power":
            h = h**self.k
        elif self.nonlinearity == "square":
            h = h**2
        elif self.nonlinearity == "relu":
            h = torch.relu(h)
        elif self.nonlinearity == "linear":
            pass
        elif self.nonlinearity == "tanh":
            h = torch.tanh(h)
        elif self.nonlinearity == "gelu":
            h = torch.nn.functional.gelu(h)
        else:
            raise ValueError(f"Invalid nonlinearity '{self.nonlinearity}' provided.")

        y = h @ self.W_out.T * self.output_scale
        return y


class QuadraticRNN(nn.Module):
    """Quadratic recurrent network for sequential group composition.

    Recurrence::

        h_0   = x_1 @ W_in.T
        h_1   = (h_0 + x_2 @ W_drive.T) ** 2
        h_t   = (h_{t-1} @ W_mix.T + x_{t+1} @ W_drive.T) ** 2   for t >= 2
        y     = h_{k-1} @ W_out.T

    Attributes
    ----------
    k : int
        Expected sequence length (number of group elements).
    group_size : int
        Dimension of each group-element vector.
    hidden_dim : int
        Hidden-state width.
    """

    def __init__(
        self,
        group_size: int,
        hidden_dim: int,
        k: int,
        init_scale: float = 1e-2,
        return_all_outputs: bool = False,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.hidden_dim = hidden_dim
        self.k = k
        self.init_scale = init_scale
        self.return_all_outputs = return_all_outputs

        self.W_in = nn.Parameter(
            init_scale * torch.randn(hidden_dim, group_size) / torch.sqrt(torch.tensor(group_size))
        )
        self.W_mix = nn.Parameter(
            init_scale * torch.randn(hidden_dim, hidden_dim) / torch.sqrt(torch.tensor(hidden_dim))
        )
        self.W_drive = nn.Parameter(
            init_scale * torch.randn(hidden_dim, group_size) / torch.sqrt(torch.tensor(group_size))
        )
        self.W_out = nn.Parameter(
            init_scale * torch.randn(group_size, hidden_dim) / torch.sqrt(torch.tensor(hidden_dim))
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_seq : torch.Tensor, shape (batch, k, group_size)

        Returns
        -------
        If ``return_all_outputs=False``: shape ``(batch, group_size)``.
        If ``return_all_outputs=True``:  shape ``(batch, k-1, group_size)``.
        """
        k = x_seq.shape[1]
        assert k >= 2, "Sequence length must be at least 2"

        h_0 = x_seq[:, 0, :] @ self.W_in.T
        h_1 = x_seq[:, 1, :] @ self.W_drive.T
        h = (h_0 + h_1) ** 2

        if self.return_all_outputs:
            outputs = [h @ self.W_out.T]
            for t in range(2, k):
                xt = x_seq[:, t, :]
                h = (h @ self.W_mix.T + xt @ self.W_drive.T) ** 2
                outputs.append(h @ self.W_out.T)
            return torch.stack(outputs, dim=1)

        for t in range(2, k):
            xt = x_seq[:, t, :]
            h = (h @ self.W_mix.T + xt @ self.W_drive.T) ** 2

        return h @ self.W_out.T
