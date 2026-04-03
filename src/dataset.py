import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class GroupCompositionDataset(Dataset):
    """Single entry point for all group composition datasets.

    For offline mode (default): a map-style Dataset storing pre-generated
    input-output pairs.

    For online mode: returns an IterableDataset that generates batches
    on-the-fly.  Only supported for ``group_name`` in ``('cn', 'cnxcn')``.

    Parameters
    ----------
    group_name : str
        One of ``'cn'``, ``'cnxcn'``, ``'dihedral'``, ``'octahedral'``,
        ``'A5'``.
    online : bool
        If True, returns an IterableDataset for on-the-fly generation.
    **kwargs
        Group-specific parameters forwarded to the build logic:

        - cn: ``group_size, template, k, mode, num_samples, return_all_outputs``
        - cnxcn: ``p1, p2, template, k, mode, num_samples, return_all_outputs``
        - generic groups: ``template, k, group, mode, num_samples, return_all_outputs``
        - online adds: ``batch_size, device``

    Attributes (offline only)
    -------------------------
    X : torch.Tensor
        Input data, shape ``(N, k, group_size)``.
    Y : torch.Tensor
        Target data, shape ``(N, group_size)`` or ``(N, k-1, group_size)``
        when ``return_all_outputs=True``.
    sequence : np.ndarray or None
        Integer element indices / shifts used to build each sample.
    """

    def __new__(cls, group_name, *, online=False, **kwargs):
        if online:
            if group_name not in ("cn", "cnxcn"):
                raise ValueError(
                    f"Online mode only supported for 'cn' and 'cnxcn', got '{group_name}'"
                )
            if group_name == "cn":
                return _OnlineModularAdditionDataset1D(**kwargs)
            return _OnlineModularAdditionDataset2D(**kwargs)
        return super().__new__(cls)

    def __init__(self, group_name, *, online=False, **kwargs):
        if online:
            return
        super().__init__()
        if group_name == "cn":
            X, Y, sequence = self._build_cn(**kwargs)
        elif group_name == "cnxcn":
            X, Y, sequence = self._build_cnxcn(**kwargs)
        else:
            X, Y, sequence = self._build_group(**kwargs)
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.Y = torch.tensor(np.asarray(Y), dtype=torch.float32)
        self.sequence = sequence

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    # ------------------------------------------------------------------
    # Offline build methods
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cn(
        group_size,
        template,
        k,
        mode="sampled",
        num_samples=65536,
        return_all_outputs=False,
    ):
        """Build dataset arrays for cyclic group C_{group_size} via 1D np.roll."""
        assert template.shape == (group_size,), (
            f"template must be ({group_size},), got {template.shape}"
        )

        if mode == "exhaustive":
            total = group_size**k
            if total > 1_000_000:
                raise ValueError(f"group_size^k = {total} is huge; use mode='sampled' instead.")
            N = total
            sequence = np.zeros((N, k), dtype=np.int64)
            for idx in range(N):
                for t in range(k):
                    sequence[idx, t] = (idx // (group_size**t)) % group_size
        else:
            N = int(num_samples)
            sequence = np.random.randint(0, group_size, size=(N, k), dtype=np.int64)

        X = np.zeros((N, k, group_size), dtype=np.float32)
        Y = np.zeros((N, k, group_size), dtype=np.float32)

        for i in range(N):
            cumsum = 0
            for t in range(k):
                shift = int(sequence[i, t])
                X[i, t, :] = np.roll(template, shift)
                cumsum = (cumsum + shift) % group_size
                Y[i, t, :] = np.roll(template, cumsum)

        if not return_all_outputs:
            Y = Y[:, -1, :]
        else:
            Y = Y[:, 1:, :]

        return X, Y, sequence

    @staticmethod
    def _build_cnxcn(
        p1,
        p2,
        template,
        k,
        mode="sampled",
        num_samples=65536,
        return_all_outputs=False,
    ):
        r"""Build dataset arrays for product group C_{p1} \times C_{p2} via 2D np.roll."""
        assert template.shape == (p1, p2), f"template must be ({p1}, {p2}), got {template.shape}"
        group_size = p1 * p2

        if mode == "exhaustive":
            total = group_size**k
            if total > 1_000_000:
                raise ValueError(f"(p1*p2)^k = {total} is huge; use mode='sampled' instead.")
            N = total
            sequence_xy = np.zeros((N, k, 2), dtype=np.int64)
            for idx in range(N):
                for t in range(k):
                    flat_idx = (idx // (group_size**t)) % group_size
                    sequence_xy[idx, t, 0] = flat_idx // p2
                    sequence_xy[idx, t, 1] = flat_idx % p2
        else:
            N = int(num_samples)
            sequence_xy = np.empty((N, k, 2), dtype=np.int64)
            sequence_xy[:, :, 0] = np.random.randint(0, p1, size=(N, k))
            sequence_xy[:, :, 1] = np.random.randint(0, p2, size=(N, k))

        X = np.zeros((N, k, group_size), dtype=np.float32)
        Y = np.zeros((N, k, group_size), dtype=np.float32)

        for i in range(N):
            sx, sy = 0, 0
            for t in range(k):
                ax = int(sequence_xy[i, t, 0])
                ay = int(sequence_xy[i, t, 1])
                rolled = np.roll(np.roll(template, shift=ax, axis=0), shift=ay, axis=1)
                X[i, t, :] = rolled.ravel()
                sx = (sx + ax) % p1
                sy = (sy + ay) % p2
                Y[i, t, :] = np.roll(np.roll(template, shift=sx, axis=0), shift=sy, axis=1).ravel()

        if not return_all_outputs:
            Y = Y[:, -1, :]

        return X, Y, sequence_xy

    @staticmethod
    def _build_group(
        template,
        k,
        group,
        mode="sampled",
        num_samples=65536,
        return_all_outputs=False,
    ):
        """Build dataset arrays for any group with a regular representation."""
        group_size = group.order

        assert template.shape == (group_size,), (
            f"template must be ({group_size},), got {template.shape}"
        )

        n_elements = group_size
        rep_matrices = group.regular_rep()

        if mode == "exhaustive":
            total = n_elements**k
            if total > 1_000_000:
                raise ValueError(f"n_elements^k = {total} is huge; use mode='sampled' instead.")
            N = total
            sequence = np.zeros((N, k), dtype=np.int64)
            for idx in range(N):
                for t in range(k):
                    sequence[idx, t] = (idx // (n_elements**t)) % n_elements
        else:
            N = int(num_samples)
            sequence = np.random.randint(0, n_elements, size=(N, k), dtype=np.int64)

        X = np.zeros((N, k, group_size), dtype=np.float32)
        Y = np.zeros((N, k, group_size), dtype=np.float32)

        for i in range(N):
            cumulative_rep = np.eye(group_size)
            for t in range(k):
                elem_idx = sequence[i, t]
                g_rep = rep_matrices[elem_idx]
                X[i, t, :] = g_rep @ template
                cumulative_rep = g_rep @ cumulative_rep
                Y[i, t, :] = cumulative_rep @ template

        if not return_all_outputs:
            Y = Y[:, -1, :]
        else:
            Y = Y[:, 1:, :]

        return X, Y, sequence


# ---------------------------------------------------------------------------
# Private online dataset classes
# ---------------------------------------------------------------------------


class _OnlineModularAdditionDataset2D(IterableDataset):
    """Online dataset that generates 2D modular addition samples on-the-fly.

    Fully GPU-accelerated for maximum throughput.
    """

    def __init__(
        self,
        p1: int,
        p2: int,
        template: np.ndarray,
        k: int,
        batch_size: int,
        device: str,
        return_all_outputs: bool = False,
    ):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.k = k
        self.batch_size = batch_size
        self.group_size = p1 * p2
        self.device = device
        self.return_all_outputs = return_all_outputs

        self.template_gpu = torch.tensor(template, device=device, dtype=torch.float32)

        x_coords = torch.arange(p1, device=device)
        y_coords = torch.arange(p2, device=device)
        self.x_grid, self.y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")

    def _roll_2d_batch(self, shifts_x, shifts_y):
        """Roll the template by different amounts for each sample in a batch.

        Args:
            shifts_x: (batch_size,) or (batch_size, k) tensor of row shifts
            shifts_y: (batch_size,) or (batch_size, k) tensor of col shifts

        Returns:
            Rolled templates: (batch_size, p1, p2) or (batch_size, k, p1, p2)
        """
        if shifts_x.dim() == 1:
            batch_size = shifts_x.shape[0]
            x_grid = self.x_grid.unsqueeze(0)
            y_grid = self.y_grid.unsqueeze(0)
            shifts_x = shifts_x.view(batch_size, 1, 1)
            shifts_y = shifts_y.view(batch_size, 1, 1)
        else:
            batch_size, k = shifts_x.shape
            x_grid = self.x_grid.unsqueeze(0).unsqueeze(0)
            y_grid = self.y_grid.unsqueeze(0).unsqueeze(0)
            shifts_x = shifts_x.view(batch_size, k, 1, 1)
            shifts_y = shifts_y.view(batch_size, k, 1, 1)

        x_shifted = (x_grid - shifts_x) % self.p1
        y_shifted = (y_grid - shifts_y) % self.p2

        return self.template_gpu[x_shifted.long(), y_shifted.long()]

    def __iter__(self):
        while True:
            shifts_x = torch.randint(
                0, self.p1, (self.batch_size, self.k), device=self.device, dtype=torch.long
            )
            shifts_y = torch.randint(
                0, self.p2, (self.batch_size, self.k), device=self.device, dtype=torch.long
            )

            X_rolled = self._roll_2d_batch(shifts_x, shifts_y)
            X = X_rolled.reshape(self.batch_size, self.k, self.group_size)

            if self.return_all_outputs:
                sx_cumsum = torch.cumsum(shifts_x, dim=1) % self.p1
                sy_cumsum = torch.cumsum(shifts_y, dim=1) % self.p2
                Y_rolled = self._roll_2d_batch(sx_cumsum, sy_cumsum)
                Y = Y_rolled.reshape(self.batch_size, self.k, self.group_size)
                Y = Y[:, 1:, :]
            else:
                sx_cumsum = shifts_x.sum(dim=1) % self.p1
                sy_cumsum = shifts_y.sum(dim=1) % self.p2
                Y_rolled = self._roll_2d_batch(sx_cumsum, sy_cumsum)
                Y = Y_rolled.reshape(self.batch_size, self.group_size)

            yield X, Y


class _OnlineModularAdditionDataset1D(IterableDataset):
    """Online dataset that generates 1D modular addition samples on-the-fly.

    Fully GPU-accelerated for maximum throughput.
    """

    def __init__(
        self,
        group_size: int,
        template: np.ndarray,
        k: int,
        batch_size: int,
        device: str,
        return_all_outputs: bool = False,
    ):
        super().__init__()
        self.group_size = group_size
        self.k = k
        self.batch_size = batch_size
        self.device = device
        self.return_all_outputs = return_all_outputs

        self.template_gpu = torch.tensor(template, device=device, dtype=torch.float32)

    def _roll_1d_batch(self, shifts):
        """Roll the 1D template by different amounts for each sample in a batch.

        Args:
            shifts: (batch_size,) or (batch_size, k) tensor of shifts

        Returns:
            Rolled templates: (batch_size, group_size) or (batch_size, k, group_size)
        """
        if shifts.dim() == 1:
            indices = (
                torch.arange(self.group_size, device=self.device).unsqueeze(0) - shifts.unsqueeze(1)
            ) % self.group_size
        else:
            indices = (
                torch.arange(self.group_size, device=self.device).unsqueeze(0).unsqueeze(0)
                - shifts.unsqueeze(2)
            ) % self.group_size

        return self.template_gpu[indices.long()]

    def __iter__(self):
        while True:
            shifts = torch.randint(
                0,
                self.group_size,
                (self.batch_size, self.k),
                device=self.device,
                dtype=torch.long,
            )

            X = self._roll_1d_batch(shifts)

            if self.return_all_outputs:
                shifts_cumsum = torch.cumsum(shifts, dim=1) % self.group_size
                Y = self._roll_1d_batch(shifts_cumsum)
                Y = Y[:, 1:, :]
            else:
                shifts_cumsum = shifts.sum(dim=1) % self.group_size
                Y = self._roll_1d_batch(shifts_cumsum)

            yield X, Y
