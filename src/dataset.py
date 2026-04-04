import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from src.groups.cn import CyclicGroup
from src.groups.cnxcn import ProductCyclicGroup


class GroupCompositionDataset(Dataset):
    """Single entry point for all group composition datasets.

    For offline mode (default): a map-style Dataset storing pre-generated
    input-output pairs.  Uses ``group.regular_rep()`` to compose elements,
    so it works uniformly for any ``Group`` subclass.

    For online mode: returns an IterableDataset that generates batches
    on-the-fly on GPU.  Only supported for ``CyclicGroup`` and
    ``ProductCyclicGroup`` (exploits modular addition structure for speed).

    Parameters
    ----------
    group : Group
        The group object.
    online : bool
        If True, returns an IterableDataset for on-the-fly generation.
    **kwargs
        Parameters forwarded to the build logic:

        - offline: ``template, k, mode, num_samples, return_all_outputs``
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

    def __new__(cls, group, *, online=False, **kwargs):
        if online:
            if isinstance(group, CyclicGroup):
                return _OnlineModularAdditionDataset1D(group_size=group.order, **kwargs)
            if isinstance(group, ProductCyclicGroup):
                return _OnlineModularAdditionDataset2D(p1=group._p1, p2=group._p2, **kwargs)
            raise ValueError(
                f"Online mode only supported for CyclicGroup and ProductCyclicGroup, "
                f"got {type(group).__name__}"
            )
        return super().__new__(cls)

    def __init__(self, group, *, online=False, **kwargs):
        if online:
            return
        super().__init__()
        X, Y, sequence = self._build_group(group=group, **kwargs)
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.Y = torch.tensor(np.asarray(Y), dtype=torch.float32)
        self.sequence = sequence

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    @staticmethod
    def _build_group(
        template,
        k,
        group,
        mode="sampled",
        num_samples=65536,
        return_all_outputs=False,
    ):
        """Build dataset arrays for any group with a regular representation.

        Replaces the former ``_build_cn`` (which used ``np.roll``) and
        ``_build_cnxcn`` (which used 2D ``np.roll``).  Equivalence for
        ``CyclicGroup`` and ``ProductCyclicGroup`` is verified in
        ``test/test_refactor_equivalence.py``.
        """
        template = np.asarray(template).ravel()
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

        template_2d = np.asarray(template).reshape(p1, p2)
        self.template_gpu = torch.tensor(template_2d, device=device, dtype=torch.float32)

        x_coords = torch.arange(p1, device=device)
        y_coords = torch.arange(p2, device=device)
        self.x_grid, self.y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")

    def _roll_2d_batch(self, shifts_x, shifts_y):
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
