import numpy as np
import torch
from torch.utils.data import IterableDataset


class OnlineModularAdditionDataset2D(IterableDataset):
    """
    Online dataset that generates 2D modular addition samples on-the-fly.
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
        self.p_flat = p1 * p2
        self.device = device
        self.return_all_outputs = return_all_outputs

        # Store template on GPU for fast rolling
        self.template_gpu = torch.tensor(template, device=device, dtype=torch.float32)

        # Pre-compute coordinate grids on GPU for efficient rolling
        x_coords = torch.arange(p1, device=device)
        y_coords = torch.arange(p2, device=device)
        self.x_grid, self.y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")

    def _roll_2d_batch(self, shifts_x, shifts_y):
        """
        Roll the template by different amounts for each sample in a batch.
        Fully vectorized on GPU.

        Args:
            shifts_x: (batch_size,) or (batch_size, k) tensor of row shifts
            shifts_y: (batch_size,) or (batch_size, k) tensor of col shifts

        Returns:
            Rolled templates: (batch_size, p1, p2) or (batch_size, k, p1, p2)
        """
        # Determine output shape based on input
        if shifts_x.dim() == 1:
            # Single roll per sample: (batch_size,)
            batch_size = shifts_x.shape[0]
            # Broadcast: (1, p1, p2) -> (batch_size, p1, p2)
            x_grid = self.x_grid.unsqueeze(0)  # (1, p1, p2)
            y_grid = self.y_grid.unsqueeze(0)  # (1, p1, p2)
            shifts_x = shifts_x.view(batch_size, 1, 1)  # (batch_size, 1, 1)
            shifts_y = shifts_y.view(batch_size, 1, 1)  # (batch_size, 1, 1)
        else:
            # Multiple rolls per sample: (batch_size, k)
            batch_size, k = shifts_x.shape
            # Broadcast: (1, 1, p1, p2) -> (batch_size, k, p1, p2)
            x_grid = self.x_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, p1, p2)
            y_grid = self.y_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, p1, p2)
            shifts_x = shifts_x.view(batch_size, k, 1, 1)  # (batch_size, k, 1, 1)
            shifts_y = shifts_y.view(batch_size, k, 1, 1)  # (batch_size, k, 1, 1)

        # Compute shifted coordinates with modular arithmetic
        x_shifted = (x_grid - shifts_x) % self.p1
        y_shifted = (y_grid - shifts_y) % self.p2

        # Index into template using advanced indexing
        rolled = self.template_gpu[x_shifted.long(), y_shifted.long()]

        return rolled

    def __iter__(self):
        """Generate batches indefinitely on GPU."""
        while True:
            # Generate random shifts on GPU: (batch_size, k)
            shifts_x = torch.randint(
                0, self.p1, (self.batch_size, self.k), device=self.device, dtype=torch.long
            )
            shifts_y = torch.randint(
                0, self.p2, (self.batch_size, self.k), device=self.device, dtype=torch.long
            )

            # Generate X: roll template for each time step
            # Shape: (batch_size, k, p1, p2)
            X_rolled = self._roll_2d_batch(shifts_x, shifts_y)

            # Reshape to (batch_size, k, p_flat)
            X = X_rolled.reshape(self.batch_size, self.k, self.p_flat)

            if self.return_all_outputs:
                # Generate Y for ALL cumulative sums (intermediate targets)
                # Compute cumulative sum at each timestep
                sx_cumsum = torch.cumsum(shifts_x, dim=1) % self.p1  # (batch_size, k)
                sy_cumsum = torch.cumsum(shifts_y, dim=1) % self.p2  # (batch_size, k)

                # Roll by all cumulative sums: (batch_size, k, p1, p2)
                Y_rolled = self._roll_2d_batch(sx_cumsum, sy_cumsum)

                # Reshape to (batch_size, k, p_flat)
                Y = Y_rolled.reshape(self.batch_size, self.k, self.p_flat)
                Y = Y[:, 1:, :]

            else:
                # Generate Y: only final cumulative sum (current behavior)
                sx_cumsum = shifts_x.sum(dim=1) % self.p1  # (batch_size,)
                sy_cumsum = shifts_y.sum(dim=1) % self.p2  # (batch_size,)

                # Shape: (batch_size, p1, p2)
                Y_rolled = self._roll_2d_batch(sx_cumsum, sy_cumsum)

                # Reshape to (batch_size, p_flat)
                Y = Y_rolled.reshape(self.batch_size, self.p_flat)

            yield X, Y

    @staticmethod
    def generate_dataset(
        p1: int,
        p2: int,
        template: np.ndarray,
        k: int,
        mode: str = "sampled",
        num_samples: int = 65536,
        return_all_outputs: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a fixed 2D modular addition dataset.

        Args:
            p1: height (rows) dimension
            p2: width  (cols) dimension
            template: (p1, p2) template array
            k: sequence length
            mode: "sampled" or "exhaustive"
            num_samples: number of samples for "sampled" mode
            return_all_outputs: if True, return intermediate outputs

        Returns:
            X:           (N, k, p1*p2) input sequences (flattened rolled templates)
            Y:           (N, p1*p2) or (N, k, p1*p2) targets
            sequence_xy: (N, k, 2) integer group elements (ax_t, ay_t) per token
        """
        assert template.shape == (p1, p2), f"template must be ({p1}, {p2}), got {template.shape}"
        p_flat = p1 * p2

        if mode == "exhaustive":
            total = (p1 * p2) ** k
            if total > 1_000_000:
                raise ValueError(
                    f"(p1*p2)**k = {total} is huge; use mode='sampled' instead."
                )
            N = total
            sequence_xy = np.zeros((N, k, 2), dtype=np.int64)
            for idx in range(N):
                for t in range(k):
                    flat_idx = (idx // (p_flat**t)) % p_flat
                    ax = flat_idx // p2
                    ay = flat_idx % p2
                    sequence_xy[idx, t, 0] = ax
                    sequence_xy[idx, t, 1] = ay
        else:
            N = int(num_samples)
            sequence_xy = np.empty((N, k, 2), dtype=np.int64)
            sequence_xy[:, :, 0] = np.random.randint(0, p1, size=(N, k))
            sequence_xy[:, :, 1] = np.random.randint(0, p2, size=(N, k))

        X = np.zeros((N, k, p_flat), dtype=np.float32)
        Y = np.zeros((N, k, p_flat), dtype=np.float32)

        for i in range(N):
            sx, sy = 0, 0
            for t in range(k):
                ax, ay = int(sequence_xy[i, t, 0]), int(sequence_xy[i, t, 1])
                rolled = np.roll(np.roll(template, shift=ax, axis=0), shift=ay, axis=1)
                X[i, t, :] = rolled.ravel()
                sx = (sx + ax) % p1
                sy = (sy + ay) % p2
                Y[i, t, :] = np.roll(
                    np.roll(template, shift=sx, axis=0), shift=sy, axis=1
                ).ravel()

        if not return_all_outputs:
            Y = Y[:, -1, :]

        return X, Y, sequence_xy


class OnlineModularAdditionDataset1D(IterableDataset):
    """
    Online dataset that generates 1D modular addition samples on-the-fly.
    Fully GPU-accelerated for maximum throughput.
    """

    def __init__(
        self,
        p: int,
        template: np.ndarray,
        k: int,
        batch_size: int,
        device: str,
        return_all_outputs: bool = False,
    ):
        super().__init__()
        self.p = p
        self.k = k
        self.batch_size = batch_size
        self.device = device
        self.return_all_outputs = return_all_outputs

        # Store template on GPU for fast rolling
        self.template_gpu = torch.tensor(template, device=device, dtype=torch.float32)

    def _roll_1d_batch(self, shifts):
        """
        Roll the 1D template by different amounts for each sample in a batch.
        Fully vectorized on GPU.

        Args:
            shifts: (batch_size,) or (batch_size, k) tensor of shifts

        Returns:
            Rolled templates: (batch_size, p) or (batch_size, k, p)
        """
        if shifts.dim() == 1:
            # Single roll per sample: (batch_size,)
            batch_size = shifts.shape[0]
            # Use advanced indexing
            indices = (
                torch.arange(self.p, device=self.device).unsqueeze(0) - shifts.unsqueeze(1)
            ) % self.p
            rolled = self.template_gpu[indices.long()]
        else:
            # Multiple rolls per sample: (batch_size, k)
            batch_size, k = shifts.shape
            indices = (
                torch.arange(self.p, device=self.device).unsqueeze(0).unsqueeze(0)
                - shifts.unsqueeze(2)
            ) % self.p
            rolled = self.template_gpu[indices.long()]

        return rolled

    def __iter__(self):
        """Generate batches indefinitely on GPU."""
        while True:
            # Generate random shifts on GPU: (batch_size, k)
            shifts = torch.randint(
                0, self.p, (self.batch_size, self.k), device=self.device, dtype=torch.long
            )

            # Generate X: roll template for each time step
            # Shape: (batch_size, k, p)
            X = self._roll_1d_batch(shifts)

            if self.return_all_outputs:
                # Generate Y for ALL cumulative sums (intermediate targets)
                shifts_cumsum = torch.cumsum(shifts, dim=1) % self.p  # (batch_size, k)

                # Roll by all cumulative sums: (batch_size, k, p)
                Y = self._roll_1d_batch(shifts_cumsum)
                Y = Y[:, 1:, :]  # Remove first timestep
            else:
                # Generate Y: only final cumulative sum
                shifts_cumsum = shifts.sum(dim=1) % self.p  # (batch_size,)

                # Shape: (batch_size, p)
                Y = self._roll_1d_batch(shifts_cumsum)

            yield X, Y

    @staticmethod
    def generate_dataset(
        p: int,
        template: np.ndarray,
        k: int,
        mode: str = "sampled",
        num_samples: int = 65536,
        return_all_outputs: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a fixed 1D modular addition dataset for cyclic group C_p.

        Args:
            p: dimension of cyclic group
            template: (p,) template array
            k: sequence length
            mode: "sampled" or "exhaustive"
            num_samples: number of samples for "sampled" mode
            return_all_outputs: if True, return intermediate outputs

        Returns:
            X: (N, k, p) where token t is template rolled by shift_t
            Y: (N, p) or (N, k-1, p) target rolled by cumulative sum
            sequence: (N, k) integer group elements (shifts) per token
        """
        assert template.shape == (p,), f"template must be ({p},), got {template.shape}"

        if mode == "exhaustive":
            total = p**k
            if total > 1_000_000:
                raise ValueError(f"p^k = {total} is huge; use mode='sampled' instead.")
            N = total
            sequence = np.zeros((N, k), dtype=np.int64)
            for idx in range(N):
                for t in range(k):
                    sequence[idx, t] = (idx // (p**t)) % p
        else:
            N = int(num_samples)
            sequence = np.random.randint(0, p, size=(N, k), dtype=np.int64)

        X = np.zeros((N, k, p), dtype=np.float32)
        Y = np.zeros((N, k, p), dtype=np.float32)

        for i in range(N):
            cumsum = 0
            for t in range(k):
                shift = int(sequence[i, t])
                X[i, t, :] = np.roll(template, shift)
                cumsum = (cumsum + shift) % p
                Y[i, t, :] = np.roll(template, cumsum)

        if not return_all_outputs:
            Y = Y[:, -1, :]
        else:
            Y = Y[:, 1:, :]

        return X, Y, sequence


def build_modular_addition_sequence_dataset_generic(
    template: np.ndarray,
    k: int,
    group,
    mode: str = "sampled",
    num_samples: int = 65536,
    return_all_outputs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build generic group composition dataset for sequence length k.

    Works with any escnn group that has a regular representation.
    For a sequence of k group elements (g1, g2, ..., gk), we compute:
    - X[i, t, :] = regular_rep(g_t) @ template  (template transformed by g_t)
    - Y[i, :] = regular_rep(g1 * g2 * ... * gk) @ template  (template transformed by composition)

    Args:
        template: (group_order,) template array
        k: sequence length (number of group elements to compose)
        group: escnn group object (e.g., Octahedral(), Icosahedral())
        mode: "sampled" or "exhaustive"
        num_samples: number of samples for "sampled" mode
        return_all_outputs: if True, return intermediate outputs after each composition

    Returns:
        X: (N, k, group_order) input sequences
        Y: (N, group_order) or (N, k-1, group_order) target outputs
        sequence: (N, k) integer indices of group elements per token
    """
    group_order = group.order()

    assert template.shape == (group_order,), (
        f"template must be ({group_order},), got {template.shape}"
    )

    # Get regular representation and list of elements
    regular_rep = group.representations["regular"]
    elements = list(group.elements)
    n_elements = len(elements)

    # Pre-compute representation matrices for all elements
    rep_matrices = np.array([regular_rep(g) for g in elements])

    if mode == "exhaustive":
        total = n_elements**k
        if total > 1_000_000:
            raise ValueError(f"n_elements^k = {total} is huge; use mode='sampled' instead.")
        N = total

        # Generate all possible sequences of k element indices
        sequence = np.zeros((N, k), dtype=np.int64)
        for idx in range(N):
            for t in range(k):
                sequence[idx, t] = (idx // (n_elements**t)) % n_elements
    else:
        N = int(num_samples)
        sequence = np.random.randint(0, n_elements, size=(N, k), dtype=np.int64)

    # Initialize output arrays
    X = np.zeros((N, k, group_order), dtype=np.float32)
    Y = np.zeros((N, k, group_order), dtype=np.float32)

    for i in range(N):
        cumulative_rep = np.eye(group_order)

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


def sequence_to_paths_xy(sequence_xy: np.ndarray, p1: int, p2: int) -> np.ndarray:
    """
    Convert a sequence of group elements (ax_t, ay_t) into cumulative positions
    after each token, modulo (p1, p2).

    Args:
        sequence_xy: (N, k, 2) integers, NOT cumulative
        p1, p2: moduli for rows/cols

    Returns:
        paths_xy: (N, k, 2) where paths_xy[n, t] = (sum_{u<=t} ax_u mod p1, sum_{u<=t} ay_u mod p2)
    """
    seq = sequence_xy.astype(np.int64, copy=False)
    N, k, _ = seq.shape
    paths_xy = np.empty_like(seq)

    # cumulative along time axis, modulo each dimension
    paths_xy[:, :, 0] = np.mod(np.cumsum(seq[:, :, 0], axis=1, dtype=np.int64), p1)
    paths_xy[:, :, 1] = np.mod(np.cumsum(seq[:, :, 1], axis=1, dtype=np.int64), p2)
    return paths_xy


# ---------------------------------------------------------------------------
# Dataset functions moved from datasets.py
# ---------------------------------------------------------------------------


def group_dataset(group, template):
    """Generate a dataset of group elements acting on the template.

    Using the regular representation.

    Parameters
    ----------
    group : Group (escnn object)
        The group.
    template : np.ndarray, shape=[group.order()]
        The template to generate the dataset from.

    Returns
    -------
    X : np.ndarray, shape=[group.order()**2, 2, group.order()]
    Y : np.ndarray, shape=[group.order()**2, group.order()]
    """
    group_order = group.order()
    assert len(template) == group_order, "template must have the same length as the group order"
    n_samples = group_order**2
    X = np.zeros((n_samples, 2, group_order))
    Y = np.zeros((n_samples, group_order))
    regular_rep = group.representations["regular"]

    idx = 0
    for g1 in group.elements:
        for g2 in group.elements:
            g1_rep = regular_rep(g1)
            g2_rep = regular_rep(g2)
            g12_rep = g1_rep @ g2_rep

            X[idx, 0, :] = g1_rep @ template
            X[idx, 1, :] = g2_rep @ template
            Y[idx, :] = g12_rep @ template
            idx += 1

    return X, Y


def cn_dataset(template):
    """Generate a dataset for the cyclic group C_n modular addition operation."""
    group_size = len(template)
    X = np.zeros((group_size * group_size, 2, group_size))
    Y = np.zeros((group_size * group_size, group_size))

    idx = 0
    for a in range(group_size):
        for b in range(group_size):
            q = (a + b) % group_size
            X[idx, 0, :] = np.roll(template, a)
            X[idx, 1, :] = np.roll(template, b)
            Y[idx, :] = np.roll(template, q)
            idx += 1

    return X, Y


def cnxcn_dataset(template):
    r"""Generate a dataset for the 2D modular addition operation.

    Parameters
    ----------
    template : np.ndarray
        A flattened 2D square image of shape (image_length*image_length,).

    Returns
    -------
    X : np.ndarray
        Input data of shape (image_length^4, 2, image_length*image_length).
    Y : np.ndarray
        Output data of shape (image_length^4, image_length*image_length).
    """
    image_length = int(np.sqrt(len(template)))
    X = np.zeros((image_length**4, 2, image_length * image_length))
    Y = np.zeros((image_length**4, image_length * image_length))
    translations = np.zeros((image_length**4, 3, 2), dtype=int)

    idx = 0
    template_2d = template.reshape((image_length, image_length))
    for a_x in range(image_length):
        for a_y in range(image_length):
            for b_x in range(image_length):
                for b_y in range(image_length):
                    q_x = (a_x + b_x) % image_length
                    q_y = (a_y + b_y) % image_length
                    X[idx, 0, :] = np.roll(np.roll(template_2d, a_x, axis=0), a_y, axis=1).flatten()
                    X[idx, 1, :] = np.roll(np.roll(template_2d, b_x, axis=0), b_y, axis=1).flatten()
                    Y[idx, :] = np.roll(np.roll(template_2d, q_x, axis=0), q_y, axis=1).flatten()
                    translations[idx, 0, :] = (a_x, a_y)
                    translations[idx, 1, :] = (b_x, b_y)
                    translations[idx, 2, :] = (q_x, q_y)
                    idx += 1

    return X, Y


def move_dataset_to_device_and_flatten(X, Y, device=None):
    """Move dataset tensors to available or specified device.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (num_samples, 2, p*p)
    Y : np.ndarray
        Target data of shape (num_samples, p*p)
    device : torch.device, optional
        Device to move tensors to.

    Returns
    -------
    X : torch.Tensor
        Input data tensor on specified device, flattened to (num_samples, 2*p*p)
    Y : torch.Tensor
        Target data tensor on specified device, flattened to (num_samples, p*p)
    """
    num_data_features = len(X[0][0])
    X_flat = X.reshape(X.shape[0], 2 * num_data_features)
    Y_flat = Y.reshape(Y.shape[0], num_data_features)
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_flat, dtype=torch.float32)

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available. Using CUDA.")
        else:
            device = torch.device("cpu")
            print("GPU is not available. Using CPU.")

    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    return X_tensor, Y_tensor, device
