"""Triangular-lattice geometry for ``Z_n² ⋊ C_3`` experiments."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors


def triangular_coordinates(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Euclidean centers for an ``n × n`` triangular lattice."""
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return x + 0.5 * y, (np.sqrt(3) / 2) * y


def signal_to_tensor(group, signal: np.ndarray) -> np.ndarray:
    """Reshape a flat group signal to ``(rotation, x, y)``."""
    signal = np.asarray(signal)
    if signal.shape != (group.order,):
        raise ValueError(f"signal must have shape ({group.order},), got {signal.shape}")
    return signal.reshape(group.m, group.n, group.n)


def tensor_to_signal(group, tensor: np.ndarray) -> np.ndarray:
    """Flatten a ``(rotation, x, y)`` tensor using the group index convention."""
    tensor = np.asarray(tensor)
    expected = (group.m, group.n, group.n)
    if tensor.shape != expected:
        raise ValueError(f"tensor must have shape {expected}, got {tensor.shape}")
    return tensor.reshape(-1)


def align_rotation_slice(group, values: np.ndarray, rotation: int) -> np.ndarray:
    """Move a heading-relative slice into the allocentric lattice frame."""
    values = np.asarray(values)
    if values.shape != (group.n, group.n):
        raise ValueError(
            f"rotation slice must have shape ({group.n}, {group.n}), got {values.shape}"
        )
    matrix = group.rotation_matrix(rotation)
    x, y = np.meshgrid(np.arange(group.n), np.arange(group.n), indexing="ij")
    x_rotated = (matrix[0, 0] * x + matrix[0, 1] * y) % group.n
    y_rotated = (matrix[1, 0] * x + matrix[1, 1] * y) % group.n
    aligned = np.empty_like(values)
    aligned[x_rotated, y_rotated] = values
    return aligned


def align_rotation_slices(group, tensor: np.ndarray) -> np.ndarray:
    """Align all rotation slices of a group-signal tensor."""
    tensor = np.asarray(tensor)
    expected = (group.m, group.n, group.n)
    if tensor.shape != expected:
        raise ValueError(f"tensor must have shape {expected}, got {tensor.shape}")
    return np.asarray(
        [align_rotation_slice(group, tensor[r], r) for r in range(group.m)]
    )


def spatial_marginal(
    group,
    signal: np.ndarray,
    *,
    align_rotations: bool = False,
    reduction: str = "sum",
) -> np.ndarray:
    """Reduce a group signal over its rotation coordinate."""
    tensor = signal_to_tensor(group, signal)
    if align_rotations:
        tensor = align_rotation_slices(group, tensor)
    if reduction == "sum":
        return tensor.sum(axis=0)
    if reduction == "mean":
        return tensor.mean(axis=0)
    raise ValueError("reduction must be 'sum' or 'mean'")


def periodic_distance_squared(
    n: int,
    point: tuple[int, int],
    center: tuple[int, int],
) -> float:
    """Shortest squared Euclidean distance in the periodic triangular lattice."""
    x, y = point
    center_x, center_y = center
    return min(
        (x - center_x - shift_x * n) ** 2
        + (x - center_x - shift_x * n) * (y - center_y - shift_y * n)
        + (y - center_y - shift_y * n) ** 2
        for shift_x in (-1, 0, 1)
        for shift_y in (-1, 0, 1)
    )


def gaussian_bump(
    group,
    *,
    center: tuple[int, int] = (2, 2),
    sigma: float = 1.2,
    amplitude: float = 1.0,
    baseline: float = 0.0,
) -> np.ndarray:
    """Return a spatial Gaussian copied across every rotation slice."""
    spatial = np.empty((group.n, group.n), dtype=float)
    for x in range(group.n):
        for y in range(group.n):
            distance_squared = periodic_distance_squared(group.n, (x, y), center)
            spatial[x, y] = baseline + amplitude * np.exp(
                -0.5 * distance_squared / sigma**2
            )
    return tensor_to_signal(group, np.repeat(spatial[None, :, :], group.m, axis=0))


def decode_spatial_argmax(group, signal: np.ndarray) -> tuple[int, int]:
    """Decode position after summing the signal over rotations."""
    spatial = spatial_marginal(group, signal)
    return tuple(int(value) for value in np.unravel_index(np.argmax(spatial), spatial.shape))


def transformed_center(
    group,
    element: int,
    original_center: tuple[int, int],
) -> tuple[int, int]:
    """Return the center obtained by applying ``element`` to a spatial bump."""
    translation_x, translation_y, rotation = group.decode(element)
    center_x, center_y = group.apply_rotation(rotation, *original_center)
    return (
        (translation_x + center_x) % group.n,
        (translation_y + center_y) % group.n,
    )


def plot_lattice_scalar(
    values: np.ndarray,
    *,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
):
    """Plot a scalar field using one consistent triangular coordinate system."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"values must be a square two-dimensional array, got {values.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    x, y = triangular_coordinates(values.shape[0])
    norm = mcolors.Normalize(
        vmin=float(values.min()) if vmin is None else vmin,
        vmax=float(values.max()) if vmax is None else vmax,
    )
    artist = ax.scatter(
        x.ravel(),
        y.ravel(),
        c=values.ravel(),
        cmap=cmap,
        norm=norm,
        marker="h",
        s=650,
        linewidths=0.4,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_group_signal(
    group,
    signal: np.ndarray,
    *,
    title: str | None = None,
    align_rotations: bool = False,
    reduction: str | None = None,
    cmap: str = "viridis",
):
    """Plot rotation slices or one rotation-reduced spatial field."""
    tensor = signal_to_tensor(group, signal)
    if align_rotations:
        tensor = align_rotation_slices(group, tensor)
    if reduction is not None:
        if reduction == "sum":
            values = tensor.sum(axis=0)
        elif reduction == "mean":
            values = tensor.mean(axis=0)
        else:
            raise ValueError("reduction must be None, 'sum', or 'mean'")
        return plot_lattice_scalar(values, title=title, cmap=cmap)

    figure, axes = plt.subplots(
        1, group.m, figsize=(4.5 * group.m, 4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    vmin, vmax = float(tensor.min()), float(tensor.max())
    for rotation, ax in enumerate(axes):
        plot_lattice_scalar(
            tensor[rotation],
            ax=ax,
            title=f"rotation {rotation}",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=rotation == group.m - 1,
        )
    if title:
        figure.suptitle(title)
    return axes
