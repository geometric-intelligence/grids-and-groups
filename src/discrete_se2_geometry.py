"""Triangular-lattice geometry for ``Z_n² ⋊ C_3`` experiments."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon


def triangular_coordinates(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Euclidean centers for an ``n × n`` triangular lattice."""
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return x + 0.5 * y, (np.sqrt(3) / 2) * y


def offset_coordinates(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a wrapped rectangular display of an ``n × n`` triangular lattice."""
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    column = (x + y // 2) % n
    return column + 0.5 * (y % 2), (np.sqrt(3) / 2) * y


def lattice_coordinates(
    n: int, *, mode: str = "offset"
) -> tuple[np.ndarray, np.ndarray]:
    """Return lattice centers in a wrapped-offset or raw axial display."""
    if mode == "offset":
        return offset_coordinates(n)
    if mode == "axial":
        return triangular_coordinates(n)
    raise ValueError("mode must be 'offset' or 'axial'")


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
    coordinate_mode: str = "offset",
):
    """Plot a scalar field as a tightly packed triangular lattice of hexagons."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"values must be a square two-dimensional array, got {values.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    x, y = lattice_coordinates(values.shape[0], mode=coordinate_mode)
    norm = mcolors.Normalize(
        vmin=float(values.min()) if vmin is None else vmin,
        vmax=float(values.max()) if vmax is None else vmax,
    )
    patches = [
        RegularPolygon(
            (center_x, center_y),
            numVertices=6,
            radius=1 / np.sqrt(3),
            orientation=np.pi / 6,
        )
        for center_x, center_y in zip(x.ravel(), y.ravel())
    ]
    artist = PatchCollection(
        patches,
        array=values.ravel(),
        cmap=cmap,
        norm=norm,
        edgecolor=(0.15, 0.15, 0.15, 0.35),
        linewidth=0.25,
    )
    ax.add_collection(artist)
    radius = 1 / np.sqrt(3)
    ax.set_xlim(float(x.min()) - radius, float(x.max()) + radius)
    ax.set_ylim(float(y.min()) - radius, float(y.max()) + radius)
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
    coordinate_mode: str = "offset",
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
        return plot_lattice_scalar(
            values,
            title=title,
            cmap=cmap,
            coordinate_mode=coordinate_mode,
        )

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
            coordinate_mode=coordinate_mode,
        )
    if title:
        figure.suptitle(title)
    return axes


def lattice_path_coordinates(
    points: np.ndarray, n: int, *, mode: str = "offset"
) -> np.ndarray:
    """Map integer lattice points to display coordinates."""
    points = np.asarray(points, dtype=int)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (steps, 2), got {points.shape}")
    x, y = lattice_coordinates(n, mode=mode)
    wrapped = points % n
    return np.column_stack(
        (x[wrapped[:, 0], wrapped[:, 1]], y[wrapped[:, 0], wrapped[:, 1]])
    )


TRACK_COLOR = "#E45756"


def make_momentum_motion_sequence(
    group,
    *,
    steps: int = 100,
    seed: int = 0,
    include_rotations: bool = False,
    momentum: bool = True,
    turn_probability: float = 0.18,
    stay_probability: float = 0.04,
    start_xy: tuple[int, int] | None = None,
    margin: int = 2,
    max_resample: int = 20,
) -> np.ndarray:
    """Generate local relative motions while keeping the displayed path in bounds."""
    rng = np.random.default_rng(seed)
    n = group.n
    if start_xy is None:
        start_xy = (n // 2, n // 2)
    x = int(np.clip(start_xy[0], margin, n - 1 - margin))
    y = int(np.clip(start_xy[1], margin, n - 1 - margin))
    directions = np.asarray(
        [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    )

    def inside_bounds(x_new, y_new):
        return (
            margin <= x_new <= n - 1 - margin
            and margin <= y_new <= n - 1 - margin
        )

    sequence = [group.encode(x, y, 0)]
    direction_index = int(rng.integers(0, len(directions)))
    for _ in range(steps - 1):
        if rng.random() < stay_probability:
            dx, dy = 0, 0
        else:
            accepted = False
            for _ in range(max_resample):
                proposed = direction_index
                if not momentum or rng.random() < turn_probability:
                    proposed = (direction_index + rng.choice([-1, 1])) % len(
                        directions
                    )
                dx, dy = directions[proposed]
                if inside_bounds(x + dx, y + dy):
                    direction_index = proposed
                    accepted = True
                    break
                direction_index = (
                    direction_index + rng.choice([-1, 1])
                ) % len(directions)
            if not accepted:
                dx, dy = 0, 0
        x += int(dx)
        y += int(dy)
        rotation = int(rng.choice([0, 0, 0, 1, 2])) if include_rotations else 0
        sequence.append(group.encode(int(dx), int(dy), rotation))
    return np.asarray(sequence, dtype=int)


def true_centers_from_cumulative_states(
    group,
    original_center: tuple[int, int],
    cumulative_states: np.ndarray,
) -> np.ndarray:
    """Decode exact bump centers from cumulative group elements."""
    return np.asarray(
        [
            transformed_center(group, int(state), original_center)
            for state in cumulative_states
        ],
        dtype=int,
    )


def decode_centers_from_outputs(
    group, outputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Decode spatial centers and direction marginals from output signals."""
    centers = []
    direction_marginals = []
    for output in outputs:
        tensor = signal_to_tensor(group, output)
        direction_marginals.append(tensor.sum(axis=(1, 2)))
        centers.append(decode_spatial_argmax(group, output))
    return np.asarray(centers, dtype=int), np.asarray(direction_marginals)


def center_errors_periodic_triangular(
    group, predicted: np.ndarray, exact: np.ndarray
) -> np.ndarray:
    """Return shortest triangular-lattice distances between decoded centers."""
    return np.asarray(
        [
            np.sqrt(
                periodic_distance_squared(
                    group.n,
                    tuple(int(value) for value in pred),
                    tuple(int(value) for value in target),
                )
            )
            for pred, target in zip(predicted, exact)
        ]
    )


def plot_lattice_trajectory(
    group,
    exact_points: np.ndarray,
    predicted_points: np.ndarray,
    *,
    title: str = "Bump center trajectory",
    coordinate_mode: str = "offset",
    save_path: str | None = None,
):
    """Plot a tracked trajectory using the original neutral-lattice aesthetic."""
    figure, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    x, y = lattice_coordinates(group.n, mode=coordinate_mode)
    patches = [
        RegularPolygon(
            (center_x, center_y),
            numVertices=6,
            radius=1 / np.sqrt(3),
            orientation=np.pi / 6,
        )
        for center_x, center_y in zip(x.ravel(), y.ravel())
    ]
    ax.add_collection(
        PatchCollection(
            patches,
            facecolor=(0.92, 0.92, 0.92, 1.0),
            edgecolor=(1.0, 1.0, 1.0, 1.0),
            linewidth=0.7,
        )
    )
    radius = 1 / np.sqrt(3)
    ax.set_xlim(float(x.min()) - radius, float(x.max()) + radius)
    ax.set_ylim(float(y.min()) - radius, float(y.max()) + radius)

    exact_xy = lattice_path_coordinates(
        exact_points, group.n, mode=coordinate_mode
    )
    predicted_xy = lattice_path_coordinates(
        predicted_points, group.n, mode=coordinate_mode
    )
    ax.scatter(
        exact_xy[:, 0],
        exact_xy[:, 1],
        s=40,
        color=TRACK_COLOR,
        alpha=0.40,
        linewidths=0,
        label="true bump path",
        zorder=2,
    )
    ax.plot(
        exact_xy[:, 0],
        exact_xy[:, 1],
        color=TRACK_COLOR,
        linewidth=2.4,
        alpha=0.95,
        label="true bump center",
        zorder=3,
    )
    ax.plot(
        predicted_xy[:, 0],
        predicted_xy[:, 1],
        "k--",
        linewidth=2.0,
        alpha=0.95,
        label="predicted theory peak",
        zorder=4,
    )
    ax.scatter(
        *exact_xy[0],
        s=120,
        marker="o",
        color=TRACK_COLOR,
        edgecolors="black",
        linewidths=0.8,
        label="start",
        zorder=5,
    )
    ax.scatter(
        *exact_xy[-1],
        s=160,
        marker="*",
        color=TRACK_COLOR,
        edgecolors="black",
        linewidths=0.8,
        label="end",
        zorder=6,
    )
    ax.set(aspect="equal", xticks=[], yticks=[], title=title)
    ax.set_frame_on(False)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.94, 0.94))
    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight", dpi=300)
    return ax
