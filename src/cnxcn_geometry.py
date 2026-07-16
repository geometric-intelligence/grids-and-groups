"""Square-torus geometry for ``C_n × C_n`` experiments."""

import matplotlib.pyplot as plt
import numpy as np

TRACK_COLOR = "#E45756"


def signal_to_grid(group, signal: np.ndarray) -> np.ndarray:
    """Reshape a flat group signal to ``(p1, p2)``."""
    signal = np.asarray(signal)
    if signal.shape != (group.order,):
        raise ValueError(f"signal must have shape ({group.order},), got {signal.shape}")
    return signal.reshape(group.p1, group.p2)


def grid_to_signal(group, grid: np.ndarray) -> np.ndarray:
    """Flatten a square-torus field using the group index convention."""
    grid = np.asarray(grid)
    expected = (group.p1, group.p2)
    if grid.shape != expected:
        raise ValueError(f"grid must have shape {expected}, got {grid.shape}")
    return grid.reshape(-1)


def periodic_delta(period: int, coordinate: int, center: int) -> int:
    """Shortest signed displacement on one cyclic coordinate."""
    delta = (int(coordinate) - int(center)) % period
    return delta - period if delta > period // 2 else delta


def periodic_distance_squared(
    group,
    point: tuple[int, int],
    center: tuple[int, int],
) -> float:
    """Squared Euclidean distance on the periodic rectangular lattice."""
    dx = periodic_delta(group.p1, point[0], center[0])
    dy = periodic_delta(group.p2, point[1], center[1])
    return float(dx * dx + dy * dy)


def gaussian_bump(
    group,
    *,
    center: tuple[int, int] = (2, 2),
    sigma: float | tuple[float, float] = 1.0,
    amplitude: float = 1.0,
    baseline: float = 0.0,
) -> np.ndarray:
    """Return a periodic Gaussian encoding on ``C_p1 × C_p2``."""
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (2,))
    if np.any(sigma <= 0):
        raise ValueError("sigma values must be positive")
    grid = np.empty((group.p1, group.p2), dtype=float)
    for x in range(group.p1):
        for y in range(group.p2):
            delta = np.asarray(
                (
                    periodic_delta(group.p1, x, center[0]),
                    periodic_delta(group.p2, y, center[1]),
                )
            )
            grid[x, y] = baseline + amplitude * np.exp(
                -0.5 * np.sum((delta / sigma) ** 2)
            )
    return grid_to_signal(group, grid)


def decode_spatial_argmax(group, signal: np.ndarray) -> tuple[int, int]:
    """Decode a signal by the maximum grid entry."""
    return tuple(
        int(value) for value in np.unravel_index(np.argmax(signal), (group.p1, group.p2))
    )


def transformed_center(
    group,
    element: int,
    original_center: tuple[int, int],
) -> tuple[int, int]:
    """Translate a center by one product-group element."""
    shift_x, shift_y = group.decode(element)
    return (
        (shift_x + original_center[0]) % group.p1,
        (shift_y + original_center[1]) % group.p2,
    )


def plot_grid_scalar(
    values: np.ndarray,
    *,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
):
    """Plot a scalar field as a square array of lattice cells."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"values must be two-dimensional, got {values.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    artist = ax.imshow(
        values.T,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.grid(which="minor", color=(1, 1, 1, 0.32), linewidth=0.3)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    return ax


def make_momentum_motion_sequence(
    group,
    *,
    steps: int = 250,
    seed: int = 1,
    turn_probability: float = 0.18,
    stay_probability: float = 0.04,
    start_xy: tuple[int, int] | None = None,
    margin: int = 2,
    max_resample: int = 20,
) -> np.ndarray:
    """Generate a bounded persistent walk as relative translation elements."""
    rng = np.random.default_rng(seed)
    if start_xy is None:
        start_xy = (group.p1 // 2, group.p2 // 2)
    x = int(np.clip(start_xy[0], margin, group.p1 - 1 - margin))
    y = int(np.clip(start_xy[1], margin, group.p2 - 1 - margin))
    directions = np.asarray([(1, 0), (0, 1), (-1, 0), (0, -1)])

    def inside_bounds(x_new, y_new):
        return (
            margin <= x_new <= group.p1 - 1 - margin
            and margin <= y_new <= group.p2 - 1 - margin
        )

    sequence = [group.encode(x, y)]
    direction_index = int(rng.integers(0, len(directions)))
    for _ in range(steps - 1):
        if rng.random() < stay_probability:
            dx, dy = 0, 0
        else:
            accepted = False
            for _ in range(max_resample):
                proposed = direction_index
                if rng.random() < turn_probability:
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
        sequence.append(group.encode(int(dx), int(dy)))
    return np.asarray(sequence, dtype=int)


def center_errors(
    group, predicted: np.ndarray, exact: np.ndarray
) -> np.ndarray:
    """Return periodic Euclidean errors between decoded centers."""
    return np.asarray(
        [
            np.sqrt(
                periodic_distance_squared(
                    group,
                    tuple(int(value) for value in pred),
                    tuple(int(value) for value in target),
                )
            )
            for pred, target in zip(predicted, exact)
        ]
    )


def plot_grid_trajectory(
    group,
    exact_points: np.ndarray,
    predicted_points: np.ndarray,
    *,
    title: str = "Bump center trajectory",
    save_path: str | None = None,
):
    """Overlay exact and decoded paths on a neutral square lattice."""
    figure, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    background = np.full((group.p1, group.p2), 0.92)
    plot_grid_scalar(
        background,
        ax=ax,
        cmap="Greys_r",
        vmin=0,
        vmax=1,
        colorbar=False,
    )
    exact_points = np.asarray(exact_points)
    predicted_points = np.asarray(predicted_points)
    ax.scatter(
        exact_points[:, 0],
        exact_points[:, 1],
        s=36,
        color=TRACK_COLOR,
        alpha=0.4,
        linewidths=0,
        label="true bump path",
        zorder=2,
    )
    ax.plot(
        exact_points[:, 0],
        exact_points[:, 1],
        color=TRACK_COLOR,
        linewidth=2.4,
        label="true bump center",
        zorder=3,
    )
    ax.plot(
        predicted_points[:, 0],
        predicted_points[:, 1],
        "k--",
        linewidth=2,
        label="predicted theory peak",
        zorder=4,
    )
    ax.scatter(
        *exact_points[0],
        s=120,
        color=TRACK_COLOR,
        edgecolors="black",
        linewidths=0.8,
        label="start",
        zorder=5,
    )
    ax.scatter(
        *exact_points[-1],
        s=160,
        marker="*",
        color=TRACK_COLOR,
        edgecolors="black",
        linewidths=0.8,
        label="end",
        zorder=6,
    )
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight", dpi=300)
    return ax
