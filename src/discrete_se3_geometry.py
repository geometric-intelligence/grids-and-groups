"""Cubic-lattice geometry for ``Z_n³ ⋊ O`` experiments."""

import matplotlib.pyplot as plt
import numpy as np


def signal_to_tensor(group, signal: np.ndarray) -> np.ndarray:
    """Reshape a flat signal to ``(rotation, x, y, z)``."""
    signal = np.asarray(signal)
    if signal.shape != (group.order,):
        raise ValueError(f"signal must have shape ({group.order},), got {signal.shape}")
    return signal.reshape(group.num_rotations, group.n, group.n, group.n)


def tensor_to_signal(group, tensor: np.ndarray) -> np.ndarray:
    """Flatten a ``(rotation, x, y, z)`` tensor using group index order."""
    tensor = np.asarray(tensor)
    expected = (group.num_rotations, group.n, group.n, group.n)
    if tensor.shape != expected:
        raise ValueError(f"tensor must have shape {expected}, got {tensor.shape}")
    return tensor.reshape(-1)


def periodic_delta(n: int, coordinate: int, center: int) -> int:
    """Shortest signed displacement on one cyclic coordinate."""
    delta = (int(coordinate) - int(center)) % n
    return delta - n if delta > n // 2 else delta


def periodic_distance_squared(
    n: int,
    point: tuple[int, int, int],
    center: tuple[int, int, int],
) -> float:
    """Squared Euclidean distance on the periodic cubic lattice."""
    delta = np.asarray(
        [periodic_delta(n, coordinate, origin) for coordinate, origin in zip(point, center)]
    )
    return float(delta @ delta)


def gaussian_landmark(
    group,
    *,
    center: tuple[int, int, int],
    sigma: float | tuple[float, float, float] = 0.75,
    orientation_weights: np.ndarray | None = None,
    amplitude: float = 1.0,
    baseline: float = 0.0,
) -> np.ndarray:
    """Return an anisotropic periodic Gaussian with an orientation profile.

    Passing unequal values in ``sigma`` makes spatial rotations observable.
    Passing nonuniform ``orientation_weights`` makes the rotation coordinate
    directly observable.
    """
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (3,))
    if np.any(sigma <= 0):
        raise ValueError("sigma values must be positive")
    if orientation_weights is None:
        orientation_weights = np.ones(group.num_rotations)
    orientation_weights = np.asarray(orientation_weights, dtype=float)
    if orientation_weights.shape != (group.num_rotations,):
        raise ValueError(
            f"orientation_weights must have shape ({group.num_rotations},), "
            f"got {orientation_weights.shape}"
        )

    spatial = np.empty((group.n, group.n, group.n), dtype=float)
    for x in range(group.n):
        for y in range(group.n):
            for z in range(group.n):
                delta = np.asarray(
                    [
                        periodic_delta(group.n, coordinate, origin)
                        for coordinate, origin in zip((x, y, z), center)
                    ],
                    dtype=float,
                )
                spatial[x, y, z] = np.exp(-0.5 * np.sum((delta / sigma) ** 2))
    tensor = baseline + amplitude * orientation_weights[:, None, None, None] * spatial
    return tensor_to_signal(group, tensor)


def peaked_orientation_weights(
    group,
    *,
    rotation: int = 0,
    peak: float = 1.0,
    floor: float = 0.05,
) -> np.ndarray:
    """Return a simple orientation profile peaked at one cubic rotation."""
    weights = np.full(group.num_rotations, floor, dtype=float)
    weights[int(rotation) % group.num_rotations] = peak
    return weights


def align_rotation_slice(group, values: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate one heading-relative volume into the allocentric frame."""
    values = np.asarray(values)
    expected = (group.n, group.n, group.n)
    if values.shape != expected:
        raise ValueError(f"rotation slice must have shape {expected}, got {values.shape}")
    matrix = group.rotation_matrix(rotation)
    coordinates = np.indices(expected).reshape(3, -1)
    rotated = (matrix @ coordinates) % group.n
    aligned = np.empty_like(values)
    aligned[tuple(rotated)] = values.reshape(-1)
    return aligned


def align_rotation_slices(group, tensor: np.ndarray) -> np.ndarray:
    """Align all orientation slices to a common allocentric frame."""
    tensor = np.asarray(tensor)
    expected = (group.num_rotations, group.n, group.n, group.n)
    if tensor.shape != expected:
        raise ValueError(f"tensor must have shape {expected}, got {tensor.shape}")
    return np.asarray(
        [
            align_rotation_slice(group, tensor[rotation], rotation)
            for rotation in range(group.num_rotations)
        ]
    )


def spatial_marginal(
    group,
    signal: np.ndarray,
    *,
    align_rotations: bool = False,
) -> np.ndarray:
    """Sum a group signal over its rotation coordinate."""
    tensor = signal_to_tensor(group, signal)
    if align_rotations:
        tensor = align_rotation_slices(group, tensor)
    return tensor.sum(axis=0)


def orientation_marginal(group, signal: np.ndarray) -> np.ndarray:
    """Sum a group signal over translation coordinates."""
    return signal_to_tensor(group, signal).sum(axis=(1, 2, 3))


def decode_spatial_argmax(group, signal: np.ndarray) -> tuple[int, int, int]:
    """Decode the spatial center after marginalizing over rotations."""
    spatial = spatial_marginal(group, signal)
    return tuple(int(value) for value in np.unravel_index(np.argmax(spatial), spatial.shape))


def decode_orientation_argmax(group, signal: np.ndarray) -> int:
    """Decode the most active cubic rotation."""
    return int(np.argmax(orientation_marginal(group, signal)))


def decode_pose(group, signal: np.ndarray) -> tuple[int, int, int, int]:
    """Decode spatial and orientation marginals as one pose tuple."""
    return (*decode_spatial_argmax(group, signal), decode_orientation_argmax(group, signal))


def transformed_pose(
    group,
    element: int,
    original_pose: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Apply ``element`` to a pose using the exact semidirect-product law."""
    pose_element = group.encode(*original_pose)
    return group.decode(group.compose(element, pose_element))


def rotation_error(group, predicted: int, target: int) -> float:
    """Geodesic rotation-angle error in radians between two cubic rotations."""
    predicted_matrix = group.rotation_matrix(predicted)
    target_matrix = group.rotation_matrix(target)
    relative = predicted_matrix @ target_matrix.T
    cosine = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
    return float(np.arccos(cosine))


def plot_orthogonal_slices(
    volume: np.ndarray,
    *,
    center: tuple[int, int, int] | None = None,
    title: str | None = None,
    cmap: str = "viridis",
):
    """Plot orthogonal slices through a cubic scalar volume."""
    volume = np.asarray(volume)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"volume must be cubic, got {volume.shape}")
    if center is None:
        center = tuple(int(value) for value in np.unravel_index(np.argmax(volume), volume.shape))
    x, y, z = center
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    slices = (
        (volume[:, :, z], f"xy at z={z}"),
        (volume[:, y, :], f"xz at y={y}"),
        (volume[x, :, :], f"yz at x={x}"),
    )
    for ax, (values, label) in zip(axes, slices):
        image = ax.imshow(values.T, origin="lower", cmap=cmap)
        ax.set_title(label)
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if title:
        figure.suptitle(title)
    return axes


def plot_orientation_marginal(
    group,
    signal: np.ndarray,
    *,
    ax=None,
    title: str = "Orientation marginal",
):
    """Plot activity across the 24 cubic rotations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    values = orientation_marginal(group, signal)
    ax.bar(np.arange(group.num_rotations), values)
    ax.set_xlabel("Rotation index")
    ax.set_ylabel("Summed activity")
    ax.set_title(title)
    return ax


def plot_trajectory(
    positions: np.ndarray,
    *,
    ax=None,
    title: str = "Decoded spatial trajectory",
):
    """Plot a three-dimensional sequence of decoded positions."""
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (steps, 3), got {positions.shape}")
    if ax is None:
        figure = plt.figure(figsize=(6, 5), constrained_layout=True)
        ax = figure.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    return ax
