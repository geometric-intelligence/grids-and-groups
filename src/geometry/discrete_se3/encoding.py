"""Signal encoders for discrete SE(3) groups."""

import numpy as np

from .core import periodic_delta, tensor_to_signal


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
