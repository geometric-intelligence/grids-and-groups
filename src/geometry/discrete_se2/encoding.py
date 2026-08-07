"""Signal encoders for discrete SE(2) geometry."""

import numpy as np

from .core import periodic_distance_squared, tensor_to_signal


def gaussian_bump(
    group,
    *,
    center: tuple[int, int] = (2, 2),
    sigma: float = 1.2,
    orientation_weights: np.ndarray | None = None,
    amplitude: float = 1.0,
    baseline: float = 0.0,
) -> np.ndarray:
    """Return a periodic Gaussian with an optional orientation profile."""
    spatial = np.empty((group.n, group.n), dtype=float)
    for x in range(group.n):
        for y in range(group.n):
            distance_squared = periodic_distance_squared(group.n, (x, y), center)
            spatial[x, y] = baseline + amplitude * np.exp(
                -0.5 * distance_squared / sigma**2
            )
    if orientation_weights is None:
        orientation_weights = np.ones(group.m)
    orientation_weights = np.asarray(orientation_weights, dtype=float)
    if orientation_weights.shape != (group.m,):
        raise ValueError(
            f"orientation_weights must have shape ({group.m},), "
            f"got {orientation_weights.shape}"
        )
    tensor = baseline + orientation_weights[:, None, None] * (spatial - baseline)
    return tensor_to_signal(group, tensor)
