"""Signal encodings for square-torus geometry."""

import numpy as np

from .core import grid_to_signal, periodic_delta


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
