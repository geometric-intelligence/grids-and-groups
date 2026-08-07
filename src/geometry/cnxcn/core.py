"""Core square-torus geometry for ``C_n × C_n`` experiments."""

import numpy as np


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
