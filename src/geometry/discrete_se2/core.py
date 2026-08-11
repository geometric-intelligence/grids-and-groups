"""Core triangular-lattice geometry for ``Z_n² ⋊ C_3`` experiments."""

import numpy as np


def triangular_coordinates(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Euclidean centers for an ``n × n`` triangular lattice."""
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return x + 0.5 * y, (np.sqrt(3) / 2) * y


def centered_triangular_coordinates(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return axial lattice centers using residues centered around zero."""
    indices = np.arange(n)
    centered = (indices + n // 2) % n - n // 2
    x, y = np.meshgrid(centered, centered, indexing="ij")
    return x + 0.5 * y, (np.sqrt(3) / 2) * y


def offset_coordinates(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a wrapped rectangular display of an ``n × n`` triangular lattice."""
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    column = (x + y // 2) % n
    return column + 0.5 * (y % 2), (np.sqrt(3) / 2) * y


def lattice_coordinates(
    n: int, *, mode: str = "offset"
) -> tuple[np.ndarray, np.ndarray]:
    """Return lattice centers in a wrapped, raw, or identity-centered display."""
    if mode == "offset":
        return offset_coordinates(n)
    if mode == "axial":
        return triangular_coordinates(n)
    if mode == "centered_axial":
        return centered_triangular_coordinates(n)
    raise ValueError("mode must be 'offset', 'axial', or 'centered_axial'")


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


def transformed_pose(
    group,
    element: int,
    original_pose: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Apply an element to a pose using the semidirect-product law."""
    return group.decode(group.compose(element, group.encode(*original_pose)))


def advanced_pose(
    group,
    original_pose: tuple[int, int, int],
    body_motion: int,
) -> tuple[int, int, int]:
    """Advance a pose by a body-frame motion using ``pose * motion``."""
    return group.decode(group.compose(group.encode(*original_pose), body_motion))


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
