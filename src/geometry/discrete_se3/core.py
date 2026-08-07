"""Core signal and geometric operations for discrete SE(3) groups."""

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


def transformed_pose(
    group,
    element: int,
    original_pose: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Apply ``element`` to a pose using the exact semidirect-product law."""
    pose_element = group.encode(*original_pose)
    return group.decode(group.compose(element, pose_element))


def advanced_pose(
    group,
    original_pose: tuple[int, int, int, int],
    body_motion: int,
) -> tuple[int, int, int, int]:
    """Advance a pose by a body-frame motion using ``pose * motion``."""
    pose_element = group.encode(*original_pose)
    return group.decode(group.compose(pose_element, body_motion))


def rotation_error(group, predicted: int, target: int) -> float:
    """Geodesic rotation-angle error in radians between two cubic rotations."""
    predicted_matrix = group.rotation_matrix(predicted)
    target_matrix = group.rotation_matrix(target)
    relative = predicted_matrix @ target_matrix.T
    cosine = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
    return float(np.arccos(cosine))
