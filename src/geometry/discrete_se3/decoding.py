"""Signal marginals, energies, and decoders for discrete SE(3) groups."""

import numpy as np

from .core import align_rotation_slices, signal_to_tensor


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


def spatial_energy(group, signal: np.ndarray) -> np.ndarray:
    """Return root-mean-square signal magnitude over cubic rotations."""
    tensor = signal_to_tensor(group, signal)
    return np.sqrt(np.mean(np.abs(tensor) ** 2, axis=0))


def orientation_energy(group, signal: np.ndarray) -> np.ndarray:
    """Return root-mean-square signal magnitude for each cubic rotation."""
    tensor = signal_to_tensor(group, signal)
    return np.sqrt(np.mean(np.abs(tensor) ** 2, axis=(1, 2, 3)))


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
