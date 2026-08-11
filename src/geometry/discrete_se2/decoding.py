"""Signal decoders and error metrics for discrete SE(2) geometry."""

import numpy as np

from .core import (
    align_rotation_slices,
    periodic_distance_squared,
    signal_to_tensor,
    transformed_center,
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


def decode_spatial_argmax(group, signal: np.ndarray) -> tuple[int, int]:
    """Decode position after summing the signal over rotations."""
    spatial = spatial_marginal(group, signal)
    return tuple(int(value) for value in np.unravel_index(np.argmax(spatial), spatial.shape))


def orientation_marginal(group, signal: np.ndarray) -> np.ndarray:
    """Sum a group signal over translation coordinates."""
    return signal_to_tensor(group, signal).sum(axis=(1, 2))


def decode_orientation_argmax(group, signal: np.ndarray) -> int:
    """Decode the most active discrete orientation."""
    return int(np.argmax(orientation_marginal(group, signal)))


def decode_pose(group, signal: np.ndarray) -> tuple[int, int, int]:
    """Decode position and orientation marginals as one pose."""
    return (*decode_spatial_argmax(group, signal), decode_orientation_argmax(group, signal))


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
