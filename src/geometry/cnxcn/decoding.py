"""Signal decoders and errors for square-torus geometry."""

import numpy as np

from .core import periodic_distance_squared


def decode_spatial_argmax(group, signal: np.ndarray) -> tuple[int, int]:
    """Decode a signal by the maximum grid entry."""
    return tuple(
        int(value) for value in np.unravel_index(np.argmax(signal), (group.p1, group.p2))
    )


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
