"""Cubic-lattice geometry for ``Z_n³ ⋊ O`` experiments."""

from .core import (
    advanced_pose,
    align_rotation_slice,
    align_rotation_slices,
    periodic_delta,
    periodic_distance_squared,
    rotation_error,
    signal_to_tensor,
    tensor_to_signal,
    transformed_pose,
)
from .decoding import (
    decode_orientation_argmax,
    decode_pose,
    decode_spatial_argmax,
    orientation_energy,
    orientation_marginal,
    spatial_energy,
    spatial_marginal,
)
from .encoding import gaussian_landmark, peaked_orientation_weights
from .plotting import (
    plot_orientation_marginal,
    plot_orthogonal_slices,
    plot_pose_trajectory,
    plot_trajectory,
    plot_volume_scatter,
)

__all__ = [
    "advanced_pose",
    "align_rotation_slice",
    "align_rotation_slices",
    "decode_orientation_argmax",
    "decode_pose",
    "decode_spatial_argmax",
    "gaussian_landmark",
    "orientation_energy",
    "orientation_marginal",
    "peaked_orientation_weights",
    "periodic_delta",
    "periodic_distance_squared",
    "plot_orientation_marginal",
    "plot_orthogonal_slices",
    "plot_pose_trajectory",
    "plot_trajectory",
    "plot_volume_scatter",
    "rotation_error",
    "signal_to_tensor",
    "spatial_energy",
    "spatial_marginal",
    "tensor_to_signal",
    "transformed_pose",
]
