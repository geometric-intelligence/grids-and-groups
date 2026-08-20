"""Triangular-lattice geometry for ``Z_n² ⋊ C_3`` experiments."""

from .core import (
    advanced_pose,
    align_rotation_slice,
    align_rotation_slices,
    lattice_coordinates,
    lattice_path_coordinates,
    offset_coordinates,
    periodic_distance_squared,
    periodic_spatial_autocorrelation,
    signal_to_tensor,
    tensor_to_signal,
    transformed_center,
    transformed_pose,
    triangular_coordinates,
)
from .decoding import (
    center_errors_periodic_triangular,
    decode_centers_from_outputs,
    decode_orientation_argmax,
    decode_pose,
    decode_spatial_argmax,
    orientation_marginal,
    spatial_marginal,
    true_centers_from_cumulative_states,
)
from .encoding import gaussian_bump
from .plotting import (
    linked_plotly_html,
    plot_group_signal,
    plot_lattice_scalar,
    plot_lattice_trajectory,
    plotly_heading_stacks,
)
from .trajectories import TRACK_COLOR, make_momentum_motion_sequence

__all__ = [
    "advanced_pose",
    "TRACK_COLOR",
    "align_rotation_slice",
    "align_rotation_slices",
    "center_errors_periodic_triangular",
    "decode_centers_from_outputs",
    "decode_orientation_argmax",
    "decode_pose",
    "decode_spatial_argmax",
    "gaussian_bump",
    "lattice_coordinates",
    "lattice_path_coordinates",
    "linked_plotly_html",
    "make_momentum_motion_sequence",
    "offset_coordinates",
    "orientation_marginal",
    "periodic_distance_squared",
    "periodic_spatial_autocorrelation",
    "plot_group_signal",
    "plotly_heading_stacks",
    "plot_lattice_scalar",
    "plot_lattice_trajectory",
    "signal_to_tensor",
    "spatial_marginal",
    "tensor_to_signal",
    "transformed_center",
    "transformed_pose",
    "triangular_coordinates",
    "true_centers_from_cumulative_states",
]
