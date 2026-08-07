"""Square-torus geometry for ``C_n × C_n`` experiments."""

from .core import (
    grid_to_signal,
    periodic_delta,
    periodic_distance_squared,
    signal_to_grid,
    transformed_center,
)
from .decoding import center_errors, decode_spatial_argmax
from .encoding import gaussian_bump
from .plotting import TRACK_COLOR, plot_grid_scalar, plot_grid_trajectory
from .trajectories import make_momentum_motion_sequence

__all__ = [
    "TRACK_COLOR",
    "center_errors",
    "decode_spatial_argmax",
    "gaussian_bump",
    "grid_to_signal",
    "make_momentum_motion_sequence",
    "periodic_delta",
    "periodic_distance_squared",
    "plot_grid_scalar",
    "plot_grid_trajectory",
    "signal_to_grid",
    "transformed_center",
]
