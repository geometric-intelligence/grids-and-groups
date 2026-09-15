# %%
# Percent-format notebook source. Regenerate the paired .ipynb from this file.

# %% [markdown]
# # Figure 7 — accurate 2D path integration
#
# Run the setup once, then run Panels A and B in order. The final cell writes
# `figure_7.svg` to `artifacts/constructed_networks/discrete_se2_c6/paper_figures/`.

# %%
import gc
import sys
import time
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display
from matplotlib.collections import LineCollection

project_root = next(folder for folder in (Path.cwd(), *Path.cwd().parents) if (folder / "src").is_dir())
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.experiments.discrete_se2 import (  # noqa: E402
    DiscreteSE2ExperimentConfig,
    DiscreteSE2RolloutConfig,
    build_discrete_se2_experiment,
    run_discrete_se2_rollout,
)
from src.geometry.discrete_se2.core import lattice_path_coordinates, lattice_path_segments  # noqa: E402
from src.geometry.discrete_se2.plotting import plot_lattice_scalar  # noqa: E402
from src.geometry.discrete_se2.trajectories import NaturalisticMotionConfig  # noqa: E402

plt.rcParams["svg.fonttype"] = "none"

figures = project_root / "artifacts" / "constructed_networks" / "discrete_se2_c6" / "paper_figures"
figures.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Setup — group, network, trajectory, and plotting parameters
#
# The position group is $C_n^2$ and the heading group is $C_6$. Change
# `n_spatial` to change the spatial group size. The five translation
# probabilities sum to one: stay, forward, forward-left/right, backward-left/right,
# and backward. `turn_persistence` controls the probability of retaining the
# previous nonzero turn direction.

# %%
# Group and inputs
n_spatial = 21
n_orientations = 6
initial_pose = (2, 2, 0)
allocentric_encoding = "gaussian space custom orientation"
allocentric_sigma = 1.0
custom_orientation_weights = (1.0, 0.8, 0.4, 0.2, 0.4, 0.8)
encoding_seed = 10
action_side = "right"

# Full closed-form construction
q_rho = 3
amplitude_multipliers = (1.0, 1.0, 1.0)
num_selected_irreps = None
max_hidden_width = None  # None keeps every selected irrep.

# Naturalistic local trajectory
stay_probability = 0.05
forward_probability = 0.625
forward_left_or_right_probability = 0.1175
backward_left_or_right_probability = 0.0125
backward_probability = 0.065
turn_probability = 0.12
turn_persistence = 0.275
wall_lookahead = 3
wall_avoidance_strength = 2.0
minimum_wall_weight = 0.05
periodic_boundaries = False
rollout_steps = 100
rollout_seed = 3
rollout_start_xy = (n_spatial // 2, n_spatial // 2)
rollout_margin = 1

# Figure appearance
figure_size = (10.8, 7.0)
time_colormap_name = "viridis"
grid_facecolor = "#F4F6F8"
grid_edgecolor = "#D4DAE2"
decoded_color = "#59636F"
trajectory_display_shift = (3, 0)  # A torus cut that keeps seed 3 continuous.

experiment_config = DiscreteSE2ExperimentConfig(
    n_spatial=n_spatial,
    n_orientations=n_orientations,
    initial_pose=initial_pose,
    allocentric_encoding=allocentric_encoding,
    sigma=allocentric_sigma,
    custom_orientation_weights=custom_orientation_weights,
    encoding_seed=encoding_seed,
    action_side=action_side,
    irrep_selection="power",
    num_selected_irreps=num_selected_irreps,
    max_hidden_width=max_hidden_width,
    normalize_power_by_dim=True,
    always_include_trivial=True,
    power_ranking="power",
    q_rho=q_rho,
    amplitude_mode="balanced",
    amplitude_multipliers=amplitude_multipliers,
    materialize_mix=False,
)
motion_config = NaturalisticMotionConfig(
    stay_probability=stay_probability,
    forward_probability=forward_probability,
    forward_left_or_right_probability=forward_left_or_right_probability,
    backward_left_or_right_probability=backward_left_or_right_probability,
    backward_probability=backward_probability,
    turn_probability=turn_probability,
    turn_persistence=turn_persistence,
    wall_lookahead=wall_lookahead,
    wall_avoidance_strength=wall_avoidance_strength,
    minimum_wall_weight=minimum_wall_weight,
    periodic_boundaries=periodic_boundaries,
)
rollout_config = DiscreteSE2RolloutConfig(
    steps=rollout_steps,
    seed=rollout_seed,
    margin=rollout_margin,
    start_xy=rollout_start_xy,
    arrow_stride=2,
    snapshot_steps=(),
)

if "experiment" in globals():
    del experiment
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

started = time.perf_counter()
experiment = build_discrete_se2_experiment(experiment_config)
if torch.cuda.is_available():
    experiment.model.to("cuda")
rollout = run_discrete_se2_rollout(experiment, rollout_config, motion_config)
print(
    f"Built n={n_spatial} in {time.perf_counter() - started:.1f} s; "
    f"hidden width = {experiment.model.hidden_dim:,}; "
    f"unique positions = {rollout.unique_positions}/{rollout_steps}."
)

# %% [markdown]
# ## Panel A — position and heading trajectory
#
# The true position and heading use the same time colormap. The dashed path is
# decoded from the recurrent state. The display shift changes only where the
# periodic chart is cut.

# %%
def path_segments(points, n):
    """Return consecutive path segments that do not cross the plotting seam."""
    coordinates = lattice_path_coordinates(points, n, mode="offset")
    segments = np.stack((coordinates[:-1], coordinates[1:]), axis=1)
    keep = np.linalg.norm(np.diff(coordinates, axis=0), axis=1) <= 1.5
    return segments[keep], np.arange(len(segments))[keep] + 1


def heading_segments(steps, heading):
    """Return the horizontal and vertical segments of a mid-step heading trace."""
    boundaries = np.r_[steps[0], (steps[:-1] + steps[1:]) / 2, steps[-1]]
    horizontal = np.stack(
        (np.column_stack((boundaries[:-1], heading)), np.column_stack((boundaries[1:], heading))),
        axis=1,
    )
    vertical = np.stack(
        (np.column_stack((boundaries[1:-1], heading[:-1])), np.column_stack((boundaries[1:-1], heading[1:]))),
        axis=1,
    )
    return np.concatenate((horizontal, vertical)), np.concatenate((steps, steps[1:]))


figure_7 = plt.figure(figsize=figure_size, layout="constrained")
outer_grid = figure_7.add_gridspec(1, 2, width_ratios=(1.8, 1.0), wspace=0.08)
left_grid = outer_grid[0].subgridspec(2, 1, height_ratios=(2.5, 1.0), hspace=0.05)
trajectory_ax = figure_7.add_subplot(left_grid[0])
heading_ax = figure_7.add_subplot(left_grid[1])

steps = np.arange(1, rollout_steps + 1)
time_colormap = plt.colormaps[time_colormap_name]
time_normalization = mcolors.Normalize(steps[0], steps[-1])
display_shift = np.asarray(trajectory_display_shift)
exact_centers = (rollout.exact_centers - display_shift) % n_spatial
predicted_centers = (rollout.predicted_centers - display_shift) % n_spatial

plot_lattice_scalar(
    np.zeros((n_spatial, n_spatial)),
    ax=trajectory_ax,
    cmap=mcolors.ListedColormap([grid_facecolor]),
    vmin=0,
    vmax=1,
    colorbar=False,
    coordinate_mode="offset",
    wrap_periodic_edges=True,
)
trajectory_ax.collections[-1].set(edgecolor=grid_edgecolor, linewidth=0.35)

segments, segment_times = path_segments(exact_centers, n_spatial)
spatial_trace = LineCollection(
    segments,
    cmap=time_colormap,
    norm=time_normalization,
    array=segment_times,
    linewidth=2.8,
    capstyle="round",
)
trajectory_ax.add_collection(spatial_trace)
for segment in lattice_path_segments(predicted_centers, n_spatial, mode="offset"):
    trajectory_ax.plot(segment[:, 0], segment[:, 1], color=decoded_color, linewidth=0.8, linestyle=(0, (4, 2)))

exact_display = lattice_path_coordinates(exact_centers, n_spatial, mode="offset")
trajectory_ax.scatter(*exact_display[0], s=72, color=time_colormap(0), edgecolors="#29323C", linewidths=0.7, label="start")
trajectory_ax.scatter(*exact_display[-1], s=105, marker="*", color=time_colormap(1), edgecolors="#29323C", linewidths=0.6, label="end")
trajectory_ax.plot([], [], color=time_colormap(0.7), linewidth=2.8, label="true pose (color = time)")
trajectory_ax.plot([], [], color=decoded_color, linewidth=0.8, linestyle=(0, (4, 2)), label="decoded pose")
trajectory_ax.set(title="A   Accurate 2D path integration", aspect="equal", xticks=[], yticks=[])
trajectory_ax.set_frame_on(False)
handles, labels = trajectory_ax.get_legend_handles_labels()
trajectory_ax.legend([handles[index] for index in (2, 3, 0, 1)], [labels[index] for index in (2, 3, 0, 1)],
                     loc="upper right", frameon=True, facecolor="white", edgecolor=grid_edgecolor, fontsize=8)

true_heading = 360 * rollout.exact_poses[:, 2] / n_orientations
decoded_heading = 360 * rollout.predicted_poses[:, 2] / n_orientations
segments, segment_times = heading_segments(steps, true_heading)
heading_ax.add_collection(LineCollection(segments, cmap=time_colormap, norm=time_normalization, array=segment_times, linewidth=2.4))
heading_ax.step(steps, decoded_heading, where="mid", color=decoded_color, linewidth=0.7, linestyle=(0, (4, 2)))
heading_ax.set(xlabel="time step", ylabel="heading", xlim=(steps[0], steps[-1]), ylim=(-15, 315),
               yticks=(0, 120, 240), yticklabels=(r"$0^\circ$", r"$120^\circ$", r"$240^\circ$"))
heading_ax.grid(alpha=0.18, linewidth=0.6)
heading_ax.spines[["top", "right"]].set_visible(False)
figure_7.colorbar(spatial_trace, ax=(trajectory_ax, heading_ax), orientation="horizontal", fraction=0.035, pad=0.06, label="time step")

# %% [markdown]
# ## Panel B — hidden activity
#
# One unit is shown from each of eight translation-sensitive irreps. Within an
# irrep, the selected unit has the largest variance over the rollout; the eight
# irreps with the largest such variance are shown. Each trace is normalized to
# its own range $[0,1]$ for display and uses the same time colormap as Panel A.

# %%
activity_grid = outer_grid[1].subgridspec(8, 1, hspace=0.08)
activity_axes = [figure_7.add_subplot(activity_grid[row]) for row in range(8)]

variances = rollout.hidden_states.var(axis=0)
units_by_irrep = {}
for unit, labels in enumerate(experiment.model.metadata):
    if labels["irrep_dim"] > 1:
        units_by_irrep.setdefault(int(labels["irrep_index"]), []).append(unit)

representatives = []
for units in units_by_irrep.values():
    best_variance = variances[units].max()
    representatives.append(min(unit for unit in units if np.isclose(variances[unit], best_variance)))
representatives = sorted(representatives, key=lambda unit: (-variances[unit], unit))[:8]

activity = rollout.hidden_states[:, representatives]
activity = (activity - activity.min(axis=0)) / np.maximum(np.ptp(activity, axis=0), 1e-15)

for column, (ax, unit) in enumerate(zip(activity_axes, representatives, strict=True)):
    segments = np.stack(
        (
            np.column_stack((steps[:-1], activity[:-1, column])),
            np.column_stack((steps[1:], activity[1:, column])),
        ),
        axis=1,
    )
    ax.add_collection(LineCollection(segments, cmap=time_colormap, norm=time_normalization, array=steps[:-1], linewidth=1.4))
    ax.set(xlim=(steps[0], steps[-1]), ylim=(-0.04, 1.04), yticks=(0, 1),
           ylabel=rf"$\rho={experiment.model.metadata[unit]['irrep_index']}$\nunit {unit}")
    ax.tick_params(axis="both", labelsize=7, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    if column < len(activity_axes) - 1:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("time step")
activity_axes[0].set_title("B   Hidden activity")
display(figure_7)

# %% [markdown]
# ## Save Figure 7

# %%
path = figures / "figure_7.svg"
figure_7.savefig(path, bbox_inches="tight")
plt.show()
print(f"Saved SVG: {path}")
