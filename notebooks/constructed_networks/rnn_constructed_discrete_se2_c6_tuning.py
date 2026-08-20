# %%
# Percent-format notebook source. Regenerate the paired .ipynb with Jupytext.

# %% [markdown]
# # Tuning analysis for the constructed $C_6$ RNN
#
# This notebook owns empirical trajectory tuning and both exhaustive theoretical
# definitions: all-pairs and local-arrival tuning. It intentionally excludes
# group-action pedagogy and neural-manifold analysis.
#
# ## Execution contract
#
# 1. Run **Build and select neurons** after changing the network or unit-selection
#    settings. This usually takes seconds.
# 2. Run **Compute or load empirical trajectory tuning** only when its cache key
#    changes or when `recompute_tuning=True`.
# 3. The exhaustive theoretical calculations use only the selected summary units
#    and can be rerun independently.
# 4. Plot edits never require occupancy recomputation.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

project_root = next(
    parent
    for parent in (Path.cwd(), *Path.cwd().parents)
    if (parent / "src").is_dir()
)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis import (  # noqa: E402
    TrajectoryTuningConfig,
    compute_all_pairs_tuning,
    compute_local_arrival_tuning,
    load_or_compute_empirical_trajectory_tuning,
    masked_periodic_spatial_autocorrelation,
)
from src.experiments.discrete_se2 import (  # noqa: E402
    DiscreteSE2ExperimentConfig,
    DiscreteSE2RolloutConfig,
    build_discrete_se2_experiment,
    run_discrete_se2_rollout,
)
from src.geometry.discrete_se2 import (  # noqa: E402
    NaturalisticMotionConfig,
    lattice_coordinates,
    lattice_path_coordinates,
    plot_lattice_scalar,
)
from src.neural_manifold import build_module_orbits  # noqa: E402

# %% [markdown]
# ## 1. Build and select neurons
#
# The model is reconstructed deterministically rather than serialized. Static
# probing and the 52-step primary rollout determine the irrep representatives
# and matched high-variance neurons before the expensive trajectory pool begins.
#
# Every experimental and computational choice is listed in the next cell. The
# dataclasses only validate and package these visible values.

# %%
# ----------------------------
# Group and signal encoding
# ----------------------------
n_spatial = 10  # Number of lattice sites along each periodic spatial axis.
n_orientations = 6  # Number of headings, separated by 60 degrees.
initial_pose = (2, 2, 0)  # Initial x, y, and heading index.
allocentric_encoding = "gaussian space custom orientation"
allocentric_sigma = 1.0  # Spatial Gaussian width in lattice units.
custom_orientation_weights = (1.0, 0.8, 0.4, 0.2, 0.4, 0.8)
egocentric_encoding_seed = 10  # Seed for the invertible drive encoding.
action_side = "right"  # "right": body-frame s*g; "left": world-frame g*s.

# ----------------------------
# Closed-form network
# ----------------------------
irrep_selection = "power"  # Rank irreps by allocentric-signal spectral power.
num_selected_irreps = None  # No fixed count; the width limit determines selection.
max_hidden_width = 24_000  # Maximum number of recurrent hidden units.
normalize_power_by_dimension = True  # Compare average rather than total irrep power.
always_include_trivial_irrep = True  # Retain the constant representation.
power_ranking = "power"  # Use raw spectral power for irrep ranking.
q_rho = 3  # Number of recurrent copies allocated per selected irrep.
amplitude_mode = "balanced"  # Balance signal amplitudes across recurrence terms.
amplitude_multipliers = (1.0, 1.0, 1.0)  # Input, recurrent, and output scales.
materialize_recurrent_matrix = False  # Apply the structured recurrence implicitly.

# ----------------------------
# Naturalistic body-motion policy
# Translation probabilities sum to one:
# stay + forward + 2*forward-left/right + 2*backward-left/right + backward.
# ----------------------------
stay_probability = 0.05  # Remain at the current position for one step.
forward_probability = 0.70  # Move in the current heading direction.
forward_left_or_right_probability = 0.115  # Each direction at ±60°.
backward_left_or_right_probability = 0.005  # Each direction at ±120°.
backward_probability = 0.01  # Move directly opposite the current heading.
turn_probability = 0.12  # Each left or right 60° heading change.
turn_persistence = 0.20  # Extra probability mass for repeating the previous turn.
immediate_reversal_weight = 0.05  # Multiplier for immediate backtracking.
wall_lookahead = 3  # Number of forward cells checked for an approaching wall.
wall_avoidance_strength = 2.0  # Strength of steering away from nearby walls.
minimum_wall_weight = 0.05  # Lowest pre-exponent weight for a wall-facing action.

# ----------------------------
# Primary trajectory used to select and summarize neurons
# ----------------------------
num_rollout_steps = 52  # Number of poses in the displayed trajectory.
rollout_seed = 1  # Random seed for the displayed trajectory.
rollout_margin = 1  # Excluded cells along each edge of the displayed arena.
rollout_start_xy = (n_spatial // 2, n_spatial // 2)
orientation_arrow_stride = 2  # Draw one heading arrow every two poses.
snapshot_steps = (0, num_rollout_steps // 2, num_rollout_steps - 1)

# ----------------------------
# Occupancy-normalized trajectory tuning
# ----------------------------
num_tuning_trajectories = 100  # Independent trajectories pooled for tuning.
steps_per_tuning_trajectory = 160  # Steps generated in each pooled trajectory.
tuning_burn_in_steps = 10  # Initial steps omitted from each trajectory.
tuning_seed = 101  # Base seed for the pooled trajectory collection.
minimum_bin_occupancy = 5  # Samples required to retain a pose or position bin.
tuning_margin = 0  # Excluded cells along each tuning-trajectory arena edge.
tuning_batch_size = 4  # Trajectories evaluated together per recurrent batch.
cache_schema_version = 1  # Increment after changing cached artifact semantics.

# ----------------------------
# Neuron and module selection
# ----------------------------
num_summary_neurons = 6  # Irreps represented by their highest-variance unit.
num_tuning_irreps_to_plot = 5  # Highest-power retained modules to inspect.
num_tuning_neurons_per_irrep = 10  # Units retained from each inspected module.
include_conjugate_irreps = True  # Treat conjugate irreps as one real module.
skip_trivial_irrep = True  # Exclude the spatially constant module.
exhaustive_drive_batch_size = 32  # Drives evaluated together in theoretical sums.

# ----------------------------
# Cache controls
# ----------------------------
use_tuning_cache = True  # Load and save matching trajectory-tuning artifacts.
recompute_tuning = False  # Ignore a matching artifact and replace it.
tuning_cache_directory = (
    project_root / "artifacts" / "constructed_networks" / "discrete_se2_c6"
)

# Package the explicit values above. No defaults are relied upon here.
experiment_config = DiscreteSE2ExperimentConfig(
    n_spatial=n_spatial,
    n_orientations=n_orientations,
    initial_pose=initial_pose,
    allocentric_encoding=allocentric_encoding,
    sigma=allocentric_sigma,
    custom_orientation_weights=custom_orientation_weights,
    encoding_seed=egocentric_encoding_seed,
    action_side=action_side,
    irrep_selection=irrep_selection,
    num_selected_irreps=num_selected_irreps,
    max_hidden_width=max_hidden_width,
    normalize_power_by_dim=normalize_power_by_dimension,
    always_include_trivial=always_include_trivial_irrep,
    power_ranking=power_ranking,
    q_rho=q_rho,
    amplitude_mode=amplitude_mode,
    amplitude_multipliers=amplitude_multipliers,
    materialize_mix=materialize_recurrent_matrix,
)
motion_config = NaturalisticMotionConfig(
    stay_probability=stay_probability,
    forward_probability=forward_probability,
    forward_left_or_right_probability=forward_left_or_right_probability,
    backward_left_or_right_probability=backward_left_or_right_probability,
    backward_probability=backward_probability,
    turn_probability=turn_probability,
    turn_persistence=turn_persistence,
    reversal_weight=immediate_reversal_weight,
    wall_lookahead=wall_lookahead,
    wall_avoidance_strength=wall_avoidance_strength,
    minimum_wall_weight=minimum_wall_weight,
)
rollout_config = DiscreteSE2RolloutConfig(
    steps=num_rollout_steps,
    seed=rollout_seed,
    margin=rollout_margin,
    start_xy=rollout_start_xy,
    arrow_stride=orientation_arrow_stride,
    snapshot_steps=snapshot_steps,
)
tuning_config = TrajectoryTuningConfig(
    num_trajectories=num_tuning_trajectories,
    steps_per_trajectory=steps_per_tuning_trajectory,
    burn_in_steps=tuning_burn_in_steps,
    seed=tuning_seed,
    min_occupancy=minimum_bin_occupancy,
    margin=tuning_margin,
    batch_size=tuning_batch_size,
    cache_schema_version=cache_schema_version,
)

experiment = build_discrete_se2_experiment(experiment_config)
rollout = run_discrete_se2_rollout(
    experiment,
    rollout_config,
    motion_config,
)
G = experiment.group
params = experiment.model

static_hidden = (
    params.probe_hidden_states(experiment.x_allo).detach().cpu().numpy()
)
units_by_irrep = {}
for unit, metadata in enumerate(params.metadata):
    units_by_irrep.setdefault(int(metadata["irrep_index"]), []).append(unit)

module_orbits = build_module_orbits(
    params,
    static_hidden,
    include_conjugates=include_conjugate_irreps,
    skip_trivial=skip_trivial_irrep,
)
representative_irreps = [module.irrep_indices[0] for module in module_orbits]
power = G.power_spectrum(experiment.x_allo)
tuning_irreps = sorted(
    representative_irreps,
    key=lambda index: float(power[index]),
    reverse=True,
)[:num_tuning_irreps_to_plot]
tuning_units_by_irrep = {
    index: np.asarray(
        units_by_irrep[index][:num_tuning_neurons_per_irrep],
        dtype=int,
    )
    for index in tuning_irreps
    if index in units_by_irrep
}

trajectory_variances = np.var(rollout.hidden_states, axis=0)
best_unit_by_irrep = {
    irrep_index: max(
        units,
        key=lambda unit: float(trajectory_variances[unit]),
    )
    for irrep_index, units in units_by_irrep.items()
}
ranked_irrep_representatives = sorted(
    best_unit_by_irrep.items(),
    key=lambda item: float(trajectory_variances[item[1]]),
    reverse=True,
)
summary_irrep_indices = np.asarray(
    [
        irrep_index
        for irrep_index, _ in ranked_irrep_representatives[
            :num_summary_neurons
        ]
    ],
    dtype=int,
)
summary_units = np.asarray(
    [
        unit
        for _, unit in ranked_irrep_representatives[:num_summary_neurons]
    ],
    dtype=int,
)
selected_units = np.asarray(
    sorted(
        {
            int(unit)
            for units in tuning_units_by_irrep.values()
            for unit in units
        }
        | {int(unit) for unit in summary_units}
    ),
    dtype=int,
)
print(f"hidden width: {params.hidden_dim:,}")
print(
    "summary irrep/unit pairs:",
    list(zip(summary_irrep_indices, summary_units)),
)
print("trajectory-tuning units:", selected_units)
print("trajectory tuning configuration:", tuning_config)

# %% [markdown]
# ## 2. Static initialization tuning
#
# `probe_hidden_states` evaluates every transformed allocentric input with an
# identity drive. This is not a recurrent trajectory average and has no
# occupancy bias.

# %%
heading_degrees = 360 * np.arange(G.m) / G.m
static_figure, static_axes = plt.subplots(
    len(summary_units),
    G.m + 1,
    figsize=(2.25 * (G.m + 1), 2.2 * len(summary_units)),
    constrained_layout=True,
    squeeze=False,
)
for row, unit in enumerate(summary_units):
    tensor = static_hidden[:, unit].reshape(G.m, G.n, G.n)
    for rotation in range(G.m):
        plot_lattice_scalar(
            tensor[rotation],
            ax=static_axes[row, rotation],
            title=rf"$\theta={heading_degrees[rotation]:.0f}^\circ$",
            colorbar=False,
            coordinate_mode="axial",
        )
    plot_lattice_scalar(
        tensor.mean(axis=0),
        ax=static_axes[row, G.m],
        title=f"unit {unit}: heading mean",
        colorbar=False,
        coordinate_mode="axial",
    )
static_figure.suptitle("Static initialization responses")
plt.show()

# %% [markdown]
# ## 3. Compute or load empirical trajectory tuning
#
# This is the only five-minute-class stage. It runs batched recurrent
# trajectories, records only `selected_units`, and accumulates sufficient
# statistics online. The cache key includes the complete experiment, motion,
# tuning, selected-unit, and schema configurations.
#
# Set `recompute_tuning=True` only to deliberately replace the matching artifact.

# %%
empirical_tuning = load_or_compute_empirical_trajectory_tuning(
    experiment,
    selected_units,
    motion_config=motion_config,
    tuning_config=tuning_config,
    cache_directory=tuning_cache_directory,
    recompute=recompute_tuning,
    use_cache=use_tuning_cache,
)
status = "loaded" if empirical_tuning.cache_hit else "computed"
print(f"empirical trajectory tuning: {status}")
print(f"cache key: {empirical_tuning.cache_key}")
print(f"cache path: {empirical_tuning.cache_path}")
print(f"selected units: {len(empirical_tuning.unit_indices)}")
print(f"retained samples: {empirical_tuning.pose_occupancy.sum():,}")
print(
    "pose occupancy: "
    f"min={empirical_tuning.pose_occupancy.min()}, "
    f"median={np.median(empirical_tuning.pose_occupancy):.0f}, "
    f"max={empirical_tuning.pose_occupancy.max()}"
)

# %% [markdown]
# ## 4. Matched trajectory, activity, and spatial tuning
#
# All four panels use the same representatives: the highest-variance neuron
# within each of the six highest-variance irreps. Panels C and D compare
# empirical trajectory tuning with exhaustive local-arrival tuning.

# %%
summary_activity = rollout.hidden_states[:, summary_units]
activity_min = summary_activity.min(axis=0, keepdims=True)
activity_range = np.ptp(summary_activity, axis=0, keepdims=True)
normalized_activity = (summary_activity - activity_min) / np.where(
    activity_range > 0,
    activity_range,
    1,
)

def normalize_tuning_map(tuning_map):
    finite = np.isfinite(tuning_map)
    normalized = np.full_like(tuning_map, np.nan)
    if np.any(finite):
        minimum = float(tuning_map[finite].min())
        value_range = float(np.ptp(tuning_map[finite]))
        normalized[finite] = (
            (tuning_map[finite] - minimum) / value_range
            if value_range > 0
            else 0
        )
    return normalized


empirical_summary_maps = [
    normalize_tuning_map(
        empirical_tuning.position_tuning[
            ..., empirical_tuning.local_unit_index(int(unit))
        ]
    )
    for unit in summary_units
]
local_arrival = compute_local_arrival_tuning(
    experiment,
    summary_units,
    drive_batch_size=exhaustive_drive_batch_size,
)
local_arrival_summary_maps = [
    normalize_tuning_map(local_arrival.position_mean[..., column])
    for column in range(len(summary_units))
]

summary_figure = plt.figure(figsize=(23, 6), layout="constrained")
(
    trajectory_subfigure,
    activity_subfigure,
    empirical_subfigure,
    local_arrival_subfigure,
) = (
    summary_figure.subfigures(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.6, 1.6),
        wspace=0.04,
    )
)

trajectory_ax = trajectory_subfigure.subplots()
lattice_x, lattice_y = lattice_coordinates(G.n, mode="axial")
trajectory_points = lattice_path_coordinates(
    rollout.exact_centers,
    G.n,
    mode="axial",
)
steps = np.arange(1, len(trajectory_points) + 1)
trajectory_ax.scatter(lattice_x, lattice_y, s=8, color="0.88", linewidths=0)
trajectory_ax.plot(
    trajectory_points[:, 0],
    trajectory_points[:, 1],
    color="0.65",
    linewidth=1,
)
trajectory_artist = trajectory_ax.scatter(
    trajectory_points[:, 0],
    trajectory_points[:, 1],
    c=steps,
    cmap="viridis",
    s=25,
    linewidths=0,
)
trajectory_ax.set(title="A. Spatial trajectory", aspect="equal", xticks=[], yticks=[])
trajectory_ax.set_frame_on(False)
trajectory_subfigure.colorbar(
    trajectory_artist,
    ax=trajectory_ax,
    orientation="horizontal",
    fraction=0.06,
    pad=0.08,
    label="time step",
)

activity_steps = np.arange(len(normalized_activity))
activity_axes = np.asarray(
    activity_subfigure.subplots(
        len(summary_units),
        1,
        sharex=True,
        squeeze=False,
    )
).ravel()
for column, (ax, irrep_index, unit) in enumerate(
    zip(activity_axes, summary_irrep_indices, summary_units)
):
    ax.plot(
        activity_steps,
        normalized_activity[:, column],
        color="0.15",
        linewidth=1.6,
    )
    ax.set(
        ylabel=f"irrep {irrep_index}\nunit {unit}",
        ylim=(-0.03, 1.03),
        yticks=(0, 1),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
activity_axes[0].set_title("B. Hidden activity")
activity_axes[-1].set_xlabel("time step")

columns = 2
rows = int(np.ceil(len(summary_units) / columns))
for subfigure, tuning_maps, title in (
    (
        empirical_subfigure,
        empirical_summary_maps,
        "C. Empirical trajectory tuning",
    ),
    (
        local_arrival_subfigure,
        local_arrival_summary_maps,
        "D. Exhaustive local-arrival tuning",
    ),
):
    tuning_axes = np.asarray(
        subfigure.subplots(rows, columns, squeeze=False)
    )
    for ax, irrep_index, unit, tuning_map in zip(
        tuning_axes.ravel(),
        summary_irrep_indices,
        summary_units,
        tuning_maps,
    ):
        plot_lattice_scalar(
            tuning_map,
            ax=ax,
            title=f"irrep {irrep_index}, unit {unit}",
            cmap="viridis",
            vmin=0,
            vmax=1,
            colorbar=False,
            coordinate_mode="axial",
        )
    for ax in tuning_axes.ravel()[len(summary_units) :]:
        ax.set_visible(False)
    subfigure.suptitle(title)
summary_figure.suptitle("Matched behavior and representation")
plt.show()

# %% [markdown]
# ## 5. Theoretical tuning and three-definition comparison
#
# For every target pose $g$ and drive $j$, both theoretical definitions solve
# $i * j = g$ for the unique predecessor $i$, evaluate the one-step response
# $h(i,j)$, and average over incoming transitions. **All pairs** uses every
# $j\in G$; **local arrivals** uses the uniform 21-element local drive set.
#
# The figure compares spatial marginals after independently scaling each map to
# $[0,1]$. The printed correlations use the full pose tuning curves on $G$,
# including heading, and retain the empirical occupancy mask.

# %%
all_pairs_tuning = compute_all_pairs_tuning(
    experiment,
    summary_units,
    drive_batch_size=exhaustive_drive_batch_size,
)

empirical_pose_tuning = np.stack(
    [
        empirical_tuning.pose_tuning[
            ..., empirical_tuning.local_unit_index(int(unit))
        ]
        for unit in summary_units
    ],
    axis=-1,
)
all_pairs_pose_tuning = all_pairs_tuning.pose_mean.reshape(
    G.m,
    G.n,
    G.n,
    len(summary_units),
)
local_arrival_pose_tuning = local_arrival.pose_mean.reshape(
    G.m,
    G.n,
    G.n,
    len(summary_units),
)

position_tuning_definitions = (
    ("Empirical trajectories", np.nanmean(empirical_pose_tuning, axis=0)),
    ("Exhaustive all pairs", all_pairs_tuning.position_mean),
    ("Exhaustive local arrivals", local_arrival.position_mean),
)


def normalize_maps(values):
    minimum = np.nanmin(values, axis=(0, 1), keepdims=True)
    span = np.nanmax(values, axis=(0, 1), keepdims=True) - minimum
    return (values - minimum) / np.where(span > 0, span, 1)


def masked_correlation(first, second):
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2:
        return np.nan
    first_valid = first[valid]
    second_valid = second[valid]
    if np.ptp(first_valid) == 0 or np.ptp(second_valid) == 0:
        return np.nan
    return float(np.corrcoef(first_valid, second_valid)[0, 1])


comparison_figure, comparison_axes = plt.subplots(
    len(position_tuning_definitions),
    len(summary_units),
    figsize=(2.55 * len(summary_units), 7.5),
    constrained_layout=True,
    squeeze=False,
)
for row, (definition, maps) in enumerate(position_tuning_definitions):
    normalized_maps = normalize_maps(maps)
    for column, (irrep_index, unit) in enumerate(
        zip(summary_irrep_indices, summary_units)
    ):
        plot_lattice_scalar(
            normalized_maps[..., column],
            ax=comparison_axes[row, column],
            title=(
                f"irrep {irrep_index}, unit {unit}"
                if row == 0
                else None
            ),
            cmap="viridis",
            vmin=0,
            vmax=1,
            colorbar=False,
            coordinate_mode="axial",
        )
    comparison_axes[row, 0].annotate(
        definition,
        xy=(-0.08, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="right",
        va="center",
    )
comparison_figure.suptitle(
    "Spatial tuning under three definitions (independently normalized)"
)
plt.show()

print(
    f"all pairs: {all_pairs_tuning.num_drives} drives per target; "
    f"local arrivals: {local_arrival.num_drives} drives per target"
)
for column, unit in enumerate(summary_units):
    empirical = empirical_pose_tuning[..., column]
    all_pairs = all_pairs_pose_tuning[..., column]
    local = local_arrival_pose_tuning[..., column]
    print(
        f"unit {unit}: "
        f"corr(empirical, all pairs)="
        f"{masked_correlation(empirical, all_pairs):.3f}, "
        f"corr(empirical, local)="
        f"{masked_correlation(empirical, local):.3f}, "
        f"corr(all pairs, local)="
        f"{masked_correlation(all_pairs, local):.3f}"
    )

# %% [markdown]
# ## 6. Occupancy and autocorrelation diagnostics

# %%
occupancy_figure, occupancy_axes = plt.subplots(
    1,
    G.m + 1,
    figsize=(2.35 * (G.m + 1), 2.5),
    constrained_layout=True,
)
for rotation in range(G.m):
    plot_lattice_scalar(
        empirical_tuning.pose_occupancy[rotation],
        ax=occupancy_axes[rotation],
        title=rf"$\theta={heading_degrees[rotation]:.0f}^\circ$",
        colorbar=False,
        coordinate_mode="axial",
    )
plot_lattice_scalar(
    empirical_tuning.position_occupancy,
    ax=occupancy_axes[G.m],
    title="all headings",
    colorbar=False,
    coordinate_mode="axial",
)
occupancy_figure.suptitle("Trajectory sample occupancy")
plt.show()

# %%
autocorrelation_figure, autocorrelation_axes = plt.subplots(
    1,
    len(summary_units),
    figsize=(2.6 * len(summary_units), 2.7),
    constrained_layout=True,
    squeeze=False,
)
for column, unit in enumerate(summary_units):
    local_index = empirical_tuning.local_unit_index(int(unit))
    autocorrelation = masked_periodic_spatial_autocorrelation(
        empirical_tuning.position_tuning[..., local_index]
    )
    plot_lattice_scalar(
        autocorrelation,
        ax=autocorrelation_axes[0, column],
        title=f"unit {unit}",
        cmap="viridis",
        vmin=-1,
        vmax=1,
        colorbar=False,
        coordinate_mode="centered_axial",
    )
autocorrelation_figure.suptitle("Masked periodic spatial autocorrelation")
plt.show()

# %% [markdown]
# ## Summary
#
# Empirical trajectory, exhaustive all-pairs, and exhaustive local-arrival tuning
# now have separate result objects and a matched comparison. The empirical cache
# remains valid across all plotting edits.
