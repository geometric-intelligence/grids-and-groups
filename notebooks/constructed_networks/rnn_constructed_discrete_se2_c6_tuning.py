# %%
# Percent-format notebook source. Regenerate the paired .ipynb with Jupytext.

# %% [markdown]
# # Tuning analysis for the constructed $C_6$ RNN
#
# This notebook owns empirical trajectory tuning and both exact one-step
# definitions: all drive and local drive tuning. It intentionally excludes
# group-action pedagogy and neural-manifold analysis.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from matplotlib.patches import Rectangle

plt.rcParams["svg.fonttype"] = "none"

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

project_root = next(
    parent for parent in (Path.cwd(), *Path.cwd().parents) if (parent / "src").is_dir()
)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis.tuning import (  # noqa: E402
    compute_empirical_trajectory_tuning,
    compute_one_step_tuning,
    occupancy_normalized_activity,
)
from src.experiments.discrete_se2 import (  # noqa: E402
    DiscreteSE2ExperimentConfig,
    DiscreteSE2RolloutConfig,
    build_discrete_se2_experiment,
    run_discrete_se2_rollout,
)
from src.geometry.discrete_se2.core import lattice_coordinates, lattice_path_coordinates  # noqa: E402
from src.geometry.discrete_se2.plotting import plot_lattice_scalar  # noqa: E402
from src.geometry.discrete_se2.trajectories import NaturalisticMotionConfig  # noqa: E402
from src.groups.znxzn_cm import DiscreteSE2Group  # noqa: E402
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
wall_lookahead = 3  # Number of forward cells checked for an approaching wall.
wall_avoidance_strength = 2.0  # Strength of steering away from nearby walls.
minimum_wall_weight = 0.05  # Lowest pre-exponent weight for a wall-facing action.

# ----------------------------
# Primary trajectory used to select and summarize neurons
# ----------------------------
num_rollout_steps = 100  # Number of poses in the displayed trajectory.
rollout_seed = 1  # Random seed for the displayed trajectory.
rollout_margin = 1  # Excluded cells along each edge of the displayed arena.
rollout_start_xy = (n_spatial // 2, n_spatial // 2)
orientation_arrow_stride = 2  # Draw one heading arrow every two poses.
snapshot_steps = (0, num_rollout_steps // 2, num_rollout_steps - 1)

# ----------------------------
# Occupancy-normalized trajectory tuning
# ----------------------------
num_tuning_trajectories = 300  # Independent trajectories pooled for tuning.
steps_per_tuning_trajectory = 160  # Steps generated in each pooled trajectory.
tuning_burn_in_steps = 10  # Initial steps omitted from each trajectory.
tuning_seed = 101  # Base seed for the pooled trajectory collection.
minimum_bin_occupancy = 5  # Samples required to retain a pose or position bin.
tuning_margin = 0  # Excluded cells along each tuning-trajectory arena edge.
tuning_batch_size = 4  # Trajectories evaluated together per recurrent batch.

# ----------------------------
# Panel A periodic random walk
# ----------------------------
panel_a_direction_probability = 1 / 6  # Uniform over six neighboring cells.
panel_a_turn_probability = 1 / 3  # Uniform over left, straight, and right turns.
panel_a_trajectory_counts = (10, 100, 1000)
panel_a_steps_per_trajectory = 110
panel_a_burn_in_steps = 10
panel_a_seed = 101
panel_a_batch_size = 8

# ----------------------------
# Neuron and module selection
# ----------------------------
num_summary_neurons = 8  # Conjugate modules represented by one high-variance unit.
num_tuning_irreps_to_plot = 5  # Highest-power retained modules to inspect.
num_tuning_neurons_per_irrep = 10  # Units retained from each inspected module.
include_conjugate_irreps = True  # Treat conjugate irreps as one real module.
skip_trivial_irrep = True  # Exclude the spatially constant module.
one_step_drive_batch_size = 32  # Drives evaluated together in exact one-step sums.

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
    wall_lookahead=wall_lookahead,
    wall_avoidance_strength=wall_avoidance_strength,
    minimum_wall_weight=minimum_wall_weight,
    periodic_boundaries=False,
)
panel_a_motion_config = NaturalisticMotionConfig(
    stay_probability=0,
    forward_probability=panel_a_direction_probability,
    forward_left_or_right_probability=panel_a_direction_probability,
    backward_left_or_right_probability=panel_a_direction_probability,
    backward_probability=panel_a_direction_probability,
    turn_probability=panel_a_turn_probability,
    turn_persistence=0,
    wall_lookahead=1,
    wall_avoidance_strength=0,
    minimum_wall_weight=1,
    periodic_boundaries=True,
)
rollout_config = DiscreteSE2RolloutConfig(
    steps=num_rollout_steps,
    seed=rollout_seed,
    margin=rollout_margin,
    start_xy=rollout_start_xy,
    arrow_stride=orientation_arrow_stride,
    snapshot_steps=snapshot_steps,
)
experiment = build_discrete_se2_experiment(experiment_config)
rollout = run_discrete_se2_rollout(
    experiment,
    rollout_config,
    motion_config,
)
G = experiment.group
params = experiment.model

static_hidden = params.probe_hidden_states(experiment.x_allo).detach().cpu().numpy()
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


def highest_variance_unit(unit_indices):
    """Choose the lowest-index unit when symmetry makes variances effectively tie."""
    unit_indices = np.asarray(unit_indices, dtype=int)
    values = trajectory_variances[unit_indices]
    maximum = values.max()
    tied = unit_indices[np.isclose(values, maximum, rtol=1e-10, atol=1e-12)]
    return int(tied.min())


ranked_module_representatives = sorted(
    [
        (
            module,
            highest_variance_unit(module.unit_indices),
        )
        for module in module_orbits
    ],
    key=lambda item: (
        -round(float(trajectory_variances[item[1]]), 12),
        item[1],
    ),
)
summary_modules = [module for module, _ in ranked_module_representatives[:num_summary_neurons]]
summary_irrep_groups = [module.irrep_indices for module in summary_modules]
summary_units = np.asarray(
    [unit for _, unit in ranked_module_representatives[:num_summary_neurons]],
    dtype=int,
)


def irrep_mode_label(irrep_indices):
    """Describe whether an irrep module depends on position, heading, or both."""
    irrep = params.all_irreps[irrep_indices[0]]
    identity = np.eye(irrep.dim)
    translation_dependent = any(
        not np.allclose(
            irrep(G.encode(x, y, 0)),
            identity,
            atol=1e-10,
        )
        for x in range(G.n)
        for y in range(G.n)
    )
    orientation_dependent = any(
        not np.allclose(
            irrep(G.encode(0, 0, rotation)),
            identity,
            atol=1e-10,
        )
        for rotation in range(G.m)
    )
    if translation_dependent and orientation_dependent:
        return "conjunctive"
    if translation_dependent:
        return "spatial-only"
    if orientation_dependent:
        return "orientation-only"
    return "constant"


summary_mode_labels = [irrep_mode_label(irrep_group) for irrep_group in summary_irrep_groups]
summary_irrep_labels = [
    (
        f"irreps {'+'.join(map(str, irrep_group))}"
        if len(irrep_group) > 1
        else f"irrep {irrep_group[0]}"
    )
    for irrep_group in summary_irrep_groups
]
selected_units = np.asarray(
    sorted(
        {int(unit) for units in tuning_units_by_irrep.values() for unit in units}
        | {int(unit) for unit in summary_units}
    ),
    dtype=int,
)
print(f"hidden width: {params.hidden_dim:,}")
print("summary module representatives:")
for irrep_label, mode_label, unit in zip(
    summary_irrep_labels,
    summary_mode_labels,
    summary_units,
):
    print(f"  {irrep_label}: unit {unit} ({mode_label})")
print("trajectory-tuning units:", selected_units)

# %% [markdown]
# ## 2. Static initialization tuning
#
# `probe_hidden_states` evaluates every transformed allocentric input with an
# identity drive. This is not a recurrent trajectory average and has no
# occupancy bias.

# %%
heading_degrees = 360 * np.arange(G.m) / G.m
# Use one rectangular fundamental domain for every spatial tuning map. Boundary
# cells retain their hexagonal shape and are split across the periodic seams.
tuning_coordinate_mode = "offset"
wrap_tuning_edges = True
static_figure, static_axes = plt.subplots(
    len(summary_units),
    G.m + 1,
    figsize=(2.25 * (G.m + 1), 2.2 * len(summary_units)),
    constrained_layout=True,
    squeeze=False,
)
for row, (irrep_label, mode_label, unit) in enumerate(
    zip(
        summary_irrep_labels,
        summary_mode_labels,
        summary_units,
    )
):
    tensor = static_hidden[:, unit].reshape(G.m, G.n, G.n)
    unit_minimum = float(tensor.min())
    unit_maximum = float(tensor.max())
    for rotation in range(G.m):
        plot_lattice_scalar(
            tensor[rotation],
            ax=static_axes[row, rotation],
            title=rf"$\theta={heading_degrees[rotation]:.0f}^\circ$",
            vmin=unit_minimum,
            vmax=unit_maximum,
            colorbar=False,
            coordinate_mode=tuning_coordinate_mode,
            wrap_periodic_edges=wrap_tuning_edges,
        )
    plot_lattice_scalar(
        tensor.mean(axis=0),
        ax=static_axes[row, G.m],
        title=f"{irrep_label}\nunit {unit}\n{mode_label}\nheading mean",
        vmin=unit_minimum,
        vmax=unit_maximum,
        colorbar=False,
        coordinate_mode=tuning_coordinate_mode,
        wrap_periodic_edges=wrap_tuning_edges,
    )
static_figure.suptitle("Static initialization responses")
plt.show()

# %% [markdown]
# ## 3. Compute empirical trajectory tuning
#
# This is the only five-minute-class stage. It runs batched recurrent
# trajectories, records only `selected_units`, and accumulates sufficient
# statistics online.

# %%
empirical_tuning = compute_empirical_trajectory_tuning(
    experiment,
    selected_units,
    motion_config=motion_config,
    num_trajectories=num_tuning_trajectories,
    steps_per_trajectory=steps_per_tuning_trajectory,
    burn_in_steps=tuning_burn_in_steps,
    seed=tuning_seed,
    margin=tuning_margin,
    batch_size=tuning_batch_size,
)
empirical_column = {
    int(unit): column
    for column, unit in enumerate(empirical_tuning.unit_indices)
}
empirical_pose_tuning = occupancy_normalized_activity(
    empirical_tuning.pose_activity_sums,
    empirical_tuning.pose_occupancy,
    min_occupancy=minimum_bin_occupancy,
)
empirical_position_tuning = occupancy_normalized_activity(
    empirical_tuning.pose_activity_sums.sum(axis=0),
    empirical_tuning.pose_occupancy.sum(axis=0),
    min_occupancy=minimum_bin_occupancy,
)
empirical_position_occupancy = empirical_tuning.pose_occupancy.sum(axis=0)
print("empirical trajectory tuning: computed")
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
# All five panels use the same representatives: the highest-variance neuron
# within each of the eight highest-variance real modules, with complex-conjugate
# irreps treated as one module. Panels C–E scale each nonconstant map independently
# so its spatial structure remains visible.

# %%
summary_activity = rollout.hidden_states[:, summary_units]
activity_min = summary_activity.min(axis=0, keepdims=True)
activity_range = np.ptp(summary_activity, axis=0, keepdims=True)
normalized_activity = (summary_activity - activity_min) / np.where(
    activity_range > 0,
    activity_range,
    1,
)


def normalize_tuning_maps(definition_maps):
    """Scale each nonconstant map independently without amplifying roundoff."""
    values = np.asarray(definition_maps, dtype=float)
    minimum = np.nanmin(values, axis=(1, 2), keepdims=True)
    maximum = np.nanmax(values, axis=(1, 2), keepdims=True)
    span = maximum - minimum
    scale = np.maximum(np.abs(minimum), np.abs(maximum))
    tolerance = 1e-12 + 1e-10 * scale
    stable_span = np.where(span > tolerance, span, 1)
    normalized = (values - minimum) / stable_span
    normalized = np.where(span > tolerance, normalized, 0.5)
    return normalized, tolerance


empirical_summary_maps = np.stack(
    [
        empirical_position_tuning[..., empirical_column[int(unit)]]
        for unit in summary_units
    ],
    axis=-1,
)
local_drive_tuning = compute_one_step_tuning(
    experiment,
    summary_units,
    experiment.local_egocentric_elements,
    drive_batch_size=one_step_drive_batch_size,
)
all_drive_tuning = compute_one_step_tuning(
    experiment,
    summary_units,
    experiment.group.elements(),
    drive_batch_size=one_step_drive_batch_size,
)
summary_definition_maps = np.stack(
    [
        empirical_summary_maps,
        local_drive_tuning.position_mean,
        all_drive_tuning.position_mean,
    ],
    axis=0,
)
normalized_summary_maps, _ = normalize_tuning_maps(summary_definition_maps)

summary_figure = plt.figure(
    figsize=(29, max(6, 1.15 * len(summary_units))),
    layout="constrained",
)
(
    trajectory_subfigure,
    activity_subfigure,
    empirical_subfigure,
    local_drive_subfigure,
    all_drive_subfigure,
) = summary_figure.subfigures(
    1,
    5,
    width_ratios=(1.0, 1.0, 1.6, 1.6, 1.6),
    wspace=0.04,
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
for column, (ax, irrep_label, mode_label, unit) in enumerate(
    zip(
        activity_axes,
        summary_irrep_labels,
        summary_mode_labels,
        summary_units,
    )
):
    ax.plot(
        activity_steps,
        normalized_activity[:, column],
        color="0.15",
        linewidth=1.6,
    )
    ax.set(
        ylabel=f"{irrep_label}\nunit {unit}\n{mode_label}",
        ylim=(-0.03, 1.03),
        yticks=(0, 1),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
activity_axes[0].set_title("B. Hidden activity")
activity_axes[-1].set_xlabel("time step")

columns = 2
rows = int(np.ceil(len(summary_units) / columns))
for definition_index, (subfigure, title) in enumerate(
    (
        (
            empirical_subfigure,
            "C. Empirical trajectory tuning",
        ),
        (
            local_drive_subfigure,
            "D. Exact local one-step tuning",
        ),
        (
            all_drive_subfigure,
            "E. Exact all-pairs one-step tuning",
        ),
    )
):
    tuning_axes = np.asarray(subfigure.subplots(rows, columns, squeeze=False))
    for column, (ax, irrep_label, mode_label, unit) in enumerate(
        zip(
            tuning_axes.ravel(),
            summary_irrep_labels,
            summary_mode_labels,
            summary_units,
        )
    ):
        plot_lattice_scalar(
            normalized_summary_maps[definition_index, ..., column],
            ax=ax,
            title=f"{irrep_label}\nunit {unit}\n{mode_label}",
            cmap="viridis",
            vmin=0,
            vmax=1,
            colorbar=False,
            coordinate_mode=tuning_coordinate_mode,
            wrap_periodic_edges=wrap_tuning_edges,
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
# $h(i,j)$, and average over incoming transitions. **All drive** uses every
# $j\in G$; **local drive** uses the uniform 21-element local drive set.
#
# The figure scales each nonconstant spatial marginal independently to preserve
# visible structure. A tolerance keeps genuinely invariant maps uniform instead
# of stretching floating-point noise across the colormap. The printed
# correlations use the full pose tuning curves on $G$, including heading, and
# retain the empirical occupancy mask.

empirical_pose_tuning = np.stack(
    [
        empirical_pose_tuning[..., empirical_column[int(unit)]]
        for unit in summary_units
    ],
    axis=-1,
)
all_drive_pose_tuning = all_drive_tuning.pose_mean.reshape(
    G.m,
    G.n,
    G.n,
    len(summary_units),
)
local_drive_pose_tuning = local_drive_tuning.pose_mean.reshape(
    G.m,
    G.n,
    G.n,
    len(summary_units),
)

position_tuning_definitions = (
    ("Empirical trajectories", np.nanmean(empirical_pose_tuning, axis=0)),
    ("Exact local one-step", local_drive_tuning.position_mean),
    ("Exact all-pairs one-step", all_drive_tuning.position_mean),
)

tuning_definition_maps = np.stack(
    [maps for _, maps in position_tuning_definitions],
    axis=0,
)
normalized_tuning_maps, _ = normalize_tuning_maps(tuning_definition_maps)


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
    for column, (irrep_label, mode_label, unit) in enumerate(
        zip(
            summary_irrep_labels,
            summary_mode_labels,
            summary_units,
        )
    ):
        plot_lattice_scalar(
            normalized_tuning_maps[row, ..., column],
            ax=comparison_axes[row, column],
            title=(f"{irrep_label}\nunit {unit}\n{mode_label}" if row == 0 else None),
            cmap="viridis",
            vmin=0,
            vmax=1,
            colorbar=False,
            coordinate_mode=tuning_coordinate_mode,
            wrap_periodic_edges=wrap_tuning_edges,
        )
    comparison_axes[row, 0].annotate(
        definition,
        xy=(-0.08, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="right",
        va="center",
    )
comparison_figure.suptitle("Spatial tuning under three definitions (stable per-map normalization)")
plt.show()

print(
    f"all drive: {all_drive_tuning.num_drives} drives per target; "
    f"local drive: {local_drive_tuning.num_drives} drives per target"
)
for column, (irrep_label, mode_label, unit) in enumerate(
    zip(
        summary_irrep_labels,
        summary_mode_labels,
        summary_units,
    )
):
    empirical = empirical_pose_tuning[..., column]
    all_drive = all_drive_pose_tuning[..., column]
    local_drive = local_drive_pose_tuning[..., column]
    print(
        f"{irrep_label}, unit {unit} ({mode_label}): "
        f"corr(empirical, all drive)="
        f"{masked_correlation(empirical, all_drive):.3f}, "
        f"corr(empirical, local drive)="
        f"{masked_correlation(empirical, local_drive):.3f}, "
        f"corr(all drive, local drive)="
        f"{masked_correlation(all_drive, local_drive):.3f}"
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
    empirical_position_occupancy,
    ax=occupancy_axes[G.m],
    title="all headings",
    colorbar=False,
    coordinate_mode="axial",
)
occupancy_figure.suptitle("Trajectory sample occupancy")
plt.show()

# %%
def masked_periodic_spatial_autocorrelation(values):
    """Correlate each periodic spatial shift using only observed bin pairs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional spatial field")
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.full_like(values, np.nan)
    centered = np.where(valid, values - values[valid].mean(), 0.0)
    zero_lag = np.mean(centered[valid] ** 2)
    if np.isclose(zero_lag, 0):
        return np.zeros_like(values)

    autocorrelation = np.full_like(values, np.nan)
    for shift_x in range(values.shape[0]):
        for shift_y in range(values.shape[1]):
            shifted = np.roll(centered, (shift_x, shift_y), axis=(0, 1))
            paired = valid & np.roll(valid, (shift_x, shift_y), axis=(0, 1))
            if np.any(paired):
                autocorrelation[shift_x, shift_y] = (
                    np.mean(centered[paired] * shifted[paired]) / zero_lag
                )
    return np.clip(np.fft.fftshift(autocorrelation), -1, 1)


autocorrelation_figure, autocorrelation_axes = plt.subplots(
    1,
    len(summary_units),
    figsize=(2.6 * len(summary_units), 2.7),
    constrained_layout=True,
    squeeze=False,
)
for column, (irrep_label, mode_label, unit) in enumerate(
    zip(
        summary_irrep_labels,
        summary_mode_labels,
        summary_units,
    )
):
    local_index = empirical_column[int(unit)]
    autocorrelation = masked_periodic_spatial_autocorrelation(
        empirical_position_tuning[..., local_index]
    )
    plot_lattice_scalar(
        autocorrelation,
        ax=autocorrelation_axes[0, column],
        title=f"{irrep_label}\nunit {unit}\n{mode_label}",
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
# Empirical trajectory, exact all-pairs, and exact local one-step tuning
# now have separate result objects and a matched comparison.

# %% [markdown]
# ## 7. Paper Figure 8 draft
#
# The current panel-by-panel implementation has moved to `figure_8.ipynb`,
# which is the canonical Figure 8 notebook. The code below is retained as a
# legacy draft for comparison and should not be used for manuscript exports.
#
# This cell follows the four-panel manuscript template.  Spatial fields use the
# wrapped rectangular chart of the periodic triangular lattice.  Panel A uses
# progressively larger periodic random-walk ensembles for one matched neuron
# and, as a separate theoretical reference, its all-pairs tuning.  The latter is
# not the infinite-data limit of the 18-action random walk. Panel C organizes
# theoretical all-pairs tuning by irrep and construction phase. Panel D shows
# position and heading marginals of selected Panel C neurons using that same
# all-pairs definition.

# %%
paper_figure_directory = (
    project_root / "artifacts" / "constructed_networks" / "discrete_se2_c6" / "paper_figures"
)
paper_figure_directory.mkdir(parents=True, exist_ok=True)


def normalize_spatial_map(values):
    """Normalize one possibly masked map without inventing missing samples."""
    values = np.asarray(values, dtype=float)
    normalized = np.full_like(values, np.nan)
    valid = np.isfinite(values)
    if not np.any(valid):
        return normalized
    minimum = float(values[valid].min())
    maximum = float(values[valid].max())
    if np.isclose(minimum, maximum):
        normalized[valid] = 0.5
    else:
        normalized[valid] = (values[valid] - minimum) / (maximum - minimum)
    return normalized


# Show two visibly modulated translation-sensitive summary neurons from
# different irreps.  The three empirical columns pool independent periodic
# random walks with iid uniform local moves: six nonzero neighboring
# translations crossed with turns in {-1, 0, +1}.  The rightmost column is
# instead the all-pairs reference, averaged over all 600 group drives.
panel_a_relative_directions = (
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
)
panel_a_drive_elements = tuple(
    G.encode(*G.apply_rotation(turn, dx, dy), turn)
    for turn in (-1, 0, 1)
    for dx, dy in panel_a_relative_directions
)
panel_a_limit_tuning = compute_one_step_tuning(
    experiment,
    summary_units,
    panel_a_drive_elements,
    drive_batch_size=one_step_drive_batch_size,
)
panel_a_units = np.asarray((2095, 8229), dtype=int)
panel_a_irreps = tuple(int(params.metadata[int(unit)]["irrep_index"]) for unit in panel_a_units)
if len(set(panel_a_irreps)) != len(panel_a_units):
    raise RuntimeError("Panel A neurons must come from distinct irreps")
panel_a_limit_lookup = {
    int(unit): index for index, unit in enumerate(panel_a_limit_tuning.unit_indices)
}
panel_a_results = [
    [
        compute_empirical_trajectory_tuning(
            experiment,
            [int(unit)],
            motion_config=panel_a_motion_config,
            num_trajectories=count,
            steps_per_trajectory=panel_a_steps_per_trajectory,
            burn_in_steps=panel_a_burn_in_steps,
            seed=panel_a_seed,
            margin=0,
            batch_size=panel_a_batch_size,
        )
        for count in panel_a_trajectory_counts
    ]
    for unit in panel_a_units
]
panel_a_sample_counts = tuple(
    count * (panel_a_steps_per_trajectory - panel_a_burn_in_steps)
    for count in panel_a_trajectory_counts
)
panel_a_maps = []
panel_a_all_pairs_indices = []
for unit, empirical_results in zip(panel_a_units, panel_a_results, strict=True):
    all_pairs_index = int(np.flatnonzero(all_drive_tuning.unit_indices == unit)[0])
    panel_a_all_pairs_indices.append(all_pairs_index)
    unit_maps = [
        occupancy_normalized_activity(
            result.pose_activity_sums.sum(axis=0),
            result.pose_occupancy.sum(axis=0),
            min_occupancy=1,
        )[..., 0]
        for result in empirical_results
    ]
    unit_maps.append(
        np.roll(
            all_drive_tuning.position_mean[..., all_pairs_index],
            initial_pose[:2],
            axis=(0, 1),
        )
    )
    panel_a_maps.append(unit_maps)

# Four high-power translation-sensitive irreps.  Panel C places irreps in
# columns and orders its six rows by the effective phase generated jointly by
# delta and eps2.  eps1 and the matrix indices are fixed.
panel_c_irreps = tuning_irreps[:4]
panel_c_phase_specs = (
    (0, 1, 1, 0),
    (2, 1, -1, 60),
    (1, 1, 1, 120),
    (0, 1, -1, 180),
    (2, 1, 1, 240),
    (1, 1, -1, 300),
)
panel_c_row_specs = tuple(specification[:3] for specification in panel_c_phase_specs)
panel_c_units_by_irrep = {}
for irrep_index in panel_c_irreps:
    selected_for_irrep = []
    for delta, eps1, eps2 in panel_c_row_specs:
        unit = next(
            unit
            for unit, metadata in enumerate(params.metadata)
            if int(metadata["irrep_index"]) == int(irrep_index)
            and metadata["delta"] == delta
            and metadata["eps1"] == eps1
            and metadata["eps2"] == eps2
            and metadata["k0"] == 0
            and metadata["k1"] == 0
            and metadata["k2"] == 0
        )
        selected_for_irrep.append(unit)
    panel_c_units_by_irrep[irrep_index] = np.asarray(selected_for_irrep, dtype=int)
panel_c_units = np.concatenate(list(panel_c_units_by_irrep.values()))
panel_c_tuning = compute_one_step_tuning(
    experiment,
    panel_c_units,
    experiment.group.elements(),
    drive_batch_size=one_step_drive_batch_size,
)
panel_c_lookup = {int(unit): index for index, unit in enumerate(panel_c_tuning.unit_indices)}

# Alternative Panel C organization: hold the construction phase fixed and vary
# the output matrix row k0.  Under the uniform all-pairs average, changing k1
# or k2 alone leaves the positional tuning unchanged; k0 is the matrix index
# that produces distinct spatial maps.  The selected six-dimensional irreps
# therefore supply one row for every k0 in {0, ..., 5}.
panel_c_k0_values = tuple(range(G.m))
panel_c_k0_units_by_irrep = {}
for irrep_index in panel_c_irreps:
    selected_for_irrep = []
    for k0 in panel_c_k0_values:
        unit = next(
            unit
            for unit, metadata in enumerate(params.metadata)
            if int(metadata["irrep_index"]) == int(irrep_index)
            and metadata["delta"] == 0
            and metadata["eps1"] == 1
            and metadata["eps2"] == 1
            and metadata["k0"] == k0
            and metadata["k1"] == 0
            and metadata["k2"] == 0
        )
        selected_for_irrep.append(unit)
    panel_c_k0_units_by_irrep[irrep_index] = np.asarray(
        selected_for_irrep,
        dtype=int,
    )
panel_c_k0_units = np.concatenate(list(panel_c_k0_units_by_irrep.values()))
panel_c_k0_tuning = compute_one_step_tuning(
    experiment,
    panel_c_k0_units,
    experiment.group.elements(),
    drive_batch_size=one_step_drive_batch_size,
)
panel_c_k0_lookup = {int(unit): index for index, unit in enumerate(panel_c_k0_tuning.unit_indices)}

# Complete sign/phase sweep.  There are twelve construction triples because
# eps1 has two values for each of the six effective output phases.  Keeping
# both signs visible is useful: all-pairs averaging makes the pair redundant
# for some irreps, but not for every irrep (notably rho=9 in this experiment).
panel_c_phase_sign_specs = tuple(
    (delta, eps1, eps2, phase) for delta, _, eps2, phase in panel_c_phase_specs for eps1 in (1, -1)
)
panel_c_phase_sign_units_by_irrep = {}
for irrep_index in panel_c_irreps:
    selected_for_irrep = []
    for delta, eps1, eps2, _ in panel_c_phase_sign_specs:
        unit = next(
            unit
            for unit, metadata in enumerate(params.metadata)
            if int(metadata["irrep_index"]) == int(irrep_index)
            and metadata["delta"] == delta
            and metadata["eps1"] == eps1
            and metadata["eps2"] == eps2
            and metadata["k0"] == 0
            and metadata["k1"] == 0
            and metadata["k2"] == 0
        )
        selected_for_irrep.append(unit)
    panel_c_phase_sign_units_by_irrep[irrep_index] = np.asarray(
        selected_for_irrep,
        dtype=int,
    )
panel_c_phase_sign_units = np.concatenate(list(panel_c_phase_sign_units_by_irrep.values()))
panel_c_phase_sign_tuning = compute_one_step_tuning(
    experiment,
    panel_c_phase_sign_units,
    experiment.group.elements(),
    drive_batch_size=one_step_drive_batch_size,
)
panel_c_phase_sign_lookup = {
    int(unit): index for index, unit in enumerate(panel_c_phase_sign_tuning.unit_indices)
}


def relative_modulation(values):
    """Peak-to-peak modulation relative to a nonnegative curve's maximum."""
    values = np.asarray(values, dtype=float)
    return float(np.ptp(values) / max(float(np.max(np.abs(values))), 1e-12))


# Panel D contrasts three response types under the same all-pairs definition.
# These unit IDs are the deterministic result of the exact 23,724-unit
# audit documented alongside the figure.  Reuse them during ordinary figure
# renders so work on another panel does not repeat that expensive search.
flat_modulation_threshold = 1e-6
visible_modulation_threshold = 1e-3
panel_d_category_units = np.asarray(
    [6010, 23, 21287],
    dtype=int,
)
panel_d_category_labels = (
    "Spatial only",
    "Orientation only",
    "Conjunctive",
)
panel_d_tuning = compute_one_step_tuning(
    experiment,
    panel_d_category_units,
    experiment.group.elements(),
    drive_batch_size=one_step_drive_batch_size,
)
panel_d_pose_tuning = panel_d_tuning.pose_mean.reshape(
    G.m,
    G.n,
    G.n,
    len(panel_d_category_units),
)
panel_d_position_maps = panel_d_tuning.position_mean
panel_d_heading_curves = panel_d_pose_tuning.mean(axis=(1, 2))
panel_d_modulations = tuple(
    (
        relative_modulation(panel_d_position_maps[..., local_index]),
        relative_modulation(panel_d_heading_curves[..., local_index]),
    )
    for local_index in range(len(panel_d_category_units))
)

figure_8 = plt.figure(figsize=(18.0, 8.2))

# Panel A: empirical convergence for two neurons in different irreps.
panel_a_lefts = np.linspace(0.035, 0.285, 4)
panel_a_bottoms = (0.765, 0.615)
panel_a_axes = np.asarray(
    [
        [figure_8.add_axes((left, bottom, 0.075, 0.125)) for left in panel_a_lefts]
        for bottom in panel_a_bottoms
    ]
)
panel_a_titles = (
    *(rf"$N={sample_count:,}$" for sample_count in panel_a_sample_counts),
    "all-pairs\n(600 drives)",
)
for row, (unit, irrep_index, maps) in enumerate(
    zip(panel_a_units, panel_a_irreps, panel_a_maps, strict=True)
):
    for column, (ax, values) in enumerate(zip(panel_a_axes[row], maps, strict=True)):
        plot_lattice_scalar(
            normalize_spatial_map(values),
            ax=ax,
            title=panel_a_titles[column] if row == 0 else None,
            cmap="viridis",
            vmin=0,
            vmax=1,
            colorbar=False,
            coordinate_mode=tuning_coordinate_mode,
            wrap_periodic_edges=wrap_tuning_edges,
        )
    panel_a_axes[row, 0].annotate(
        f"unit {int(unit)}\n" + rf"$\rho={irrep_index}$",
        xy=(-0.16, 0.5),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.5,
    )
figure_8.text(0.025, 0.94, "A", fontsize=24, fontweight="bold", va="top")
figure_8.text(
    0.052,
    0.94,
    "Periodic random-walk tuning and all-pairs reference",
    fontsize=13,
    fontweight="semibold",
    va="top",
)

# Panel B: all one-dimensional translation characters, one C6 orbit repeated
# explicitly as a row, and the scalar spatial field obtained by summing that
# orbit.  We use odd n=11 so the frequency grid has a unique central cell.
panel_b_n = 11
panel_b_m = 6
panel_b_group = DiscreteSE2Group(panel_b_n, panel_b_m)
panel_b_frequency_labels = np.arange(-(panel_b_n // 2), panel_b_n // 2 + 1)
panel_b_position_y, panel_b_position_x = np.meshgrid(
    np.arange(panel_b_n),
    np.arange(panel_b_n),
    indexing="ij",
)


def panel_b_character(frequency):
    """Real part of one translation character on the spatial lattice."""
    first, second = frequency
    return np.cos(
        2 * np.pi * (first * panel_b_position_x + second * panel_b_position_y) / panel_b_n
    )


def panel_b_pair_key(frequency):
    """Canonical key shared by a centered frequency and its conjugate."""
    frequency = tuple(int(value) for value in frequency)
    conjugate = tuple(-value for value in frequency)
    return tuple(sorted((frequency, conjugate)))


panel_b_character_sheet = np.empty((panel_b_n**2, panel_b_n**2), dtype=float)
for tile_row, k2 in enumerate(panel_b_frequency_labels):
    for tile_column, k1 in enumerate(panel_b_frequency_labels):
        character_real_part = panel_b_character((k1, k2))
        panel_b_character_sheet[
            tile_row * panel_b_n : (tile_row + 1) * panel_b_n,
            tile_column * panel_b_n : (tile_column + 1) * panel_b_n,
        ] = character_real_part

panel_b_seed_frequency = (2, 0)
panel_b_orbit = next(
    orbit for orbit in panel_b_group.orbit_dict()[panel_b_m] if panel_b_seed_frequency in orbit
)
panel_b_centered_orbit = tuple(
    (
        (first + panel_b_n // 2) % panel_b_n - panel_b_n // 2,
        (second + panel_b_n // 2) % panel_b_n - panel_b_n // 2,
    )
    for first, second in panel_b_orbit
)
panel_b_pair_keys = []
for frequency in panel_b_centered_orbit:
    pair_key = panel_b_pair_key(frequency)
    if pair_key not in panel_b_pair_keys:
        panel_b_pair_keys.append(pair_key)
panel_b_pair_colors = dict(
    zip(
        panel_b_pair_keys,
        ("#0072B2", "#D55E00", "#009E73"),
        strict=True,
    )
)
panel_b_ordered_orbit = tuple(frequency for pair in panel_b_pair_keys for frequency in pair)
panel_b_orbit_pattern = sum(panel_b_character(frequency) for frequency in panel_b_centered_orbit)


def draw_character_orbit_panel(
    figure,
    character_bounds,
    orbit_bounds,
    pattern_bounds,
    *,
    compact=False,
):
    """Draw the character bank, its selected orbit, and the orbit sum."""
    character_ax = figure.add_axes(character_bounds)
    pattern_ax = figure.add_axes(pattern_bounds)
    minimum_frequency = int(panel_b_frequency_labels[0])
    maximum_frequency = int(panel_b_frequency_labels[-1])
    character_ax.imshow(
        panel_b_character_sheet,
        origin="lower",
        extent=(
            minimum_frequency - 0.5,
            maximum_frequency + 0.5,
            minimum_frequency - 0.5,
            maximum_frequency + 0.5,
        ),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    for boundary in panel_b_frequency_labels[:-1] + 0.5:
        character_ax.axhline(boundary, color="white", linewidth=0.22, alpha=0.55)
        character_ax.axvline(boundary, color="white", linewidth=0.22, alpha=0.55)
    for first, second in panel_b_centered_orbit:
        character_ax.add_patch(
            Rectangle(
                (first - 0.5, second - 0.5),
                1,
                1,
                fill=False,
                edgecolor=panel_b_pair_colors[panel_b_pair_key((first, second))],
                linewidth=2.0,
                zorder=4,
            )
        )
    for index, point in enumerate(panel_b_centered_orbit):
        next_point = panel_b_centered_orbit[(index + 1) % len(panel_b_centered_orbit)]
        character_ax.annotate(
            "",
            xy=next_point,
            xytext=point,
            arrowprops={
                "arrowstyle": "->",
                "color": "0.15",
                "lw": 0.7,
                "shrinkA": 8,
                "shrinkB": 8,
            },
            zorder=5,
        )
    character_ax.set(
        title=rf"All $C_{{{panel_b_n}}}^2$ characters $\chi_k$",
        xlabel=r"frequency $k_1$",
        ylabel=r"frequency $k_2$",
        xticks=(minimum_frequency, 0, maximum_frequency),
        yticks=(minimum_frequency, 0, maximum_frequency),
    )
    character_ax.tick_params(labelsize=5 if compact else 8, length=2)
    character_ax.xaxis.label.set_size(6 if compact else 9)
    character_ax.yaxis.label.set_size(6 if compact else 9)
    character_ax.title.set_size(7 if compact else 11)

    orbit_left, orbit_bottom, orbit_width, orbit_height = orbit_bounds
    orbit_gap = orbit_height * 0.02
    orbit_tile_height = (orbit_height - 5 * orbit_gap) / 6
    orbit_axes = []
    for row, frequency in enumerate(panel_b_ordered_orbit):
        orbit_ax = figure.add_axes(
            (
                orbit_left,
                orbit_bottom + (5 - row) * (orbit_tile_height + orbit_gap),
                orbit_width,
                orbit_tile_height,
            )
        )
        orbit_ax.imshow(
            panel_b_character(frequency),
            origin="lower",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
            aspect="equal",
        )
        color = panel_b_pair_colors[panel_b_pair_key(frequency)]
        for spine in orbit_ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5 if compact else 2.4)
        orbit_ax.set_xticks([])
        orbit_ax.set_yticks([])
        if not compact:
            orbit_ax.annotate(
                rf"$k=({frequency[0]},{frequency[1]})$",
                xy=(1.12, 0.5),
                xycoords="axes fraction",
                ha="left",
                va="center",
                fontsize=8,
            )
        orbit_axes.append(orbit_ax)
    figure.text(
        orbit_left + orbit_width / 2,
        orbit_bottom + orbit_height + (0.018 if compact else 0.045),
        "One $C_6$ orbit\nthree conjugate pairs",
        ha="center",
        va="bottom",
        fontsize=6 if compact else 11,
    )

    plot_lattice_scalar(
        normalize_spatial_map(panel_b_orbit_pattern),
        ax=pattern_ax,
        title=r"Orbit sum $\mathrm{Re}\!\sum_{q\in\mathcal{O}_k}\chi_q(x)$",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        colorbar=False,
        coordinate_mode="offset",
        wrap_periodic_edges=True,
    )
    pattern_ax.title.set_size(7 if compact else 11)

    first_arrow_x = (character_bounds[0] + character_bounds[2] + orbit_left) / 2
    second_arrow_x = (orbit_left + orbit_width + pattern_bounds[0]) / 2
    arrow_y = orbit_bottom + orbit_height / 2
    for arrow_x in (first_arrow_x, second_arrow_x):
        figure.text(
            arrow_x,
            arrow_y,
            r"$\longrightarrow$",
            fontsize=11 if compact else 20,
            ha="center",
            va="center",
        )
    return character_ax, orbit_axes, pattern_ax


frequency_ax, orbit_axes, pattern_ax = draw_character_orbit_panel(
    figure_8,
    (0.045, 0.145, 0.100, 0.33),
    (0.185, 0.160, 0.035, 0.30),
    (0.260, 0.145, 0.100, 0.33),
    compact=True,
)
figure_8.text(0.025, 0.585, "B", fontsize=24, fontweight="bold", va="top")
figure_8.text(
    0.052,
    0.585,
    rf"A $C_6$ orbit of $C_{{{panel_b_n}}}^2$ characters forms a spatial irrep",
    fontsize=13,
    fontweight="semibold",
    va="top",
)

# Panel C: columns are irreps and rows follow the cyclic phase order.  All six
# maps within an irrep share one color scale so phase shifts can be compared
# without erasing response-amplitude differences.
panel_c_axes = np.asarray(
    [
        [
            figure_8.add_axes(
                (
                    0.425 + 0.088 * column,
                    0.720 - 0.120 * row,
                    0.078,
                    0.105,
                )
            )
            for column in range(len(panel_c_irreps))
        ]
        for row in range(len(panel_c_phase_specs))
    ]
)
panel_c_display_maps = {}
for irrep_index in panel_c_irreps:
    irrep_maps = np.stack(
        [
            panel_c_tuning.position_mean[..., panel_c_lookup[int(unit)]]
            for unit in panel_c_units_by_irrep[irrep_index]
        ]
    )
    irrep_minimum = float(np.nanmin(irrep_maps))
    irrep_maximum = float(np.nanmax(irrep_maps))
    panel_c_display_maps[irrep_index] = (
        np.full_like(irrep_maps, 0.5)
        if np.isclose(irrep_minimum, irrep_maximum)
        else (irrep_maps - irrep_minimum) / (irrep_maximum - irrep_minimum)
    )
for column, irrep_index in enumerate(panel_c_irreps):
    for row, (delta, _, eps2, phase) in enumerate(panel_c_phase_specs):
        plot_lattice_scalar(
            panel_c_display_maps[irrep_index][row],
            ax=panel_c_axes[row, column],
            title=rf"$\rho={irrep_index}$" if row == 0 else None,
            cmap="viridis",
            vmin=0,
            vmax=1,
            colorbar=False,
            coordinate_mode=tuning_coordinate_mode,
            wrap_periodic_edges=wrap_tuning_edges,
        )
        if column == 0:
            panel_c_axes[row, column].annotate(
                rf"$\phi={phase}^\circ$" + "\n" + rf"$(\delta,\epsilon_2)=({delta},{eps2:+d})$",
                xy=(-0.10, 0.5),
                xycoords="axes fraction",
                ha="right",
                va="center",
                fontsize=6.5,
            )
figure_8.text(0.385, 0.94, "C", fontsize=24, fontweight="bold", va="top")
figure_8.text(
    0.412,
    0.94,
    "All-pairs spatial tuning ordered by effective phase",
    fontsize=13,
    fontweight="semibold",
    va="top",
)
figure_8.text(
    0.585,
    0.075,
    r"$\epsilon_1=+1$; $k_0=k_1=k_2=0$; shared scale within each $\rho$",
    fontsize=7.5,
    color="0.35",
    ha="center",
)

# Panel D: three qualitatively different all-pairs selectivity profiles.
figure_8.text(0.795, 0.94, "D", fontsize=24, fontweight="bold", va="top")
figure_8.text(
    0.822,
    0.94,
    "Spatial and orientation selectivity",
    fontsize=13,
    fontweight="semibold",
    va="top",
)
figure_8.text(0.858, 0.875, "Position", fontsize=10, ha="center")
figure_8.text(0.950, 0.875, "Orientation", fontsize=10, ha="center")
panel_d_position_axes = []
panel_d_heading_axes = []
for row, (category, unit) in enumerate(
    zip(panel_d_category_labels, panel_d_category_units, strict=True)
):
    bottom = 0.630 - 0.235 * row
    position_ax = figure_8.add_axes((0.825, bottom, 0.068, 0.145))
    heading_ax = figure_8.add_axes((0.915, bottom + 0.015, 0.070, 0.115))
    panel_d_position_axes.append(position_ax)
    panel_d_heading_axes.append(heading_ax)
    plot_lattice_scalar(
        normalize_spatial_map(panel_d_position_maps[..., row]),
        ax=position_ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        colorbar=False,
        coordinate_mode=tuning_coordinate_mode,
        wrap_periodic_edges=wrap_tuning_edges,
    )
    heading_values = panel_d_heading_curves[..., row]
    heading_values = normalize_spatial_map(heading_values[:, None]).ravel()
    closed_headings = np.arange(G.m + 1)
    closed_values = np.append(heading_values, heading_values[0])
    heading_ax.plot(
        closed_headings,
        closed_values,
        color="0.2",
        linewidth=1.3,
    )
    if panel_d_modulations[row][1] > visible_modulation_threshold:
        preferred_heading = int(np.nanargmax(heading_values))
        heading_ax.scatter(
            preferred_heading,
            heading_values[preferred_heading],
            s=18,
            color="#df6b66",
            edgecolor="none",
            zorder=3,
        )
    heading_ax.set(
        xlim=(0, G.m),
        ylim=(0, 1.05),
        xticks=(0, G.m // 2, G.m),
        xticklabels=(r"$0^\circ$", r"$180^\circ$", r"$360^\circ$"),
        yticks=[],
    )
    heading_ax.tick_params(axis="x", labelsize=6, length=2, pad=1)
    heading_ax.spines[["top", "right", "left"]].set_visible(False)
    heading_ax.spines["bottom"].set_color("0.6")
    metadata = params.metadata[int(unit)]
    spatial_modulation, heading_modulation = panel_d_modulations[row]
    figure_8.text(
        0.815,
        bottom + 0.072,
        category
        + "\n"
        + rf"$\rho={metadata['irrep_index']}$, unit {int(unit)}"
        + "\n"
        + rf"$M_x={spatial_modulation:.3f},\ M_\theta={heading_modulation:.3f}$",
        ha="right",
        va="center",
        fontsize=6.5,
    )

for suffix in ("pdf", "png", "svg"):
    figure_8.savefig(
        paper_figure_directory / f"figure_8_draft.{suffix}",
        dpi=300,
        bbox_inches="tight",
    )

# Export Panel B separately for manuscript assembly and Illustrator editing.
figure_8_panel_b = plt.figure(figsize=(11.5, 7.0))
draw_character_orbit_panel(
    figure_8_panel_b,
    (0.055, 0.16, 0.300, 0.68),
    (0.475, 0.16, 0.075, 0.68),
    (0.680, 0.16, 0.270, 0.68),
)
figure_8_panel_b.text(
    0.01,
    0.96,
    "B",
    fontsize=24,
    fontweight="bold",
    va="top",
)
figure_8_panel_b.text(
    0.045,
    0.95,
    rf"A $C_6$ orbit of $C_{{{panel_b_n}}}^2$ characters forms a spatial irrep",
    fontsize=14,
    fontweight="semibold",
    va="top",
)
for suffix in ("pdf", "png", "svg"):
    figure_8_panel_b.savefig(
        paper_figure_directory / f"figure_8_panel_b.{suffix}",
        dpi=300,
        bbox_inches="tight",
    )

# Side-by-side Panel C comparison.  The left block varies the effective phase
# at fixed matrix indices; the right block varies k0 at fixed phase parameters.
figure_8_panel_c_comparison, panel_c_comparison_axes = plt.subplots(
    len(panel_c_phase_specs),
    9,
    figsize=(13.5, 8.5),
    gridspec_kw={"width_ratios": (1, 1, 1, 1, 0.18, 1, 1, 1, 1)},
)
figure_8_panel_c_comparison.subplots_adjust(
    left=0.065,
    right=0.985,
    bottom=0.045,
    top=0.875,
    wspace=0.10,
    hspace=0.12,
)
for row in range(len(panel_c_phase_specs)):
    panel_c_comparison_axes[row, 4].remove()
for column, irrep_index in enumerate(panel_c_irreps):
    for row, (delta, eps1, eps2, phase) in enumerate(panel_c_phase_specs):
        unit = next(
            int(unit)
            for unit in panel_c_units_by_irrep[irrep_index]
            if params.metadata[int(unit)]["delta"] == delta
            and params.metadata[int(unit)]["eps1"] == eps1
            and params.metadata[int(unit)]["eps2"] == eps2
        )
        position_map = panel_c_tuning.position_mean[
            ...,
            panel_c_lookup[int(unit)],
        ]
        plot_lattice_scalar(
            normalize_spatial_map(position_map),
            ax=panel_c_comparison_axes[row, column],
            title=(rf"$\rho={irrep_index}$" if row == 0 else None),
            cmap="viridis",
            vmin=0,
            vmax=1,
            colorbar=False,
            coordinate_mode=tuning_coordinate_mode,
            wrap_periodic_edges=wrap_tuning_edges,
        )
        if column == 0:
            panel_c_comparison_axes[row, column].annotate(
                rf"$\phi={phase}^\circ$",
                xy=(-0.08, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="right",
                va="center",
                fontsize=9,
            )

    comparison_column = column + 5
    for row, unit in enumerate(panel_c_k0_units_by_irrep[irrep_index]):
        position_map = panel_c_k0_tuning.position_mean[
            ...,
            panel_c_k0_lookup[int(unit)],
        ]
        plot_lattice_scalar(
            normalize_spatial_map(position_map),
            ax=panel_c_comparison_axes[row, comparison_column],
            title=(rf"$\rho={irrep_index}$" if row == 0 else None),
            cmap="viridis",
            vmin=0,
            vmax=1,
            colorbar=False,
            coordinate_mode=tuning_coordinate_mode,
            wrap_periodic_edges=wrap_tuning_edges,
        )
        if column == 0:
            panel_c_comparison_axes[row, comparison_column].annotate(
                rf"$k_0={panel_c_k0_values[row]}$",
                xy=(-0.08, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="right",
                va="center",
                fontsize=9,
            )

figure_8_panel_c_comparison.text(
    0.265,
    0.96,
    r"Vary phase; fix $k_0=k_1=k_2=0$",
    ha="center",
    va="top",
    fontsize=15,
    fontweight="semibold",
)
figure_8_panel_c_comparison.text(
    0.755,
    0.96,
    r"Vary $k_0$; fix $(\delta,\epsilon_1,\epsilon_2)=(0,+1,+1)$ and $k_1=k_2=0$",
    ha="center",
    va="top",
    fontsize=15,
    fontweight="semibold",
)
for suffix in ("png", "svg"):
    figure_8_panel_c_comparison.savefig(
        paper_figure_directory / f"figure_8_panel_c_phase_vs_k0.{suffix}",
        dpi=300,
        bbox_inches="tight",
    )

# Complete 12-row sign/phase diagnostic at fixed matrix indices.
figure_8_panel_c_phase_sign, panel_c_phase_sign_axes = plt.subplots(
    len(panel_c_phase_sign_specs),
    len(panel_c_irreps),
    figsize=(7.2, 15.0),
)
figure_8_panel_c_phase_sign.subplots_adjust(
    left=0.14,
    right=0.985,
    bottom=0.025,
    top=0.925,
    wspace=0.10,
    hspace=0.10,
)
for column, irrep_index in enumerate(panel_c_irreps):
    irrep_phase_sign_maps = np.stack(
        [
            panel_c_phase_sign_tuning.position_mean[
                ...,
                panel_c_phase_sign_lookup[int(unit)],
            ]
            for unit in panel_c_phase_sign_units_by_irrep[irrep_index]
        ]
    )
    irrep_minimum = float(np.nanmin(irrep_phase_sign_maps))
    irrep_maximum = float(np.nanmax(irrep_phase_sign_maps))
    for row, unit in enumerate(panel_c_phase_sign_units_by_irrep[irrep_index]):
        position_map = panel_c_phase_sign_tuning.position_mean[
            ...,
            panel_c_phase_sign_lookup[int(unit)],
        ]
        if np.isclose(irrep_minimum, irrep_maximum):
            displayed_position_map = np.full_like(position_map, 0.5)
        else:
            displayed_position_map = (position_map - irrep_minimum) / (
                irrep_maximum - irrep_minimum
            )
        plot_lattice_scalar(
            displayed_position_map,
            ax=panel_c_phase_sign_axes[row, column],
            title=(rf"$\rho={irrep_index}$" if row == 0 else None),
            cmap="viridis",
            vmin=0,
            vmax=1,
            colorbar=False,
            coordinate_mode=tuning_coordinate_mode,
            wrap_periodic_edges=wrap_tuning_edges,
        )
        if column == 0:
            delta, eps1, eps2, phase = panel_c_phase_sign_specs[row]
            panel_c_phase_sign_axes[row, column].annotate(
                rf"$\phi={phase}^\circ,\ \epsilon_1={eps1:+d}$"
                + "\n"
                + rf"$(\delta,\epsilon_2)=({delta},{eps2:+d})$",
                xy=(-0.08, 0.5),
                xycoords="axes fraction",
                rotation=90,
                ha="right",
                va="center",
                fontsize=8,
            )
figure_8_panel_c_phase_sign.suptitle(
    r"All 12 $(\delta,\epsilon_1,\epsilon_2)$ combinations; "
    r"fix $k_0=k_1=k_2=0$; shared scale within each $\rho$",
    y=0.975,
    fontsize=15,
    fontweight="semibold",
)
for suffix in ("png", "svg"):
    figure_8_panel_c_phase_sign.savefig(
        paper_figure_directory / f"figure_8_panel_c_all_phase_signs.{suffix}",
        dpi=300,
        bbox_inches="tight",
    )
plt.show()

print("Figure 8 Panel A units and irreps:", tuple(zip(panel_a_units, panel_a_irreps)))
for unit, empirical_results, all_pairs_index in zip(
    panel_a_units,
    panel_a_results,
    panel_a_all_pairs_indices,
    strict=True,
):
    local_index = panel_a_limit_lookup[int(unit)]
    local_pose = panel_a_limit_tuning.pose_mean[..., local_index].reshape(
        G.m,
        G.n,
        G.n,
    )
    local_pose = np.roll(local_pose, initial_pose[:2], axis=(1, 2))
    all_pairs_pose = all_drive_tuning.pose_mean[..., all_pairs_index].reshape(
        G.m,
        G.n,
        G.n,
    )
    all_pairs_pose = np.roll(all_pairs_pose, initial_pose[:2], axis=(1, 2))
    empirical_pose = occupancy_normalized_activity(
        empirical_results[-1].pose_activity_sums,
        empirical_results[-1].pose_occupancy,
        min_occupancy=1,
    )[..., 0]
    print(
        f"  unit {int(unit)}: N=100,000 correlation with "
        f"exact walk={masked_correlation(empirical_pose, local_pose):.3f}, "
        f"all pairs={masked_correlation(empirical_pose, all_pairs_pose):.3f}"
    )
print("Figure 8 panel A exact sample counts:", panel_a_sample_counts)
print("Figure 8 panel A local increments:", len(panel_a_drive_elements))
print("Figure 8 panel C irreps:", panel_c_irreps)
print("Figure 8 panel C construction-grid units:", panel_c_units.tolist())
print("Figure 8 panel C k0-sweep units:", panel_c_k0_units.tolist())
print("Figure 8 panel D categories:")
for category, unit, (spatial_modulation, heading_modulation) in zip(
    panel_d_category_labels,
    panel_d_category_units,
    panel_d_modulations,
    strict=True,
):
    print(
        f"  {category}: unit {int(unit)}, "
        f"rho={params.metadata[int(unit)]['irrep_index']}, "
        f"spatial={spatial_modulation:.6f}, heading={heading_modulation:.6f}"
    )
