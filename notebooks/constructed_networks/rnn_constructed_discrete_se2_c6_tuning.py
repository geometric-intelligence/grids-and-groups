# %%
# Percent-format notebook source. Regenerate the paired .ipynb with Jupytext.

# %% [markdown]
# # Tuning analysis for the constructed $C_6$ RNN
#
# This notebook owns empirical trajectory tuning and both exhaustive theoretical
# definitions: all drive and local drive tuning. It intentionally excludes
# group-action pedagogy and neural-manifold analysis.

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
    parent for parent in (Path.cwd(), *Path.cwd().parents) if (parent / "src").is_dir()
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
cache_schema_version = 1  # Increment after changing cached artifact semantics.

# ----------------------------
# Neuron and module selection
# ----------------------------
num_summary_neurons = 8  # Conjugate modules represented by one high-variance unit.
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
tuning_cache_directory = project_root / "artifacts" / "constructed_networks" / "discrete_se2_c6"

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
ranked_module_representatives = sorted(
    [
        (
            module,
            max(
                module.unit_indices,
                key=lambda unit: float(trajectory_variances[unit]),
            ),
        )
        for module in module_orbits
    ],
    key=lambda item: float(trajectory_variances[item[1]]),
    reverse=True,
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
            coordinate_mode="axial",
        )
    plot_lattice_scalar(
        tensor.mean(axis=0),
        ax=static_axes[row, G.m],
        title=f"{irrep_label}\nunit {unit}\n{mode_label}\nheading mean",
        vmin=unit_minimum,
        vmax=unit_maximum,
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
        empirical_tuning.position_tuning[..., empirical_tuning.local_unit_index(int(unit))]
        for unit in summary_units
    ],
    axis=-1,
)
local_drive_tuning = compute_local_arrival_tuning(
    experiment,
    summary_units,
    drive_batch_size=exhaustive_drive_batch_size,
)
all_drive_tuning = compute_all_pairs_tuning(
    experiment,
    summary_units,
    drive_batch_size=exhaustive_drive_batch_size,
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
            "D. Exhaustive local drive tuning",
        ),
        (
            all_drive_subfigure,
            "E. Exhaustive all drive tuning",
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
        empirical_tuning.pose_tuning[..., empirical_tuning.local_unit_index(int(unit))]
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
    ("Exhaustive local drive", local_drive_tuning.position_mean),
    ("Exhaustive all drive", all_drive_tuning.position_mean),
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
for column, (irrep_label, mode_label, unit) in enumerate(
    zip(
        summary_irrep_labels,
        summary_mode_labels,
        summary_units,
    )
):
    local_index = empirical_tuning.local_unit_index(int(unit))
    autocorrelation = masked_periodic_spatial_autocorrelation(
        empirical_tuning.position_tuning[..., local_index]
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
# Empirical trajectory, exhaustive all drive, and exhaustive local drive tuning
# now have separate result objects and a matched comparison. The empirical cache
# remains valid across all plotting edits.
