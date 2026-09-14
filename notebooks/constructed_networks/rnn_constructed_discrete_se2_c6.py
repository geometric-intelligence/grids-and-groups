# %%
# Percent-format notebook source. Regenerate the paired .ipynb with Jupytext.

# %% [markdown]
# # Closed-form RNN on $\mathbb{Z}_n^2\rtimes C_6$
#
# This notebook contains only group conventions, signal encodings, analytical
# network construction, and one naturalistic rollout. Tuning and neural-manifold
# analyses live in their own notebooks.

# %%
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from IPython import get_ipython
from IPython.display import HTML, display

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

from src.experiments.discrete_se2 import (  # noqa: E402
    DiscreteSE2ExperimentConfig,
    DiscreteSE2RolloutConfig,
    build_discrete_se2_experiment,
    run_discrete_se2_rollout,
)
from src.geometry.discrete_se2.core import (  # noqa: E402
    lattice_coordinates,
    lattice_path_coordinates,
    lattice_path_segments,
)
from src.geometry.discrete_se2.plotting import (  # noqa: E402
    linked_plotly_html,
    plot_lattice_scalar,
    plotly_heading_stacks,
)
from src.geometry.discrete_se2.trajectories import NaturalisticMotionConfig  # noqa: E402
from src.groups.opposite import as_action_group  # noqa: E402

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1. Configuration and construction
#
# $C_6$ supplies six headings separated by $60^\circ$, aligned with the six
# nearest-neighbor directions of the triangular lattice. The right action uses
# body-frame updates $s\mapsto sg$; the left action uses world-frame updates
# $s\mapsto gs$.
#
# Every scientific and computational choice is listed in the next cell. The
# dataclasses only validate and package these visible values; they do not supply
# hidden defaults.

# %%
# ----------------------------
# Group and signal encoding
# ----------------------------
n_spatial = 21
n_orientations = 6
initial_pose = (2, 2, 0)
allocentric_encoding = "gaussian space custom orientation"
allocentric_sigma = 1.0
custom_orientation_weights = (1.0, 0.8, 0.4, 0.2, 0.4, 0.8)
egocentric_encoding_seed = 10
action_side = "right"  # "right": body-frame s*g; "left": world-frame g*s.

# ----------------------------
# Closed-form network
# ----------------------------
irrep_selection = "power"
num_selected_irreps = None
max_hidden_width = None  # Full construction, matching Figure 8 (H = 189,576).
normalize_power_by_dimension = True
always_include_trivial_irrep = True
power_ranking = "power"
q_rho = 3
amplitude_mode = "balanced"
amplitude_multipliers = (1.0, 1.0, 1.0)
materialize_recurrent_matrix = False

# ----------------------------
# Pure periodic local random walk, matching Figure 8. The seven translations
# (stay plus six neighbours) and three relative turns are each uniform.
# ----------------------------
stay_probability = 1 / 7
forward_probability = 1 / 7
forward_left_or_right_probability = 1 / 7
backward_left_or_right_probability = 1 / 7
backward_probability = 1 / 7
turn_probability = 1 / 3
turn_persistence = 0
wall_lookahead = 1
wall_avoidance_strength = 0
minimum_wall_weight = 1

# ----------------------------
# Primary rollout
# ----------------------------
num_rollout_steps = 100
rollout_seed = 31
rollout_margin = 0  # Required for the periodic Figure 8 random walk.
rollout_start_xy = (n_spatial // 2, n_spatial // 2)

# ----------------------------
# Rollout visualization
# ----------------------------
orientation_arrow_stride = 2

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
    periodic_boundaries=True,
)
rollout_config = DiscreteSE2RolloutConfig(
    steps=num_rollout_steps,
    seed=rollout_seed,
    margin=rollout_margin,
    start_xy=rollout_start_xy,
    arrow_stride=orientation_arrow_stride,
    snapshot_steps=(),
)

# %%
experiment = build_discrete_se2_experiment(experiment_config)
G = experiment.group
params = experiment.model
x_allo = experiment.x_allo
x_ego = experiment.x_ego
if torch.cuda.is_available():
    params.to("cuda")
    print("Moved the full construction to CUDA.")

power = G.power_spectrum(x_allo)
retained_power = power[params.selected_irrep_indices].sum() / power.sum()
perfect_reconstruction_width = sum(
    4 * experiment_config.q_rho * irrep.dim**2 for irrep in experiment.irreps
)
print(f"|G|: {G.order}")
print(f"action side: {experiment_config.action_side}")
print(f"selected irreps: {len(params.irreps)}/{len(experiment.irreps)}")
print(f"hidden width: {params.hidden_dim:,}")
print(f"theoretical perfect-reconstruction width (4q Σρ dim(ρ)²): {perfect_reconstruction_width:,}")
print(f"retained Fourier power: {retained_power:.4%}")
print("experiment configuration:", experiment_config)

# %% [markdown]
# ## 2. Template and regular actions
#
# For $g=(t,r)$, multiplication is
#
# $$(t_1,r_1)(t_2,r_2)=(t_1+A^{r_1}t_2,r_1+r_2),\qquad
# A=\begin{pmatrix}0&-1\\1&1\end{pmatrix}.$$
#
# The left action is $(L_gx)(h)=x(g^{-1}h)$; the right action is
# $(R_gx)(h)=x(hg^{-1})$. The views below share a linked camera.

# %%
g_x = G.encode(1, 0, 0)
g_y = G.encode(0, 1, 0)
g_rotation = G.encode(0, 0, 1)
right_group = as_action_group(G, "right")

left_signals = [
    x_allo,
    G.left_action(g_x, x_allo),
    G.left_action(g_y, x_allo),
    G.left_action(g_rotation, x_allo),
]
right_signals = [
    x_allo,
    right_group.left_action(g_x, x_allo),
    right_group.left_action(g_y, x_allo),
    right_group.left_action(g_rotation, x_allo),
]
action_titles = [
    "Original x<sub>allo</sub>",
    "unit x translation",
    "unit y translation",
    "60° rotation",
]

# %%
left_figure = plotly_heading_stacks(
    G,
    left_signals,
    titles=[f"Left: {title}" for title in action_titles],
    arrangement="horizontal",
    width=1050,
    height=560,
)
display(
    HTML(
        linked_plotly_html(
            left_figure,
            [("scene", f"scene{index}") for index in range(2, 5)],
        )
    )
)

# %%
right_figure = plotly_heading_stacks(
    G,
    right_signals,
    titles=[f"Right: {title}" for title in action_titles],
    arrangement="horizontal",
    width=1050,
    height=560,
)
display(
    HTML(
        linked_plotly_html(
            right_figure,
            [("scene", f"scene{index}") for index in range(2, 5)],
        )
    )
)

# %% [markdown]
# ## 3. Pure periodic random-walk policy
#
# This is the Figure 8 motion distribution: at every step, choose one of the
# seven local translations (stay plus six neighbours) and one of the three
# relative turns independently and uniformly. Thus every one of the
# $7\times3=21$ local egocentric actions has probability $1/21$. There is no
# momentum, wall avoidance, or turn persistence; the spatial domain is periodic.

# %%
print("motion configuration:", motion_config)
local_support = np.zeros(G.order)
local_support[list(experiment.local_egocentric_elements)] = 1.0
support_figure = plotly_heading_stacks(
    G,
    [local_support],
    titles=[f"Right-action local support ({len(experiment.local_egocentric_elements)} elements)"],
    highlighted_elements=[experiment.local_egocentric_elements],
    coordinate_mode="centered_axial",
    width=720,
    height=560,
)
display(HTML(linked_plotly_html(support_figure)))

# %% [markdown]
# ## 4. Pure random-walk rollout
#
# This is the only cell to rerun after changing the motion or rollout
# configuration. It constructs the trajectory, evaluates the network, decodes
# pose, and packages all diagnostics in one object.

# %%
rollout = run_discrete_se2_rollout(
    experiment,
    rollout_config,
    motion_config,
)
print(f"steps: {len(rollout.sequence)}")
print(
    "relative output error: "
    f"mean={rollout.relative_output_errors.mean():.3e}, "
    f"max={rollout.relative_output_errors.max():.3e}"
)
print(
    f"center error: mean={rollout.center_errors.mean():.3e}, max={rollout.center_errors.max():.3e}"
)
print(f"unique positions: {rollout.unique_positions}/{len(rollout.sequence)}")
print(f"heading changes: {rollout.heading_changes}")
print(f"stationary steps: {rollout.stationary_steps}")
print(f"immediate reversals: {rollout.immediate_reversals}")

# %%
figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
lattice_x, lattice_y = lattice_coordinates(G.n, mode="axial")
steps = np.arange(1, len(rollout.sequence) + 1)
time_norm = mcolors.Normalize(steps[0], steps[-1])

for ax, centers, title in (
    (axes[0], rollout.exact_centers, "Ground-truth pose trajectory"),
    (
        axes[1],
        rollout.predicted_centers,
        "Joint template-orbit decoded trajectory",
    ),
):
    points = lattice_path_coordinates(centers, G.n, mode="axial")
    ax.scatter(lattice_x, lattice_y, s=8, color="0.88", linewidths=0)
    ax.plot(points[:, 0], points[:, 1], color="0.65", linewidth=1)
    artist = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=steps,
        cmap="viridis",
        norm=time_norm,
        s=28,
        linewidths=0,
    )
    ax.scatter(*points[0], s=70, facecolors="none", edgecolors="black")
    ax.scatter(*points[-1], s=90, marker="*", color="black")
    ax.set(title=title, aspect="equal", xticks=[], yticks=[])
    ax.set_frame_on(False)
figure.colorbar(artist, ax=axes, fraction=0.03, label="time step")
figure.suptitle("Pose tracking: exact group composition vs. joint signal decoding")
plt.show()

# %% [markdown]
# ### Rollout accuracy
#
# Spatial and orientation accuracy are exact-match indicators for the jointly
# decoded pose. Reconstruction accuracy is
#
# $$a_t=\max\left(0,\ 1-
# \frac{\|\hat{x}_t-x_t\|_2}{\|x_t\|_2}\right).$$
#
# Each panel shows the per-step value faintly and its cumulative mean as a solid
# line, so both transient failures and overall rollout performance remain visible.

# %%
spatial_accuracy = np.isclose(
    rollout.center_errors,
    0,
    atol=1e-10,
).astype(float)
orientation_accuracy = np.isclose(
    rollout.orientation_errors,
    0,
    atol=1e-10,
).astype(float)
reconstruction_accuracy = np.clip(
    1 - rollout.relative_output_errors,
    0,
    1,
)

figure, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4),
    constrained_layout=True,
    sharex=True,
    sharey=True,
)
for ax, accuracy, title in zip(
    axes,
    (
        spatial_accuracy,
        orientation_accuracy,
        reconstruction_accuracy,
    ),
    (
        "Decoded spatial accuracy",
        "Decoded orientation accuracy",
        "Signal reconstruction accuracy",
    ),
):
    cumulative_accuracy = np.cumsum(accuracy) / steps
    ax.plot(
        steps,
        accuracy,
        color="0.7",
        linewidth=1,
        marker=".",
        markersize=4,
        label="per step",
    )
    ax.plot(
        steps,
        cumulative_accuracy,
        color="tab:blue",
        linewidth=2.2,
        label="cumulative mean",
    )
    ax.set(
        title=f"{title}\nfinal cumulative={cumulative_accuracy[-1]:.2%}",
        xlabel="time step",
        ylim=(-0.05, 1.05),
    )
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
axes[0].set_ylabel("accuracy")
figure.suptitle("Decoded pose and full-signal reconstruction over the rollout")
plt.show()

# %% [markdown]
# ## 5. Paper Figure 7 draft
#
# The paper layout overlays the true and decoded paths in the wrapped rectangular
# chart of the periodic triangular lattice.  The background uses the same
# wrapped offset hex-grid renderer as the Figure 8 tuning curves. Lines are split at chart seams, so
# the cut-and-paste display never introduces a false long trajectory segment.
# The lower-left panel summarizes decoding accuracy, while the right column shows
# one high-variance neuron from each of eight translation-sensitive irreps.

# %%
paper_figure_directory = (
    project_root
    / "artifacts"
    / "constructed_networks"
    / "discrete_se2_c6"
    / "paper_figures"
)
paper_figure_directory.mkdir(parents=True, exist_ok=True)

trajectory_variances = np.var(rollout.hidden_states, axis=0)
units_by_irrep = {}
for unit, metadata in enumerate(params.metadata):
    if metadata["irrep_dim"] > 1:
        units_by_irrep.setdefault(int(metadata["irrep_index"]), []).append(unit)


def highest_variance_unit(unit_indices):
    """Choose the lowest-index unit when symmetry makes variances effectively tie."""
    unit_indices = np.asarray(unit_indices, dtype=int)
    values = trajectory_variances[unit_indices]
    maximum = values.max()
    tied = unit_indices[np.isclose(values, maximum, rtol=1e-10, atol=1e-12)]
    return int(tied.min())


figure_7_representatives = sorted(
    (highest_variance_unit(units) for units in units_by_irrep.values()),
    key=lambda unit: (-round(float(trajectory_variances[unit]), 12), unit),
)[:8]

figure_7_activity = rollout.hidden_states[:, figure_7_representatives]
activity_minimum = figure_7_activity.min(axis=0, keepdims=True)
activity_span = np.ptp(figure_7_activity, axis=0, keepdims=True)
figure_7_activity = (figure_7_activity - activity_minimum) / np.where(
    activity_span > 0,
    activity_span,
    1,
)


def time_colored_path(points, n, *, colormap, normalization):
    """Return seam-safe display segments and their associated trajectory times."""
    coordinates = lattice_path_coordinates(points, n, mode="offset")
    segments = np.stack((coordinates[:-1], coordinates[1:]), axis=1)
    keep = np.linalg.norm(np.diff(coordinates, axis=0), axis=1) <= 1.5
    return segments[keep], np.arange(len(segments))[keep] + 1


def time_colored_heading(steps, values, *, colormap, normalization):
    """Return horizontal and vertical segments for a time-coloured mid-step trace."""
    boundaries = np.r_[steps[0], (steps[:-1] + steps[1:]) / 2, steps[-1]]
    horizontal = np.stack(
        (np.column_stack((boundaries[:-1], values)), np.column_stack((boundaries[1:], values))),
        axis=1,
    )
    vertical = np.stack(
        (np.column_stack((boundaries[1:-1], values[:-1])), np.column_stack((boundaries[1:-1], values[1:]))),
        axis=1,
    )
    return np.concatenate((horizontal, vertical)), np.concatenate((steps, steps[1:]))

figure_7 = plt.figure(figsize=(10.8, 7.0), layout="constrained")
outer_grid = figure_7.add_gridspec(1, 2, width_ratios=(1.8, 1.0), wspace=0.08)
left_grid = outer_grid[0].subgridspec(2, 1, height_ratios=(2.5, 1.0), hspace=0.05)
trajectory_ax = figure_7.add_subplot(left_grid[0])
heading_ax = figure_7.add_subplot(left_grid[1])
activity_grid = outer_grid[1].subgridspec(len(figure_7_representatives), 1, hspace=0.08)
activity_axes = [figure_7.add_subplot(activity_grid[index]) for index in range(len(figure_7_representatives))]
steps = np.arange(1, len(rollout.exact_centers) + 1)
time_normalization = mcolors.Normalize(vmin=steps[0], vmax=steps[-1])
time_colormap = plt.colormaps["viridis"]

plot_lattice_scalar(
    np.zeros((G.n, G.n)),
    ax=trajectory_ax,
    cmap=mcolors.ListedColormap(["#08192D"]),
    vmin=0,
    vmax=1,
    colorbar=False,
    coordinate_mode="offset",
    wrap_periodic_edges=True,
)
trajectory_ax.collections[-1].set(edgecolor="#4D6D8D", linewidth=0.35)
spatial_segments, spatial_times = time_colored_path(
    rollout.exact_centers, G.n, colormap=time_colormap, normalization=time_normalization
)
spatial_trace = LineCollection(
    spatial_segments,
    cmap=time_colormap,
    norm=time_normalization,
    array=spatial_times,
    linewidth=2.8,
    capstyle="round",
    zorder=2,
)
trajectory_ax.add_collection(spatial_trace)
for segment in lattice_path_segments(rollout.predicted_centers, G.n, mode="offset"):
    trajectory_ax.plot(
        segment[:, 0],
        segment[:, 1],
        color="#DCE7F2",
        linewidth=0.8,
        linestyle=(0, (4, 2)),
        alpha=0.8,
        solid_capstyle="round",
        zorder=3,
    )
exact_display = lattice_path_coordinates(rollout.exact_centers, G.n, mode="offset")
trajectory_ax.scatter(
    *exact_display[0],
    s=72,
    color=time_colormap(time_normalization(steps[0])),
    edgecolors="#EAF2FF",
    linewidths=0.7,
    zorder=4,
    label="start",
)
trajectory_ax.scatter(
    *exact_display[-1],
    s=105,
    marker="*",
    color=time_colormap(time_normalization(steps[-1])),
    edgecolors="#EAF2FF",
    linewidths=0.6,
    zorder=5,
    label="end",
)
trajectory_ax.plot([], [], color=time_colormap(0.7), linewidth=2.8, label="true pose (color = time)")
trajectory_ax.plot([], [], color="#DCE7F2", linewidth=0.8, linestyle=(0, (4, 2)), label="decoded pose")
trajectory_ax.set(
    title="A   Accurate 2D path integration",
    aspect="equal",
    xticks=[],
    yticks=[],
)
trajectory_ax.set_frame_on(False)
handles, labels = trajectory_ax.get_legend_handles_labels()
legend_order = [2, 3, 0, 1]
trajectory_ax.legend(
    [handles[index] for index in legend_order],
    [labels[index] for index in legend_order],
    loc="upper right",
    frameon=True,
    facecolor="#08192D",
    edgecolor="#4D6D8D",
    labelcolor="white",
    fontsize=8,
)

heading_degrees = 360 * rollout.exact_poses[:, 2] / G.m
decoded_heading_degrees = 360 * rollout.predicted_poses[:, 2] / G.m
heading_segments, heading_times = time_colored_heading(
    steps, heading_degrees, colormap=time_colormap, normalization=time_normalization
)
heading_ax.add_collection(
    LineCollection(heading_segments, cmap=time_colormap, norm=time_normalization, array=heading_times, linewidth=2.4)
)
heading_ax.step(
    steps,
    decoded_heading_degrees,
    where="mid",
    label="decoded heading",
    color="#607080",
    linewidth=0.7,
    linestyle=(0, (4, 2)),
)
heading_ax.set(
    xlabel="time step",
    ylabel="heading",
    xlim=(steps[0], steps[-1]),
    ylim=(-15, 315),
    yticks=(0, 120, 240),
    yticklabels=(r"$0^\circ$", r"$120^\circ$", r"$240^\circ$"),
)
heading_ax.grid(alpha=0.18, linewidth=0.6)
heading_ax.spines[["top", "right"]].set_visible(False)
heading_ax.plot([], [], color=time_colormap(0.7), linewidth=2.4, label="true heading (color = time)")
heading_ax.legend(loc="upper right", frameon=False, ncols=2, fontsize=7)
colorbar = figure_7.colorbar(spatial_trace, ax=(trajectory_ax, heading_ax), orientation="horizontal", fraction=0.035, pad=0.06)
colorbar.set_label("time step", fontsize=8)
colorbar.ax.tick_params(labelsize=7)
heading_ax.text(
    0.98,
    0.08,
    (
        f"position accuracy {spatial_accuracy.mean():.0%}   "
        f"heading accuracy {orientation_accuracy.mean():.0%}"
    ),
    transform=heading_ax.transAxes,
    fontsize=8,
    ha="right",
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
)

for column, (ax, unit) in enumerate(zip(activity_axes, figure_7_representatives)):
    metadata = params.metadata[unit]
    ax.plot(steps, figure_7_activity[:, column], color="0.12", linewidth=1.15)
    ax.set(
        xlim=(steps[0], steps[-1]),
        ylim=(-0.04, 1.04),
        yticks=(0, 1),
        ylabel=f"$\\rho={metadata['irrep_index']}$\nunit {unit}",
    )
    ax.tick_params(axis="both", labelsize=7, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    if column < len(activity_axes) - 1:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("time step")
activity_axes[0].set_title("B   Hidden activity")

for suffix in ("pdf", "png", "svg"):
    figure_7.savefig(
        paper_figure_directory / f"figure_7_draft.{suffix}",
        dpi=300,
        bbox_inches="tight",
    )
plt.show()
