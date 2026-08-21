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
from IPython import get_ipython
from IPython.display import HTML, display

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
from src.geometry.discrete_se2 import (  # noqa: E402
    NaturalisticMotionConfig,
    lattice_coordinates,
    lattice_path_coordinates,
    linked_plotly_html,
    plotly_heading_stacks,
)
from src.groups import as_action_group  # noqa: E402

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
n_spatial = 25
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
max_hidden_width = 24_000
normalize_power_by_dimension = True
always_include_trivial_irrep = True
power_ranking = "power"
q_rho = 3
amplitude_mode = "balanced"
amplitude_multipliers = (1.0, 1.0, 1.0)
materialize_recurrent_matrix = False

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
# Primary rollout
# ----------------------------
num_rollout_steps = 52
rollout_seed = 1
rollout_margin = 1
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
    right_group.right_action(g_x, x_allo),
    right_group.right_action(g_y, x_allo),
    right_group.right_action(g_rotation, x_allo),
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
# ## 3. Naturalistic trajectory policy
#
# Every step samples a persistent turn in $\{-60^\circ,0,+60^\circ\}$ and a
# translation relative to the updated heading. The explicit default prior is:
#
# - stay: 5%;
# - forward: 70%;
# - forward-oblique: 11.5% in each direction;
# - rear-oblique: 0.5% in each direction;
# - backward: 1%.
#
# Candidate actions crossing the boundary are removed, and headings are
# reweighted using three-cell forward lookahead. The first token only relocates
# the allocentric template to the requested rollout start.

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
# ## 4. Naturalistic rollout
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
