# Percent-format notebook source.
# Run cells marked with # %% in Cursor or another compatible editor.
# Outputs and notebook runtime metadata are intentionally omitted.

# %% [markdown]
# # Closed-form RNNs for discrete $SE(2)$
#
# A vector $x\in\mathbb{R}^{|G|}$ is a scalar function on $G=\mathbb{Z}_n^2\rtimes C_3$. This notebook constructs a parameterized quadratic RNN with an explicit choice of left- or right-regular updates, visualizes its encodings and analytical operators, follows one long pose rollout, and analyzes tuning and nonabelian irrep modules.

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

from src.geometry.discrete_se2 import (
    TRACK_COLOR,
    advanced_pose,
    align_rotation_slice,
    center_errors_periodic_triangular,
    decode_centers_from_outputs,
    decode_pose,
    decode_spatial_argmax,
    gaussian_bump,
    make_momentum_motion_sequence,
    periodic_distance_squared,
    periodic_spatial_autocorrelation,
    plot_lattice_scalar,
    spatial_marginal,
    transformed_pose,
)
from src.finite_group_rnn import (
    build_finite_group_rnn,
    random_invertible_encoding,
)
from src.groups import DiscreteSE2Group, as_action_group
from src.neural_manifold import (
    analyze_module_orbits,
    build_module_orbits,
    coordinate_colors,
    fixed_point_embedding,
    plot_manifold_analysis,
)

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## Key hyperparameters
#
# This cell collects the principal choices controlling the group, signal encodings, closed-form network, rollout, and downstream analyses. Plot styling remains local to the relevant figure cells.

# %%
# Group and signal construction
n_spatial = 10  # Side length of the periodic triangular translation lattice Z_n².
n_orientations = 6  # Number of discrete headings; this notebook uses C_m.
initial_pose = (2, 2, 0)  # Initial allocentric state (x, y, heading index).
allocentric_encoding = "one-hot space custom orientation"  # Profile used for x_allo.
sigma = 1.0  # Spatial width when allocentric_encoding uses a Gaussian bump.
custom_orientation_weights = (1.0, 0.7, 0.4, 0.3, 0.6, 0.8)  # One weight per discrete heading.
encoding_seed = 10  # Seed for the random invertible egocentric template x_ego.
action_side = "right"  # "right": body-frame s*g; "left": world-frame g*s.

# Closed-form network construction
irrep_selection = "power"  # Rank retained irreps by Fourier power in x_allo.
num_selected_irreps = None  # None retains every irrep allowed by the selection rule.
max_hidden_width = None  # Optional total-width budget used during irrep selection.
normalize_power_by_dim = True  # Compare average Fourier power per irrep dimension.
always_include_trivial = True  # Keep the constant mode under truncated selection.
power_ranking = "power"  # Rank by total power rather than reconstruction gain.
q_rho = 3  # Per-irrep phase/multiplicity controlling hidden width.
amplitude_mode = "balanced"  # Balance the three closed-form factor amplitudes.
amplitude_multipliers = (1.0, 1.0, 1.0)  # Product-one rescaling of those factors.
materialize_mix = False  # Keep W_mix factored as W_in @ W_out to save memory.

# Primary rollout
num_long_steps = 52  # Number of incoming group elements in the rollout.
rollout_seed = 1  # Seed controlling the sampled momentum walk.
rollout_include_rotations = True  # Allow heading changes as well as translations.
rollout_momentum = True  # Correlate successive translation directions.
rollout_turn_probability = 0.18  # Probability of turning to an adjacent direction.
rollout_stay_probability = 0.04  # Probability of no translation on a step.
rollout_margin = 2  # Keep displayed positions this many cells from each boundary.
start_xy = (n_spatial // 2, n_spatial // 2)  # Spatial start of the displayed walk.

# Visualization and tuning analysis
arrow_stride = 2  # Draw one orientation arrow every this many rollout steps.
snapshot_steps = [0, num_long_steps // 2, num_long_steps - 1]
num_tuning_irreps_to_plot = 5  # Highest-power nontrivial irrep modules to show.
num_tuning_neurons_per_irrep = 10  # Optional neuron cap per tuning figure.

# Module-manifold analysis
num_manifold_modules_to_plot = 6  # Highest-power retained modules to analyze.
manifold_spatial_samples = 12  # Samples per spatial axis before removing duplicates.
fixed_point_tolerance = 1e-8  # Convergence tolerance for identity-update iteration.
fixed_point_max_iterations = 50  # Iteration cap for recurrent fixed points.
max_persistence_points = 300  # Point cap for persistent-homology computation.
max_homology_dimension = 2  # Highest homology dimension computed.
manifold_random_seed = 11  # Seed for stochastic manifold-analysis components.
umap_components = 3  # Embedding dimension used only for UMAP visualization.

# %% [markdown]
# ## 1. Group and action conventions
#
# Write $g=(t,r)$ with $t\in\mathbb{Z}_n^2$ and $r\in C_3$. For
#
# $$A=\begin{pmatrix}-1&-1\\1&0\end{pmatrix}\pmod n,$$
#
# the semidirect-product law is
#
# $$(t_1,r_1)(t_2,r_2)=(t_1+A^{r_1}t_2,\ r_1+r_2).$$
#
# The `action_side` hyperparameter below chooses between two conventions:
#
# - `"right"` uses $(R_gx)[h]=x[h g^{-1}]$, so a state $s$ moves to $sg$ and drives are body-frame motions;
# - `"left"` uses $(L_gx)[h]=x[g^{-1}h]$, so a state $s$ moves to $gs$ and drives are spatial/world-frame motions.
#
# For the right action, the implementation applies the closed-form theorem to the opposite group $G^{\mathrm{op}}$, whose irreps are $\rho^{\mathrm{op}}(g)=\rho(g)^\top$. The initial allocentric state is supplied explicitly as `x_allo`; every sequence token is interpreted according to the selected action convention.

# %% [markdown]
# ## 2. Group and signal encodings
#
# The allocentric encoding can independently use a one-hot or periodic-Gaussian spatial profile and a one-hot, uniform, or manually specified orientation profile. `custom_orientation_weights` supplies one value per discrete heading, allowing orientation to remain graded but decodable. The special `"one-hot"` option is one-hot in both space and orientation. The egocentric code is sampled until every irrep Fourier block is invertible.

# %%
encoding_options = {
    "one-hot",
    "one-hot space uniform orientation",
    "one-hot space custom orientation",
    "gaussian space one-hot orientation",
    "gaussian space uniform orientation",
    "gaussian space custom orientation",
}
if allocentric_encoding not in encoding_options:
    raise ValueError(
        f"allocentric_encoding must be one of {sorted(encoding_options)}, "
        f"got {allocentric_encoding!r}"
    )
if action_side not in {"left", "right"}:
    raise ValueError(
        f"action_side must be 'left' or 'right', got {action_side!r}"
    )

G = DiscreteSE2Group(n=n_spatial, m=n_orientations)
center_xy = initial_pose[:2]
one_hot_space = allocentric_encoding in {
    "one-hot",
    "one-hot space uniform orientation",
    "one-hot space custom orientation",
}
one_hot_orientation = allocentric_encoding in {
    "one-hot",
    "gaussian space one-hot orientation",
}
custom_orientation = allocentric_encoding in {
    "one-hot space custom orientation",
    "gaussian space custom orientation",
}

if one_hot_orientation:
    orientation_weights = np.zeros(G.m)
    orientation_weights[initial_pose[2]] = 1.0
elif custom_orientation:
    orientation_weights = np.asarray(custom_orientation_weights, dtype=float)
    if orientation_weights.shape != (G.m,):
        raise ValueError(
            f"custom_orientation_weights must have shape ({G.m},), "
            f"got {orientation_weights.shape}"
        )
    if not np.all(np.isfinite(orientation_weights)) or np.any(orientation_weights < 0):
        raise ValueError("custom_orientation_weights must be finite and nonnegative")
    if np.allclose(orientation_weights, orientation_weights[0]):
        raise ValueError("custom_orientation_weights must vary across headings")
else:
    orientation_weights = np.ones(G.m)

if one_hot_space:
    x_allo = np.zeros(G.order)
    for orientation, weight in enumerate(orientation_weights):
        x_allo[G.encode(*center_xy, orientation)] = weight
else:
    x_allo = gaussian_bump(
        G,
        center=center_xy,
        sigma=sigma,
        orientation_weights=orientation_weights,
    )

irreps = G.irreps()
x_ego = random_invertible_encoding(G, irreps, seed=encoding_seed)

# %% [markdown]
# ## 3. Visualizing the template and its left-regular group orbit
#
# The first figure shows the scalar template $x_{\mathrm{allo}}:G\to\mathbb R$ itself. The red cell is the identity element $(0,0,0)\in G$. The following three comparisons apply exemplar generators through the left-regular action
#
# $$(L_gx)(h)=x(g^{-1}h).$$

# %%
from IPython.display import HTML, display

from src.geometry.discrete_se2 import linked_plotly_html, plotly_heading_stacks

template_figure = plotly_heading_stacks(
    G,
    [x_allo],
    titles=["Template x<sub>allo</sub>"],
    highlighted_elements=[{G.identity()}],
    width=750,
    height=650,
)
display(HTML(linked_plotly_html(template_figure)))

# %%
# For m=3, rotation index 1 is 120 degrees.
g_x = G.encode(1, 0, 0)
g_y = G.encode(0, 1, 0)
g_rotation = G.encode(0, 0, 1)

x_translated = G.left_action(g_x, x_allo)
y_translated = G.left_action(g_y, x_allo)
rotated = G.left_action(g_rotation, x_allo)

# %% [markdown]
# ### Unit translation along the lattice $x$ direction
#
# Compare the original template with $L_{(1,0,0)}x_{\mathrm{allo}}$.

# %%
x_figure = plotly_heading_stacks(
    G,
    [x_allo, x_translated],
    titles=["Original x<sub>allo</sub>", "L<sub>(1,0,0)</sub> x<sub>allo</sub>"],
    arrangement="vertical",
    width=750,
    height=1250,
)
display(HTML(linked_plotly_html(x_figure, [("scene", "scene2")])))

# %% [markdown]
# ### Unit translation along the lattice $y$ direction
#
# Compare the original template with $L_{(0,1,0)}x_{\mathrm{allo}}$.

# %%
y_figure = plotly_heading_stacks(
    G,
    [x_allo, y_translated],
    titles=["Original x<sub>allo</sub>", "L<sub>(0,1,0)</sub> x<sub>allo</sub>"],
    arrangement="vertical",
    width=750,
    height=1250,
)
display(HTML(linked_plotly_html(y_figure, [("scene", "scene2")])))

# %% [markdown]
# ### Rotation by $120^\circ$
#
# Compare the original template with $L_{(0,0,120^\circ)}x_{\mathrm{allo}}$.

# %%
rotation_figure = plotly_heading_stacks(
    G,
    [x_allo, rotated],
    titles=[
        "Original x<sub>allo</sub>",
        "L<sub>(0,0,120°)</sub> x<sub>allo</sub>",
    ],
    arrangement="vertical",
    width=750,
    height=1250,
)
display(HTML(linked_plotly_html(rotation_figure, [("scene", "scene2")])))

# %% [markdown]
# ## 4. Visualizing the right-regular action on the template
#
# The right-regular action is
#
# $$(R_gx)(h)=x(hg^{-1}).$$
#
# It moves a one-hot state at $s$ to $sg$, so translations are interpreted in the state's current body frame. The same three group elements below make the contrast with the left-regular action explicit. With the current initial heading $r=0$, pure left and right translations happen to agree; set the heading in `initial_pose` to `1` or `2` to expose their body-frame difference directly.

# %%
right_action_group = as_action_group(G, action_side="right")
right_x_translated = right_action_group.right_action(g_x, x_allo)
right_y_translated = right_action_group.right_action(g_y, x_allo)
right_rotated = right_action_group.right_action(g_rotation, x_allo)

# %% [markdown]
# ### Right action by a unit lattice $x$ translation
#
# Compare the original template with $R_{(1,0,0)}x_{\mathrm{allo}}$.

# %%
right_x_figure = plotly_heading_stacks(
    G,
    [x_allo, right_x_translated],
    titles=["Original x<sub>allo</sub>", "R<sub>(1,0,0)</sub> x<sub>allo</sub>"],
    arrangement="vertical",
    width=750,
    height=1250,
)
display(HTML(linked_plotly_html(right_x_figure, [("scene", "scene2")])))

# %% [markdown]
# ### Right action by a unit lattice $y$ translation
#
# Compare the original template with $R_{(0,1,0)}x_{\mathrm{allo}}$.

# %%
right_y_figure = plotly_heading_stacks(
    G,
    [x_allo, right_y_translated],
    titles=["Original x<sub>allo</sub>", "R<sub>(0,1,0)</sub> x<sub>allo</sub>"],
    arrangement="vertical",
    width=750,
    height=1250,
)
display(HTML(linked_plotly_html(right_y_figure, [("scene", "scene2")])))

# %% [markdown]
# ### Right action by a $120^\circ$ rotation
#
# Compare the original template with $R_{(0,0,120^\circ)}x_{\mathrm{allo}}$.

# %%
right_rotation_figure = plotly_heading_stacks(
    G,
    [x_allo, right_rotated],
    titles=[
        "Original x<sub>allo</sub>",
        "R<sub>(0,0,120°)</sub> x<sub>allo</sub>",
    ],
    arrangement="vertical",
    width=750,
    height=1250,
)
display(HTML(linked_plotly_html(right_rotation_figure, [("scene", "scene2")])))

# %%
# Construct the closed-form RNN after inspecting the signal encodings.
params = build_finite_group_rnn(
    G,
    x_ego,
    x_allo=x_allo,
    irrep_selection=irrep_selection,
    num_irreps=num_selected_irreps,
    max_hidden_width=max_hidden_width,
    normalize_power_by_dim=normalize_power_by_dim,
    always_include_trivial=always_include_trivial,
    power_ranking=power_ranking,
    q_rho=q_rho,
    amplitude_mode=amplitude_mode,
    amplitude_multipliers=amplitude_multipliers,
    materialize_mix=materialize_mix,
    action_side=action_side,
)

all_dimensions, all_dimension_counts = np.unique(
    [rho.dim for rho in irreps], return_counts=True
)
selected_dimensions, selected_dimension_counts = np.unique(
    [rho.dim for rho in params.irreps], return_counts=True
)

print(f"|G|={G.order}, all irreps={len(irreps)}")
print(
    "all irreps by dimension: "
    + ", ".join(
        f"{count} of dimension {dimension}"
        for dimension, count in zip(all_dimensions, all_dimension_counts)
    )
)
print("selected irrep indices:", params.selected_irrep_indices)
print(
    f"action side: {params.action_side} "
    f"({'body-frame' if params.action_side == 'right' else 'world-frame'} updates)"
)
print(
    "selected irreps by dimension: "
    + ", ".join(
        f"{count} of dimension {dimension}"
        for dimension, count in zip(selected_dimensions, selected_dimension_counts)
    )
)
print("hidden width:", params.hidden_dim)
print("W_mix stored:", params.W_mix is not None, "(materialized below only for plotting)")

# %% [markdown]
# ## 5. Closed-form weight operators
#
# The four panels use one consistent block ordering from `params.metadata`. $W_{\rm mix}$ is materialized only for this visualization; rollout evaluation continues to apply $W_{\rm in}(W_{\rm out}h)$.

# %%
# # Analytical weight operators. W_mix is materialized locally for inspection;
# # recurrent evaluation continues to use its factored form.
# W_mix_for_plot = params.W_in @ params.W_out
# weight_matrices = {
#     r"$W_{in}$": params.W_in,
#     r"$W_{drive}$": params.W_drive,
#     r"$W_{out}$": params.W_out,
#     r"$W_{mix}=W_{in}W_{out}$": W_mix_for_plot,
# }
# metadata_irreps = np.asarray([item["irrep_index"] for item in params.metadata])
# block_boundaries = np.flatnonzero(np.diff(metadata_irreps)) + 1

# figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
# for panel, (ax, (name, matrix)) in enumerate(
#     zip(axes.ravel(), weight_matrices.items())
# ):
#     scale = np.percentile(np.abs(matrix), 99)
#     image = ax.imshow(
#         matrix,
#         aspect="auto",
#         interpolation="nearest",
#         cmap="coolwarm",
#         vmin=-scale,
#         vmax=scale,
#     )
#     if panel in (0, 1, 3):
#         for boundary in block_boundaries:
#             ax.axhline(boundary - 0.5, color="black", linewidth=0.35, alpha=0.7)
#     if panel in (2, 3):
#         for boundary in block_boundaries:
#             ax.axvline(boundary - 0.5, color="black", linewidth=0.35, alpha=0.7)
#     ax.set_title(f"{name}  shape={matrix.shape}")
#     ax.set_xlabel("column")
#     ax.set_ylabel("row")
#     figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
# figure.suptitle("Closed-form RNN weight matrices; lines mark irrep blocks")
# plt.show()

# del W_mix_for_plot

# %% [markdown]
# ## 6. Primary long rollout and decoding error
#
# A single parameterized momentum walk supplies both translations and discrete rotations. The rollout compares the full reconstructed group signal with the regular-action target, then decodes position and heading from the same outputs. Sparse arrows use the columns of the implemented lattice rotation action, so exact and decoded orientation are shown without clutter.

# %% [markdown]
# ### Restricted egocentric input support
#
# The RNN construction can encode every element of $G$, but the momentum-walk sampler restricts each ordinary step to a local motion. Its translation is either zero or one of the six nearest-neighbor displacements
#
# $$D=\{(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)\},$$
#
# while its rotation increment can be any element of $C_3$. Thus the ordinary-step support is
#
# $$S_{\mathrm{ego}}=D\times C_3,\qquad |S_{\mathrm{ego}}|=7\cdot 3=21.$$
#
# For an incoming element $g\in S_{\mathrm{ego}}$, the actual egocentric drive presented to the RNN is the transformed signal $L_gx_{\mathrm{ego}}$. The figure below shows the support mask on $G$: each layer fixes the rotation increment, and the red cells mark the seven allowed translations. We display periodic coordinates using their representatives centered around zero, placing the identity at the origin and keeping its six neighbors together.
#
# The first element of the generated sequence is a deliberate exception: it relocates `initial_pose` to `start_xy`. All subsequent elements belong to the 21-element support shown here. Momentum and boundary rejection make their sampling probabilities nonuniform.

# %%
local_step_translations = [
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
]
local_egocentric_elements = {
    G.encode(dx, dy, rotation)
    for dx, dy in local_step_translations
    for rotation in range(G.m)
}
local_egocentric_support = np.zeros(G.order)
local_egocentric_support[list(local_egocentric_elements)] = 1.0

local_egocentric_figure = plotly_heading_stacks(
    G,
    [local_egocentric_support],
    titles=[
        f"Ordinary-step egocentric support "
        f"({len(local_egocentric_elements)} of {G.order} elements)"
    ],
    highlighted_elements=[local_egocentric_elements],
    coordinate_mode="centered_axial",
    width=750,
    height=650,
)
display(HTML(linked_plotly_html(local_egocentric_figure)))

# %%
long_sequence = make_momentum_motion_sequence(
    G,
    steps=num_long_steps,
    seed=rollout_seed,
    include_rotations=rollout_include_rotations,
    momentum=rollout_momentum,
    turn_probability=rollout_turn_probability,
    stay_probability=rollout_stay_probability,
    start_xy=start_xy,
    initial_pose=initial_pose,
    margin=rollout_margin,
)
long_result = {
    key: value.detach().cpu().numpy()
    for key, value in params.rollout(x_allo, long_sequence).items()
}

if action_side == "right":
    exact_poses = np.asarray([
        advanced_pose(G, initial_pose, int(state))
        for state in long_result["cumulative_states"]
    ])
else:
    exact_poses = np.asarray([
        transformed_pose(G, int(state), initial_pose)
        for state in long_result["cumulative_states"]
    ])
exact_centers = exact_poses[:, :2]
predicted_centers, predicted_direction_marginals = decode_centers_from_outputs(
    G, long_result["predicted_outputs"]
)
predicted_poses = np.asarray([
    decode_pose(G, output) for output in long_result["predicted_outputs"]
])
absolute_errors = np.linalg.norm(
    long_result["predicted_outputs"] - long_result["true_outputs"], axis=1
)
long_relative_errors = absolute_errors / np.linalg.norm(
    long_result["true_outputs"], axis=1
)
long_center_errors = center_errors_periodic_triangular(
    G, predicted_centers, exact_centers
)
orientation_steps = np.abs(predicted_poses[:, 2] - exact_poses[:, 2]) % G.m
orientation_steps = np.minimum(orientation_steps, G.m - orientation_steps)
long_orientation_errors = 2 * np.pi * orientation_steps / G.m

print("Long-sequence tracking summary")
print("------------------------------")
print("max output error:          ", absolute_errors.max())
print("mean output error:         ", absolute_errors.mean())
print("max center error:          ", long_center_errors.max())
print("mean center error:         ", long_center_errors.mean())
print("max orientation error (°): ", np.degrees(long_orientation_errors).max())

# %%
import matplotlib.colors as mcolors
from src.geometry.discrete_se2 import lattice_coordinates, lattice_path_coordinates

steps = np.arange(1, len(exact_poses) + 1)
time_cmap = plt.colormaps["viridis"]
time_norm = mcolors.Normalize(vmin=steps[0], vmax=steps[-1])

orientation_colors = ["#0072B2", "#E69F00", "#009E73"]
orientation_cmap = mcolors.ListedColormap(orientation_colors[:G.m])
orientation_norm = mcolors.BoundaryNorm(np.arange(-0.5, G.m + 0.5), G.m)

def orientation_vectors(rotations):
    angles = 2 * np.pi * np.asarray(rotations) / G.m
    return np.column_stack([np.cos(angles), np.sin(angles)])

figure = plt.figure(figsize=(12.5, 11), constrained_layout=True)
grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
spatial_axes = [figure.add_subplot(grid[0, column]) for column in range(2)]
orientation_axes = [
    figure.add_subplot(grid[1, column], projection="3d") for column in range(2)
]

lattice_x, lattice_y = lattice_coordinates(G.n, mode="axial")
spatial_panels = (
    (exact_centers, "True position"),
    (predicted_centers, "Predicted position"),
)
for ax, (centers, title) in zip(spatial_axes, spatial_panels):
    points = lattice_path_coordinates(centers, G.n, mode="axial")
    ax.scatter(lattice_x, lattice_y, s=7, color="0.86", linewidths=0, zorder=0)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=steps,
        cmap=time_cmap,
        norm=time_norm,
        s=22,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(
        *points[0],
        s=65,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1,
        zorder=3,
    )
    ax.scatter(*points[-1], s=85, marker="*", color="black", zorder=3)
    ax.set(title=title, aspect="equal", xticks=[], yticks=[])
    ax.set_frame_on(False)

spatial_colorbar = figure.colorbar(
    plt.cm.ScalarMappable(norm=time_norm, cmap=time_cmap),
    ax=spatial_axes,
    fraction=0.025,
    pad=0.02,
)
spatial_colorbar.set_label("time step")

arrow_indices = np.arange(0, len(steps), arrow_stride)
orientation_panels = (
    (exact_poses[:, 2], "True orientation"),
    (predicted_poses[:, 2], "Predicted orientation"),
)
for ax, (rotations, title) in zip(orientation_axes, orientation_panels):
    directions = orientation_vectors(rotations[arrow_indices])
    arrow_colors = orientation_cmap(orientation_norm(rotations[arrow_indices]))

    ax.plot(
        np.zeros_like(steps),
        np.zeros_like(steps),
        steps,
        color="0.55",
        linewidth=1,
    )
    ax.quiver(
        np.zeros(len(arrow_indices)),
        np.zeros(len(arrow_indices)),
        steps[arrow_indices],
        0.75 * directions[:, 0],
        0.75 * directions[:, 1],
        np.zeros(len(arrow_indices)),
        color=arrow_colors,
        arrow_length_ratio=0.16,
        linewidth=1.25,
    )

    ax.set(
        title=title,
        xlim=(-1, 1),
        ylim=(-1, 1),
        zlim=(steps[0], steps[-1]),
        xticks=[],
        yticks=[],
        xlabel="",
        ylabel="",
        zlabel="time step",
    )
    ax.set_box_aspect((1, 1, 2.2))
    ax.view_init(elev=36, azim=-35)
    ax.grid(False)
    ax.set_facecolor("none")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
    ax.xaxis.line.set_color((0, 0, 0, 0))
    ax.yaxis.line.set_color((0, 0, 0, 0))
    ax.zaxis.line.set_color("0.45")

orientation_colorbar = figure.colorbar(
    plt.cm.ScalarMappable(cmap=orientation_cmap, norm=orientation_norm),
    ax=orientation_axes,
    ticks=np.arange(G.m),
    fraction=0.025,
    pad=0.02,
)
orientation_colorbar.set_label("orientation")

figure.suptitle("True and decoded SE(2) trajectory", fontsize=15)
plt.show()

# %%
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots

steps = np.arange(1, len(exact_poses) + 1)
arrow_length = 0.9

orientation_colors = [
    "#0072B2",
    "#E69F00",
    "#009E73",
]


def orientation_vectors(rotations):
    angles = 2 * np.pi * np.asarray(rotations) / G.m
    return np.column_stack([np.cos(angles), np.sin(angles)])


figure = go.FigureWidget(
    make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("True pose", "Decoded pose"),
        horizontal_spacing=0.01,
    )
)

panels = (
    (exact_centers, exact_poses[:, 2], 1),
    (predicted_centers, predicted_poses[:, 2], 2),
)

all_points = np.vstack(
    [
        lattice_path_coordinates(centers, G.n, mode="axial")
        for centers, _, _ in panels
    ]
)

plot_padding = 0.8
x_range = [
    all_points[:, 0].min() - plot_padding,
    all_points[:, 0].max() + plot_padding,
]
y_range = [
    all_points[:, 1].min() - plot_padding,
    all_points[:, 1].max() + plot_padding,
]

for centers, rotations, column in panels:
    points = lattice_path_coordinates(centers, G.n, mode="axial")
    directions = orientation_vectors(rotations)

    figure.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=steps,
            mode="lines+markers",
            line={
                "color": "rgba(80, 80, 80, 0.45)",
                "width": 3,
            },
            marker={
                "size": 4,
                "color": steps,
                "colorscale": "Viridis",
                "showscale": column == 2,
                "colorbar": {
                    "title": "time step",
                    "x": 0.98,
                    "len": 0.78,
                    "thickness": 12,
                },
            },
            customdata=np.column_stack([centers, rotations]),
            hovertemplate=(
                "time: %{z}<br>"
                "position: (%{customdata[0]}, %{customdata[1]})<br>"
                "orientation: %{customdata[2]}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=column,
    )

    for orientation in range(G.m):
        mask = rotations == orientation
        if not np.any(mask):
            continue

        bases = points[mask]
        direction = directions[mask]
        color = orientation_colors[orientation]

        arrow_tips = bases + arrow_length * direction
        separators = np.full(mask.sum(), np.nan)

        shaft_x = np.column_stack(
            [bases[:, 0], arrow_tips[:, 0], separators]
        ).ravel()
        shaft_y = np.column_stack(
            [bases[:, 1], arrow_tips[:, 1], separators]
        ).ravel()
        shaft_z = np.column_stack(
            [steps[mask], steps[mask], separators]
        ).ravel()

        # Thick, explicitly drawn arrow shafts.
        figure.add_trace(
            go.Scatter3d(
                x=shaft_x,
                y=shaft_y,
                z=shaft_z,
                mode="lines",
                line={"color": color, "width": 5},
                name=f"orientation {orientation}",
                legendgroup=f"orientation-{orientation}",
                showlegend=column == 1,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )

        # Cone arrowheads at the shaft tips.
        head_fraction = 0.30
        head_bases = (
            bases
            + (1 - head_fraction) * arrow_length * direction
        )

        figure.add_trace(
            go.Cone(
                x=head_bases[:, 0],
                y=head_bases[:, 1],
                z=steps[mask],
                u=head_fraction * arrow_length * direction[:, 0],
                v=head_fraction * arrow_length * direction[:, 1],
                w=np.zeros(mask.sum()),
                anchor="tail",
                sizemode="raw",
                sizeref=1.6,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )

figure.update_scenes(
    xaxis_title="grid x",
    yaxis_title="grid y",
    zaxis_title="time step",
    xaxis_range=x_range,
    yaxis_range=y_range,
    zaxis_range=[steps[0] - 1, steps[-1] + 1],
    dragmode="orbit",
    aspectmode="manual",
    aspectratio={"x": 1, "y": 1, "z": 1.35},
    camera={"eye": {"x": 1.65, "y": 1.65, "z": 1.15}},
)

figure.update_layout(
    title={
        "text": "SE(2) pose trajectory through time",
        "x": 0.47,
        "y": 0.98,
    },
    autosize=False,
    width=820,
    height=460,
    margin={"l": 0, "r": 42, "t": 52, "b": 0},
    scene={"domain": {"x": [0.00, 0.46], "y": [0.00, 1.00]}},
    scene2={"domain": {"x": [0.48, 0.94], "y": [0.00, 1.00]}},
    legend={
        "title": "",
        "orientation": "h",
        "x": 0.47,
        "xanchor": "center",
        "y": 1.01,
        "yanchor": "bottom",
        "font": {"size": 10},
    },
)

# Synchronize the two cameras in both directions.
camera_sync = {"active": False}


def copy_camera(target_scene):
    def callback(_, camera):
        if camera_sync["active"]:
            return

        camera_sync["active"] = True
        try:
            target_scene.camera = camera
        finally:
            camera_sync["active"] = False

    return callback


figure.layout.scene.on_change(
    copy_camera(figure.layout.scene2),
    "camera",
)
figure.layout.scene2.on_change(
    copy_camera(figure.layout.scene),
    "camera",
)

display(figure)

# %%
time = np.arange(1, num_long_steps + 1)

figure, axes = plt.subplots(1, 3, figsize=(14, 3.2), constrained_layout=True)
error_series = (
    (
        long_relative_errors,
        r"$\|y_t-y_t^\star\|_2/\|y_t^\star\|_2$",
        "Relative signal reconstruction error",
    ),
    (
        long_center_errors,
        "lattice distance",
        "Decoded position error",
    ),
    (
        np.degrees(long_orientation_errors),
        "degrees",
        "Decoded orientation error",
    ),
)
for ax, (values, ylabel, title) in zip(axes, error_series):
    ax.plot(time, values, color=TRACK_COLOR, linewidth=1.8)
    ax.set(xlabel="time step", ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.show()

figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
all_snapshot_values = [
    spatial_marginal(G, long_result[key][step])
    for key in ("true_outputs", "predicted_outputs")
    for step in snapshot_steps
]
vmin = min(float(values.min()) for values in all_snapshot_values)
vmax = max(float(values.max()) for values in all_snapshot_values)
for column, step in enumerate(snapshot_steps):
    plot_lattice_scalar(
        spatial_marginal(G, long_result["true_outputs"][step]),
        ax=axes[0, column],
        title=f"target, step {step + 1}",
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
    )
    plot_lattice_scalar(
        spatial_marginal(G, long_result["predicted_outputs"][step]),
        ax=axes[1, column],
        title=f"reconstructed, step {step + 1}",
        vmin=vmin,
        vmax=vmax,
        colorbar=column == 2,
    )
figure.suptitle("Long-rollout reconstruction snapshots")
plt.show()

# %% [markdown]
# ## 7. Representation-stratified hidden-state tuning
#
# These tuning curves are computed by translating the allocentric input through every group element while holding the egocentric drive fixed at the identity. Following the $C_n\times C_n$ notebook, we keep one canonical representative from each retained conjugate pair and order those nontrivial irreps deterministically by their Fourier-power contribution to `x_allo`, from largest to smallest. Each figure contains many neurons from one irrep; the variables below can cap either the irreps or neurons displayed.
#
# For every neuron, the first three panels show the complete orientation-conditioned spatial tuning $f(x,y,\theta)$ at the three discrete headings. Each slice is aligned to the common allocentric lattice frame, and all conditioned maps in an irrep share one raw-activation color scale. The fourth panel is the spatial marginal $\sum_\theta f(x,y,\theta)$, with a separate shared scale because summation changes its range. The final three-bar panel is the direction marginal $\sum_{x,y} f(x,y,\theta)$. The conditioned maps preserve conjunctive position-direction structure that either marginal alone would discard.
#
# A second figure for each irrep shows periodic spatial autocorrelograms for every orientation-conditioned map and for the orientation-summed map. Before correlation, each map's spatial mean is removed; the result is normalized to one at zero displacement, which is displayed at the center of the tilted axial grid.

# %%
# Sweep translated allocentric inputs while holding the egocentric drive fixed.
tuning_hidden = params.probe_hidden_states(x_allo).detach().cpu().numpy()
power = G.power_spectrum(x_allo)
units_by_irrep = {}
for unit, metadata in enumerate(params.metadata):
    units_by_irrep.setdefault(int(metadata["irrep_index"]), []).append(unit)

# build_module_orbits identifies conjugate groups; keep their lowest-index member.
representative_tuning_irreps = [
    module.irrep_indices[0]
    for module in build_module_orbits(params, tuning_hidden)
]
tuning_irreps = sorted(
    representative_tuning_irreps,
    key=lambda irrep_index: (-power[irrep_index], irrep_index),
)
if num_tuning_irreps_to_plot is not None:
    if num_tuning_irreps_to_plot < 1:
        raise ValueError("num_tuning_irreps_to_plot must be positive or None")
    tuning_irreps = tuning_irreps[:num_tuning_irreps_to_plot]

# Retained for the rollout diagnostic in the next cell.
representatives = [int(units_by_irrep[index][0]) for index in tuning_irreps]
heading_degrees = 360 * np.arange(G.m) / G.m
heading_colors = plt.get_cmap("tab10")(np.arange(G.m))

for power_rank, irrep_index in enumerate(tuning_irreps, start=1):
    units = np.asarray(units_by_irrep[irrep_index], dtype=int)
    if num_tuning_neurons_per_irrep is not None:
        if num_tuning_neurons_per_irrep < 1:
            raise ValueError("num_tuning_neurons_per_irrep must be positive or None")
        units = units[:num_tuning_neurons_per_irrep]

    conditioned_maps = []
    spatial_marginals = []
    direction_marginals = []
    for unit in units:
        tensor = tuning_hidden[:, unit].reshape(G.m, G.n, G.n)
        aligned_slices = np.asarray(
            [align_rotation_slice(G, tensor[r], r) for r in range(G.m)]
        )
        conditioned_maps.append(aligned_slices)
        spatial_marginals.append(aligned_slices.sum(axis=0))
        direction_marginals.append(tensor.sum(axis=(1, 2)))
    conditioned_maps = np.asarray(conditioned_maps)
    spatial_marginals = np.asarray(spatial_marginals)
    direction_marginals = np.asarray(direction_marginals)
    conditioned_autocorrelations = np.asarray(
        [
            [
                periodic_spatial_autocorrelation(conditioned_map)
                for conditioned_map in unit_maps
            ]
            for unit_maps in conditioned_maps
        ]
    )
    spatial_autocorrelations = np.asarray(
        [
            periodic_spatial_autocorrelation(spatial_marginal)
            for spatial_marginal in spatial_marginals
        ]
    )

    conditioned_vmin = float(conditioned_maps.min())
    conditioned_vmax = float(conditioned_maps.max())
    marginal_vmin = float(spatial_marginals.min())
    marginal_vmax = float(spatial_marginals.max())
    direction_max = float(direction_marginals.max())

    figure, axes = plt.subplots(
        len(units),
        G.m + 2,
        figsize=(2.65 * (G.m + 2), 2.45 * len(units)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, unit in enumerate(units):
        metadata = params.metadata[int(unit)]
        for rotation in range(G.m):
            title = rf"$\theta={heading_degrees[rotation]:.0f}^\circ$"
            if rotation == 0:
                title = f"unit {int(unit)}, $\\delta={metadata['delta']}$\n" + title
            plot_lattice_scalar(
                conditioned_maps[row, rotation],
                ax=axes[row, rotation],
                title=title,
                vmin=conditioned_vmin,
                vmax=conditioned_vmax,
                colorbar=False,
                coordinate_mode="axial",
            )
        plot_lattice_scalar(
            spatial_marginals[row],
            ax=axes[row, G.m],
            title=rf"unit {int(unit)}: $\sum_\theta f$",
            vmin=marginal_vmin,
            vmax=marginal_vmax,
            colorbar=False,
            coordinate_mode="axial",
        )
        direction_ax = axes[row, G.m + 1]
        direction_ax.bar(
            np.arange(G.m),
            direction_marginals[row],
            color=heading_colors,
            width=0.72,
        )
        direction_ax.set(
            xticks=np.arange(G.m),
            xticklabels=[rf"${angle:.0f}^\circ$" for angle in heading_degrees],
            ylim=(0, 1.05 * direction_max if direction_max > 0 else 1),
            title=rf"unit {int(unit)}: $\sum_{{x,y}} f$",
            ylabel="total activity",
        )
        direction_ax.spines[["top", "right"]].set_visible(False)

    conditioned_colorbar = plt.cm.ScalarMappable(
        norm=plt.Normalize(conditioned_vmin, conditioned_vmax),
        cmap="viridis",
    )
    conditioned_colorbar.set_array([])
    figure.colorbar(
        conditioned_colorbar,
        ax=axes[:, :G.m].ravel(),
        fraction=0.012,
        pad=0.01,
        label="conditioned activity",
    )
    marginal_colorbar = plt.cm.ScalarMappable(
        norm=plt.Normalize(marginal_vmin, marginal_vmax),
        cmap="viridis",
    )
    marginal_colorbar.set_array([])
    figure.colorbar(
        marginal_colorbar,
        ax=axes[:, G.m],
        fraction=0.035,
        pad=0.01,
        label="summed activity",
    )

    irrep_power_fraction = power[irrep_index] / power.sum()
    figure.suptitle(
        f"Power rank {power_rank}: irrep {irrep_index}; "
        f"x_allo power={irrep_power_fraction:.2%}; {len(units)} neurons\n"
        "Conjunctive position-direction tuning",
        fontsize=14,
    )
    plt.show()

    autocorrelation_figure, autocorrelation_axes = plt.subplots(
        len(units),
        G.m + 1,
        figsize=(2.65 * (G.m + 1), 2.45 * len(units)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, unit in enumerate(units):
        metadata = params.metadata[int(unit)]
        for rotation in range(G.m):
            title = rf"$\theta={heading_degrees[rotation]:.0f}^\circ$"
            if rotation == 0:
                title = (
                    f"unit {int(unit)}, $\\delta={metadata['delta']}$\n" + title
                )
            plot_lattice_scalar(
                conditioned_autocorrelations[row, rotation],
                ax=autocorrelation_axes[row, rotation],
                title=title,
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                colorbar=False,
                coordinate_mode="centered_axial",
            )
        plot_lattice_scalar(
            spatial_autocorrelations[row],
            ax=autocorrelation_axes[row, G.m],
            title=rf"unit {int(unit)}: autocorr of $\sum_\theta f$",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            colorbar=False,
            coordinate_mode="centered_axial",
        )

    autocorrelation_colorbar = plt.cm.ScalarMappable(
        norm=plt.Normalize(-1.0, 1.0),
        cmap="coolwarm",
    )
    autocorrelation_colorbar.set_array([])
    autocorrelation_figure.colorbar(
        autocorrelation_colorbar,
        ax=autocorrelation_axes.ravel(),
        fraction=0.012,
        pad=0.01,
        label="normalized periodic autocorrelation",
    )
    autocorrelation_figure.suptitle(
        f"Power rank {power_rank}: irrep {irrep_index}; "
        f"x_allo power={irrep_power_fraction:.2%}; {len(units)} neurons\n"
        "Spatial autocorrelograms",
        fontsize=14,
    )
    plt.show()

# %%
power = G.power_spectrum(x_allo)
retained_fraction = power[params.selected_irrep_indices].sum() / power.sum()
print(f"selected Fourier power fraction: {retained_fraction:.3%}")

recurrent_unit = representatives[0]
recurrent_response = long_result["hidden_states"][:, recurrent_unit]
print(
    f"unit {recurrent_unit} tuning-sweep range={np.ptp(tuning_hidden[:, recurrent_unit]):.3e}; "
    f"long-rollout recurrent range={np.ptp(recurrent_response):.3e}"
)
print("Fourier power and reconstruction diagnostics quantify the selected module budget.")

# %% [markdown]
# ## 8. Module-restricted neural manifolds
#
# We now evaluate the prediction
#
# $$
# \mathcal M_{\rho,x_{\mathrm{allo}}}
# =
# \{\Pi_\rho\Phi(g\cdot x_{\mathrm{allo}}):g\in G\}
# $$
#
# for the configured number of highest-power retained nontrivial modules. Conjugate irreps are combined when both are present. The probe uses a stratified spatial sample at every discrete orientation. Every sampled population state is iterated to a recurrent fixed point, and the final residual is reported explicitly.
#
# Each embedding is shown three times, colored separately by the ground-truth $x$ coordinate, $y$ coordinate, and discrete orientation. Separate colorings avoid conflating the coordinates in one RGB mixture. UMAP is visualization only. Persistent homology is computed after PCA in neural space with deterministic farthest-point subsampling at the configured point cap.
#
# An induced three-dimensional irrep contains a $C_3$ orbit of translation frequencies, while this expanded probe also exposes variation over the finite orientation coordinate. The resulting topology remains a property of the module-restricted, finitely sampled orbit—not a claim that the full finite group is a continuous manifold.

# %%
sample_coordinates_1d = np.unique(
    np.linspace(0, G.n - 1, manifold_spatial_samples, dtype=int)
)
pose_coordinates = np.asarray(
    [
        (x, y, rotation)
        for rotation in range(G.m)
        for x in sample_coordinates_1d
        for y in sample_coordinates_1d
    ]
)
pose_elements = np.asarray(
    [G.encode(int(x), int(y), int(rotation)) for x, y, rotation in pose_coordinates]
)
pose_initial_hidden = tuning_hidden[pose_elements]
pose_fixed = fixed_point_embedding(
    params,
    pose_initial_hidden,
    tolerance=fixed_point_tolerance,
    max_iterations=fixed_point_max_iterations,
)
print(
    "identity-update fixed-point iteration: "
    f"converged={pose_fixed.converged}, "
    f"iterations={pose_fixed.iterations}, "
    f"max residual={pose_fixed.residuals.max():.3e}"
)

power = G.power_spectrum(x_allo)
all_pose_module_orbits = build_module_orbits(params, pose_fixed.states)
pose_module_orbits = sorted(
    all_pose_module_orbits,
    key=lambda module: (
        -sum(power[index] for index in module.irrep_indices),
        module.irrep_indices,
    ),
)[:num_manifold_modules_to_plot]
pose_manifold_analyses = analyze_module_orbits(
    pose_module_orbits,
    max_persistence_points=min(max_persistence_points, len(pose_coordinates)),
    max_homology_dimension=max_homology_dimension,
    random_state=manifold_random_seed,
    umap_components=umap_components,
)

x_colors = plt.get_cmap("viridis")(
    pose_coordinates[:, 0] / max(G.n - 1, 1)
)[:, :3]
y_colors = plt.get_cmap("viridis")(
    pose_coordinates[:, 1] / max(G.n - 1, 1)
)[:, :3]
angle_colors = plt.get_cmap("tab10")(
    pose_coordinates[:, 2].astype(int)
)[:, :3]
pose_colorings = (
    ("x coordinate", x_colors),
    ("y coordinate", y_colors),
    ("orientation", angle_colors),
)

print("highest-power module-restricted pose orbits:")
for analysis in pose_manifold_analyses:
    print(
        f"  {analysis.module.label}: PCA d={analysis.pca_dimension}, "
        f"variance={analysis.explained_variance:.2%}, "
        f"PH points={analysis.persistence_sample_size}"
    )
    for color_label, colors in pose_colorings:
        plot_manifold_analysis(
            analysis,
            colors,
            title=f"{analysis.module.label} — colored by {color_label}",
        )
        plt.show()

# %% [markdown]
# ## 9. Summary
#
# - The parameterized discrete-$SE(2)$ construction uses matrix-valued nonabelian irreps and keeps the recurrent mixing operator factored.
# - Allocentric position and orientation are both observable; the primary rollout reports signal, position, and heading errors and overlays exact/decoded pose arrows.
# - Tuning is grouped by retained irrep, while module-restricted manifolds show 3D PCA, UMAP, and $H_0/H_1/H_2$ Vietoris–Rips persistence from the same sampled activity.
# - The topology remains probe- and sampling-dependent: the pose probe samples a $12\times12$ spatial grid at all three discrete orientations and displays separate $x$, $y$, and orientation colorings.
