# Percent-format notebook source.
# Run cells marked with # %% in Cursor or another compatible editor.
# Outputs and notebook runtime metadata are intentionally omitted.

# %% [markdown]
# # Closed-form RNNs for discrete $SE(3)$
#
# This notebook constructs a fixed-weight PyTorch `nn.Module` for $G=\mathbb{Z}_n^3\rtimes O$, where $O$ is the 24-element orientation-preserving cubic rotation group. Signals transform by the regular left action $(g\cdot x)[h]=x[g^{-1}h]$. Its analytical weights are registered buffers, so no optimization or training occurs. We visualize the network, follow one long six-degree-of-freedom pose rollout, and analyze tuning and module-restricted neural geometry.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = next(
    parent
    for parent in (Path.cwd(), *Path.cwd().parents)
    if (parent / "src").is_dir()
)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.geometry.discrete_se3 import (
    advanced_pose,
    decode_pose,
    gaussian_landmark,
    orientation_energy,
    orientation_marginal,
    peaked_orientation_weights,
    periodic_distance_squared,
    plot_orientation_marginal,
    plot_orthogonal_slices,
    plot_pose_trajectory,
    plot_volume_scatter,
    rotation_error,
    spatial_energy,
    spatial_marginal,
    transformed_pose,
)
from src.finite_group_rnn import (
    build_finite_group_rnn,
    hidden_width,
    random_invertible_encoding,
    select_irreps_by_power,
)
from src.groups import DiscreteSE3Group
from src.neural_manifold import (
    analyze_module_orbits,
    build_module_orbits,
    coordinate_colors,
    fixed_point_embedding,
    plot_manifold_analysis,
)

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1. Group and representation structure
#
# Write an element as $g=(t,r)$, with $t\in\mathbb{Z}_n^3$ and $r\in O$. If $R_r$ is the signed permutation matrix for rotation $r$, then
#
# $$(t_1,r_1)(t_2,r_2)=(t_1+R_{r_1}t_2,\ r_1r_2).$$
#
# Elements are flattened with rotation as the most significant coordinate:
#
# $$\operatorname{idx}(x,y,z,r)=rn^3+xn^2+yn+z.$$
#
# Irreps are constructed by the little-group method: cubic rotations act on translation frequencies $k\in\mathbb{Z}_n^3$, and stabilizer irreps are induced to the full semidirect product. The implementation checks $\sum_\rho d_\rho^2=|G|$.

# %%


# %% [markdown]
# ## 2. Group construction and RNN build
#
# For each generally matrix-valued nonabelian irrep,
#
# $$\widehat{x}(\rho)=\sum_{g\in G}x(g)\rho(g)^\dagger.$$
#
# Let $\phi(z)=\operatorname{ReLU}(z)^2$. For relative drives $g_1,\ldots,g_T$,
# we use the right-regular body-frame action, so the cumulative physical motion is $g_1\cdots g_T$.
#
# $$h_1=\phi(W_{\rm in}x_{\rm allo}+W_{\rm drive}(g_1\cdot x_{\rm ego})),$$
#
# $$h_t=\phi(W_{\rm mix}h_{t-1}+W_{\rm drive}(g_t\cdot x_{\rm ego})),\qquad y_t=W_{\rm out}h_t.$$
#
# The trace-feature construction contributes $4q_\rho d_\rho^3$ hidden units per selected irrep. We apply
#
# $$W_{\rm mix}h=W_{\rm in}(W_{\rm out}h)$$
#
# without materializing a hidden-by-hidden matrix. The editable configuration controls the irrep count and width budget; power-per-hidden selection is part of this configured model, not a separate exact/truncated experiment.

# %%


# %% [markdown]


# %%
def rotation_index(group, matrix):
    """Look up a cubic rotation by its implemented allocentric matrix."""
    matrix = np.asarray(matrix)
    return next(
        rotation
        for rotation in range(group.num_rotations)
        if np.array_equal(group.rotation_matrix(rotation), matrix)
    )

# %% [markdown]
# ## 3. Allocentric and egocentric encodings
#
# The allocentric code is an off-center anisotropic periodic Gaussian with a peaked cubic-orientation profile, making both translation and rotation observable. The egocentric code is random and Fourier-invertible on every selected nonabelian block.
#
# Orthogonal slices summarize translation, a 24-bin marginal summarizes orientation, and RMS energy avoids cancellation for signed signals. These are compact diagnostics of the full signal on $\mathbb Z_n^3\rtimes O$.

# %%
G = DiscreteSE3Group(n=3)
initial_pose = (1, 0, 2, 0)
x_allo = gaussian_landmark(
    G,
    center=initial_pose[:3],
    sigma=(0.4, 0.7, 1.0),
    orientation_weights=peaked_orientation_weights(G, rotation=initial_pose[3], floor=0.05),
)

all_irreps = G.irreps()
selected_irreps, selected_indices = select_irreps_by_power(
    all_irreps,
    x_allo,
    num_irreps=6,
    max_hidden_width=10_000,
    ranking="power_per_hidden",
)
x_ego = random_invertible_encoding(G, selected_irreps, seed=19)
model = build_finite_group_rnn(
    G,
    x_ego,
    irreps=all_irreps,
    x_allo=x_allo,
    irrep_selection="power",
    num_irreps=6,
    max_hidden_width=10_000,
    power_ranking="power_per_hidden",
    materialize_mix=False,
)

print(f"|G|={G.order}; all irreps={len(all_irreps)}")
print("selected global indices:", model.selected_irrep_indices)
print("selected dimensions:", [irrep.dim for irrep in model.irreps])
print("hidden width:", model.hidden_dim)

# %%
# Allocentric encoding: signed marginals are meaningful because this signal is
# nonnegative and has an intentionally peaked orientation profile.
spatial = spatial_marginal(G, x_allo)
plot_orthogonal_slices(
    spatial,
    center=initial_pose[:3],
    title="Allocentric encoding $x_{allo}$: spatial marginal",
)
plt.show()

plot_orientation_marginal(
    G,
    x_allo,
    title="Allocentric encoding $x_{allo}$: orientation marginal",
)
plt.show()

# Egocentric encoding: use RMS magnitude so positive and negative random entries
# do not disappear through cancellation.
x_ego_spatial_energy = spatial_energy(G, x_ego)
plot_orthogonal_slices(
    x_ego_spatial_energy,
    title="Egocentric encoding $x_{ego}$: spatial RMS energy",
    cmap="magma",
)
plt.show()

figure, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
ax.bar(np.arange(G.num_rotations), orientation_energy(G, x_ego), color="#4C78A8")
ax.set(
    xlabel="Cubic rotation index",
    ylabel="RMS magnitude",
    title="Egocentric encoding $x_{ego}$: orientation energy",
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()

# %% [markdown]
# ## 4. Closed-form weight operators
#
# The four panels share the irrep-block order in `model.metadata`. $W_{\rm mix}$ is materialized only for inspection; recurrent evaluation keeps the product factored.

# %%
# Closed-form weight operators for the configured model. W_mix remains
# factored during recurrence and is materialized only for this diagnostic.
W_mix_for_plot = (model.W_in @ model.W_out).detach().cpu().numpy()
weight_matrices = {
    r"$W_{in}$": model.W_in.detach().cpu().numpy(),
    r"$W_{drive}$": model.W_drive.detach().cpu().numpy(),
    r"$W_{out}$": model.W_out.detach().cpu().numpy(),
    r"$W_{mix}=W_{in}W_{out}$": W_mix_for_plot,
}
metadata_irreps = np.asarray([item["irrep_index"] for item in model.metadata])
block_boundaries = np.flatnonzero(np.diff(metadata_irreps)) + 1

figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
for panel, (ax, (name, matrix)) in enumerate(
    zip(axes.ravel(), weight_matrices.items())
):
    scale = np.percentile(np.abs(matrix), 99)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
    )
    if panel in (0, 1, 3):
        for boundary in block_boundaries:
            ax.axhline(boundary - 0.5, color="black", linewidth=0.35, alpha=0.7)
    if panel in (2, 3):
        for boundary in block_boundaries:
            ax.axvline(boundary - 0.5, color="black", linewidth=0.35, alpha=0.7)
    ax.set_title(f"{name}  shape={matrix.shape}")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
figure.suptitle("Closed-form SE(3) RNN weights; lines mark irrep blocks")
plt.show()

del W_mix_for_plot

# %% [markdown]


# %%
def summarize_rollout(name, sequence):
    result = {
        key: value.detach().cpu().numpy()
        for key, value in model.rollout(x_allo, sequence).items()
    }
    signal_errors = np.linalg.norm(
        result["predicted_outputs"] - result["true_outputs"], axis=1
    ) / np.linalg.norm(result["true_outputs"], axis=1)

    predicted_poses = []
    target_poses = []
    position_errors = []
    orientation_errors = []
    for predicted, state in zip(
        result["predicted_outputs"], result["cumulative_states"]
    ):
        predicted_pose = decode_pose(G, predicted)
        target_pose = advanced_pose(G, initial_pose, int(state))
        predicted_poses.append(predicted_pose)
        target_poses.append(target_pose)
        position_errors.append(
            np.sqrt(periodic_distance_squared(G.n, predicted_pose[:3], target_pose[:3]))
        )
        orientation_errors.append(
            rotation_error(G, predicted_pose[3], target_pose[3])
        )

    print(
        f"{name:16s} final signal={signal_errors[-1]:.3e}; "
        f"max position={max(position_errors):.3f}; "
        f"max rotation={np.degrees(max(orientation_errors)):.1f}°"
    )
    return {
        "rollout": result,
        "signal_errors": np.asarray(signal_errors),
        "position_errors": np.asarray(position_errors),
        "orientation_errors": np.asarray(orientation_errors),
        "predicted_poses": np.asarray(predicted_poses),
        "target_poses": np.asarray(target_poses),
    }


rotation_x_90 = rotation_index(
    G,
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
)
rotation_z_90 = rotation_index(
    G,
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
)

# %%


# %% [markdown]
# ## 5. Primary long rollout and decoding error
#
# One parameterized mixed-motion sequence drives persistent Cartesian translations, stays, and cubic rotations. Wrapped path segments avoid false chords across the periodic cube. Sparse RGB orientation frames are the columns of `G.rotation_matrix(r)`: solid frames are exact and dashed frames are decoded, while geodesic rotation error remains the quantitative metric.

# %%
rng_long = np.random.default_rng(31)
translation_steps = np.asarray(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
)
num_long_steps = 120
long_sequence = []
direction_index = 0
for _ in range(num_long_steps):
    draw = rng_long.random()
    if draw < 0.05:
        long_sequence.append(G.identity())
    elif draw < 0.17:
        rotation = rotation_x_90 if rng_long.random() < 0.5 else rotation_z_90
        long_sequence.append(G.encode(0, 0, 0, rotation))
    else:
        if rng_long.random() < 0.25:
            direction_index = int(rng_long.integers(0, len(translation_steps)))
        dx, dy, dz = translation_steps[direction_index]
        long_sequence.append(G.encode(int(dx), int(dy), int(dz), 0))

long = summarize_rollout("long mixed", long_sequence)
long_absolute_errors = np.linalg.norm(
    long["rollout"]["predicted_outputs"] - long["rollout"]["true_outputs"],
    axis=1,
)

print("Long-sequence tracking summary")
print("------------------------------")
print("max output error:          ", long_absolute_errors.max())
print("mean output error:         ", long_absolute_errors.mean())
print("max position error:        ", long["position_errors"].max())
print("mean position error:       ", long["position_errors"].mean())
print("max rotation error (deg):  ", np.degrees(long["orientation_errors"]).max())
print("mean rotation error (deg): ", np.degrees(long["orientation_errors"]).mean())

plot_pose_trajectory(
    G,
    long["target_poses"][:, :3],
    long["predicted_poses"][:, :3],
    exact_rotations=long["target_poses"][:, 3],
    predicted_rotations=long["predicted_poses"][:, 3],
    title="Exact and decoded SE(3) pose",
)
plt.show()

# %%
time = np.arange(1, num_long_steps + 1)
error_series = (
    (long_absolute_errors, r"$\|y_t-y_t^\star\|_2$", "Output reconstruction error"),
    (long["position_errors"], "position error", "Decoded spatial-center error"),
    (
        np.degrees(long["orientation_errors"]),
        "rotation error (degrees)",
        "Decoded orientation error",
    ),
)
for values, ylabel, title in error_series:
    figure, ax = plt.subplots(figsize=(10, 2.8), constrained_layout=True)
    ax.plot(time, values, color="#E45756", linewidth=1.8)
    ax.set(xlabel="time step", ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

final_target = spatial_marginal(G, long["rollout"]["true_outputs"][-1])
final_prediction = spatial_marginal(G, long["rollout"]["predicted_outputs"][-1])
plot_orthogonal_slices(
    final_target,
    center=tuple(long["target_poses"][-1, :3]),
    title="Final target spatial marginal",
)
plt.show()
plot_orthogonal_slices(
    final_prediction,
    center=tuple(long["predicted_poses"][-1, :3]),
    title="Final reconstructed spatial marginal",
)
plt.show()

# %% [markdown]
# ## 6. Hidden-state tuning by irrep and power
#
# These tuning curves are computed by transforming `x_allo` through every group element while holding the egocentric drive fixed at the identity. We show one unit from each distinct nontrivial selected irrep block, up to the visualization limit set in the cell below. One-dimensional blocks are omitted because they do not provide informative pose tuning.
#
# A full tuning curve lives on 648 poses and cannot be displayed in one ordinary plot. We therefore use two complementary projections:
#
# 1. rotation slices are aligned to the allocentric frame and summed, producing one $3\times3\times3$ spatial tuning volume per neuron;
# 2. translation coordinates are summed, producing a representative-neuron by 24-orientation heatmap.
#
# The spatial volumes are rendered as colored, activity-scaled lattice points. All panels use a shared raw-activation scale.

# %%
num_tuning_units_to_plot = 6

# These tuning curves sweep transformed allocentric inputs while holding the
# egocentric drive at the identity.
# Note: a trajectory-based analysis could instead sample movements from the
# data distribution and average each neuron's activity at a given group element
# across trajectories.
tuning_hidden = model.probe_hidden_states(x_allo).detach().cpu().numpy()
representatives = []
seen_irreps = set()
for unit, metadata in enumerate(model.metadata):
    irrep_index = metadata["irrep_index"]
    if metadata["irrep_dim"] == 1 or irrep_index in seen_irreps:
        continue
    representatives.append(unit)
    seen_irreps.add(irrep_index)
    if len(representatives) == num_tuning_units_to_plot:
        break

spatial_tuning = [
    spatial_marginal(G, tuning_hidden[:, unit], align_rotations=True)
    for unit in representatives
]
vmin = min(float(volume.min()) for volume in spatial_tuning)
vmax = max(float(volume.max()) for volume in spatial_tuning)

num_columns = 2 if len(representatives) <= 4 else 3
num_rows = int(np.ceil(len(representatives) / num_columns))
figure = plt.figure(
    figsize=(5 * num_columns, 4.5 * num_rows),
    constrained_layout=True,
)
artists = []
for panel, (unit, volume) in enumerate(zip(representatives, spatial_tuning), start=1):
    metadata = model.metadata[unit]
    ax = figure.add_subplot(num_rows, num_columns, panel, projection="3d")
    artists.append(
        plot_volume_scatter(
            volume,
            ax=ax,
            title=f"unit {unit}; irrep {metadata['irrep_index']} (d={metadata['irrep_dim']})",
            vmin=vmin,
            vmax=vmax,
        )
    )
figure.colorbar(artists[-1], ax=figure.axes, fraction=0.02, pad=0.02)
figure.suptitle("Representative spatial tuning volumes (aligned rotation sum)")
plt.show()

orientation_tuning = np.asarray(
    [orientation_marginal(G, tuning_hidden[:, unit]) for unit in representatives]
)
figure, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
image = ax.imshow(orientation_tuning, aspect="auto", cmap="viridis")
ax.set(
    xlabel="Cubic rotation index",
    ylabel="Representative hidden unit",
    title="Orientation tuning after summing translations",
)
ax.set_yticks(np.arange(len(representatives)), labels=representatives)
figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="summed activation")
plt.show()

print("Selected tuning units")
print("---------------------")
for unit in representatives:
    print(unit, model.metadata[unit])

# %%
power = G.power_spectrum(x_allo)
retained_fraction = power[model.selected_irrep_indices].sum() / power.sum()
all_width = sum(hidden_width(irrep) for irrep in all_irreps)

print(f"retained Fourier power: {retained_fraction:.2%}")
print(f"selected hidden width: {model.hidden_dim:,}")
print(f"all-irrep hidden width: {all_width:,}")
print(f"width reduction: {1 - model.hidden_dim / all_width:.2%}")

# %% [markdown]
# ## 7. Module-restricted neural manifolds
#
# We evaluate the module-restricted orbit separately for two probes:
#
# 1. **translation probe:** fix cubic orientation at the identity and vary all $3^3=27$ positions;
# 2. **orientation probe:** fix translation at the origin and vary all 24 cubic rotations.
#
# For each retained nontrivial module, conjugate irreps are combined when needed. We verify the identity-update fixed-point residual before interpreting the hidden orbit as $\Phi$. UMAP is used only for visualization, while persistent homology is computed after PCA in neural space.
#
# These results are necessarily exploratory. Twenty-seven spatial samples are far too sparse to establish the topology of a continuous three-torus, and the 24-element cubic rotation group is a finite noncommutative set—not a densely sampled copy of $SO(3)$ or a circle. The two probes are plotted separately precisely because their topology and interpretation differ.

# %%
translation_elements = np.asarray(
    [
        G.encode(x, y, z, 0)
        for x in range(G.n)
        for y in range(G.n)
        for z in range(G.n)
    ]
)
translation_coordinates = np.asarray(
    [
        (x, y, z)
        for x in range(G.n)
        for y in range(G.n)
        for z in range(G.n)
    ]
)
orientation_elements = np.asarray(
    [G.encode(0, 0, 0, rotation) for rotation in range(G.num_rotations)]
)
translation_initial_hidden = tuning_hidden[translation_elements]
orientation_initial_hidden = tuning_hidden[orientation_elements]

fixed_probe_states = np.concatenate(
    (translation_initial_hidden, orientation_initial_hidden),
    axis=0,
)
fixed_probe = fixed_point_embedding(
    model,
    fixed_probe_states,
    tolerance=1e-10,
    max_iterations=20,
)
print(
    "identity-update fixed-point iteration: "
    f"converged={fixed_probe.converged}, "
    f"iterations={fixed_probe.iterations}, "
    f"max residual={fixed_probe.residuals.max():.3e}"
)
translation_fixed_hidden = fixed_probe.states[: len(translation_elements)]
orientation_fixed_hidden = fixed_probe.states[len(translation_elements) :]

translation_modules = build_module_orbits(model, translation_fixed_hidden)
orientation_modules = build_module_orbits(model, orientation_fixed_hidden)
translation_analyses = analyze_module_orbits(
    translation_modules,
    max_persistence_points=27,
    max_homology_dimension=2,
    random_state=13,
)
orientation_analyses = analyze_module_orbits(
    orientation_modules,
    max_persistence_points=24,
    max_homology_dimension=2,
    random_state=17,
)
translation_colors = coordinate_colors(
    translation_coordinates,
    (G.n, G.n, G.n),
)
orientation_colors = plt.get_cmap("turbo")(
    np.linspace(0, 1, G.num_rotations, endpoint=False)
)[:, :3]

for analysis in translation_analyses:
    plot_manifold_analysis(
        analysis,
        translation_colors,
        title=f"Translation probe — {analysis.module.label}",
    )
    plt.show()
for analysis in orientation_analyses:
    plot_manifold_analysis(
        analysis,
        orientation_colors,
        title=f"Finite orientation probe — {analysis.module.label}",
    )
    plt.show()

# %% [markdown]
# ## 8. Summary
#
# - The parameterized discrete-$SE(3)$ model uses induced matrix-valued irreps, cost-aware power selection, and factored recurrent mixing.
# - Position and cubic orientation are both observable; the primary rollout reports signal, periodic-position, and geodesic-rotation errors.
# - The pose panel overlays exact and decoded paths with sparse orientation frames derived directly from `G.rotation_matrix`.
# - Per-irrep translation and orientation probes each show 3D PCA, UMAP, and $H_0/H_1/H_2$ Vietoris–Rips persistence from the same sampled activity. These finite probes are diagnostics, not claims about continuous $\mathbb T^3$ or $SO(3)$ topology.
