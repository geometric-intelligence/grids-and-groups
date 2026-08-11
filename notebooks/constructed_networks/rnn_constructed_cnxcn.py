# %%
# Percent-format notebook source.
# Run cells marked with # %% in Cursor or another compatible editor.
# Outputs and notebook runtime metadata are intentionally omitted.

# %% [markdown]
# # Closed-form RNNs for $C_n\times C_n$
#
# This notebook constructs and analyzes a fixed-weight PyTorch `nn.Module` whose recurrent state realizes translation on the finite square torus
#
# $$G=C_n\times C_n.$$
#
# Its analytical weights are registered buffers, and no optimization or training occurs.
#
# A vector $x\in\mathbb R^{n^2}$ is identified with a scalar signal $x:G\to\mathbb R$. Because this group is abelian, every irreducible representation is a one-dimensional Fourier character. We build allocentric and egocentric encodings, inspect the closed-form RNN weights, evaluate a long translation trajectory, and analyze hidden-neuron tuning and irrep-restricted population geometry.
#
# For computational tractability, the configured model may retain only a power-selected subset of the available irreps. The configuration cell below controls the group size, encoding parameters, irrep count, and phase multiplicity.

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

from src.finite_group_rnn import (
    build_finite_group_rnn,
    hidden_width,
    random_invertible_encoding,
)
from src.geometry.cnxcn import (
    TRACK_COLOR,
    center_errors,
    decode_spatial_argmax,
    gaussian_bump,
    make_momentum_motion_sequence,
    plot_grid_scalar,
    plot_grid_trajectory,
    signal_to_grid,
    transformed_center,
)
from src.groups import ProductCyclicGroup
from src.neural_manifold import (
    analyze_module_orbit,
    build_module_orbits,
    combine_module_orbits,
    fixed_point_embedding,
)

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1. Group, action, and Fourier structure
#
# Write $g=(a,b)$ with $a,b\in C_n$. The group law is componentwise addition,
#
# $$(a,b)(c,d)=(a+c,b+d)\pmod n,$$
#
# and the regular left action is
#
# $$(g\cdot x)[h]=x[g^{-1}h].$$
#
# Equivalently, $g$ translates the $n\times n$ signal on a periodic square grid. The irreps are one-dimensional characters, corresponding to the familiar Fourier modes
#
# $$\rho_{k,\ell}(a,b)=\exp\left[2\pi i\left(\frac{ka}{n}+\frac{\ell b}{n}\right)\right].$$
#
# There are $n^2$ one-dimensional irreps, so $\sum_\rho d_\rho^2=n^2=|G|$. This group is abelian.

# %% [markdown]
# ## 2. Group construction and RNN
#
# The configuration below constructs $G=C_n\times C_n$, an allocentric Gaussian bump, and a random Fourier-invertible egocentric encoding. It then builds the closed-form RNN from a selected set of irreps.
#
# The model uses `irrep_selection="power"`: Fourier characters are ranked by their contribution to the allocentric signal's power, and the requested number are retained, with the trivial/DC character always included. Thus the model is truncated whenever `num_selected_irreps < len(all_irreps)`, but truncation is a configurable implementation choice rather than the main object of the notebook.
#
# The hidden-width formula is $H=4\sum_\rho q_\rho d_\rho^3$. Every $C_n\times C_n$ irrep has $d_\rho=1$, so each selected irrep contributes $4q_\rho$ hidden units.

# %%
n_spatial = 50
num_selected_irreps = 500
q_rho = 3
center_xy = (2, 2)
sigma = 1.0
ego_encoding_seed = 10

G = ProductCyclicGroup(n_spatial, n_spatial)
all_irreps = G.irreps()

x_allo = gaussian_bump(G, center=center_xy, sigma=sigma)
x_ego = random_invertible_encoding(G, all_irreps, seed=ego_encoding_seed)

model = build_finite_group_rnn(
    G,
    x_ego,
    x_allo=x_allo,
    irreps=all_irreps,
    irrep_selection="power",
    num_irreps=num_selected_irreps,
    q_rho=q_rho,
    materialize_mix=False,
)

print(f"G = C_{G.p1} x C_{G.p2}; |G|={G.order}")
print(f"selected irreps: {len(model.irreps)} / {len(all_irreps)}")
print(f"q_rho={model.q_rho}; hidden width={model.hidden_dim}")
print("W_mix stored:", model.W_mix is not None)

# %% [markdown]
# ## 3. Allocentric and egocentric encodings
#
# The allocentric encoding is a Gaussian bump on the periodic grid. The egocentric encoding is a random real signal whose selected Fourier blocks are invertible. We visualize both encodings and the allocentric Fourier power used to select irreps.

# %%
plot_grid_scalar(
    signal_to_grid(G, x_allo),
    title="Allocentric Gaussian encoding $x_{allo}$",
)
plt.show()

x_ego_grid = signal_to_grid(G, x_ego)
ego_scale = np.percentile(np.abs(x_ego_grid), 99)
plot_grid_scalar(
    x_ego_grid,
    title="Egocentric random encoding $x_{ego}$",
    cmap="coolwarm",
    vmin=-ego_scale,
    vmax=ego_scale,
)
plt.show()

power = G.power_spectrum(x_allo)
power_grid = power.reshape(G.p1, G.p2)
ax = plot_grid_scalar(
    np.log10(power_grid + 1e-14),
    title="Allocentric Fourier power (log scale); selected modes marked",
    cmap="magma",
)
selected_frequencies = np.asarray(
    [G.decode(index) for index in model.selected_irrep_indices]
)
ax.scatter(
    selected_frequencies[:, 0],
    selected_frequencies[:, 1],
    s=55,
    marker="o",
    facecolors="none",
    edgecolors="cyan",
    linewidths=1.2,
)
plt.show()

# %% [markdown]
# ## 4. Closed-form RNN weights
#
# We visualize $W_{\mathrm{in}}$, $W_{\mathrm{drive}}$, $W_{\mathrm{out}}$, and the implied recurrent matrix $W_{\mathrm{mix}}=W_{\mathrm{in}}W_{\mathrm{out}}$. Block boundaries mark the selected Fourier irreps.

# %%
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
figure.suptitle("Closed-form $C_n \\times C_n$ RNN weights; lines mark Fourier modes")
plt.show()

del W_mix_for_plot

# %% [markdown]
# ## 5. Long trajectory rollout and reconstruction error
#
# The long trajectory is the notebook's rollout test. We report both full-signal reconstruction error and decoded-center error: a power-selected model can preserve the Gaussian peak location even when truncation changes the bump's detailed shape or amplitude.

# %%
def summarize_rollout(name, sequence):
    result = {
        key: value.detach().cpu().numpy()
        for key, value in model.rollout(x_allo, sequence).items()
    }
    absolute_errors = np.linalg.norm(
        result["predicted_outputs"] - result["true_outputs"], axis=1
    )
    relative_errors = absolute_errors / np.linalg.norm(
        result["true_outputs"], axis=1
    )
    exact_centers = np.asarray(
        [transformed_center(G, int(state), center_xy) for state in result["cumulative_states"]]
    )
    predicted_centers = np.asarray(
        [decode_spatial_argmax(G, output) for output in result["predicted_outputs"]]
    )
    position_errors = center_errors(G, predicted_centers, exact_centers)
    print(
        f"{name:16s} final relative signal={relative_errors[-1]:.3e}; "
        f"max center={position_errors.max():.3f}"
    )
    return {
        "rollout": result,
        "absolute_errors": absolute_errors,
        "relative_errors": relative_errors,
        "position_errors": position_errors,
        "exact_centers": exact_centers,
        "predicted_centers": predicted_centers,
    }

# %% [markdown]
# The bounded random walk has `num_long_steps` updates. Its first relative translation moves the bump near the grid center; subsequent local translations have momentum with occasional turns and stays. Exact and decoded centers are overlaid on the periodic lattice, and reconstruction snapshots compare the target and RNN output at selected times.

# %%
num_long_steps = 250
start_xy = (
    (G.p1 // 2 - center_xy[0]) % G.p1,
    (G.p2 // 2 - center_xy[1]) % G.p2,
)
long_sequence = make_momentum_motion_sequence(
    G,
    steps=num_long_steps,
    seed=1,
    turn_probability=0.18,
    stay_probability=0.04,
    start_xy=start_xy,
)
long = summarize_rollout("long trajectory", long_sequence)

print("Long-sequence tracking summary")
print("------------------------------")
print("max output error:   ", long["absolute_errors"].max())
print("mean output error:  ", long["absolute_errors"].mean())
print("max center error:   ", long["position_errors"].max())
print("mean center error:  ", long["position_errors"].mean())

plot_grid_trajectory(
    G,
    long["exact_centers"],
    long["predicted_centers"],
    title="Tracking a Gaussian bump on $C_n\\times C_n$",
)
plt.show()

# %%
time = np.arange(1, num_long_steps + 1)
for values, ylabel, title in (
    (
        long["relative_errors"],
        r"$\|y_t-y_t^\star\|_2 / \|y_t^\star\|_2$",
        "Relative output reconstruction error",
    ),
    (long["position_errors"], "center error", "Decoded bump-center error"),
):
    figure, ax = plt.subplots(figsize=(10, 2.8), constrained_layout=True)
    ax.plot(time, values, color=TRACK_COLOR, linewidth=1.8)
    ax.set(xlabel="time step", ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

snapshot_steps = [0, num_long_steps // 2, num_long_steps - 1]
figure, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
snapshot_values = [
    signal_to_grid(G, long["rollout"][key][step])
    for key in ("true_outputs", "predicted_outputs")
    for step in snapshot_steps
]
vmin = min(float(values.min()) for values in snapshot_values)
vmax = max(float(values.max()) for values in snapshot_values)
for column, step in enumerate(snapshot_steps):
    plot_grid_scalar(
        signal_to_grid(G, long["rollout"]["true_outputs"][step]),
        ax=axes[0, column],
        title=f"target, step {step + 1}",
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
    )
    plot_grid_scalar(
        signal_to_grid(G, long["rollout"]["predicted_outputs"][step]),
        ax=axes[1, column],
        title=f"reconstructed, step {step + 1}",
        vmin=vmin,
        vmax=vmax,
        colorbar=column == 2,
    )
figure.suptitle("Long-rollout reconstruction snapshots")
plt.show()

# %% [markdown]
# ## 6. Analysis: neuron tuning by irrep
#
# These tuning curves are computed by translating the allocentric input through every group element while holding the egocentric drive fixed at the identity. Since $C_n\times C_n$ has no orientation coordinate, each neuron's tuning curve is directly displayed as a square-grid field.
#
# The selected nontrivial irreps are first filtered to one canonical representative from each available conjugate pair: the lower global irrep index is kept and its partner is excluded. Those representatives are then ordered deterministically by their Fourier-power contribution to `x_allo`, from largest to smallest; there is no random irrep sampling. Each figure contains many neurons from one irrep and uses a shared raw-activation color scale within that irrep. The variables in the cell below control how many irreps and how many neurons per irrep are shown.

# %%
# Use None to show every selected nontrivial irrep, or a positive integer to cap it.
num_tuning_irreps_to_plot = 12
# Use None to show every neuron in each irrep, or a positive integer to cap it.
num_tuning_neurons_per_irrep = None
max_tuning_columns = 6

# Sweep translated allocentric inputs while holding the egocentric drive fixed.
tuning_hidden = model.probe_hidden_states(x_allo).detach().cpu().numpy()
units_by_irrep = {}
for unit, metadata in enumerate(model.metadata):
    irrep_index = int(metadata["irrep_index"])
    units_by_irrep.setdefault(irrep_index, []).append(unit)

nontrivial_irrep_indices = [
    int(irrep_index)
    for irrep_index in model.selected_irrep_indices
    if irrep_index != 0
]
available_nontrivial_irreps = set(nontrivial_irrep_indices)


def tuning_conjugate_irrep_index(irrep_index):
    k, ell = G.decode(int(irrep_index))
    return G.encode(-k, -ell)


# Keep the lower-index member when both conjugate irreps are available.
representative_tuning_irreps = [
    irrep_index
    for irrep_index in nontrivial_irrep_indices
    if (
        tuning_conjugate_irrep_index(irrep_index)
        not in available_nontrivial_irreps
        or irrep_index <= tuning_conjugate_irrep_index(irrep_index)
    )
]
ranked_tuning_irreps = sorted(
    representative_tuning_irreps,
    key=lambda irrep_index: (-power[irrep_index], irrep_index),
)
if num_tuning_irreps_to_plot is None:
    tuning_irreps_to_plot = ranked_tuning_irreps
else:
    if num_tuning_irreps_to_plot < 1:
        raise ValueError("num_tuning_irreps_to_plot must be positive or None")
    tuning_irreps_to_plot = ranked_tuning_irreps[:num_tuning_irreps_to_plot]

for power_rank, irrep_index in enumerate(tuning_irreps_to_plot, start=1):
    units = np.asarray(units_by_irrep[irrep_index], dtype=int)
    if num_tuning_neurons_per_irrep is not None:
        if num_tuning_neurons_per_irrep < 1:
            raise ValueError("num_tuning_neurons_per_irrep must be positive or None")
        units = units[:num_tuning_neurons_per_irrep]

    tuning_fields = [signal_to_grid(G, tuning_hidden[:, unit]) for unit in units]
    vmin = min(float(field.min()) for field in tuning_fields)
    vmax = max(float(field.max()) for field in tuning_fields)
    num_columns = min(max_tuning_columns, len(units))
    num_rows = int(np.ceil(len(units) / num_columns))
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(2.6 * num_columns, 2.55 * num_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, unit, field in zip(axes, units, tuning_fields):
        metadata = model.metadata[int(unit)]
        plot_grid_scalar(
            field,
            ax=ax,
            title=(
                f"unit {int(unit)}\n"
                f"eps=({metadata['eps1']},{metadata['eps2']}), "
                f"delta={metadata['delta']}"
            ),
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
        )
    for ax in axes[len(units) :]:
        ax.axis("off")

    colorbar = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
        cmap=plt.get_cmap("viridis"),
    )
    colorbar.set_array([])
    figure.colorbar(colorbar, ax=axes[: len(units)], fraction=0.025, pad=0.02)
    frequency = G.decode(irrep_index)
    power_fraction = power[irrep_index] / power.sum()
    figure.suptitle(
        f"Power rank {power_rank}: irrep {irrep_index}, "
        rf"$(k,\ell)={frequency}$; "
        f"x_allo power={power_fraction:.2%}; {len(units)} neurons"
    )
    plt.show()

# %%
retained_fraction = power[model.selected_irrep_indices].sum() / power.sum()
all_width = sum(hidden_width(irrep) for irrep in all_irreps)
print(f"retained Fourier power: {retained_fraction:.2%}")
print(f"selected hidden width: {model.hidden_dim:,}")
print(f"all-irrep hidden width: {all_width:,}")
print(f"width reduction: {1 - model.hidden_dim / all_width:.2%}")

# %% [markdown]
# ## 7. Module-restricted neural manifolds
#
# For each displayed Fourier irrep, we form the population orbit
#
# $$
# \mathcal M_{\rho,x_{\mathrm{allo}}}
# =
# \{\Pi_\rho\Phi(g\cdot x_{\mathrm{allo}}):g\in C_n\times C_n\}.
# $$
#
# We do not aggregate conjugate irreps. For each available pair $\rho_{k,\ell}$ and $\rho_{-k,-\ell}$, we keep the representative with the smaller global irrep index and exclude its partner. Self-conjugate irreps, or irreps whose partner was not retained by the RNN, are kept.
#
# We sample `num_sample_points_per_axis` translations along each cyclic coordinate, producing its square in sampled group elements. Every sampled population state is iterated to a recurrent fixed point, and the final residual is reported explicitly.
#
# The displayed irreps are ordered deterministically by wrapped Fourier radius $\sqrt{\min(k,n-k)^2+\min(\ell,n-\ell)^2}$, from lowest to higher spatial frequency. This ordering is intentionally different from the power ordering used in the neuron-tuning section.
#
# The final combined-pair plot concatenates two low-frequency representative irreps whose Fourier vectors span both translation directions: one irrep gives one circular phase, while two independent phases can reveal the torus.
#
# The focused cells below first show 3D PCA point clouds colored separately by the two ground-truth translation coordinates, then show theory-guided 2D circle projections for the individual irreps, and finally show dedicated Vietoris–Rips persistence diagrams. Averaging the duplicate $\epsilon_1$ neurons and differencing the $\epsilon_2=\pm1$ responses isolates the first harmonic, with radius $4A^2R$ under the projection used below. Centering and summing those responses isolates the second harmonic, with radius $A^2R^2$. Two $\delta$ phases are linearly de-skewed into cosine/sine coordinates. All harmonic panels share axis limits, so their relative sizes are preserved. Each circle is shown twice, colored separately by the ground-truth $x$ and $y$ coordinates. These coordinate colors are diagnostic rather than intrinsic: whenever several positions have the same irrep phase, their projected points coincide. A standard scatter plot can display only the last-drawn color at that location, so an apparently uniform high-coordinate color can be an overplotting artifact rather than evidence that the underlying samples changed coordinates.

# %%
# Valid values: any integer from 2 through min(G.p1, G.p2).
# The plot contains num_sample_points_per_axis**2 points. Use the full period
# to show every group element when both cyclic factors have the same size.
num_sample_points_per_axis = 20
if not 2 <= num_sample_points_per_axis <= min(G.p1, G.p2):
    raise ValueError("num_sample_points_per_axis must be between 2 and min(G.p1, G.p2)")

sample_coordinates_1d = np.unique(
    np.linspace(0, G.p1 - 1, num_sample_points_per_axis, dtype=int)
)
translation_coordinates = np.asarray(
    [(x, y) for x in sample_coordinates_1d for y in sample_coordinates_1d]
)
translation_elements = np.asarray(
    [G.encode(int(x), int(y)) for x, y in translation_coordinates]
)
translation_fixed = fixed_point_embedding(
    model,
    tuning_hidden[translation_elements],
    tolerance=1e-8,
    max_iterations=50,
)
print(
    "identity-update fixed-point iteration: "
    f"converged={translation_fixed.converged}, "
    f"iterations={translation_fixed.iterations}, "
    f"max residual={translation_fixed.residuals.max():.3e}"
)

individual_module_orbits = build_module_orbits(
    model,
    translation_fixed.states,
    include_conjugates=False,
)
available_irrep_indices = {
    int(module.irrep_indices[0]) for module in individual_module_orbits
}


def conjugate_irrep_index(irrep_index):
    k, ell = G.decode(int(irrep_index))
    return G.encode(-k, -ell)


# Keep the lower-index representative when both members of a conjugate pair
# are available. Keep an irrep unchanged when its partner is absent.
all_module_orbits = [
    module
    for module in individual_module_orbits
    if (
        conjugate_irrep_index(module.irrep_indices[0])
        not in available_irrep_indices
        or module.irrep_indices[0]
        <= conjugate_irrep_index(module.irrep_indices[0])
    )
]
print(
    f"conjugate filtering: {len(individual_module_orbits)} individual irreps -> "
    f"{len(all_module_orbits)} representatives"
)


def wrapped_frequency(frequency):
    frequency = np.asarray(frequency, dtype=int)
    periods = np.asarray((G.p1, G.p2), dtype=int)
    return np.minimum(frequency, periods - frequency)


def module_frequency_summary(module):
    raw_frequencies = np.asarray(
        [G.decode(int(irrep_index)) for irrep_index in module.irrep_indices]
    )
    wrapped_frequencies = np.asarray(
        [wrapped_frequency(frequency) for frequency in raw_frequencies]
    )
    radii_squared = np.sum(wrapped_frequencies**2, axis=1)
    representative = int(np.argmin(radii_squared))
    return raw_frequencies, wrapped_frequencies, float(radii_squared[representative])


def module_frequency_key(module):
    _, wrapped_frequencies, radius_squared = module_frequency_summary(module)
    nearest_frequency = wrapped_frequencies[np.argmin(np.sum(wrapped_frequencies**2, axis=1))]
    return (
        radius_squared,
        int(np.sum(nearest_frequency)),
        tuple(int(index) for index in module.irrep_indices),
    )


num_modules_to_plot = 6
low_frequency_modules = sorted(all_module_orbits, key=module_frequency_key)
manifold_modules = low_frequency_modules[:num_modules_to_plot]

independent_pair = None
for first_index, first_module in enumerate(low_frequency_modules):
    first_frequency = G.decode(first_module.irrep_indices[0])
    for second_module in low_frequency_modules[first_index + 1 :]:
        second_frequency = G.decode(second_module.irrep_indices[0])
        determinant = (
            first_frequency[0] * second_frequency[1]
            - first_frequency[1] * second_frequency[0]
        )
        if np.gcd(abs(determinant), G.p1) == 1:
            independent_pair = (first_module, second_module)
            break
    if independent_pair is not None:
        break
if independent_pair is not None:
    manifold_modules.append(combine_module_orbits(list(independent_pair)))


def module_title(module):
    joined = "+".join(str(index) for index in module.irrep_indices)
    prefix = "Irrep" if len(module.irrep_indices) == 1 else "Combined irreps"
    return f"{prefix} {joined} ({len(module.unit_indices)} neurons)"


print("low-frequency manifold inputs shown:")
for module in manifold_modules:
    raw_frequencies, wrapped_frequencies, radius_squared = module_frequency_summary(module)
    print(
        f"  {module_title(module)}: frequencies={raw_frequencies.tolist()}, "
        f"wrapped={wrapped_frequencies.tolist()}, "
        f"radius={np.sqrt(radius_squared):.3g}"
    )


def pca_3d_coordinates(activity):
    centered = np.asarray(activity, dtype=float)
    centered = centered - centered.mean(axis=0, keepdims=True)
    active_columns = np.std(centered, axis=0) > 1e-12
    centered = centered[:, active_columns]
    if centered.shape[1] == 0 or len(centered) < 2:
        return np.zeros((len(activity), 3)), 0.0
    scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    if scale > 0:
        centered = centered / scale
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    num_components = min(3, len(singular_values))
    coordinates = centered @ components[:num_components].T
    if num_components < 3:
        coordinates = np.column_stack(
            [coordinates, np.zeros((len(centered), 3 - num_components))]
        )
    variances = singular_values**2
    explained_variance = (
        float(np.sum(variances[:num_components]) / np.sum(variances))
        if np.sum(variances) > 0
        else 0.0
    )
    return coordinates, explained_variance


color_specs = [
    ("x coordinate", translation_coordinates[:, 0], "viridis", "x"),
    ("y coordinate", translation_coordinates[:, 1], "plasma", "y"),
]


def plot_module_pca3d(module, title):
    coordinates, explained_variance = pca_3d_coordinates(module.activity)
    figure = plt.figure(figsize=(8.2, 4.2), constrained_layout=True)
    axes = []
    for panel, (color_title, values, cmap, colorbar_label) in enumerate(color_specs, start=1):
        ax = figure.add_subplot(1, len(color_specs), panel, projection="3d")
        artist = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            c=values,
            cmap=cmap,
            s=18,
            alpha=0.9,
            linewidths=0,
        )
        figure.colorbar(
            artist,
            ax=ax,
            fraction=0.045,
            pad=0.02,
            label=colorbar_label,
        )
        ax.set(
            xlabel="PC1",
            ylabel="PC2",
            zlabel="PC3",
            title=color_title,
        )
        ax.view_init(elev=22, azim=45)
        axes.append(ax)
    figure.suptitle(f"{title}; 3D PCA variance={explained_variance:.1%}")
    return axes


def mean_activity_for_sign_and_phase(module, rnn_model, eps2, delta):
    local_columns = [
        local_column
        for local_column, unit in enumerate(module.unit_indices)
        if rnn_model.metadata[int(unit)]["eps2"] == eps2
        and rnn_model.metadata[int(unit)]["delta"] == delta
    ]
    if not local_columns:
        raise ValueError(
            f"missing neurons for eps2={eps2}, delta={delta}, "
            f"irrep={module.irrep_indices}"
        )
    # eps1 produces duplicate tuning curves after the squared nonlinearity.
    return module.activity[:, local_columns].mean(axis=1)


def deskew_phase_pair(first_curve, second_curve, phase_offset):
    sine = np.sin(phase_offset)
    if abs(sine) < 1e-12:
        raise ValueError("phase offset must provide two independent directions")
    return np.column_stack(
        [
            first_curve,
            (first_curve * np.cos(phase_offset) - second_curve) / sine,
        ]
    )


def harmonic_circle_projections(module, rnn_model):
    if len(module.irrep_indices) != 1:
        raise ValueError("harmonic projections require one individual irrep")
    if rnn_model.q_rho < 3:
        raise ValueError("harmonic projections require q_rho >= 3")

    difference_curves = []
    sum_curves = []
    for delta in (0, 1):
        plus = mean_activity_for_sign_and_phase(
            module, rnn_model, eps2=1, delta=delta
        )
        minus = mean_activity_for_sign_and_phase(
            module, rnn_model, eps2=-1, delta=delta
        )
        phi_delta = np.pi * delta / rnn_model.q_rho
        difference_curves.append((plus - minus) / np.cos(phi_delta))
        summed = plus + minus
        sum_curves.append(summed - summed.mean())

    first_harmonic = deskew_phase_pair(
        difference_curves[0],
        difference_curves[1],
        np.pi / rnn_model.q_rho,
    )
    second_harmonic = deskew_phase_pair(
        sum_curves[0],
        sum_curves[1],
        2 * np.pi / rnn_model.q_rho,
    )
    return first_harmonic, second_harmonic


def plot_harmonic_circle_projections(module, rnn_model, title):
    first_harmonic, second_harmonic = harmonic_circle_projections(
        module, rnn_model
    )
    irrep_index = int(module.irrep_indices[0])
    k, ell = G.decode(irrep_index)
    rho_squared_is_trivial = (2 * k) % G.p1 == 0 and (2 * ell) % G.p2 == 0

    all_coordinates = np.vstack([first_harmonic, second_harmonic])
    axis_limit = max(float(np.max(np.abs(all_coordinates))) * 1.08, 1e-12)
    first_radius = float(np.median(np.linalg.norm(first_harmonic, axis=1)))
    second_radius = float(np.median(np.linalg.norm(second_harmonic, axis=1)))

    harmonic_specs = [
        (
            first_harmonic,
            r"first harmonic: $4A^2R$" + "\n" + f"median radius={first_radius:.3g}",
        ),
        (
            second_harmonic,
            (
                r"second harmonic: $A^2R^2$"
                + "\n"
                + f"median radius={second_radius:.3g}"
                + (" (degenerate: $\\rho^2=1$)" if rho_squared_is_trivial else "")
            ),
        ),
    ]
    figure, axes = plt.subplots(
        len(color_specs),
        len(harmonic_specs),
        figsize=(9, 8),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(color_specs), len(harmonic_specs))
    for row, (color_title, values, cmap, colorbar_label) in enumerate(color_specs):
        for column, (coordinates, harmonic_title) in enumerate(harmonic_specs):
            ax = axes[row, column]
            artist = ax.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                c=values,
                cmap=cmap,
                s=22,
                alpha=0.65,
                linewidths=0,
            )
            ax.set(
                xlim=(-axis_limit, axis_limit),
                ylim=(-axis_limit, axis_limit),
                xlabel="projection coordinate 1",
                ylabel="projection coordinate 2",
                title=f"{harmonic_title}\ncolored by {color_title}",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.axhline(0, color="0.8", linewidth=0.6, zorder=0)
            ax.axvline(0, color="0.8", linewidth=0.6, zorder=0)
            figure.colorbar(
                artist,
                ax=ax,
                fraction=0.045,
                pad=0.02,
                label=colorbar_label,
            )
    figure.suptitle(f"{title}: theory-guided harmonic projections (shared scale)")
    return axes


# Plotting is intentionally split into the focused cells below.

# %% [markdown]
# ### 7.1 Baseline 3D PCA projections
#
# The first focused view applies 3D PCA to each selected low-frequency module. Separate panels color the same embedding by the ground-truth $x$ and $y$ coordinates. The combined independent-frequency control is included because two independent character phases can expose both directions of the translation torus.

# %%
for index, module in enumerate(manifold_modules):
    is_combined_pair = (
        index == len(manifold_modules) - 1
        and independent_pair is not None
    )
    title = (
        "Combined independent low-frequency pair"
        if is_combined_pair
        else module_title(module)
    )
    plot_module_pca3d(module, title)
    plt.show()

# %% [markdown]
# ### 7.2 Theory-guided harmonic projections
#
# For each individual irrep, signed and phase-shifted neural responses are combined to isolate the first and second character harmonics. These projections test the predicted circular phase geometry directly. Both projection coordinates use identical limits and an equal aspect ratio, so geometric distortion cannot be hidden by axis stretching.

# %%
individual_manifold_modules = [
    module for module in manifold_modules if len(module.irrep_indices) == 1
]
baseline_modules_by_irrep = {
    int(module.irrep_indices[0]): module
    for module in individual_manifold_modules
}
for module in individual_manifold_modules:
    plot_harmonic_circle_projections(
        module,
        model,
        module_title(module),
    )
    plt.show()

# %% [markdown]
# ### 7.3 Vietoris–Rips persistence
#
# Persistent homology is computed from PCA-preprocessed neural-space distances with deterministic farthest-point subsampling. The barcode panels below show each feature as a horizontal interval from birth to death for $H_0$, $H_1$, and $H_2$. Features with infinite death times end in an arrow at the plotting boundary. The combined-irrep control is excluded because it serves a different, explicitly constructed two-frequency comparison.

# %%
persistence_analyses = {}
for module in individual_manifold_modules:
    analysis = analyze_module_orbit(
        module,
        max_persistence_points=min(300, len(module.activity)),
        max_homology_dimension=2,
        random_state=19,
    )
    persistence_analyses[int(module.irrep_indices[0])] = analysis

    finite_death_arrays = [
        diagram[np.isfinite(diagram[:, 1]), 1]
        for diagram in analysis.persistence_diagrams
        if len(diagram) and np.any(np.isfinite(diagram[:, 1]))
    ]
    finite_deaths = (
        np.concatenate(finite_death_arrays)
        if finite_death_arrays
        else np.asarray([])
    )
    cap = max(float(finite_deaths.max()), 1e-8) if finite_deaths.size else 1.0
    infinity_endpoint = 1.05 * cap
    upper = 1.12 * cap
    persistence_colors = ("0.45", "#4C78A8", "#F58518")
    figure, axes = plt.subplots(
        1,
        len(analysis.persistence_diagrams),
        figsize=(13, 4.5),
        constrained_layout=True,
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for dimension, (ax, diagram) in enumerate(
        zip(axes, analysis.persistence_diagrams)
    ):
        if not len(diagram):
            ax.set(title=rf"$H_{dimension}$ (empty)")
            continue
        finite = np.isfinite(diagram[:, 1])
        display_deaths = np.where(finite, diagram[:, 1], infinity_endpoint)
        persistence = display_deaths - diagram[:, 0]
        order = np.argsort(persistence)
        births = diagram[order, 0]
        deaths = display_deaths[order]
        finite = finite[order]
        rows = np.arange(len(diagram))
        color = persistence_colors[dimension]

        ax.hlines(rows, births, deaths, color=color, linewidth=1.5, alpha=0.85)
        ax.scatter(births, rows, color=color, marker="|", s=28, linewidths=0.9)
        if np.any(~finite):
            ax.scatter(
                deaths[~finite],
                rows[~finite],
                color=color,
                marker=">",
                s=25,
                label="persists beyond plot",
            )
        ax.set(
            xlim=(-0.02 * upper, upper),
            xlabel="filtration scale",
            ylabel="feature (ordered by persistence)",
            title=rf"$H_{dimension}$ barcode",
        )
        ax.spines[["top", "right"]].set_visible(False)
        if np.any(~finite):
            ax.legend(frameon=False, fontsize=8, loc="lower right")

    figure.suptitle(f"{module_title(module)} — Vietoris–Rips persistence barcodes")
    plt.show()

# %% [markdown]
# ## 8. Rebuild with anisotropic $A_u$, $A_v$, and $A_w$
#
# The next cell constructs a new fixed-weight model from anisotropic amplitude factors, then recomputes its hidden orbit and recurrent fixed points rather than analytically rescaling the baseline activity. The positive multipliers retain product one, preserving $A_uA_vA_w=(q_\rho|G|)^{-1}$.
#
# The final comparison applies the same theory-guided 2D harmonic projections to the baseline and rebuilt networks. Each row uses one common limit on both projection coordinates and across both conditions, making circular-to-elliptical distortions directly visible.

# %%
# Multipliers are relative to the balanced amplitudes. All values must be
# positive and their product must equal one.
anisotropic_amplitude_multipliers = (2.0, 0.5, 1.0)

anisotropic_model = build_finite_group_rnn(
    G,
    x_ego,
    x_allo=x_allo,
    irreps=all_irreps,
    irrep_selection="power",
    num_irreps=num_selected_irreps,
    q_rho=q_rho,
    amplitude_mode="balanced",
    amplitude_multipliers=anisotropic_amplitude_multipliers,
    materialize_mix=False,
)
anisotropic_tuning_hidden = (
    anisotropic_model.probe_hidden_states(x_allo).detach().cpu().numpy()
)
anisotropic_fixed = fixed_point_embedding(
    anisotropic_model,
    anisotropic_tuning_hidden[translation_elements],
    tolerance=1e-8,
    max_iterations=50,
)
anisotropic_module_orbits = build_module_orbits(
    anisotropic_model,
    anisotropic_fixed.states,
    include_conjugates=False,
)
anisotropic_modules_by_irrep = {
    int(module.irrep_indices[0]): module
    for module in anisotropic_module_orbits
}

amplitude_product = 1.0 / (anisotropic_model.q_rho * G.order)
balanced_amplitude = amplitude_product ** (1 / 3)
actual_amplitudes = balanced_amplitude * np.asarray(
    anisotropic_model.amplitude_multipliers
)
amplitude_label = (
    rf"$A_u={actual_amplitudes[0]:.2e}$, "
    rf"$A_v={actual_amplitudes[1]:.2e}$, "
    rf"$A_w={actual_amplitudes[2]:.2e}$"
)
displayed_irrep_indices = [
    int(module.irrep_indices[0])
    for module in individual_manifold_modules
]

print(
    "anisotropic identity-update fixed-point iteration: "
    f"converged={anisotropic_fixed.converged}, "
    f"iterations={anisotropic_fixed.iterations}, "
    f"max residual={anisotropic_fixed.residuals.max():.3e}"
)
print(f"actual amplitudes: {actual_amplitudes}")

# %% [markdown]
# ### 8.1 Baseline versus anisotropic harmonic geometry
#
# For each displayed irrep, the columns compare the baseline and rebuilt anisotropic RNN. Rows show the first and second harmonic projections colored separately by $x$ and $y$. A single axis limit is shared across every panel for that irrep, and every panel has equal coordinate scaling. Titles report measured median radii rather than applying balanced-amplitude radius formulas to the anisotropic network.

# %%
for irrep_index in displayed_irrep_indices:
    baseline_module = baseline_modules_by_irrep[irrep_index]
    anisotropic_module = anisotropic_modules_by_irrep[irrep_index]
    baseline_harmonics = harmonic_circle_projections(baseline_module, model)
    anisotropic_harmonics = harmonic_circle_projections(
        anisotropic_module, anisotropic_model
    )

    all_coordinates = np.vstack(
        [*baseline_harmonics, *anisotropic_harmonics]
    )
    axis_limit = max(float(np.max(np.abs(all_coordinates))) * 1.08, 1e-12)
    condition_specs = (
        ("Balanced", baseline_harmonics),
        ("Anisotropic", anisotropic_harmonics),
    )
    harmonic_names = ("first harmonic", "second harmonic")
    figure, axes = plt.subplots(
        len(color_specs) * len(harmonic_names),
        len(condition_specs),
        figsize=(10, 13),
        constrained_layout=True,
        squeeze=False,
    )

    for color_index, (color_title, values, cmap, colorbar_label) in enumerate(
        color_specs
    ):
        for harmonic_index, harmonic_name in enumerate(harmonic_names):
            row = color_index * len(harmonic_names) + harmonic_index
            for column, (condition_name, harmonics) in enumerate(condition_specs):
                coordinates = harmonics[harmonic_index]
                radius = float(np.median(np.linalg.norm(coordinates, axis=1)))
                ax = axes[row, column]
                artist = ax.scatter(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    c=values,
                    cmap=cmap,
                    s=22,
                    alpha=0.65,
                    linewidths=0,
                )
                ax.set(
                    xlim=(-axis_limit, axis_limit),
                    ylim=(-axis_limit, axis_limit),
                    xlabel="projection coordinate 1",
                    ylabel="projection coordinate 2",
                    title=(
                        f"{condition_name}: {harmonic_name}\n"
                        f"colored by {color_title}; median radius={radius:.3g}"
                    ),
                )
                ax.set_aspect("equal", adjustable="box")
                ax.axhline(0, color="0.8", linewidth=0.6, zorder=0)
                ax.axvline(0, color="0.8", linewidth=0.6, zorder=0)
                figure.colorbar(
                    artist,
                    ax=ax,
                    fraction=0.045,
                    pad=0.02,
                    label=colorbar_label,
                )

    title = module_title(baseline_module)
    figure.suptitle(
        f"{title}: balanced versus anisotropic harmonic geometry\n"
        f"{amplitude_label}; shared axis limit={axis_limit:.3g}"
    )
    plt.show()

# %% [markdown]
# ## 9. Summary
#
# - The configured model retains `len(model.irreps)` Fourier characters selected by their allocentric power contribution. Signal-reconstruction and decoded-center errors measure different consequences of this configurable truncation.
# - Neuron-tuning figures keep one canonical representative from each available conjugate pair, order those representatives by allocentric power contribution, and show all neurons unless explicitly capped.
# - Manifold figures use the same conjugate-representative rule—not an aggregation of the pair—and order those representatives from low to higher wrapped spatial frequency.
# - Baseline manifold analysis is separated into 3D PCA, theory-guided first/second harmonic projections, and dedicated $H_0/H_1/H_2$ Vietoris–Rips persistence diagrams.
# - One nontrivial irrep generally exposes one circular phase, while two irreps with independent frequency vectors can expose both phases of the translation torus.
# - The anisotropic-amplitude section rebuilds the RNN and compares its harmonic geometry directly against baseline using shared equal axis scales; comparison figures are saved as SVG files.
# - The group is abelian: this notebook studies translation accumulation, not the noncommuting translation–rotation interactions of a semidirect product.
