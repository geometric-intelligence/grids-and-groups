# %%
# Percent-format notebook source.
# Regenerate the paired .ipynb with Jupytext.
# Outputs and notebook runtime metadata are intentionally omitted.

# %% [markdown]
# # Figure 8 — spatial and conjunctive tuning
#
# This is deliberately a research notebook, not a plotting library. Run the setup once, then run the panel you want. Each panel writes one editable SVG to `artifacts/constructed_networks/discrete_se2_c6/paper_figures/`.
#
# The construction is the full, untruncated $n=21$ model. Panel A uses a true periodic local random walk: all $7\times3=21$ local actions are equally likely. Panels B--E use exact all-pairs tuning unless stated otherwise.

# %%
from pathlib import Path
import copy
import csv
import gc
import sys
import time
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.ticker import MaxNLocator

project_root = next(folder for folder in (Path.cwd(), *Path.cwd().parents) if (folder / "src").is_dir())
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis import (
    TrajectoryTuningConfig,
    compute_all_pairs_tuning,
    compute_local_arrival_tuning,
    load_or_compute_empirical_trajectory_tuning,
)
from src.experiments.discrete_se2 import DiscreteSE2ExperimentConfig, build_discrete_se2_experiment
from src.geometry.discrete_se2 import NaturalisticMotionConfig, plot_lattice_scalar
from src.groups import DiscreteSE2Group

plt.rcParams["svg.fonttype"] = "none"

figures = project_root / "artifacts" / "constructed_networks" / "discrete_se2_c6" / "paper_figures"
figures.mkdir(parents=True, exist_ok=True)

# The four modules used throughout Panel C.  Their frequency representatives are
# (0,1), (1,2), (0,2), and (1,-2), with radii 1, sqrt(3), 2, and sqrt(7).
n = 21
panel_c_rhos = (9, 19, 10, 36)

configuration = DiscreteSE2ExperimentConfig(
    n_spatial=n,
    n_orientations=6,
    initial_pose=(2, 2, 0),
    allocentric_encoding="gaussian space custom orientation",
    sigma=1.0,
    custom_orientation_weights=(1.0, 0.8, 0.4, 0.2, 0.4, 0.8),
    encoding_seed=10,
    action_side="right",
    irrep_selection="power",
    max_hidden_width=None,                 # full construction: H = 189,576
    normalize_power_by_dim=True,
    always_include_trivial=True,
    power_ranking="power",
    q_rho=3,
    amplitude_mode="balanced",
    amplitude_multipliers=(1.0, 1.0, 1.0),
    materialize_mix=False,
)
# On a notebook rerun, release the previous full-width construction before
# allocating its replacement.  Otherwise both 11-GiB models briefly coexist.
if "experiment" in globals():
    del experiment
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

started = time.perf_counter()
experiment = build_discrete_se2_experiment(configuration)
model_buffer_gib = sum(buffer.numel() * buffer.element_size() for buffer in experiment.model.buffers()) / 2**30
print(f"Built the full n={n} construction in {time.perf_counter() - started:.1f} s; hidden width = {experiment.model.hidden_dim:,}; persistent buffers = {model_buffer_gib:.2f} GiB.")
if torch.cuda.is_available():
    experiment.model.to("cuda")
    print("Moved the construction to CUDA.")

# These are the seven local translations (stay plus six neighbours) and three
# relative turns. Each joint local action therefore has probability 1/21.
local_random_walk = NaturalisticMotionConfig(
    stay_probability=1 / 7,
    forward_probability=1 / 7,
    forward_left_or_right_probability=1 / 7,
    backward_left_or_right_probability=1 / 7,
    backward_probability=1 / 7,
    turn_probability=1 / 3,
    turn_persistence=0,
    wall_lookahead=1,
    wall_avoidance_strength=0,
    minimum_wall_weight=1,
    periodic_boundaries=True,
)


def display_normalized(values):
    # Map one tuning curve to [0, 1] for plotting only.
    values = np.asarray(values, dtype=float)
    low, high = np.nanmin(values), np.nanmax(values)
    return (values - low) / max(high - low, 1e-15)


def plot_orientation_pizza(values, ax, color, deviation_scale):
    # The dashed circle is each curve's mean activity. Radial deviations use
    # one shared raw-activity scale for the whole panel, not a per-cell peak.
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    radii = np.clip(0.5 + 0.45 * (values - mean) / max(deviation_scale, 1e-15), 0, 1)
    for index, radius in enumerate(radii):
        start = index * 60 - 30
        ax.add_patch(Wedge((0, 0), 1, start, start + 60, facecolor="#F2F2F2", edgecolor="white", linewidth=0.7))
        ax.add_patch(Wedge((0, 0), radius, start + 1, start + 59, facecolor=color, edgecolor="white", linewidth=0.4))
        angle = np.deg2rad(index * 60)
        ax.text(1.18 * np.cos(angle), 1.18 * np.sin(angle), rf"${index * 60}^\circ$", ha="center", va="center", fontsize=5.5)
    ax.add_patch(Circle((0, 0), 0.5, fill=False, edgecolor="0.25", linewidth=0.8, linestyle="--"))
    ax.set(xlim=(-1.32, 1.32), ylim=(-1.32, 1.32), aspect="equal", xticks=[], yticks=[])
    ax.spines[:].set_visible(False)


def unit_with_labels(rho, *, eps1=1, eps2=1, delta=0, k0=0, k1=0, k2=0):
    # Return the hidden unit with the specified algebraic labels.
    wanted = (eps1, eps2, delta, k0, k1, k2)
    for unit, labels in enumerate(experiment.model.metadata):
        observed = (labels["eps1"], labels["eps2"], labels["delta"], labels["k0"], labels["k1"], labels["k2"])
        if int(labels["irrep_index"]) == rho and observed == wanted:
            return unit
    raise ValueError(f"No unit found for rho={rho}, labels={wanted}.")


def write_runtime(name, seconds, note=""):
    # Append a small, human-readable runtime record for future planning.
    path = figures / "figure_8_run_log.csv"
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(("name", "elapsed_seconds", "model_buffer_gib", "notes"))
        writer.writerow((name, f"{seconds:.3f}", f"{model_buffer_gib:.3f}", note))

# %% [markdown]
# ## Panel A — empirical convergence to all-pairs tuning
#
# The empirical curve averages hidden activity at each position along independent periodic random walks. We use 110 steps per trajectory, discard the first 10, and therefore retain exactly $N=10^4,10^5,5\times10^5$ samples. The last column is the exact all-pairs curve,
#
# $$T_{\mathrm{all}}(x)=\mathbb E_{g,a\sim\mathrm{Unif}(G)}[h(g,a)\mid x(g)=x].$$

# %%
# Use two visually clear modules already featured in Panel C, rather than the
# first and last irreps in the construction's internal ordering.
panel_a_rhos = (panel_c_rhos[0], panel_c_rhos[1], panel_c_rhos[-1])  # rho = 9, 19, and 36
panel_a_units = [unit_with_labels(rho) for rho in panel_a_rhos]
steps, burn_in = 110, 10
sample_counts = (10_000, 100_000, 500_000)
retained_steps = steps - burn_in

empirical_curves = {unit: [] for unit in panel_a_units}
for samples in sample_counts:
    assert samples % retained_steps == 0
    result = load_or_compute_empirical_trajectory_tuning(
        experiment, panel_a_units,
        motion_config=local_random_walk,
        tuning_config=TrajectoryTuningConfig(
            num_trajectories=samples // retained_steps, steps_per_trajectory=steps, burn_in_steps=burn_in,
            seed=101, min_occupancy=1, margin=0, batch_size=8, cache_schema_version=1,
        ),
        cache_directory=figures.parent, recompute=False, use_cache=True,
    )
    for column, unit in enumerate(result.unit_indices):
        empirical_curves[unit].append(result.position_tuning[..., column])

started = time.perf_counter()
exact_a = compute_all_pairs_tuning(experiment, panel_a_units, drive_batch_size=32)
write_runtime("Panel A: exact all-pairs", time.perf_counter() - started, "three neurons")

def save_panel_a(exact_tuning, exact_title, filename):
    figure, axes = plt.subplots(3, 4, figsize=(8.4, 5.7), squeeze=False)
    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.03, top=0.87, wspace=0.12, hspace=0.14)
    titles = [rf"empirical\n$N={samples:,}$" for samples in sample_counts] + [exact_title]
    for row, (rho, unit) in enumerate(zip(panel_a_rhos, panel_a_units, strict=True)):
        curves = empirical_curves[unit] + [np.roll(exact_tuning.position_mean[..., row], (2, 2), axis=(0, 1))]
        for column, curve in enumerate(curves):
            colormap = "magma" if column < len(sample_counts) else "viridis"
            plot_lattice_scalar(display_normalized(curve), ax=axes[row, column], title=titles[column] if row == 0 else None,
                                cmap=colormap, vmin=0, vmax=1, colorbar=False, coordinate_mode="offset", wrap_periodic_edges=True)
        axes[row, 0].annotate(rf"$\rho={rho}$\nunit {unit}", (-0.16, 0.5), xycoords="axes fraction",
                              rotation=90, ha="center", va="center", fontsize=9)
    figure.text(0.01, 0.96, "A", fontsize=24, fontweight="bold", va="top")
    figure.text(0.065, 0.95, "Periodic random-walk tuning and its exact counterpart", fontsize=13, fontweight="bold", va="top")
    path = figures / filename
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved SVG: {path}")


save_panel_a(exact_a, "exact\nall-pairs", "figure_8_panel_a.svg")

started = time.perf_counter()
exact_a_local = compute_local_arrival_tuning(experiment, panel_a_units, drive_batch_size=32)
write_runtime("Panel A: exact local", time.perf_counter() - started, "three neurons; uniform average over 21 local actions")
save_panel_a(exact_a_local, "exact\nlocal", "figure_8_panel_a_exact_local.svg")

# %% [markdown]
# ## Panel B — a $C_6$ orbit of characters of $C_n^2$
#
# Translation characters are $\chi_k(x)=\cos(2\pi k\cdot x/n)$, indexed by $k\in\mathbb Z_n^2$. A $C_6$ rotation orbit collects the six Fourier components that induce one spatial irrep. The three colours identify conjugate pairs $\{k,-k\}$.

# %%
group = DiscreteSE2Group(n, 6)
frequencies = np.arange(-n // 2 + 1, n // 2 + 1)
sheet_frequencies = np.arange(-3, 4)
y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
character = lambda k: np.cos(2 * np.pi * (k[0] * x + k[1] * y) / n)
centered = lambda k: tuple((int(value) + n // 2) % n - n // 2 for value in k)

character_sheet = np.block([[character((kx, ky)) for kx in sheet_frequencies] for ky in sheet_frequencies])
orbit = tuple(centered(k) for k in next(candidate for candidate in group.orbit_dict()[6] if (2, 0) in candidate))
conjugate_pairs = []
for k in orbit:
    pair = tuple(sorted((k, tuple(-value for value in k))))
    if pair not in conjugate_pairs:
        conjugate_pairs.append(pair)
colours = dict(zip(conjugate_pairs, ("#00A6FF", "#FF5A36", "#00C853"), strict=True))
ordered_orbit = tuple(k for pair in conjugate_pairs for k in pair)

figure = plt.figure(figsize=(11.5, 7.0))
bank = figure.add_axes((0.055, 0.16, 0.30, 0.68))
bank.imshow(character_sheet, origin="lower", extent=(sheet_frequencies[0] - 0.5, sheet_frequencies[-1] + 0.5) * 2,
            cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
for boundary in sheet_frequencies[:-1] + 0.5:
    bank.axhline(boundary, color="white", linewidth=0.22, alpha=0.55)
    bank.axvline(boundary, color="white", linewidth=0.22, alpha=0.55)
for k in orbit:
    pair = tuple(sorted((k, tuple(-value for value in k))))
    box = (k[0] - 0.5, k[1] - 0.5)
    bank.add_patch(Rectangle(box, 1, 1, fill=False, edgecolor="white", linewidth=4.5, zorder=2))
    bank.add_patch(Rectangle(box, 1, 1, fill=False, edgecolor=colours[pair], linewidth=2.8, zorder=3))
bank.set(title=r"Central $7\times7$ characters $\chi_k$", xlabel=r"frequency $k_1$", ylabel=r"frequency $k_2$",
         xticks=(sheet_frequencies[0], 0, sheet_frequencies[-1]), yticks=(sheet_frequencies[0], 0, sheet_frequencies[-1]))
for row, k in enumerate(ordered_orbit):
    axis = figure.add_axes((0.475, 0.16 + (5 - row) * 0.115, 0.075, 0.105))
    axis.imshow(character(k), origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    pair = tuple(sorted((k, tuple(-value for value in k))))
    for spine in axis.spines.values():
        spine.set(color=colours[pair], linewidth=3.2)
    axis.set(xticks=[], yticks=[])
    axis.annotate(rf"$k=({k[0]},{k[1]})$", (1.12, 0.5), xycoords="axes fraction", va="center", fontsize=8)
orbit_sum = figure.add_axes((0.68, 0.16, 0.27, 0.68))
plot_lattice_scalar(display_normalized(sum(character(k) for k in orbit)), ax=orbit_sum,
                    title=r"$\mathrm{Re}\sum_{q\in\mathcal{O}_k}\chi_q(x)$", cmap="coolwarm", vmin=0, vmax=1,
                    colorbar=False, coordinate_mode="offset", wrap_periodic_edges=True)
figure.text(0.01, 0.96, "B", fontsize=24, fontweight="bold", va="top")
figure.text(0.045, 0.95, rf"A $C_6$ orbit of $C_{{{n}}}^2$ characters forms a spatial irrep", fontsize=14, fontweight="bold", va="top")
path_b = figures / "figure_8_panel_b.svg"
figure.savefig(path_b, bbox_inches="tight")
plt.close(figure)
print(f"Saved SVG: {path_b}")

# %% [markdown]
# ## Panel C — algebraic-label effects on exact all-pairs tuning
#
# Each column is a frequency module. The reference neuron has
#
# $$ (\epsilon_1,\epsilon_2,\delta;k_0,k_1,k_2)=(+1,+1,0;0,0,0). $$
#
# Each lower row changes exactly one label. The maps are the exact all-pairs spatial marginals, normalized separately only for display. Thus colour contrast does not report activity amplitude. The last two rows summarize all distinct spatial maps in each module: their translations relative to the reference, and their dominant spatial Fourier frequencies.

# %%
label_changes = (
    ("reference", {}),
    (r"flip $\epsilon_1$", {"eps1": -1}),
    (r"flip $\epsilon_2$", {"eps2": -1}),
    (r"$\delta:0\to1$", {"delta": 1}),
    (r"$k_0:0\to1$", {"k0": 1}),
    (r"$k_1:0\to1$", {"k1": 1}),
    (r"$k_2:0\to1$", {"k2": 1}),
)
panel_c_units = {
    rho: [(name, unit_with_labels(rho, **changed)) for name, changed in label_changes]
    for rho in panel_c_rhos
}
requested_units = tuple(dict.fromkeys(unit for rows in panel_c_units.values() for _, unit in rows))
started = time.perf_counter()
exact_c = compute_all_pairs_tuning(experiment, requested_units, drive_batch_size=32)
write_runtime("Panel C: exact all-pairs", time.perf_counter() - started, "four reference neurons and six one-label changes each")
maps_c = {unit: exact_c.position_mean[..., column] for column, unit in enumerate(exact_c.unit_indices)}

def best_translation_to_reference(reference, curve):
    """Return the periodic shift that maximizes centred map correlation."""
    reference = reference - reference.mean()
    curve = curve - curve.mean()
    denominator = max(np.linalg.norm(reference) * np.linalg.norm(curve), 1e-15)
    best_score, best_shift = -np.inf, (0, 0)
    for dx in range(n):
        for dy in range(n):
            score = float(np.sum(reference * np.roll(curve, (dx, dy), axis=(0, 1))) / denominator)
            signed_shift = (dx if dx <= n // 2 else dx - n, dy if dy <= n // 2 else dy - n)
            if score > best_score + 1e-12 or (np.isclose(score, best_score) and sum(value**2 for value in signed_shift) < sum(value**2 for value in best_shift)):
                best_score, best_shift = score, signed_shift
    return best_shift


# k1 and k2 do not change the spatial marginal. One representative for each
# remaining label combination therefore captures every distinct spatial curve.
translation_units = {
    rho: [unit for unit, labels in enumerate(experiment.model.metadata)
          if int(labels["irrep_index"]) == rho and labels["k1"] == 0 and labels["k2"] == 0]
    for rho in panel_c_rhos
}
all_translation_units = tuple(unit for units in translation_units.values() for unit in units)
started = time.perf_counter()
translation_tuning = compute_all_pairs_tuning(experiment, all_translation_units, drive_batch_size=32)
write_runtime("Panel C: translation shifts", time.perf_counter() - started, "all k1=k2=0 representatives in four modules")
translation_maps = {unit: translation_tuning.position_mean[..., column] for column, unit in enumerate(translation_tuning.unit_indices)}

def periodic_density(points, bandwidth=0.8):
    # Smooth a set of periodic lattice shifts without discretising the points.
    coordinates = np.linspace(-n // 2, n // 2, 121)
    y_grid, x_grid = np.meshgrid(coordinates, coordinates, indexing="ij")
    dx = (x_grid[..., None] - points[:, 0] + n / 2) % n - n / 2
    dy = (y_grid[..., None] - points[:, 1] + n / 2) % n - n / 2
    density = np.exp(-(dx**2 + dy**2) / (2 * bandwidth**2)).sum(axis=-1)
    return coordinates, density


def dominant_spatial_frequency(values):
    """Magnitude of the strongest non-constant Fourier mode on the hexagonal lattice."""
    spectrum = np.abs(np.fft.fft2(values - values.mean()))
    spectrum[0, 0] = 0
    index_y, index_x = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    k_x = index_x if index_x <= n // 2 else index_x - n
    k_y = index_y if index_y <= n // 2 else index_y - n
    return np.sqrt(k_x**2 - k_x * k_y + k_y**2)


row_labels = (r"reference\n$(+1,+1,0;0,0,0)$", *[name for name, _ in label_changes[1:]])


def save_panel_c(maps, translation_maps, tuning_label, filename):
    module_frequencies = {
        rho: np.asarray([dominant_spatial_frequency(translation_maps[unit]) for unit in translation_units[rho]])
        for rho in panel_c_rhos
    }
    maximum_frequency = max(frequencies.max() for frequencies in module_frequencies.values())
    frequency_bins = np.arange(0, np.ceil(maximum_frequency / 0.25) * 0.25 + 0.251, 0.25)
    figure, axes = plt.subplots(9, 4, figsize=(9.0, 16.0), squeeze=False,
                                gridspec_kw={"height_ratios": (1, 1, 1, 1, 1, 1, 1, 1.25, 0.72)})
    figure.subplots_adjust(left=0.17, right=0.985, bottom=0.055, top=0.895, wspace=0.055, hspace=0.24)
    for column, rho in enumerate(panel_c_rhos):
        for row, (_, unit) in enumerate(panel_c_units[rho]):
            plot_lattice_scalar(display_normalized(maps[unit]), ax=axes[row, column], cmap="viridis", vmin=0, vmax=1,
                                colorbar=False, coordinate_mode="offset", wrap_periodic_edges=True)
        axes[0, column].set_title(rf"$\rho={rho}$", fontsize=12, fontweight="bold", pad=5)

        reference = maps[panel_c_units[rho][0][1]]
        shifts = np.asarray([best_translation_to_reference(reference, translation_maps[unit]) for unit in translation_units[rho]])
        axis = axes[-2, column]
        coordinates, density = periodic_density(shifts)
        axis.contourf(coordinates, coordinates, density, levels=4, cmap="Reds", alpha=0.35, antialiased=True)
        axis.scatter(shifts[:, 0], shifts[:, 1], s=16, color="#B2182B", alpha=0.45, edgecolors="none")
        axis.axhline(0, color="0.75", linewidth=0.6, zorder=0)
        axis.axvline(0, color="0.75", linewidth=0.6, zorder=0)
        axis.set(xlim=(-n // 2 - 0.5, n // 2 + 0.5), ylim=(-n // 2 - 0.5, n // 2 + 0.5), aspect="equal",
                 xticks=(-n // 2, 0, n // 2), yticks=(-n // 2, 0, n // 2), xlabel=r"$\Delta_x$")
        if column == 0:
            axis.set_ylabel(r"$\Delta_y$")

        axis = axes[-1, column]
        axis.hist(module_frequencies[rho], bins=frequency_bins, color="#6A3D9A", edgecolor="white", linewidth=0.6)
        axis.set(xlim=(0, frequency_bins[-1]), xlabel=r"dominant $|k|$")
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        if column == 0:
            axis.set_ylabel("number\nof maps")

    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(label, (-0.15, 0.5), xycoords="axes fraction", ha="right", va="center", fontsize=8.5)
    axes[-2, 0].annotate("all best\ntranslations", (-0.15, 0.5), xycoords="axes fraction", ha="right", va="center", fontsize=8.5)
    axes[-1, 0].annotate("dominant\nfrequency", (-0.15, 0.5), xycoords="axes fraction", ha="right", va="center", fontsize=8.5)
    figure.text(0.015, 0.975, "C", fontsize=25, fontweight="bold", va="top")
    figure.text(0.075, 0.967, "Algebraic-label effects on spatial tuning", fontsize=14, fontweight="bold", va="top")
    figure.text(0.075, 0.938, f"{tuning_label}; translation shading is a periodic density estimate of raw points; dominant-frequency histograms use all distinct spatial maps.", fontsize=9, va="top")
    path = figures / filename
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved SVG: {path}")


save_panel_c(maps_c, translation_maps, "Exact all-pairs tuning", "figure_8_panel_c.svg")

# Alternative: exact one-step tuning averaged only over the 21 local actions.
started = time.perf_counter()
local_c = compute_local_arrival_tuning(experiment, requested_units, drive_batch_size=32)
local_translation_tuning = compute_local_arrival_tuning(experiment, all_translation_units, drive_batch_size=32)
write_runtime("Panel C: exact local", time.perf_counter() - started, "label effects and translation shifts; uniform average over 21 local actions")
local_maps_c = {unit: local_c.position_mean[..., column] for column, unit in enumerate(local_c.unit_indices)}
local_translation_maps = {unit: local_translation_tuning.position_mean[..., column] for column, unit in enumerate(local_translation_tuning.unit_indices)}
save_panel_c(local_maps_c, local_translation_maps, "Exact one-step local tuning", "figure_8_panel_c_exact_local.svg")

# %% [markdown]
# ## Panel D — representative spatial, orientation, and conjunctive tuning
#
# From an exact all-pairs curve $T(x,\theta)$, define spatial and heading marginals $T_x(x)=\langle T\rangle_\theta$ and $T_\theta(\theta)=\langle T\rangle_x$. Their relative modulation is
#
# $$M(v)=\frac{\max v-\min v}{\max |v|}.$$
#
# A neuron is spatial-only, orientation-only, or conjunctive according to whether $M_x$ and $M_\theta$ exceed $10^{-2}$.

# %%
units_d = tuple(
    unit for unit, labels in enumerate(experiment.model.metadata)
    if int(labels["irrep_index"]) in experiment.model.selected_irrep_indices and labels["k1"] == 0 and labels["k2"] == 0
)
started = time.perf_counter()
exact_d = compute_all_pairs_tuning(experiment, units_d, drive_batch_size=16)
write_runtime("Panel D/E: exact all-pairs", time.perf_counter() - started, "one representative per k1,k2 class")
def modulation(values):
    values = np.asarray(values)
    return float(np.ptp(values) / max(np.max(np.abs(values)), 1e-12))


classes = ("Spatial only", "Conjunctive", "Orientation only")


def classify_tuning(tuning):
    pose_curves = tuning.pose_mean.reshape(6, n, n, len(units_d))
    heading_curves = pose_curves.mean(axis=(1, 2))
    records = []
    for column, unit in enumerate(units_d):
        labels = experiment.model.metadata[unit]
        mx = modulation(tuning.position_mean[..., column])
        mtheta = modulation(heading_curves[..., column])
        if mx > 1e-2 and mtheta > 1e-2:
            category = "Conjunctive"
        elif mx > 1e-2:
            category = "Spatial only"
        elif mtheta > 1e-2:
            category = "Orientation only"
        else:
            category = "Untuned"
        records.append({"unit": unit, "column": column, "rho": int(labels["irrep_index"]), "mx": mx, "mtheta": mtheta,
                        "heading": int(np.argmax(heading_curves[..., column])), "category": category,
                        "multiplicity": int(labels["irrep_dim"]) ** 2})
    return heading_curves, records


def two_examples(records, category, preferred_units=()):
    candidates = [record for record in records if record["category"] == category]
    score = lambda record: min(record["mx"], record["mtheta"]) if category == "Conjunctive" else record["mx"] if category == "Spatial only" else record["mtheta"]
    if category == "Spatial only" and preferred_units:
        preferred = [record for unit in preferred_units for record in records if record["unit"] == unit]
        if len(preferred) == len(preferred_units) and all(record["category"] == "Spatial only" for record in preferred):
            return tuple(preferred)
    if category == "Spatial only":
        preferred = [
            [record for record in candidates if record["rho"] == rho]
            for rho in (10, 36)
        ]
        if all(preferred):
            return tuple(max(module, key=score) for module in preferred)
    first = max(candidates, key=score)
    alternatives = [record for record in candidates if record is not first]
    second = max(alternatives, key=lambda record: (record["rho"] != first["rho"], record["heading"] != first["heading"], score(record)))
    return first, second

def save_panel_d(tuning, tuning_label, filename, preferred_spatial_units=()):
    heading_curves, records = classify_tuning(tuning)
    examples = {category: two_examples(records, category, preferred_spatial_units) for category in classes}
    displayed_columns = [record["column"] for category in classes for record in examples[category]]
    heading_deviation_scale = max(
        np.max(np.abs(heading_curves[..., column] - heading_curves[..., column].mean()))
        for column in displayed_columns
    )
    figure = plt.figure(figsize=(10.5, 9.0))
    grid = figure.add_gridspec(3, 2, left=0.16, right=0.98, bottom=0.08, top=0.88, width_ratios=(1.05, 1), wspace=0.20, hspace=0.28)
    figure.text(0.025, 0.97, "D", fontsize=26, fontweight="bold", va="top")
    figure.text(0.085, 0.965, rf"Representative {tuning_label} spatial and orientation tuning ($n={n}$)", fontsize=16, fontweight="bold", va="top")
    for row, category in enumerate(classes):
        figure.text(0.145, 0.765 - row * 0.273, category, ha="right", va="center", fontsize=11, fontweight="bold")
        spatial = grid[row, 0].subgridspec(1, 2, wspace=0.08)
        heading = grid[row, 1].subgridspec(1, 2, wspace=0.22)
        for column, record in enumerate(examples[category]):
            axis = figure.add_subplot(spatial[0, column])
            plot_lattice_scalar(display_normalized(tuning.position_mean[..., record["column"]]), ax=axis, cmap="viridis", vmin=0, vmax=1,
                                colorbar=False, coordinate_mode="offset", wrap_periodic_edges=True)
            axis.set_title(rf"$\rho={record['rho']}$", fontsize=8)
            axis = figure.add_subplot(heading[0, column])
            plot_orientation_pizza(heading_curves[..., record["column"]], axis, ("#4477AA", "#CC6677")[column], heading_deviation_scale)
            axis.set_title(rf"$M_x={record['mx']:.3g},\ M_\theta={record['mtheta']:.3g}$", fontsize=8)
    path = figures / filename
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved SVG: {path}")
    return records


records_d = save_panel_d(exact_d, "all-pairs", "figure_8_panel_d.svg")

started = time.perf_counter()
exact_d_local = compute_local_arrival_tuning(experiment, units_d, drive_batch_size=16)
write_runtime("Panel D: exact local", time.perf_counter() - started, "one representative per k1,k2 class; uniform average over 21 local actions")
canonical_spatial_units = tuple(unit_with_labels(rho) for rho in (10, 36))
records_d_local = save_panel_d(exact_d_local, "one-step local", "figure_8_panel_d_exact_local.svg", canonical_spatial_units)

# %% [markdown]
# ## Panel E — selective-cell composition of the full network
#
# The exact classification from Panel D is expanded to every hidden-unit copy in the construction. These bar charts count spatial-only, conjunctive, and orientation-only neurons across the whole network, once for all-pairs tuning and once for local one-step tuning.

# %%
colours = {"Spatial only": "#4C78A8", "Conjunctive": "#F58518", "Orientation only": "#54A24B"}


def save_panel_e(records, tuning_label, filename):
    class_counts = np.asarray([
        sum(record["multiplicity"] for record in records if record["category"] == category)
        for category in classes
    ])
    total_hidden = sum(record["multiplicity"] for record in records)
    untuned = total_hidden - class_counts.sum()

    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    figure.subplots_adjust(left=0.16, right=0.96, bottom=0.19, top=0.79)
    bars = axis.bar(classes, class_counts - 1, bottom=1, width=0.62, color=[colours[category] for category in classes])
    axis.bar_label(bars, labels=[f"{count:,}" for count in class_counts], padding=4, fontsize=11, fontweight="bold")
    axis.set(yscale="log", ylabel="number of hidden neurons (log scale)", ylim=(1, class_counts.max() * 1.8))
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=11)
    figure.text(0.025, 0.96, "E", fontsize=26, fontweight="bold", va="top")
    figure.text(0.10, 0.95, rf"Selective-cell composition of the full network ($H={total_hidden:,}$)", fontsize=14, fontweight="bold", va="top")
    figure.text(0.53, 0.045, rf"{tuning_label}; $M_x,M_\theta>10^{{-2}}$ define selectivity; {untuned:,} remaining neurons are untuned.", ha="center", fontsize=8.5)
    path = figures / filename
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved SVG: {path}")


save_panel_e(records_d, "Exact all-pairs tuning", "figure_8_panel_e.svg")
save_panel_e(records_d_local, "Exact local one-step tuning (uniform average over 21 actions)", "figure_8_panel_e_exact_local.svg")

# %% [markdown]
# ## Optional diagnostic — comparing label changes by spatial symmetry
#
# These quantities are useful when interpreting Panel C, but they are not part of the manuscript panel. For centred spatial curves $\widetilde T=T-\bar T$ and $\widetilde T'$, let $r(A,B)$ be their Pearson correlation. We report
#
# $$r_0=r(T,T'),\qquad r_T=\max_\Delta r(T,\tau_\Delta T'),$$
# $$r_{TR}=\max_{q\in C_6,\Delta}r(T,\tau_\Delta R_qT'),\qquad r_I=r(T,-T')=-r_0,$$
# $$r_{\pm TR}=\max_{s\in\{-1,+1\},q\in C_6,\Delta}r(T,s\tau_\Delta R_qT').$$
#
# `r_T` asks whether two maps differ only by a translation; `r_{TR}` also allows lattice rotations; `r_{\pm TR}` additionally permits inversion. The code below computes a compact CSV for the same reference neurons used in Panel C.

# %%
def rotate_lattice(values, turns):
    # Rotate an n-by-n lattice map by `turns` of 60 degrees.
    rotated = np.empty_like(values)
    for x0 in range(n):
        for y0 in range(n):
            x1, y1 = experiment.group.apply_rotation(turns, x0, y0)
            rotated[x1, y1] = values[x0, y0]
    return rotated


def best_translation_correlation(reference, changed, turns=(0,), signs=(1,)):
    # Maximise centred correlation over the specified rotations, signs, and translations.
    reference = reference - reference.mean()
    changed = changed - changed.mean()
    denominator = max(np.linalg.norm(reference) * np.linalg.norm(changed), 1e-15)
    best = -np.inf
    for turn in turns:
        rotated = rotate_lattice(changed, turn)
        correlations = np.fft.ifft2(np.conj(np.fft.fft2(reference)) * np.fft.fft2(rotated)).real / denominator
        for sign in signs:
            best = max(best, float(np.max(sign * correlations)))
    return best


metric_rows = []
for rho, variants in panel_c_units.items():
    reference = maps_c[variants[0][1]]
    reference_centered = reference - reference.mean()
    for name, unit in variants[1:]:
        changed = maps_c[unit]
        changed_centered = changed - changed.mean()
        denominator = max(np.linalg.norm(reference_centered) * np.linalg.norm(changed_centered), 1e-15)
        r0 = float(np.sum(reference_centered * changed_centered) / denominator)
        metric_rows.append((rho, name, r0,
                            best_translation_correlation(reference, changed),
                            best_translation_correlation(reference, changed, turns=range(6)),
                            -r0,
                            best_translation_correlation(reference, changed, turns=range(6), signs=(-1, 1))))
metrics_path = figures / "figure_8_panel_c_label_correlations.csv"
with metrics_path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(("rho", "label_change", "r0", "rT", "rTR", "rI", "r_plusminus_TR"))
    writer.writerows(metric_rows)
print(metrics_path)

# %% [markdown]
# ## Optional diagnostic — empirical label effects
#
# The following cell is intentionally off by default because it requires several minutes on Daisy and a full-width model. If enabled, it recreates the four-way per-reference comparison: empirical local random walks, empirical uniform-$G$ updates, exact local one-step tuning, and exact all-pairs tuning. It uses the same uniformly distributed 21 local actions defined in the setup.

# %%
RUN_EMPIRICAL_LABEL_EFFECTS = False

if RUN_EMPIRICAL_LABEL_EFFECTS:
    retained_samples = 500_000
    if torch.cuda.is_available():
        experiment.model.to("cuda")

    # The full-group condition samples arbitrary right-action increments uniformly.
    # It is deliberately non-naturalistic; its role is to separate finite sampling
    # from the restriction to the local egocentric action set.
    def empirical_full_group_maps(units, tuning_config):
        sums = np.zeros((6, n, n, len(units)), dtype=float)
        visits = np.zeros((6, n, n), dtype=int)
        rng = np.random.default_rng(tuning_config.seed)
        start_pose = experiment.group.encode(*configuration.initial_pose)
        for first in range(0, tuning_config.num_trajectories, tuning_config.batch_size):
            batch = min(tuning_config.batch_size, tuning_config.num_trajectories - first)
            actions = rng.integers(experiment.group.order, size=(batch, tuning_config.steps_per_trajectory), dtype=np.int64)
            poses = np.empty((batch, tuning_config.steps_per_trajectory, 3), dtype=int)
            for row, trajectory in enumerate(actions):
                pose = start_pose
                for step, action in enumerate(trajectory):
                    pose = experiment.group.compose(pose, int(action))
                    poses[row, step] = experiment.group.decode(pose)
            with torch.no_grad():
                activity = experiment.model.selected_hidden_rollout(experiment.x_allo, actions, units).detach().cpu().numpy()
            x0, y0, theta0 = poses[:, tuning_config.burn_in_steps:].reshape(-1, 3).T
            np.add.at(sums, (theta0, x0, y0), activity[:, tuning_config.burn_in_steps:].reshape(-1, len(units)))
            np.add.at(visits, (theta0, x0, y0), 1)
        return {unit: np.divide(sums.sum(axis=0)[..., column], visits.sum(axis=0), out=np.full((n, n), np.nan), where=visits.sum(axis=0) > 0)
                for column, unit in enumerate(units)}

    all_units = tuple(dict.fromkeys(unit for rows in panel_c_units.values() for _, unit in rows))
    tuning_config = TrajectoryTuningConfig(num_trajectories=retained_samples // 100, steps_per_trajectory=110, burn_in_steps=10,
                                           seed=101, min_occupancy=1, margin=0, batch_size=64, cache_schema_version=1)
    local = load_or_compute_empirical_trajectory_tuning(experiment, all_units, motion_config=local_random_walk, tuning_config=tuning_config,
                                                         cache_directory=figures.parent, recompute=False, use_cache=True)
    local_maps = {unit: local.position_tuning[..., column] for column, unit in enumerate(local.unit_indices)}
    uniform_group_maps = empirical_full_group_maps(all_units, tuning_config)
    exact_local = compute_local_arrival_tuning(experiment, all_units, drive_batch_size=32)
    exact_local_maps = {unit: exact_local.position_mean[..., column] for column, unit in enumerate(exact_local.unit_indices)}

    sections = ((r"empirical local ($N=500{,}000$)", local_maps), (r"empirical uniform $G$ ($N=500{,}000$)", uniform_group_maps),
                (r"exact one-step local", exact_local_maps), (r"exact all-pairs", maps_c))
    for rho, variants in panel_c_units.items():
        figure, axes = plt.subplots(4, 7, figsize=(13, 8.5), squeeze=False)
        figure.subplots_adjust(left=0.06, right=0.99, bottom=0.03, top=0.93, wspace=0.08, hspace=0.22)
        for row, (heading, maps) in enumerate(sections):
            for column, (name, unit) in enumerate(variants):
                plot_lattice_scalar(display_normalized(maps[unit]), ax=axes[row, column], cmap="viridis", vmin=0, vmax=1,
                                    colorbar=False, coordinate_mode="offset", wrap_periodic_edges=True)
                if row == 0:
                    axes[row, column].set_title("reference" if column == 0 else name, fontsize=8)
            axes[row, 0].annotate(heading, (-0.08, 0.5), xycoords="axes fraction", ha="right", va="center", fontsize=8)
        figure.suptitle(rf"Algebraic-label effects, $\rho={rho}$", fontweight="bold")
        path = figures / f"figure_8_panel_c_label_effects_four_way_n500000_rho_{rho}.svg"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved SVG: {path}")

# %% [markdown]
# ## Assemble Figure 8
#
# Run this after Panels A--E. It embeds their SVGs in one editable composite without rasterising them.

# %%
svg_namespace = "http://www.w3.org/2000/svg"
ET.register_namespace("", svg_namespace)

def prefix_svg_ids(root, prefix):
    # SVG definitions need unique ids after several panels are embedded together.
    renamed = {}
    for node in root.iter():
        if node.get("id"):
            old = node.get("id")
            renamed[old] = f"{prefix}_{old}"
            node.set("id", renamed[old])
    for node in root.iter():
        for attribute, value in tuple(node.attrib.items()):
            for old, new in renamed.items():
                value = value.replace(f"url(#{old})", f"url(#{new})").replace(f"#{old}", f"#{new}")
            node.set(attribute, value)


layout = (("a", 35, 65, 1.25), ("b", 820, 20, 1.18), ("c", 30, 520, 1.00), ("d", 840, 560, 1.05), ("e", 255, 1280, 1.55))
canvas = ET.Element(f"{{{svg_namespace}}}svg", {"width": "1800pt", "height": "1800pt", "viewBox": "0 0 1800 1800", "version": "1.1"})
for panel, x0, y0, scale in layout:
    source = ET.parse(figures / f"figure_8_panel_{panel}.svg").getroot()
    prefix_svg_ids(source, panel)
    group = ET.SubElement(canvas, f"{{{svg_namespace}}}g", {"id": f"panel_{panel}", "transform": f"translate({x0} {y0}) scale({scale})"})
    for child in source:
        group.append(copy.deepcopy(child))
path = figures / "figure_8.svg"
ET.ElementTree(canvas).write(path, encoding="utf-8", xml_declaration=True)
print(f"Saved SVG: {path}")
