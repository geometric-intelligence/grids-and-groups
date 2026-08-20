# %%
# Percent-format notebook source. Regenerate the paired .ipynb with Jupytext.

# %% [markdown]
# # Module-restricted neural manifolds for the constructed $C_6$ RNN
#
# This notebook is independent of occupancy-normalized trajectory tuning. It
# reconstructs the deterministic network, probes all transformed allocentric
# inputs once, iterates a stratified pose sample to identity-update fixed points,
# and analyzes retained irrep modules.
#
# ## Execution contract
#
# 1. Run **Construction and static probe** after changing the network.
# 2. Run **Fixed-point pose sample** after changing spatial sampling or fixed-point
#    settings.
# 3. Run **Module topology** after changing module count, UMAP, or persistence
#    settings. No trajectory-tuning artifact is read or invalidated.

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

from src.experiments.discrete_se2 import (  # noqa: E402
    DiscreteSE2ExperimentConfig,
    DiscreteSE2ManifoldConfig,
    build_discrete_se2_experiment,
)
from src.neural_manifold import (  # noqa: E402
    analyze_module_orbits,
    build_module_orbits,
    fixed_point_embedding,
    plot_manifold_analysis,
)

# %% [markdown]
# ## 1. Construction and static probe
#
# The static probe evaluates all $g\cdot x_{\mathrm{allo}}$ with an identity
# drive. It is inexpensive relative to trajectory pooling and is reconstructed
# locally so this notebook has no dependency on another notebook's kernel state.
#
# Every experimental and computational choice is listed in the next cell. The
# dataclasses only validate and package these visible values.

# %%
# ----------------------------
# Group and signal encoding
# ----------------------------
n_spatial = 25  # Number of lattice sites along each periodic spatial axis.
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
# Pose sampling and fixed-point iteration
# ----------------------------
spatial_samples_per_axis = 12  # Stratified x and y values sampled across the arena.
fixed_point_tolerance = 1e-8  # Maximum update residual required for convergence.
fixed_point_max_iterations = 50  # Maximum identity-drive recurrence iterations.

# ----------------------------
# Module selection and topology
# ----------------------------
num_modules_to_analyze = 6  # Highest-signal-power modules retained for analysis.
include_conjugate_irreps = True  # Treat conjugate irreps as one real module.
skip_trivial_irrep = True  # Exclude the spatially constant module.
max_persistence_points = 300  # Farthest-point sample size for persistent homology.
max_homology_dimension = 2  # Compute persistence through H2.
manifold_random_seed = 11  # Seed for randomized PCA and UMAP.
umap_components = 3  # Number of UMAP coordinates used for visualization.

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
manifold_config = DiscreteSE2ManifoldConfig(
    num_modules=num_modules_to_analyze,
    spatial_samples=spatial_samples_per_axis,
    fixed_point_tolerance=fixed_point_tolerance,
    fixed_point_max_iterations=fixed_point_max_iterations,
    max_persistence_points=max_persistence_points,
    max_homology_dimension=max_homology_dimension,
    random_seed=manifold_random_seed,
    umap_components=umap_components,
)
experiment = build_discrete_se2_experiment(experiment_config)
G = experiment.group
params = experiment.model
static_hidden = (
    params.probe_hidden_states(experiment.x_allo).detach().cpu().numpy()
)

print(f"|G|: {G.order}")
print(f"hidden width: {params.hidden_dim:,}")
print("manifold configuration:", manifold_config)

# %% [markdown]
# ## 2. Fixed-point pose sample
#
# For the sampled population states $h_0$, iterate the identity-drive recurrence
# until $h_{k+1}\approx h_k$. The residual is reported rather than assuming
# convergence. Every discrete orientation is included at a stratified spatial
# sample.

# %%
sample_coordinates_1d = np.unique(
    np.linspace(
        0,
        G.n - 1,
        manifold_config.spatial_samples,
        dtype=int,
    )
)
pose_coordinates = np.asarray(
    [
        (x, y, rotation)
        for rotation in range(G.m)
        for x in sample_coordinates_1d
        for y in sample_coordinates_1d
    ],
    dtype=int,
)
pose_elements = np.asarray(
    [
        G.encode(int(x), int(y), int(rotation))
        for x, y, rotation in pose_coordinates
    ]
)
pose_initial_hidden = static_hidden[pose_elements]
pose_fixed = fixed_point_embedding(
    params,
    pose_initial_hidden,
    tolerance=manifold_config.fixed_point_tolerance,
    max_iterations=manifold_config.fixed_point_max_iterations,
)
print(
    "identity-update fixed points: "
    f"converged={pose_fixed.converged}, "
    f"iterations={pose_fixed.iterations}, "
    f"max residual={pose_fixed.residuals.max():.3e}"
)

# %% [markdown]
# ## 3. Module topology
#
# For each highest-power retained module, persistent homology is computed after
# PCA in neural space with deterministic farthest-point subsampling. UMAP is
# visualization only. Separate colorings expose $x$, $y$, and discrete heading
# without conflating them into one RGB coordinate.

# %%
power = G.power_spectrum(experiment.x_allo)
all_module_orbits = build_module_orbits(
    params,
    pose_fixed.states,
    include_conjugates=include_conjugate_irreps,
    skip_trivial=skip_trivial_irrep,
)
module_orbits = sorted(
    all_module_orbits,
    key=lambda module: (
        -sum(power[index] for index in module.irrep_indices),
        module.irrep_indices,
    ),
)[: manifold_config.num_modules]
manifold_analyses = analyze_module_orbits(
    module_orbits,
    max_persistence_points=min(
        manifold_config.max_persistence_points,
        len(pose_coordinates),
    ),
    max_homology_dimension=manifold_config.max_homology_dimension,
    random_state=manifold_config.random_seed,
    umap_components=manifold_config.umap_components,
)

x_colors = plt.get_cmap("viridis")(
    pose_coordinates[:, 0] / max(G.n - 1, 1)
)[:, :3]
y_colors = plt.get_cmap("viridis")(
    pose_coordinates[:, 1] / max(G.n - 1, 1)
)[:, :3]
heading_colors = plt.get_cmap("hsv")(
    pose_coordinates[:, 2] / G.m
)[:, :3]
colorings = (
    ("x coordinate", x_colors),
    ("y coordinate", y_colors),
    ("orientation", heading_colors),
)

for analysis in manifold_analyses:
    print(
        f"{analysis.module.label}: "
        f"PCA d={analysis.pca_dimension}, "
        f"variance={analysis.explained_variance:.2%}, "
        f"PH points={analysis.persistence_sample_size}"
    )
    for color_label, colors in colorings:
        plot_manifold_analysis(
            analysis,
            colors,
            title=f"{analysis.module.label} — colored by {color_label}",
        )
        plt.show()

# %% [markdown]
# ## Interpretation
#
# The displayed topology belongs to a module-restricted, finitely sampled
# fixed-point orbit. It is not a claim that the finite group itself is a
# continuous manifold. Induced nonabelian irreps contain a $C_6$ frequency orbit
# or a smaller stabilizer-reduced orbit, and the pose probe additionally exposes
# variation over the finite orientation coordinate.
