"""Module-restricted neural orbit and persistent-homology analysis."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from umap import UMAP

from src.finite_group_rnn import squared_relu


@dataclass
class FixedPointEmbedding:
    """Fixed-point states and convergence diagnostics."""

    states: np.ndarray
    residuals: np.ndarray
    iterations: int
    converged: bool


@dataclass
class ModuleOrbit:
    """Population orbit restricted to one irrep and its conjugate."""

    irrep_indices: tuple[int, ...]
    unit_indices: np.ndarray
    activity: np.ndarray

    @property
    def label(self) -> str:
        joined = "+".join(str(index) for index in self.irrep_indices)
        return f"irrep {joined} ({len(self.unit_indices)} units)"


@dataclass
class ManifoldAnalysis:
    """Low-dimensional visualization and persistent-homology result."""

    module: ModuleOrbit
    embedding: np.ndarray
    persistence_diagrams: list[np.ndarray]
    pca_dimension: int
    explained_variance: float
    persistence_sample_size: int


def fixed_point_embedding(
    params,
    initial_states: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> FixedPointEmbedding:
    """Iterate identity-drive recurrence until each hidden state is fixed."""
    states = np.asarray(initial_states, dtype=float).copy()
    if states.ndim != 2 or states.shape[1] != params.hidden_dim:
        raise ValueError(
            f"initial_states must have shape (samples, {params.hidden_dim}), "
            f"got {states.shape}"
        )
    group = params.group
    identity_drive = params.W_drive @ group.left_action(
        group.identity(), params.x_ego
    )
    residuals = np.full(len(states), np.inf)
    for iteration in range(1, max_iterations + 1):
        updated = squared_relu(
            params.apply_mix(states.T).T + identity_drive[None, :]
        )
        difference = np.linalg.norm(updated - states, axis=1)
        scale = np.maximum(np.linalg.norm(states, axis=1), 1.0)
        residuals = difference / scale
        states = updated
        if np.max(residuals, initial=0.0) <= tolerance:
            return FixedPointEmbedding(states, residuals, iteration, True)
        if not np.all(np.isfinite(states)):
            break
    return FixedPointEmbedding(states, residuals, iteration, False)


def _selected_characters(params, *, max_samples: int = 256) -> np.ndarray:
    group = params.group
    if group.order <= max_samples:
        elements = np.arange(group.order)
    else:
        elements = np.unique(
            np.linspace(0, group.order - 1, max_samples, dtype=int)
        )
    return np.asarray(
        [
            [np.trace(irrep(int(element))) for element in elements]
            for irrep in params.irreps
        ]
    )


def conjugate_irrep_groups(
    params,
    *,
    include_conjugates: bool = True,
    skip_trivial: bool = True,
    tolerance: float = 1e-7,
) -> list[tuple[int, ...]]:
    """Group selected global irrep indices with available conjugate partners."""
    selected = list(params.selected_irrep_indices)
    if not include_conjugates:
        return [
            (index,)
            for index in selected
            if not (skip_trivial and index == 0)
        ]

    characters = _selected_characters(params)
    dimensions = np.asarray([irrep.dim for irrep in params.irreps])
    partners: dict[int, int] = {}
    for local_index, global_index in enumerate(selected):
        candidates = np.flatnonzero(dimensions == dimensions[local_index])
        errors = np.asarray(
            [
                np.max(
                    np.abs(
                        characters[candidate]
                        - np.conjugate(characters[local_index])
                    )
                )
                for candidate in candidates
            ]
        )
        best_local = int(candidates[np.argmin(errors)])
        if errors.min() <= tolerance:
            partners[global_index] = selected[best_local]
        else:
            partners[global_index] = global_index

    groups = set()
    for index in selected:
        if skip_trivial and index == 0:
            continue
        groups.add(tuple(sorted({index, partners[index]})))
    return sorted(groups)


def build_module_orbits(
    params,
    fixed_states: np.ndarray,
    *,
    include_conjugates: bool = True,
    skip_trivial: bool = True,
) -> list[ModuleOrbit]:
    """Restrict fixed-point population activity to each selected irrep module."""
    fixed_states = np.asarray(fixed_states)
    if fixed_states.ndim != 2 or fixed_states.shape[1] != params.hidden_dim:
        raise ValueError(
            f"fixed_states must have shape (samples, {params.hidden_dim}), "
            f"got {fixed_states.shape}"
        )
    metadata_irreps = np.asarray(
        [item["irrep_index"] for item in params.metadata]
    )
    orbits = []
    for irrep_group in conjugate_irrep_groups(
        params,
        include_conjugates=include_conjugates,
        skip_trivial=skip_trivial,
    ):
        unit_indices = np.flatnonzero(np.isin(metadata_irreps, irrep_group))
        if unit_indices.size:
            orbits.append(
                ModuleOrbit(
                    irrep_indices=irrep_group,
                    unit_indices=unit_indices,
                    activity=fixed_states[:, unit_indices],
                )
            )
    return orbits


def combine_module_orbits(modules: list[ModuleOrbit]) -> ModuleOrbit:
    """Concatenate several module orbits over the same group probe."""
    if not modules:
        raise ValueError("modules must contain at least one orbit")
    num_samples = len(modules[0].activity)
    if any(len(module.activity) != num_samples for module in modules):
        raise ValueError("all module orbits must contain the same samples")
    return ModuleOrbit(
        irrep_indices=tuple(
            sorted(
                {
                    index
                    for module in modules
                    for index in module.irrep_indices
                }
            )
        ),
        unit_indices=np.concatenate(
            [module.unit_indices for module in modules]
        ),
        activity=np.concatenate(
            [module.activity for module in modules],
            axis=1,
        ),
    )


def coordinate_colors(
    coordinates: np.ndarray,
    periods: tuple[int, ...],
) -> np.ndarray:
    """Encode one-, two-, or three-dimensional group coordinates as RGB."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != len(periods):
        raise ValueError(
            "coordinates must have shape (samples, len(periods))"
        )
    normalized = np.column_stack(
        [
            coordinates[:, axis] / max(period - 1, 1)
            for axis, period in enumerate(periods)
        ]
    )
    if normalized.shape[1] == 1:
        return plt.get_cmap("hsv")(normalized[:, 0])[:, :3]
    if normalized.shape[1] == 2:
        return np.column_stack(
            (normalized[:, 0], normalized[:, 1], np.full(len(normalized), 0.35))
        )
    return normalized[:, :3]


def _pca_coordinates(
    activity: np.ndarray,
    *,
    max_components: int = 20,
    random_state: int = 0,
) -> tuple[np.ndarray, int, float]:
    centered = activity - activity.mean(axis=0, keepdims=True)
    active_columns = np.std(centered, axis=0) > 1e-12
    centered = centered[:, active_columns]
    if centered.shape[1] == 0 or len(centered) < 2:
        return np.zeros((len(centered), 1)), 0, 0.0
    scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    if scale > 0:
        centered = centered / scale
    num_components = min(
        max_components,
        centered.shape[0] - 1,
        centered.shape[1],
    )
    pca = PCA(
        n_components=num_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    coordinates = pca.fit_transform(centered)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    retained_dimension = int(np.searchsorted(cumulative, 0.99) + 1)
    retained_dimension = min(retained_dimension, num_components)
    return (
        coordinates[:, :retained_dimension],
        retained_dimension,
        float(cumulative[retained_dimension - 1]),
    )


def farthest_point_subsample(
    points: np.ndarray,
    max_points: int,
) -> np.ndarray:
    """Return deterministic farthest-point indices for topology calculations."""
    points = np.asarray(points)
    if len(points) <= max_points:
        return np.arange(len(points))
    selected = np.empty(max_points, dtype=int)
    selected[0] = 0
    minimum_distances = np.sum((points - points[0]) ** 2, axis=1)
    for index in range(1, max_points):
        selected[index] = int(np.argmax(minimum_distances))
        distances = np.sum((points - points[selected[index]]) ** 2, axis=1)
        minimum_distances = np.minimum(minimum_distances, distances)
    return selected


def analyze_module_orbit(
    module: ModuleOrbit,
    *,
    max_persistence_points: int = 300,
    max_homology_dimension: int = 2,
    random_state: int = 0,
) -> ManifoldAnalysis:
    """Compute PCA-preprocessed UMAP and Vietoris–Rips persistence."""
    coordinates, pca_dimension, explained_variance = _pca_coordinates(
        module.activity,
        random_state=random_state,
    )
    if len(coordinates) < 3 or pca_dimension == 0:
        embedding = np.zeros((len(coordinates), 2))
    else:
        embedding = UMAP(
            n_components=2,
            n_neighbors=min(30, len(coordinates) - 1),
            min_dist=0.1,
            metric="euclidean",
            init="random",
            n_jobs=1,
            random_state=random_state,
        ).fit_transform(coordinates)
    topology_coordinates = np.unique(
        np.round(coordinates, decimals=12),
        axis=0,
    )
    topology_indices = farthest_point_subsample(
        topology_coordinates, max_persistence_points
    )
    distance_matrix = pairwise_distances(
        topology_coordinates[topology_indices],
        metric="euclidean",
    )
    diagrams = ripser(
        distance_matrix,
        distance_matrix=True,
        maxdim=max_homology_dimension,
    )["dgms"]
    return ManifoldAnalysis(
        module=module,
        embedding=embedding,
        persistence_diagrams=diagrams,
        pca_dimension=pca_dimension,
        explained_variance=explained_variance,
        persistence_sample_size=len(topology_indices),
    )


def analyze_module_orbits(
    modules: list[ModuleOrbit],
    **kwargs,
) -> list[ManifoldAnalysis]:
    """Analyze every supplied module orbit."""
    return [analyze_module_orbit(module, **kwargs) for module in modules]


def _plot_persistence_diagrams(ax, diagrams: list[np.ndarray]) -> None:
    finite_arrays = [
        diagram[np.isfinite(diagram[:, 1]), 1]
        for diagram in diagrams
        if len(diagram)
    ]
    finite_deaths = (
        np.concatenate(finite_arrays) if finite_arrays else np.asarray([])
    )
    cap = float(finite_deaths.max()) if finite_deaths.size else 1.0
    cap = max(cap, 1e-8)
    colors = ("0.55", "#4C78A8", "#F58518", "#54A24B")
    for dimension, diagram in enumerate(diagrams):
        if not len(diagram):
            continue
        deaths = np.where(np.isfinite(diagram[:, 1]), diagram[:, 1], cap * 1.05)
        ax.scatter(
            diagram[:, 0],
            deaths,
            s=24,
            alpha=0.8,
            color=colors[dimension % len(colors)],
            label=rf"$H_{dimension}$",
        )
    upper = cap * 1.1
    ax.plot([0, upper], [0, upper], "k--", linewidth=0.8, alpha=0.5)
    ax.set(
        xlim=(-0.02 * upper, upper),
        ylim=(-0.02 * upper, upper),
        xlabel="birth",
        ylabel="death",
        title="Vietoris–Rips persistence",
    )
    ax.legend(frameon=False, fontsize=8)


def plot_manifold_analysis(
    analysis: ManifoldAnalysis,
    colors: np.ndarray,
    *,
    title: str | None = None,
):
    """Plot one module's UMAP embedding and persistence diagram."""
    colors = np.asarray(colors)
    if colors.shape != (len(analysis.embedding), 3):
        raise ValueError(
            f"colors must have shape ({len(analysis.embedding)}, 3), "
            f"got {colors.shape}"
        )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].scatter(
        analysis.embedding[:, 0],
        analysis.embedding[:, 1],
        c=colors,
        s=14,
        alpha=0.85,
        linewidths=0,
    )
    axes[0].set(
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        title="Module-restricted population orbit",
    )
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    _plot_persistence_diagrams(axes[1], analysis.persistence_diagrams)
    heading = analysis.module.label if title is None else title
    figure.suptitle(
        f"{heading}; PCA d={analysis.pca_dimension}, "
        f"variance={analysis.explained_variance:.1%}, "
        f"PH n={analysis.persistence_sample_size}"
    )
    return axes
