"""Module-restricted neural orbit and persistent-homology analysis."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from umap import UMAP

from src.finite_group_rnn import FiniteGroupRNN, squared_relu


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
    pca_embedding: np.ndarray
    embedding: np.ndarray
    persistence_diagrams: list[np.ndarray]
    pca_dimension: int
    explained_variance: float
    persistence_sample_size: int


def fixed_point_embedding(
    model: FiniteGroupRNN,
    initial_states: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> FixedPointEmbedding:
    """Iterate identity-drive recurrence until each hidden state is fixed."""
    states = torch.as_tensor(
        initial_states,
        dtype=model.W_in.dtype,
        device=model.W_in.device,
    )
    if states.ndim != 2 or states.shape[1] != model.hidden_dim:
        raise ValueError(
            f"initial_states must have shape (samples, {model.hidden_dim}), "
            f"got {tuple(states.shape)}"
        )
    group = model.group
    identity_signal = group.left_action(
        group.identity(), model.x_ego.detach().cpu().numpy()
    )
    identity_drive = torch.nn.functional.linear(
        torch.as_tensor(
            identity_signal,
            dtype=model.W_in.dtype,
            device=model.W_in.device,
        ),
        model.W_drive,
    )
    residuals = torch.full(
        (len(states),),
        torch.inf,
        dtype=states.dtype,
        device=states.device,
    )
    for iteration in range(1, max_iterations + 1):
        updated = squared_relu(
            model.apply_mix(states) + identity_drive.unsqueeze(0)
        )
        difference = torch.linalg.vector_norm(updated - states, dim=1)
        scale = torch.clamp(torch.linalg.vector_norm(states, dim=1), min=1.0)
        residuals = difference / scale
        states = updated
        if torch.max(residuals).item() <= tolerance:
            return FixedPointEmbedding(
                states.detach().cpu().numpy(),
                residuals.detach().cpu().numpy(),
                iteration,
                True,
            )
        if not torch.all(torch.isfinite(states)):
            break
    return FixedPointEmbedding(
        states.detach().cpu().numpy(),
        residuals.detach().cpu().numpy(),
        iteration,
        False,
    )


def _selected_characters(params, *, max_samples: int = 256) -> np.ndarray:
    group = params.group
    if group.order <= max_samples:
        elements = np.arange(group.order)
    else:
        elements = np.unique(np.linspace(0, group.order - 1, max_samples, dtype=int))
    return np.asarray(
        [[np.trace(irrep(int(element))) for element in elements] for irrep in params.irreps]
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
        return [(index,) for index in selected if not (skip_trivial and index == 0)]

    characters = _selected_characters(params)
    dimensions = np.asarray([irrep.dim for irrep in params.irreps])
    partners: dict[int, int] = {}
    for local_index, global_index in enumerate(selected):
        candidates = np.flatnonzero(dimensions == dimensions[local_index])
        errors = np.asarray(
            [
                np.max(np.abs(characters[candidate] - np.conjugate(characters[local_index])))
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
            f"fixed_states must have shape (samples, {params.hidden_dim}), got {fixed_states.shape}"
        )
    metadata_irreps = np.asarray([item["irrep_index"] for item in params.metadata])
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
            sorted({index for module in modules for index in module.irrep_indices})
        ),
        unit_indices=np.concatenate([module.unit_indices for module in modules]),
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
        raise ValueError("coordinates must have shape (samples, len(periods))")
    normalized = np.column_stack(
        [coordinates[:, axis] / max(period - 1, 1) for axis, period in enumerate(periods)]
    )
    if normalized.shape[1] == 1:
        return plt.get_cmap("hsv")(normalized[:, 0])[:, :3]
    if normalized.shape[1] == 2:
        return np.column_stack((normalized[:, 0], normalized[:, 1], np.full(len(normalized), 0.35)))
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
    umap_components: int = 2,
) -> ManifoldAnalysis:
    """Compute PCA-preprocessed UMAP and Vietoris–Rips persistence."""
    if umap_components not in (2, 3):
        raise ValueError("umap_components must be 2 or 3")
    coordinates, pca_dimension, explained_variance = _pca_coordinates(
        module.activity,
        random_state=random_state,
    )
    pca_embedding = np.zeros((len(coordinates), 3))
    retained_for_plot = min(coordinates.shape[1], 3)
    pca_embedding[:, :retained_for_plot] = coordinates[:, :retained_for_plot]
    if len(coordinates) < 3 or pca_dimension == 0:
        embedding = np.zeros((len(coordinates), umap_components))
    else:
        embedding = UMAP(
            n_components=umap_components,
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
    topology_indices = farthest_point_subsample(topology_coordinates, max_persistence_points)
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
        pca_embedding=pca_embedding,
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
    finite_arrays = [diagram[np.isfinite(diagram[:, 1]), 1] for diagram in diagrams if len(diagram)]
    finite_deaths = np.concatenate(finite_arrays) if finite_arrays else np.asarray([])
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
    """Plot one module's 3D PCA, UMAP, and persistence diagram."""
    colors = np.asarray(colors)
    if colors.shape != (len(analysis.embedding), 3):
        raise ValueError(
            f"colors must have shape ({len(analysis.embedding)}, 3), got {colors.shape}"
        )
    if analysis.embedding.shape[1] not in (2, 3):
        raise ValueError("UMAP embedding must have two or three components")
    figure = plt.figure(figsize=(14.5, 4.2), constrained_layout=True)
    pca_ax = figure.add_subplot(1, 3, 1, projection="3d")
    umap_ax = figure.add_subplot(
        1,
        3,
        2,
        projection="3d" if analysis.embedding.shape[1] == 3 else None,
    )
    persistence_ax = figure.add_subplot(1, 3, 3)
    axes = np.asarray((pca_ax, umap_ax, persistence_ax), dtype=object)
    pca_ax.scatter(
        analysis.pca_embedding[:, 0],
        analysis.pca_embedding[:, 1],
        analysis.pca_embedding[:, 2],
        c=colors,
        s=14,
        alpha=0.85,
        linewidths=0,
        depthshade=False,
    )
    pca_ax.set(
        xlabel="PC 1",
        ylabel="PC 2",
        zlabel="PC 3",
        title="3D PCA in neural space",
    )
    pca_ax.set_box_aspect((1, 1, 1))
    if analysis.embedding.shape[1] == 3:
        umap_ax.scatter(
            analysis.embedding[:, 0],
            analysis.embedding[:, 1],
            analysis.embedding[:, 2],
            c=colors,
            s=14,
            alpha=0.85,
            linewidths=0,
            depthshade=False,
        )
        umap_ax.set(
            xlabel="UMAP 1",
            ylabel="UMAP 2",
            zlabel="UMAP 3",
            title="3D UMAP",
        )
        umap_ax.set_box_aspect((1, 1, 1))
    else:
        umap_ax.scatter(
            analysis.embedding[:, 0],
            analysis.embedding[:, 1],
            c=colors,
            s=14,
            alpha=0.85,
            linewidths=0,
        )
        umap_ax.set(
            xlabel="UMAP 1",
            ylabel="UMAP 2",
            title="UMAP",
        )
        umap_ax.spines["top"].set_visible(False)
        umap_ax.spines["right"].set_visible(False)
    _plot_persistence_diagrams(persistence_ax, analysis.persistence_diagrams)
    heading = analysis.module.label if title is None else title
    figure.suptitle(
        f"{heading}; PCA d={analysis.pca_dimension}, "
        f"variance={analysis.explained_variance:.1%}, "
        f"PH n={analysis.persistence_sample_size}"
    )
    return axes
