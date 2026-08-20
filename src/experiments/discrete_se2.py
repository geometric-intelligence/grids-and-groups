"""Shared construction and rollout helpers for the discrete-SE(2) notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.finite_group_rnn import build_finite_group_rnn, random_invertible_encoding
from src.geometry.discrete_se2 import (
    NaturalisticMotionConfig,
    advanced_pose,
    center_errors_periodic_triangular,
    decode_poses_from_template_orbit,
    gaussian_bump,
    make_naturalistic_motion_sequence,
    transformed_pose,
)
from src.groups import DiscreteSE2Group

_ENCODING_OPTIONS = {
    "one-hot",
    "one-hot space uniform orientation",
    "one-hot space custom orientation",
    "gaussian space one-hot orientation",
    "gaussian space uniform orientation",
    "gaussian space custom orientation",
}
_LOCAL_TRANSLATIONS = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
)
_LOCAL_ROTATIONS = (-1, 0, 1)


@dataclass(frozen=True)
class DiscreteSE2ExperimentConfig:
    """Scientifically meaningful choices for one constructed C6 network."""

    n_spatial: int = 10
    n_orientations: int = 6
    initial_pose: tuple[int, int, int] = (2, 2, 0)
    allocentric_encoding: str = "gaussian space custom orientation"
    sigma: float = 1.0
    custom_orientation_weights: tuple[float, ...] = (
        1.0,
        0.8,
        0.4,
        0.2,
        0.4,
        0.8,
    )
    encoding_seed: int = 10
    action_side: str = "right"
    irrep_selection: str | None = "power"
    num_selected_irreps: int | None = None
    max_hidden_width: int | None = 24_000
    normalize_power_by_dim: bool = True
    always_include_trivial: bool = True
    power_ranking: str = "power"
    q_rho: int = 3
    amplitude_mode: str = "balanced"
    amplitude_multipliers: tuple[float, float, float] = (1.0, 1.0, 1.0)
    materialize_mix: bool = False

    def __post_init__(self) -> None:
        if self.n_orientations != 6:
            raise ValueError("the naturalistic discrete-SE(2) preset requires C6")
        if self.n_spatial < 2:
            raise ValueError("n_spatial must be at least 2")
        if self.allocentric_encoding not in _ENCODING_OPTIONS:
            raise ValueError(
                f"allocentric_encoding must be one of {sorted(_ENCODING_OPTIONS)}"
            )
        if self.action_side not in {"left", "right"}:
            raise ValueError("action_side must be 'left' or 'right'")
        if len(self.custom_orientation_weights) != self.n_orientations:
            raise ValueError(
                "custom_orientation_weights must have one value per orientation"
            )
        weights = np.asarray(self.custom_orientation_weights, dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError(
                "custom_orientation_weights must be finite and nonnegative"
            )
        x, y, heading = self.initial_pose
        if not (
            0 <= x < self.n_spatial
            and 0 <= y < self.n_spatial
            and 0 <= heading < self.n_orientations
        ):
            raise ValueError("initial_pose must lie inside the configured group")


@dataclass(frozen=True)
class DiscreteSE2RolloutConfig:
    """Primary naturalistic rollout and visualization sampling choices."""

    steps: int = 52
    seed: int = 1
    margin: int = 1
    start_xy: tuple[int, int] = (5, 5)
    arrow_stride: int = 2
    snapshot_steps: tuple[int, ...] = (0, 26, 51)

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.margin < 0:
            raise ValueError("margin must be nonnegative")
        if self.arrow_stride < 1:
            raise ValueError("arrow_stride must be positive")
        if any(step < 0 or step >= self.steps for step in self.snapshot_steps):
            raise ValueError("snapshot_steps must index the rollout")


@dataclass(frozen=True)
class DiscreteSE2ManifoldConfig:
    """Sampling and topology choices for module-restricted manifolds."""

    num_modules: int = 6
    spatial_samples: int = 12
    fixed_point_tolerance: float = 1e-8
    fixed_point_max_iterations: int = 50
    max_persistence_points: int = 300
    max_homology_dimension: int = 2
    random_seed: int = 11
    umap_components: int = 3

    def __post_init__(self) -> None:
        if self.num_modules < 1:
            raise ValueError("num_modules must be positive")
        if self.spatial_samples < 2:
            raise ValueError("spatial_samples must be at least 2")
        if self.fixed_point_tolerance <= 0:
            raise ValueError("fixed_point_tolerance must be positive")
        if self.fixed_point_max_iterations < 1:
            raise ValueError("fixed_point_max_iterations must be positive")
        if self.max_persistence_points < 2:
            raise ValueError("max_persistence_points must be at least 2")
        if self.max_homology_dimension < 0:
            raise ValueError("max_homology_dimension must be nonnegative")
        if self.umap_components < 1:
            raise ValueError("umap_components must be positive")


@dataclass
class DiscreteSE2Experiment:
    """Constructed group, signals, model, and local-drive support."""

    config: DiscreteSE2ExperimentConfig
    group: DiscreteSE2Group
    irreps: list
    orientation_weights: np.ndarray
    x_allo: np.ndarray
    x_ego: np.ndarray
    model: object
    local_egocentric_elements: frozenset[int]


@dataclass
class DiscreteSE2Rollout:
    """Decoded outputs and diagnostics for one primary rollout."""

    config: DiscreteSE2RolloutConfig
    sequence: np.ndarray
    cumulative_states: np.ndarray
    true_outputs: np.ndarray
    predicted_outputs: np.ndarray
    hidden_states: np.ndarray
    exact_poses: np.ndarray
    predicted_poses: np.ndarray
    exact_centers: np.ndarray
    predicted_centers: np.ndarray
    relative_output_errors: np.ndarray
    center_errors: np.ndarray
    orientation_errors: np.ndarray
    unique_positions: int
    heading_changes: int
    stationary_steps: int
    immediate_reversals: int


def default_c6_experiment_config() -> DiscreteSE2ExperimentConfig:
    """Return the canonical C6 construction."""
    return DiscreteSE2ExperimentConfig()


def default_c6_motion_config() -> NaturalisticMotionConfig:
    """Return the explicit forward-biased C6 motion policy."""
    return NaturalisticMotionConfig(
        stay_probability=0.05,
        forward_probability=0.70,
        forward_left_or_right_probability=0.115,
        backward_left_or_right_probability=0.005,
        backward_probability=0.01,
        turn_probability=0.12,
        turn_persistence=0.20,
        reversal_weight=0.05,
        wall_lookahead=3,
        wall_avoidance_strength=2.0,
        minimum_wall_weight=0.05,
    )


def default_c6_rollout_config(
    experiment_config: DiscreteSE2ExperimentConfig,
) -> DiscreteSE2RolloutConfig:
    """Return the primary rollout choices matched to an experiment."""
    steps = 52
    center = experiment_config.n_spatial // 2
    snapshots = (0, steps // 2, steps - 1)
    return DiscreteSE2RolloutConfig(
        steps=steps,
        seed=1,
        margin=1,
        start_xy=(center, center),
        arrow_stride=2,
        snapshot_steps=snapshots,
    )


def default_c6_manifold_config() -> DiscreteSE2ManifoldConfig:
    """Return the canonical manifold-analysis choices."""
    return DiscreteSE2ManifoldConfig()


def _orientation_weights(
    config: DiscreteSE2ExperimentConfig,
) -> np.ndarray:
    if "one-hot orientation" in config.allocentric_encoding or (
        config.allocentric_encoding == "one-hot"
    ):
        weights = np.zeros(config.n_orientations)
        weights[config.initial_pose[2]] = 1.0
        return weights
    if "custom orientation" in config.allocentric_encoding:
        weights = np.asarray(config.custom_orientation_weights, dtype=float)
        if np.allclose(weights, weights[0]):
            raise ValueError("custom orientation weights must vary")
        return weights
    return np.ones(config.n_orientations)


def _allocentric_signal(
    group: DiscreteSE2Group,
    config: DiscreteSE2ExperimentConfig,
    orientation_weights: np.ndarray,
) -> np.ndarray:
    center = config.initial_pose[:2]
    if config.allocentric_encoding.startswith("one-hot"):
        signal = np.zeros(group.order)
        for orientation, weight in enumerate(orientation_weights):
            signal[group.encode(*center, orientation)] = weight
        return signal
    return gaussian_bump(
        group,
        center=center,
        sigma=config.sigma,
        orientation_weights=orientation_weights,
    )


def build_discrete_se2_experiment(
    config: DiscreteSE2ExperimentConfig | None = None,
) -> DiscreteSE2Experiment:
    """Construct the deterministic C6 experiment represented by ``config``."""
    config = default_c6_experiment_config() if config is None else config
    group = DiscreteSE2Group(
        n=config.n_spatial,
        m=config.n_orientations,
    )
    orientation_weights = _orientation_weights(config)
    x_allo = _allocentric_signal(group, config, orientation_weights)
    irreps = group.irreps()
    x_ego = random_invertible_encoding(
        group,
        irreps,
        seed=config.encoding_seed,
    )
    model = build_finite_group_rnn(
        group,
        x_ego,
        x_allo=x_allo,
        irrep_selection=config.irrep_selection,
        num_irreps=config.num_selected_irreps,
        max_hidden_width=config.max_hidden_width,
        normalize_power_by_dim=config.normalize_power_by_dim,
        always_include_trivial=config.always_include_trivial,
        power_ranking=config.power_ranking,
        q_rho=config.q_rho,
        amplitude_mode=config.amplitude_mode,
        amplitude_multipliers=config.amplitude_multipliers,
        materialize_mix=config.materialize_mix,
        action_side=config.action_side,
    )
    local_elements = frozenset(
        group.encode(dx, dy, rotation)
        for dx, dy in _LOCAL_TRANSLATIONS
        for rotation in _LOCAL_ROTATIONS
    )
    return DiscreteSE2Experiment(
        config=config,
        group=group,
        irreps=irreps,
        orientation_weights=orientation_weights,
        x_allo=x_allo,
        x_ego=x_ego,
        model=model,
        local_egocentric_elements=local_elements,
    )


def run_discrete_se2_rollout(
    experiment: DiscreteSE2Experiment,
    rollout_config: DiscreteSE2RolloutConfig | None = None,
    motion_config: NaturalisticMotionConfig | None = None,
) -> DiscreteSE2Rollout:
    """Generate, evaluate, and decode one naturalistic rollout."""
    config = experiment.config
    group = experiment.group
    if rollout_config is None:
        rollout_config = default_c6_rollout_config(config)
    if motion_config is None:
        motion_config = default_c6_motion_config()
    sequence = make_naturalistic_motion_sequence(
        group,
        steps=rollout_config.steps,
        seed=rollout_config.seed,
        start_xy=rollout_config.start_xy,
        initial_pose=config.initial_pose,
        margin=rollout_config.margin,
        action_side=config.action_side,
        config=motion_config,
    )
    raw = {
        key: value.detach().cpu().numpy()
        for key, value in experiment.model.rollout(
            experiment.x_allo,
            sequence,
        ).items()
    }
    if config.action_side == "right":
        exact_poses = np.asarray(
            [
                advanced_pose(group, config.initial_pose, int(state))
                for state in raw["cumulative_states"]
            ]
        )
    else:
        exact_poses = np.asarray(
            [
                transformed_pose(group, int(state), config.initial_pose)
                for state in raw["cumulative_states"]
            ]
        )
    exact_centers = exact_poses[:, :2]
    predicted_poses = decode_poses_from_template_orbit(
        group,
        raw["predicted_outputs"],
        experiment.x_allo,
        config.initial_pose,
        action_side=config.action_side,
    )
    predicted_centers = predicted_poses[:, :2]
    absolute_errors = np.linalg.norm(
        raw["predicted_outputs"] - raw["true_outputs"],
        axis=1,
    )
    relative_errors = absolute_errors / np.linalg.norm(
        raw["true_outputs"],
        axis=1,
    )
    center_errors = center_errors_periodic_triangular(
        group,
        predicted_centers,
        exact_centers,
    )
    orientation_steps = np.abs(predicted_poses[:, 2] - exact_poses[:, 2]) % group.m
    orientation_steps = np.minimum(orientation_steps, group.m - orientation_steps)
    orientation_errors = 2 * np.pi * orientation_steps / group.m
    world_steps = np.diff(exact_centers, axis=0)
    nonzero = np.any(world_steps != 0, axis=1)
    reversals = sum(
        np.array_equal(world_steps[index], -world_steps[index - 1])
        and nonzero[index]
        and nonzero[index - 1]
        for index in range(1, len(world_steps))
    )
    return DiscreteSE2Rollout(
        config=rollout_config,
        sequence=sequence,
        cumulative_states=raw["cumulative_states"],
        true_outputs=raw["true_outputs"],
        predicted_outputs=raw["predicted_outputs"],
        hidden_states=raw["hidden_states"],
        exact_poses=exact_poses,
        predicted_poses=predicted_poses,
        exact_centers=exact_centers,
        predicted_centers=predicted_centers,
        relative_output_errors=relative_errors,
        center_errors=center_errors,
        orientation_errors=orientation_errors,
        unique_positions=len({tuple(center) for center in exact_centers}),
        heading_changes=int(np.count_nonzero(np.diff(exact_poses[:, 2]) % group.m)),
        stationary_steps=int(np.count_nonzero(~nonzero)),
        immediate_reversals=int(reversals),
    )
