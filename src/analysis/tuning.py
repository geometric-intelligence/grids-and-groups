"""Batched and cached tuning analyses for constructed finite-group RNNs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.experiments.discrete_se2 import DiscreteSE2Experiment
from src.geometry.discrete_se2 import (
    NaturalisticMotionConfig,
    make_naturalistic_motion_sequence,
)

_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrajectoryTuningConfig:
    """Sampling and masking choices for occupancy-normalized tuning."""

    num_trajectories: int = 100
    steps_per_trajectory: int = 160
    burn_in_steps: int = 10
    seed: int = 101
    min_occupancy: int = 5
    margin: int = 0
    batch_size: int = 4
    cache_schema_version: int = _CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.num_trajectories < 1:
            raise ValueError("num_trajectories must be positive")
        if self.steps_per_trajectory < 1:
            raise ValueError("steps_per_trajectory must be positive")
        if not 0 <= self.burn_in_steps < self.steps_per_trajectory:
            raise ValueError(
                "burn_in_steps must be nonnegative and smaller than "
                "steps_per_trajectory"
            )
        if self.min_occupancy < 1:
            raise ValueError("min_occupancy must be positive")
        if self.margin < 0:
            raise ValueError("margin must be nonnegative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.cache_schema_version < 1:
            raise ValueError("cache_schema_version must be positive")


@dataclass
class TrajectoryTuningResult:
    """Empirical tuning estimated from recurrent trajectory occupancy."""

    config: TrajectoryTuningConfig
    unit_indices: np.ndarray
    pose_activity_sums: np.ndarray
    pose_occupancy: np.ndarray
    cache_key: str
    cache_path: Path | None = None
    cache_hit: bool = False

    @property
    def pose_tuning(self) -> np.ndarray:
        return occupancy_normalized_activity(
            self.pose_activity_sums,
            self.pose_occupancy,
            min_occupancy=self.config.min_occupancy,
        )

    @property
    def position_activity_sums(self) -> np.ndarray:
        return self.pose_activity_sums.sum(axis=0)

    @property
    def position_occupancy(self) -> np.ndarray:
        return self.pose_occupancy.sum(axis=0)

    @property
    def position_tuning(self) -> np.ndarray:
        return occupancy_normalized_activity(
            self.position_activity_sums,
            self.position_occupancy,
            min_occupancy=self.config.min_occupancy,
        )

    @property
    def heading_activity_sums(self) -> np.ndarray:
        return self.pose_activity_sums.sum(axis=(1, 2))

    @property
    def heading_occupancy(self) -> np.ndarray:
        return self.pose_occupancy.sum(axis=(1, 2))

    @property
    def heading_tuning(self) -> np.ndarray:
        return occupancy_normalized_activity(
            self.heading_activity_sums,
            self.heading_occupancy,
            min_occupancy=self.config.min_occupancy,
        )

    def local_unit_index(self, global_unit: int) -> int:
        matches = np.flatnonzero(self.unit_indices == int(global_unit))
        if matches.size != 1:
            raise KeyError(f"unit {global_unit} is not present in this tuning result")
        return int(matches[0])


EmpiricalTrajectoryTuningResult = TrajectoryTuningResult


@dataclass
class ExhaustiveArrivalTuningResult:
    """One-step tuning moments over a specified exhaustive drive set."""

    unit_indices: np.ndarray
    pose_mean: np.ndarray
    pose_standard_deviation: np.ndarray
    position_mean: np.ndarray
    position_standard_deviation: np.ndarray
    num_drives: int
    drive_scope: str


LocalArrivalTuningResult = ExhaustiveArrivalTuningResult


def default_trajectory_tuning_config() -> TrajectoryTuningConfig:
    """Return the canonical trajectory-tuning sampler."""
    return TrajectoryTuningConfig()


def occupancy_normalized_activity(
    activity_sums: np.ndarray,
    occupancy: np.ndarray,
    *,
    min_occupancy: int = 1,
) -> np.ndarray:
    """Return conditional mean activity, masking poorly sampled bins."""
    activity_sums = np.asarray(activity_sums, dtype=float)
    occupancy = np.asarray(occupancy)
    if activity_sums.shape[:-1] != occupancy.shape:
        raise ValueError(
            "activity_sums leading dimensions must match occupancy, got "
            f"{activity_sums.shape} and {occupancy.shape}"
        )
    if min_occupancy < 1:
        raise ValueError("min_occupancy must be positive")
    return np.divide(
        activity_sums,
        occupancy[..., None],
        out=np.full_like(activity_sums, np.nan),
        where=occupancy[..., None] >= min_occupancy,
    )


def masked_periodic_spatial_autocorrelation(values: np.ndarray) -> np.ndarray:
    """Compute periodic autocorrelation using pairs of observed bins."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional spatial field")
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.full_like(values, np.nan)
    centered = np.where(valid, values - values[valid].mean(), 0.0)
    zero_lag_energy = float(np.mean(centered[valid] ** 2))
    if np.isclose(zero_lag_energy, 0.0):
        return np.zeros_like(values)

    autocorrelation = np.full_like(values, np.nan)
    for shift_x in range(values.shape[0]):
        for shift_y in range(values.shape[1]):
            shifted_centered = np.roll(
                centered,
                (shift_x, shift_y),
                axis=(0, 1),
            )
            shifted_valid = np.roll(valid, (shift_x, shift_y), axis=(0, 1))
            paired = valid & shifted_valid
            if np.any(paired):
                autocorrelation[shift_x, shift_y] = (
                    np.mean(centered[paired] * shifted_centered[paired])
                    / zero_lag_energy
                )
    return np.clip(np.fft.fftshift(autocorrelation), -1.0, 1.0)


def _canonical_payload(
    experiment: DiscreteSE2Experiment,
    motion_config: NaturalisticMotionConfig,
    tuning_config: TrajectoryTuningConfig,
    unit_indices: np.ndarray,
) -> dict:
    return {
        "experiment": asdict(experiment.config),
        "motion": asdict(motion_config),
        "tuning": asdict(tuning_config),
        "unit_indices": [int(index) for index in unit_indices],
    }


def trajectory_tuning_cache_key(
    experiment: DiscreteSE2Experiment,
    motion_config: NaturalisticMotionConfig,
    tuning_config: TrajectoryTuningConfig,
    unit_indices,
) -> str:
    """Return a stable content key for one trajectory-tuning request."""
    selected = _validated_unit_indices(experiment, unit_indices)
    encoded = json.dumps(
        _canonical_payload(experiment, motion_config, tuning_config, selected),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _validated_unit_indices(
    experiment: DiscreteSE2Experiment,
    unit_indices,
) -> np.ndarray:
    selected = np.asarray(unit_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("unit_indices must be a nonempty one-dimensional array")
    if np.unique(selected).size != selected.size:
        raise ValueError("unit_indices must not contain duplicates")
    if np.any(selected < 0) or np.any(selected >= experiment.model.hidden_dim):
        raise ValueError("unit_indices contain an invalid hidden-unit index")
    return np.ascontiguousarray(selected)


def _absolute_poses(
    experiment: DiscreteSE2Experiment,
    sequence: np.ndarray,
) -> np.ndarray:
    group = experiment.group
    pose = group.encode(*experiment.config.initial_pose)
    poses = []
    for element in sequence:
        if experiment.config.action_side == "right":
            pose = group.compose(pose, int(element))
        else:
            pose = group.compose(int(element), pose)
        poses.append(group.decode(pose))
    return np.asarray(poses, dtype=np.int64)


def compute_empirical_trajectory_tuning(
    experiment: DiscreteSE2Experiment,
    unit_indices,
    *,
    motion_config: NaturalisticMotionConfig,
    tuning_config: TrajectoryTuningConfig,
) -> TrajectoryTuningResult:
    """Estimate tuning by occupancy-normalizing recurrent trajectory activity."""
    selected = _validated_unit_indices(experiment, unit_indices)
    group = experiment.group
    if 2 * tuning_config.margin >= group.n:
        raise ValueError("tuning margin leaves no valid starting position")

    cache_key = trajectory_tuning_cache_key(
        experiment,
        motion_config,
        tuning_config,
        selected,
    )
    activity_sums = np.zeros(
        (group.m, group.n, group.n, len(selected)),
        dtype=np.float64,
    )
    occupancy = np.zeros((group.m, group.n, group.n), dtype=np.int64)
    rng = np.random.default_rng(tuning_config.seed)
    trajectory_seeds = rng.integers(
        0,
        np.iinfo(np.int32).max,
        size=tuning_config.num_trajectories,
    )
    starts = rng.integers(
        tuning_config.margin,
        group.n - tuning_config.margin,
        size=(tuning_config.num_trajectories, 2),
    )

    for first in range(0, tuning_config.num_trajectories, tuning_config.batch_size):
        last = min(first + tuning_config.batch_size, tuning_config.num_trajectories)
        sequences = np.stack(
            [
                make_naturalistic_motion_sequence(
                    group,
                    steps=tuning_config.steps_per_trajectory,
                    seed=int(trajectory_seeds[index]),
                    start_xy=tuple(int(value) for value in starts[index]),
                    initial_pose=experiment.config.initial_pose,
                    margin=tuning_config.margin,
                    action_side=experiment.config.action_side,
                    config=motion_config,
                )
                for index in range(first, last)
            ]
        )
        poses = np.stack(
            [_absolute_poses(experiment, sequence) for sequence in sequences]
        )
        with torch.no_grad():
            selected_hidden = (
                experiment.model.selected_hidden_rollout(
                    experiment.x_allo,
                    sequences,
                    selected,
                )
                .detach()
                .cpu()
                .numpy()
            )

        retained_poses = poses[:, tuning_config.burn_in_steps :].reshape(-1, 3)
        retained_hidden = selected_hidden[
            :, tuning_config.burn_in_steps :
        ].reshape(-1, len(selected))
        x_indices, y_indices, heading_indices = retained_poses.T
        np.add.at(
            activity_sums,
            (heading_indices, x_indices, y_indices),
            retained_hidden,
        )
        np.add.at(
            occupancy,
            (heading_indices, x_indices, y_indices),
            1,
        )

    return TrajectoryTuningResult(
        config=tuning_config,
        unit_indices=selected,
        pose_activity_sums=activity_sums,
        pose_occupancy=occupancy,
        cache_key=cache_key,
    )


def compute_trajectory_tuning(
    experiment: DiscreteSE2Experiment,
    unit_indices,
    *,
    motion_config: NaturalisticMotionConfig,
    tuning_config: TrajectoryTuningConfig,
) -> TrajectoryTuningResult:
    """Backward-compatible alias for empirical trajectory tuning."""
    return compute_empirical_trajectory_tuning(
        experiment,
        unit_indices,
        motion_config=motion_config,
        tuning_config=tuning_config,
    )


def _compute_exhaustive_arrival_tuning(
    experiment: DiscreteSE2Experiment,
    unit_indices,
    drive_elements,
    *,
    drive_scope: str,
    drive_batch_size: int,
) -> ExhaustiveArrivalTuningResult:
    """Average one-step responses over every supplied incoming drive."""
    selected = _validated_unit_indices(experiment, unit_indices)
    if isinstance(drive_batch_size, bool) or drive_batch_size < 1:
        raise ValueError("drive_batch_size must be a positive integer")
    if not isinstance(drive_batch_size, (int, np.integer)):
        raise ValueError("drive_batch_size must be a positive integer")

    model = experiment.model
    group = experiment.group
    drive_elements = np.asarray(
        sorted(int(element) for element in drive_elements),
        dtype=np.int64,
    )
    if drive_elements.ndim != 1 or drive_elements.size == 0:
        raise ValueError("drive_elements must be a nonempty one-dimensional array")
    if np.unique(drive_elements).size != drive_elements.size:
        raise ValueError("drive_elements must not contain duplicates")
    if np.any(drive_elements < 0) or np.any(drive_elements >= model.group_size):
        raise ValueError("drive_elements contain an invalid group element")

    targets = np.asarray(list(model.group.elements()), dtype=np.int64)
    selected_tensor = torch.as_tensor(
        selected,
        dtype=torch.long,
        device=model.W_in.device,
    )
    selected_input_weights = model.W_in.index_select(0, selected_tensor)
    selected_drive_weights = model.W_drive.index_select(0, selected_tensor)

    allocentric_linear_batches = []
    with torch.no_grad():
        for first in range(0, len(targets), drive_batch_size):
            elements = targets[first : first + drive_batch_size]
            allocentric_orbit = np.stack(
                [
                    model.group.left_action(int(element), experiment.x_allo)
                    for element in elements
                ]
            )
            allocentric_linear_batches.append(
                torch.nn.functional.linear(
                    torch.as_tensor(
                        allocentric_orbit,
                        dtype=model.W_in.dtype,
                        device=model.W_in.device,
                    ),
                    selected_input_weights,
                )
            )
        allocentric_linear = torch.cat(allocentric_linear_batches, dim=0)

    response_sum = np.zeros((model.group_size, len(selected)), dtype=np.float64)
    response_square_sum = np.zeros_like(response_sum)
    with torch.no_grad():
        egocentric_template = model.x_ego.detach().cpu().numpy()
        for first in range(0, len(drive_elements), drive_batch_size):
            drive_batch = drive_elements[first : first + drive_batch_size]
            drive_orbit = np.stack(
                [
                    model.group.left_action(int(element), egocentric_template)
                    for element in drive_batch
                ]
            )
            drive_linear = torch.nn.functional.linear(
                torch.as_tensor(
                    drive_orbit,
                    dtype=model.W_drive.dtype,
                    device=model.W_drive.device,
                ),
                selected_drive_weights,
            )
            predecessors = np.stack(
                [
                    model.group.action_permutation(int(element))
                    for element in drive_batch
                ]
            )
            predecessor_tensor = torch.as_tensor(
                predecessors,
                dtype=torch.long,
                device=model.W_in.device,
            )
            incoming = torch.clamp(
                allocentric_linear[predecessor_tensor]
                + drive_linear[:, None, :],
                min=0,
            ).square()
            response_sum += (
                incoming.sum(dim=0).detach().cpu().numpy()
            )
            response_square_sum += (
                incoming.square().sum(dim=0).detach().cpu().numpy()
            )

    pose_mean = response_sum / len(drive_elements)
    pose_second_moment = response_square_sum / len(drive_elements)
    pose_variance = np.maximum(pose_second_moment - pose_mean**2, 0)
    pose_shape = (group.m, group.n, group.n, len(selected))
    position_mean = pose_mean.reshape(pose_shape).mean(axis=0)
    position_second_moment = pose_second_moment.reshape(pose_shape).mean(axis=0)
    position_variance = np.maximum(
        position_second_moment - position_mean**2,
        0,
    )
    return ExhaustiveArrivalTuningResult(
        unit_indices=selected,
        pose_mean=pose_mean,
        pose_standard_deviation=np.sqrt(pose_variance),
        position_mean=position_mean,
        position_standard_deviation=np.sqrt(position_variance),
        num_drives=len(drive_elements),
        drive_scope=drive_scope,
    )


def compute_all_pairs_tuning(
    experiment: DiscreteSE2Experiment,
    unit_indices,
    *,
    drive_batch_size: int = 32,
) -> ExhaustiveArrivalTuningResult:
    """Average ``h(i, j)`` over every factorization ``i * j = g``."""
    return _compute_exhaustive_arrival_tuning(
        experiment,
        unit_indices,
        experiment.model.group.elements(),
        drive_scope="all pairs",
        drive_batch_size=drive_batch_size,
    )


def compute_local_arrival_tuning(
    experiment: DiscreteSE2Experiment,
    unit_indices,
    *,
    drive_batch_size: int = 32,
) -> ExhaustiveArrivalTuningResult:
    """Average over all one-step arrivals using the uniform local drive set."""
    return _compute_exhaustive_arrival_tuning(
        experiment,
        unit_indices,
        experiment.local_egocentric_elements,
        drive_scope="local arrivals",
        drive_batch_size=drive_batch_size,
    )


def _default_cache_directory() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "constructed_networks"
        / "discrete_se2_c6"
    )


def _save_result(path: Path, result: TrajectoryTuningResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache_key": result.cache_key,
        "config": asdict(result.config),
    }
    np.savez_compressed(
        path,
        unit_indices=result.unit_indices,
        pose_activity_sums=result.pose_activity_sums,
        pose_occupancy=result.pose_occupancy,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _load_result(
    path: Path,
    expected_key: str,
    expected_config: TrajectoryTuningConfig,
) -> TrajectoryTuningResult:
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        if metadata["cache_key"] != expected_key:
            raise ValueError("tuning artifact cache key does not match its filename")
        return TrajectoryTuningResult(
            config=expected_config,
            unit_indices=stored["unit_indices"].copy(),
            pose_activity_sums=stored["pose_activity_sums"].copy(),
            pose_occupancy=stored["pose_occupancy"].copy(),
            cache_key=expected_key,
            cache_path=path,
            cache_hit=True,
        )


def load_or_compute_empirical_trajectory_tuning(
    experiment: DiscreteSE2Experiment,
    unit_indices,
    *,
    motion_config: NaturalisticMotionConfig,
    tuning_config: TrajectoryTuningConfig,
    cache_directory: str | Path | None = None,
    recompute: bool = False,
    use_cache: bool = True,
) -> TrajectoryTuningResult:
    """Load a matching compact artifact or compute and optionally persist it."""
    selected = _validated_unit_indices(experiment, unit_indices)
    cache_key = trajectory_tuning_cache_key(
        experiment,
        motion_config,
        tuning_config,
        selected,
    )
    directory = (
        _default_cache_directory()
        if cache_directory is None
        else Path(cache_directory)
    )
    path = directory / f"trajectory_tuning_{cache_key}.npz"
    if use_cache and path.exists() and not recompute:
        return _load_result(path, cache_key, tuning_config)

    result = compute_empirical_trajectory_tuning(
        experiment,
        selected,
        motion_config=motion_config,
        tuning_config=tuning_config,
    )
    result.cache_path = path if use_cache else None
    if use_cache:
        _save_result(path, result)
    return result


def load_or_compute_trajectory_tuning(
    experiment: DiscreteSE2Experiment,
    unit_indices,
    *,
    motion_config: NaturalisticMotionConfig,
    tuning_config: TrajectoryTuningConfig,
    cache_directory: str | Path | None = None,
    recompute: bool = False,
    use_cache: bool = True,
) -> TrajectoryTuningResult:
    """Backward-compatible alias for empirical trajectory tuning."""
    return load_or_compute_empirical_trajectory_tuning(
        experiment,
        unit_indices,
        motion_config=motion_config,
        tuning_config=tuning_config,
        cache_directory=cache_directory,
        recompute=recompute,
        use_cache=use_cache,
    )
