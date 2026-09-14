"""Empirical and exact one-step tuning curves for constructed discrete-SE(2) RNNs.

Empirical tuning follows sampled trajectories through the recurrent network and
estimates conditional mean activity in pose bins. Exact tuning evaluates the
first recurrent update, averaging over a specified set of egocentric drives.
The latter therefore does not simulate a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.experiments.discrete_se2 import DiscreteSE2Experiment
from src.finite_group_rnn import squared_relu
from src.geometry.discrete_se2.trajectories import (
    NaturalisticMotionConfig,
    make_naturalistic_motion_sequence,
)


@dataclass
class TrajectoryTuningResult:
    """Raw empirical activity sums and pose occupancies.

    pose_activity_sums has shape (heading, x, y, unit) and pose_occupancy has
    shape (heading, x, y). Form pose, position, or heading tuning explicitly
    with occupancy_normalized_activity after summing the desired axes.
    """

    unit_indices: np.ndarray
    pose_activity_sums: np.ndarray
    pose_occupancy: np.ndarray

@dataclass
class OneStepTuningResult:
    """Exact first-step conditional means for selected hidden units.

    pose_mean has shape (heading * x * y, unit) in the group's native element
    order. position_mean reshapes it to (heading, x, y, unit) and averages
    over heading, yielding (x, y, unit). Each mean is over num_drives
    egocentric drives per target pose.
    """

    unit_indices: np.ndarray
    pose_mean: np.ndarray
    position_mean: np.ndarray
    num_drives: int


def occupancy_normalized_activity(
    activity_sums: np.ndarray,
    occupancy: np.ndarray,
    *,
    min_occupancy: int = 1,
) -> np.ndarray:
    """Return binwise conditional means, with under-sampled bins set to NaN.

    activity_sums must have shape occupancy.shape + (unit,). The division is
    performed independently for each unit.
    """
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


def compute_empirical_trajectory_tuning(
    experiment: DiscreteSE2Experiment,
    unit_indices,
    *,
    motion_config: NaturalisticMotionConfig,
    num_trajectories: int,
    steps_per_trajectory: int,
    burn_in_steps: int,
    seed: int,
    margin: int,
    batch_size: int,
) -> TrajectoryTuningResult:
    """Estimate activity conditioned on pose along independently sampled trajectories.

    Each retained recurrent update contributes its selected-unit activity to the
    pose reached after that update. The result contains the raw binwise sums
    and occupancies; normalize them later with occupancy_normalized_activity.
    """
    selected = np.asarray(unit_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("unit_indices must be a nonempty one-dimensional array")
    if np.unique(selected).size != selected.size:
        raise ValueError("unit_indices must not contain duplicates")
    if np.any(selected < 0) or np.any(selected >= experiment.model.hidden_dim):
        raise ValueError("unit_indices contain an invalid hidden-unit index")
    selected = np.ascontiguousarray(selected)
    group = experiment.group
    if num_trajectories < 1:
        raise ValueError("num_trajectories must be positive")
    if steps_per_trajectory < 1:
        raise ValueError("steps_per_trajectory must be positive")
    if not 0 <= burn_in_steps < steps_per_trajectory:
        raise ValueError(
            "burn_in_steps must be nonnegative and smaller than "
            "steps_per_trajectory"
        )
    if margin < 0 or 2 * margin >= group.n:
        raise ValueError("tuning margin leaves no valid starting position")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    activity_sums = np.zeros(
        (group.m, group.n, group.n, len(selected)), dtype=np.float64
    )
    occupancy = np.zeros((group.m, group.n, group.n), dtype=np.int64)
    rng = np.random.default_rng(seed)
    trajectory_seeds = rng.integers(
        0, np.iinfo(np.int32).max, size=num_trajectories
    )
    starts = rng.integers(
        margin,
        group.n - margin,
        size=(num_trajectories, 2),
    )

    for first in range(0, num_trajectories, batch_size):
        last = min(first + batch_size, num_trajectories)
        action_sequences = np.stack(
            [
                make_naturalistic_motion_sequence(
                    group,
                    steps=steps_per_trajectory,
                    seed=int(trajectory_seeds[index]),
                    start_xy=tuple(int(value) for value in starts[index]),
                    initial_pose=experiment.config.initial_pose,
                    margin=margin,
                    action_side=experiment.config.action_side,
                    config=motion_config,
                )
                for index in range(first, last)
            ]
        )
        poses = []
        for actions in action_sequences:
            pose = group.encode(*experiment.config.initial_pose)
            trajectory_poses = []
            for action in actions:
                if experiment.config.action_side == "right":
                    pose = group.compose(pose, int(action))
                else:
                    pose = group.compose(int(action), pose)
                trajectory_poses.append(group.decode(pose))
            poses.append(trajectory_poses)
        poses = np.asarray(poses, dtype=np.int64)
        with torch.no_grad():
            activity = (
                experiment.model.selected_hidden_rollout(
                    experiment.x_allo, action_sequences, selected
                )
                .detach()
                .cpu()
                .numpy()
            )

        retained_poses = poses[:, burn_in_steps:].reshape(-1, 3)
        retained_activity = activity[:, burn_in_steps:].reshape(
            -1, len(selected)
        )
        x_indices, y_indices, heading_indices = retained_poses.T
        np.add.at(
            activity_sums,
            (heading_indices, x_indices, y_indices),
            retained_activity,
        )
        np.add.at(occupancy, (heading_indices, x_indices, y_indices), 1)

    return TrajectoryTuningResult(
        unit_indices=selected,
        pose_activity_sums=activity_sums,
        pose_occupancy=occupancy,
    )


def compute_one_step_tuning(
    experiment: DiscreteSE2Experiment,
    unit_indices,
    drive_elements,
    *,
    drive_batch_size: int = 32,
) -> OneStepTuningResult:
    """Compute selected units' first-step mean for every target and drive.

    For target g and egocentric drive j, the predecessor is i = j^(-1) g
    (equivalently, i j = g in the action-group convention). This computes
    squared_relu(W_in x_allo[i] + W_drive x_ego[j]) only for selected hidden
    units, avoiding the full-width recurrent state.
    """
    selected = np.asarray(unit_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("unit_indices must be a nonempty one-dimensional array")
    if np.unique(selected).size != selected.size:
        raise ValueError("unit_indices must not contain duplicates")
    if np.any(selected < 0) or np.any(selected >= experiment.model.hidden_dim):
        raise ValueError("unit_indices contain an invalid hidden-unit index")
    selected = np.ascontiguousarray(selected)
    if (
        isinstance(drive_batch_size, bool)
        or not isinstance(drive_batch_size, (int, np.integer))
        or drive_batch_size < 1
    ):
        raise ValueError("drive_batch_size must be a positive integer")

    model = experiment.model
    group = experiment.group
    drive_elements = np.asarray(
        sorted(int(element) for element in drive_elements), dtype=np.int64
    )
    if drive_elements.ndim != 1 or drive_elements.size == 0:
        raise ValueError("drive_elements must be a nonempty one-dimensional array")
    if np.unique(drive_elements).size != drive_elements.size:
        raise ValueError("drive_elements must not contain duplicates")
    if np.any(drive_elements < 0) or np.any(drive_elements >= model.group_size):
        raise ValueError("drive_elements contain an invalid group element")

    selected_tensor = torch.as_tensor(
        selected, dtype=torch.long, device=model.W_in.device
    )
    input_weights = model.W_in.index_select(0, selected_tensor)
    drive_weights = model.W_drive.index_select(0, selected_tensor)
    targets = np.asarray(list(group.elements()), dtype=np.int64)

    # allocentric_linear[g, unit] = W_in[unit] x_allo[g].
    allocentric_linear = []
    with torch.no_grad():
        for first in range(0, len(targets), drive_batch_size):
            target_batch = targets[first : first + drive_batch_size]
            allocentric_signals = np.stack(
                [
                    group.left_action(int(target), experiment.x_allo)
                    for target in target_batch
                ]
            )
            allocentric_linear.append(
                torch.nn.functional.linear(
                    torch.as_tensor(
                        allocentric_signals,
                        dtype=model.W_in.dtype,
                        device=model.W_in.device,
                    ),
                    input_weights,
                )
            )
        allocentric_linear = torch.cat(allocentric_linear)

    response_sum = np.zeros((model.group_size, len(selected)), dtype=np.float64)
    egocentric_template = model.x_ego.detach().cpu().numpy()
    with torch.no_grad():
        for first in range(0, len(drive_elements), drive_batch_size):
            drive_batch = drive_elements[first : first + drive_batch_size]
            drive_signals = np.stack(
                [
                    group.left_action(int(drive), egocentric_template)
                    for drive in drive_batch
                ]
            )
            drive_linear = torch.nn.functional.linear(
                torch.as_tensor(
                    drive_signals,
                    dtype=model.W_drive.dtype,
                    device=model.W_drive.device,
                ),
                drive_weights,
            )
            # predecessor_indices[drive, target] gives i = j^(-1) g.
            predecessor_indices = torch.as_tensor(
                np.stack(
                    [group.action_permutation(int(drive)) for drive in drive_batch]
                ),
                dtype=torch.long,
                device=model.W_in.device,
            )
            response = squared_relu(
                allocentric_linear[predecessor_indices]
                + drive_linear[:, None, :]
            )
            response_sum += response.sum(dim=0).detach().cpu().numpy()

    pose_mean = response_sum / len(drive_elements)
    pose_shape = (group.m, group.n, group.n, len(selected))
    position_mean = pose_mean.reshape(pose_shape).mean(axis=0)
    return OneStepTuningResult(
        unit_indices=selected,
        pose_mean=pose_mean,
        position_mean=position_mean,
        num_drives=len(drive_elements),
    )
