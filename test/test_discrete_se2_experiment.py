"""Tests for the split constructed discrete-SE(2) experiment workflow."""

from dataclasses import replace

import numpy as np
import pytest

from src.analysis.tuning import (
    TrajectoryTuningConfig,
    compute_all_pairs_tuning,
    compute_local_arrival_tuning,
    compute_trajectory_tuning,
    load_or_compute_trajectory_tuning,
    occupancy_normalized_activity,
    trajectory_tuning_cache_key,
)
from src.experiments.discrete_se2 import (
    DiscreteSE2ExperimentConfig,
    build_discrete_se2_experiment,
    default_c6_experiment_config,
    default_c6_motion_config,
    default_c6_rollout_config,
    run_discrete_se2_rollout,
)


@pytest.fixture(scope="module")
def small_experiment():
    config = replace(
        default_c6_experiment_config(),
        n_spatial=4,
        initial_pose=(1, 1, 0),
        max_hidden_width=3_500,
    )
    return build_discrete_se2_experiment(config)


def test_c6_config_validation_and_deterministic_construction():
    with pytest.raises(ValueError, match="requires C6"):
        DiscreteSE2ExperimentConfig(n_orientations=3)

    config = replace(
        default_c6_experiment_config(),
        n_spatial=4,
        initial_pose=(1, 1, 0),
        max_hidden_width=3_500,
    )
    first = build_discrete_se2_experiment(config)
    second = build_discrete_se2_experiment(config)

    np.testing.assert_allclose(first.x_allo, second.x_allo)
    np.testing.assert_allclose(first.x_ego, second.x_ego)
    np.testing.assert_allclose(
        first.model.W_in.detach().cpu().numpy(),
        second.model.W_in.detach().cpu().numpy(),
    )
    assert first.model.selected_irrep_indices == second.model.selected_irrep_indices


def test_selected_hidden_rollout_matches_full_rollout(small_experiment):
    rollout_config = replace(
        default_c6_rollout_config(small_experiment.config),
        steps=8,
        margin=0,
        arrow_stride=1,
        snapshot_steps=(0, 4, 7),
    )
    motion = default_c6_motion_config()
    first = run_discrete_se2_rollout(
        small_experiment,
        rollout_config,
        motion,
    )
    second = run_discrete_se2_rollout(
        small_experiment,
        replace(rollout_config, seed=2),
        motion,
    )
    sequences = np.stack([first.sequence, second.sequence])
    selected = np.asarray([0, 7, small_experiment.model.hidden_dim - 1])

    batched = (
        small_experiment.model.selected_hidden_rollout(
            small_experiment.x_allo,
            sequences,
            selected,
        )
        .detach()
        .cpu()
        .numpy()
    )
    expected = np.stack(
        [
            small_experiment.model.rollout(
                small_experiment.x_allo,
                sequence,
            )["hidden_states"]
            .detach()
            .cpu()
            .numpy()[:, selected]
            for sequence in sequences
        ]
    )

    np.testing.assert_allclose(batched, expected, atol=1e-12)


def test_occupancy_normalization_masks_sparse_bins():
    sums = np.asarray([[[2.0, 4.0]], [[9.0, 12.0]]])
    occupancy = np.asarray([[2], [1]])

    normalized = occupancy_normalized_activity(
        sums,
        occupancy,
        min_occupancy=2,
    )

    np.testing.assert_allclose(normalized[0, 0], [1.0, 2.0])
    assert np.isnan(normalized[1, 0]).all()


def test_exhaustive_tuning_matches_direct_factorization_average(small_experiment):
    selected = np.asarray([0, 7])[::-1]  # Deliberately has a negative stride.
    all_pairs = compute_all_pairs_tuning(
        small_experiment,
        selected,
        drive_batch_size=7,
    )
    local = compute_local_arrival_tuning(
        small_experiment,
        selected,
        drive_batch_size=5,
    )

    model = small_experiment.model
    target = 5
    input_weights = model.W_in.detach().cpu().numpy()[selected]
    drive_weights = model.W_drive.detach().cpu().numpy()[selected]
    egocentric_template = model.x_ego.detach().cpu().numpy()

    def direct_mean(drive_elements):
        responses = []
        for drive in drive_elements:
            predecessor = model.group.action_permutation(int(drive))[target]
            allocentric_signal = model.group.left_action(
                int(predecessor),
                small_experiment.x_allo,
            )
            drive_signal = model.group.left_action(
                int(drive),
                egocentric_template,
            )
            preactivation = (
                input_weights @ allocentric_signal
                + drive_weights @ drive_signal
            )
            responses.append(np.maximum(preactivation, 0) ** 2)
        return np.mean(responses, axis=0)

    np.testing.assert_allclose(
        all_pairs.pose_mean[target],
        direct_mean(model.group.elements()),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        local.pose_mean[target],
        direct_mean(small_experiment.local_egocentric_elements),
        atol=1e-12,
    )
    assert all_pairs.num_drives == model.group_size
    assert local.num_drives == len(small_experiment.local_egocentric_elements)


def test_batched_tuning_matches_single_trajectory_batches(small_experiment):
    motion = default_c6_motion_config()
    base = TrajectoryTuningConfig(
        num_trajectories=3,
        steps_per_trajectory=9,
        burn_in_steps=1,
        seed=23,
        min_occupancy=1,
        margin=0,
        batch_size=1,
    )
    selected = [0, 1, 7]

    sequential = compute_trajectory_tuning(
        small_experiment,
        selected,
        motion_config=motion,
        tuning_config=base,
    )
    batched = compute_trajectory_tuning(
        small_experiment,
        selected,
        motion_config=motion,
        tuning_config=replace(base, batch_size=3),
    )

    np.testing.assert_array_equal(
        batched.pose_occupancy,
        sequential.pose_occupancy,
    )
    np.testing.assert_allclose(
        batched.pose_activity_sums,
        sequential.pose_activity_sums,
        atol=1e-11,
    )


def test_cache_key_changes_and_round_trip(small_experiment, tmp_path):
    motion = default_c6_motion_config()
    config = TrajectoryTuningConfig(
        num_trajectories=2,
        steps_per_trajectory=8,
        burn_in_steps=1,
        min_occupancy=1,
        batch_size=2,
    )
    selected = [0, 1]
    key = trajectory_tuning_cache_key(
        small_experiment,
        motion,
        config,
        selected,
    )
    changed = trajectory_tuning_cache_key(
        small_experiment,
        motion,
        replace(config, seed=config.seed + 1),
        selected,
    )
    assert key != changed

    computed = load_or_compute_trajectory_tuning(
        small_experiment,
        selected,
        motion_config=motion,
        tuning_config=config,
        cache_directory=tmp_path,
    )
    loaded = load_or_compute_trajectory_tuning(
        small_experiment,
        selected,
        motion_config=motion,
        tuning_config=config,
        cache_directory=tmp_path,
    )

    assert computed.cache_hit is False
    assert loaded.cache_hit is True
    np.testing.assert_allclose(
        loaded.pose_activity_sums,
        computed.pose_activity_sums,
    )
    np.testing.assert_array_equal(
        loaded.pose_occupancy,
        computed.pose_occupancy,
    )


def test_disabling_cache_never_writes_artifact(small_experiment, tmp_path):
    result = load_or_compute_trajectory_tuning(
        small_experiment,
        [0],
        motion_config=default_c6_motion_config(),
        tuning_config=TrajectoryTuningConfig(
            num_trajectories=1,
            steps_per_trajectory=4,
            burn_in_steps=1,
            min_occupancy=1,
        ),
        cache_directory=tmp_path,
        use_cache=False,
    )

    assert result.cache_path is None
    assert not list(tmp_path.iterdir())
