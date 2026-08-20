"""Tests for the closed-form finite-group QuadraticRNN construction."""

import numpy as np
import pytest
import torch
from torch import nn

from src.finite_group_rnn import (
    FiniteGroupRNN,
    build_finite_group_rnn,
    hidden_width,
    random_invertible_encoding,
    select_irreps_by_power,
)
from src.geometry.cnxcn import (
    center_errors as cnxcn_center_errors,
)
from src.geometry.cnxcn import (
    decode_spatial_argmax as decode_cnxcn_argmax,
)
from src.geometry.cnxcn import (
    gaussian_bump as cnxcn_gaussian_bump,
)
from src.geometry.cnxcn import (
    make_momentum_motion_sequence as make_cnxcn_motion_sequence,
)
from src.geometry.cnxcn import (
    transformed_center as transformed_cnxcn_center,
)
from src.geometry.discrete_se2 import (
    advanced_pose as advance_se2_pose,
)
from src.geometry.discrete_se2 import (
    align_rotation_slice,
    center_errors_periodic_triangular,
    decode_spatial_argmax,
    gaussian_bump,
    lattice_path_coordinates,
    make_momentum_motion_sequence,
    offset_coordinates,
    periodic_distance_squared,
    signal_to_tensor,
    transformed_center,
)
from src.geometry.discrete_se2 import (
    decode_pose as decode_se2_pose,
)
from src.geometry.discrete_se2 import (
    transformed_pose as transformed_se2_pose,
)
from src.geometry.discrete_se3 import (
    align_rotation_slice as align_rotation_volume,
)
from src.geometry.discrete_se3 import (
    decode_pose,
    gaussian_landmark,
    orientation_energy,
    peaked_orientation_weights,
    rotation_error,
    spatial_energy,
    transformed_pose,
)
from src.geometry.discrete_se3 import (
    periodic_distance_squared as periodic_distance_squared_3d,
)
from src.groups.cnxcn import ProductCyclicGroup
from src.groups.znxzn_cm import DiscreteSE2Group
from src.groups.znxznxzn_oh import DiscreteSE3Group


@pytest.fixture
def group():
    return DiscreteSE2Group(n=2, m=3)


def test_complete_irrep_construction_reproduces_group_action(group):
    x_ego = random_invertible_encoding(group, group.irreps(), seed=0)
    x_allo = np.random.default_rng(1).standard_normal(group.order)
    model = build_finite_group_rnn(group, x_ego, materialize_mix=False)
    sequence = [
        group.encode(1, 0, 0),
        group.encode(0, 1, 1),
        group.encode(-1, 0, -1),
    ]

    result = model.rollout(x_allo, sequence)

    np.testing.assert_allclose(
        result["predicted_outputs"],
        result["true_outputs"],
        atol=1e-12,
    )


def test_complete_cnxcn_construction_reproduces_group_action():
    group = ProductCyclicGroup(3, 3)
    x_ego = random_invertible_encoding(group, group.irreps(), seed=20)
    x_allo = np.random.default_rng(21).standard_normal(group.order)
    model = build_finite_group_rnn(group, x_ego, materialize_mix=False)
    sequence = [
        group.encode(1, 0),
        group.encode(0, -1),
        group.encode(1, 1),
    ]

    result = model.rollout(x_allo, sequence)

    np.testing.assert_allclose(
        result["predicted_outputs"],
        result["true_outputs"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        group.left_action(sequence[0], x_allo),
        group.regular_rep()[sequence[0]] @ x_allo,
    )


def test_anisotropic_amplitudes_preserve_cnxcn_group_action():
    group = ProductCyclicGroup(3, 3)
    x_ego = random_invertible_encoding(group, group.irreps(), seed=22)
    x_allo = np.random.default_rng(23).standard_normal(group.order)
    model = build_finite_group_rnn(
        group,
        x_ego,
        amplitude_multipliers=(2.0, 0.5, 1.0),
        materialize_mix=False,
    )
    sequence = [
        group.encode(1, 0),
        group.encode(0, -1),
        group.encode(1, 1),
    ]

    result = model.rollout(x_allo, sequence)

    assert model.amplitude_multipliers == (2.0, 0.5, 1.0)
    np.testing.assert_allclose(
        result["predicted_outputs"],
        result["true_outputs"],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "multipliers",
    [
        (1.0, 1.0),
        (1.0, -1.0, -1.0),
        (2.0, 2.0, 1.0),
    ],
)
def test_invalid_amplitude_multipliers_are_rejected(multipliers):
    group = ProductCyclicGroup(3, 3)
    x_ego = random_invertible_encoding(group, group.irreps(), seed=24)

    with pytest.raises(ValueError, match="amplitude_multipliers"):
        build_finite_group_rnn(
            group,
            x_ego,
            amplitude_multipliers=multipliers,
        )


def test_cnxcn_geometry_tracks_translated_gaussian():
    group = ProductCyclicGroup(8, 8)
    center = (2, 2)
    signal = cnxcn_gaussian_bump(group, center=center, sigma=0.4)
    element = group.encode(3, -1)
    transformed = group.left_action(element, signal)

    decoded = decode_cnxcn_argmax(group, transformed)
    exact = transformed_cnxcn_center(group, element, center)

    assert decoded == exact
    np.testing.assert_allclose(
        cnxcn_center_errors(group, np.asarray([decoded]), np.asarray([exact])),
        [0.0],
    )


def test_cnxcn_momentum_sequence_stays_in_bounds():
    group = ProductCyclicGroup(10, 10)
    sequence = make_cnxcn_motion_sequence(
        group,
        steps=40,
        start_xy=(5, 5),
        margin=2,
    )

    cumulative = group.identity()
    for element in sequence:
        cumulative = group.compose(cumulative, int(element))
        x, y = group.decode(cumulative)
        assert 2 <= x <= 7
        assert 2 <= y <= 7


def test_factored_and_materialized_recurrence_agree(group):
    x_ego = random_invertible_encoding(group, group.irreps(), seed=2)
    x_allo = np.random.default_rng(3).standard_normal(group.order)
    factored = build_finite_group_rnn(group, x_ego, materialize_mix=False)
    materialized = build_finite_group_rnn(group, x_ego, materialize_mix=True)
    hidden = np.random.default_rng(4).standard_normal(factored.hidden_dim)

    assert factored.W_mix is None
    np.testing.assert_allclose(
        factored.apply_mix(hidden),
        materialized.apply_mix(hidden),
        atol=1e-12,
    )


def test_constructed_rnn_is_fixed_weight_torch_module(group):
    x_ego = random_invertible_encoding(group, group.irreps(), seed=25)

    model = build_finite_group_rnn(group, x_ego)

    assert isinstance(model, FiniteGroupRNN)
    assert isinstance(model, nn.Module)
    assert list(model.parameters()) == []
    assert set(model.state_dict()) == {"x_ego", "W_in", "W_drive", "W_out"}
    assert model.W_in.dtype == torch.float64


def test_forward_accepts_encoded_drives(group):
    x_ego = random_invertible_encoding(group, group.irreps(), seed=26)
    x_allo = np.random.default_rng(27).standard_normal(group.order)
    model = build_finite_group_rnn(group, x_ego)
    sequence = [group.encode(1, 0, 0), group.encode(0, 1, 1)]
    drives = model.encode_drives(sequence)

    outputs = model(x_allo, drives, return_all_outputs=True)
    rollout_result = model.rollout(x_allo, sequence)

    assert outputs.shape == (len(sequence), group.order)
    torch.testing.assert_close(outputs, rollout_result["predicted_outputs"])


def test_hidden_width_budget_limits_power_selection(group):
    irreps = group.irreps()
    signal = np.random.default_rng(5).standard_normal(group.order)
    budget = hidden_width(irreps[0])

    selected, indices = select_irreps_by_power(
        irreps,
        signal,
        max_hidden_width=budget,
    )

    assert indices == [0]
    assert selected == [irreps[0]]


def test_periodic_triangular_distance_wraps():
    assert periodic_distance_squared(7, (0, 0), (6, 0)) == pytest.approx(1.0)


def test_offset_coordinates_form_unique_rectangular_rows():
    x, y = offset_coordinates(8)

    assert x.shape == (8, 8)
    assert y.shape == (8, 8)
    for row in range(8):
        assert np.unique(x[:, row]).size == 8
        assert np.unique(y[:, row]).size == 1


def test_lattice_path_coordinates_wrap_group_indices():
    points = np.array([[0, 0], [7, 7], [8, 8], [-1, -1]])

    coordinates = lattice_path_coordinates(points, 8)

    np.testing.assert_allclose(coordinates[0], coordinates[2])
    np.testing.assert_allclose(coordinates[1], coordinates[3])


def test_momentum_sequence_starts_requested_translation_and_stays_in_bounds():
    group = DiscreteSE2Group(n=8, m=3)
    sequence = make_momentum_motion_sequence(
        group,
        steps=30,
        seed=1,
        start_xy=(3, 4),
        margin=1,
    )

    assert len(sequence) == 30
    assert group.decode(sequence[0]) == (3, 4, 0)
    cumulative = group.identity()
    for element in sequence:
        cumulative = group.compose(cumulative, int(element))
        x, y, rotation = group.decode(cumulative)
        assert 1 <= x <= 6
        assert 1 <= y <= 6
        assert rotation == 0


def test_rotating_momentum_sequence_keeps_transformed_pose_in_bounds():
    group = DiscreteSE2Group(n=8, m=3)
    initial_pose = (2, 2, 0)
    sequence = make_momentum_motion_sequence(
        group,
        steps=30,
        seed=1,
        include_rotations=True,
        start_xy=(3, 4),
        initial_pose=initial_pose,
        margin=1,
    )

    cumulative = group.identity()
    for element in sequence:
        cumulative = group.compose(cumulative, int(element))
        x, y, _ = advance_se2_pose(group, initial_pose, cumulative)
        assert 1 <= x <= 6
        assert 1 <= y <= 6


def test_constructed_rnn_uses_body_frame_right_action():
    group = DiscreteSE2Group(n=5, m=4)
    current_pose = group.encode(2, 2, 1)
    forward_body = group.encode(1, 0, 0)
    x_allo = np.zeros(group.order)
    x_allo[current_pose] = 1.0
    x_ego = random_invertible_encoding(group, group.irreps(), seed=28)
    model = build_finite_group_rnn(group, x_ego)

    result = model.rollout(x_allo, [forward_body])
    predicted_pose = group.decode(int(result["predicted_outputs"][0].argmax()))

    assert model.action_side == "right"
    assert predicted_pose == (2, 3, 1)
    assert predicted_pose == group.decode(group.compose(current_pose, forward_body))


def test_left_action_remains_available_as_explicit_spatial_option():
    group = DiscreteSE2Group(n=5, m=4)
    current_pose = group.encode(2, 2, 1)
    world_translation = group.encode(1, 0, 0)
    x_allo = np.zeros(group.order)
    x_allo[current_pose] = 1.0
    x_ego = random_invertible_encoding(group, group.irreps(), seed=29)
    model = build_finite_group_rnn(group, x_ego, action_side="left")

    result = model.rollout(x_allo, [world_translation])
    predicted_pose = group.decode(int(result["predicted_outputs"][0].argmax()))

    assert predicted_pose == (3, 2, 1)
    assert predicted_pose == group.decode(group.compose(world_translation, current_pose))


@pytest.mark.parametrize("q_rho", [0, 1, 2, 2.5, True])
def test_q_rho_must_support_phase_cycling(group, q_rho):
    x_ego = random_invertible_encoding(group, group.irreps(), seed=30)

    with pytest.raises(ValueError, match="q_rho"):
        build_finite_group_rnn(group, x_ego, q_rho=q_rho)


def test_center_errors_use_periodic_triangular_distance():
    group = DiscreteSE2Group(n=8, m=3)
    predicted = np.asarray([(0, 0), (2, 3)])
    exact = np.asarray([(7, 0), (2, 3)])

    errors = center_errors_periodic_triangular(group, predicted, exact)

    np.testing.assert_allclose(errors, [1.0, 0.0])


def test_gaussian_is_copied_across_rotations(group):
    signal = gaussian_bump(group, center=(0, 0), sigma=0.5)
    tensor = signal_to_tensor(group, signal)

    for rotation in range(1, group.m):
        np.testing.assert_allclose(tensor[rotation], tensor[0])
    assert decode_spatial_argmax(group, signal) == (0, 0)


def test_alignment_uses_group_rotation(group):
    values = np.arange(group.n**2).reshape(group.n, group.n)
    aligned = align_rotation_slice(group, values, rotation=1)

    for x in range(group.n):
        for y in range(group.n):
            rotated = group.apply_rotation(1, x, y)
            assert aligned[rotated] == values[x, y]


def test_transformed_center_matches_left_action(group):
    center = (0, 1)
    signal = gaussian_bump(group, center=center, sigma=0.2)
    element = group.encode(1, 0, 1)

    predicted_center = decode_spatial_argmax(group, group.left_action(element, signal))

    assert predicted_center == transformed_center(group, element, center)


def test_se2_orientation_profile_tracks_semidirect_pose(group):
    pose = (0, 1, 0)
    orientation_weights = np.zeros(group.m)
    orientation_weights[pose[2]] = 1.0
    signal = gaussian_bump(
        group,
        center=pose[:2],
        sigma=0.2,
        orientation_weights=orientation_weights,
    )
    element = group.encode(1, 0, 1)

    transformed = group.left_action(element, signal)

    assert decode_se2_pose(group, transformed) == transformed_se2_pose(
        group, element, pose
    )


def test_complete_se3_construction_reproduces_group_action():
    group = DiscreteSE3Group(n=2)
    x_ego = random_invertible_encoding(group, group.irreps(), seed=10)
    x_allo = np.random.default_rng(11).standard_normal(group.order)
    model = build_finite_group_rnn(group, x_ego, materialize_mix=False)
    sequence = [
        group.encode(1, 0, 0, 0),
        group.encode(0, 0, 0, 7),
        group.encode(0, 1, 0, 0),
    ]

    result = model.rollout(x_allo, sequence)

    assert model.hidden_dim == 9_312
    np.testing.assert_allclose(
        result["predicted_outputs"],
        result["true_outputs"],
        atol=1e-12,
    )


def test_se3_landmark_pose_tracks_left_action():
    group = DiscreteSE3Group(n=3)
    pose = (1, 0, 2, 0)
    signal = gaussian_landmark(
        group,
        center=pose[:3],
        sigma=(0.35, 0.55, 0.75),
        orientation_weights=peaked_orientation_weights(group, floor=0.0),
    )
    element = group.encode(1, 0, 0, 7)

    transformed = group.left_action(element, signal)

    assert decode_pose(group, transformed) == transformed_pose(group, element, pose)


def test_se3_periodic_distance_wraps():
    assert periodic_distance_squared_3d(7, (0, 0, 0), (6, 0, 0)) == pytest.approx(1.0)


def test_se3_energy_views_do_not_cancel_signed_encoding():
    group = DiscreteSE3Group(n=2)
    tensor = np.zeros((group.num_rotations, group.n, group.n, group.n))
    tensor[0, 0, 0, 0] = 1.0
    tensor[1, 0, 0, 0] = -1.0
    signal = tensor.reshape(-1)

    spatial = spatial_energy(group, signal)
    orientation = orientation_energy(group, signal)

    assert spatial[0, 0, 0] == pytest.approx(np.sqrt(2 / group.num_rotations))
    assert orientation[0] == pytest.approx(1 / np.sqrt(group.n**3))
    assert orientation[1] == pytest.approx(1 / np.sqrt(group.n**3))


def test_se3_volume_alignment_uses_rotation():
    group = DiscreteSE3Group(n=3)
    values = np.arange(group.n**3).reshape(group.n, group.n, group.n)
    aligned = align_rotation_volume(group, values, rotation=7)

    for x in range(group.n):
        for y in range(group.n):
            for z in range(group.n):
                rotated = group.apply_rotation(7, x, y, z)
                assert aligned[rotated] == values[x, y, z]


def test_rotation_error_is_zero_only_for_matching_rotation():
    group = DiscreteSE3Group(n=2)

    assert rotation_error(group, 0, 0) == pytest.approx(0.0)
    assert rotation_error(group, 0, 1) > 0
