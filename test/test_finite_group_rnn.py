"""Tests for the closed-form finite-group QuadraticRNN construction."""

import numpy as np
import pytest

from src.cnxcn_geometry import (
    center_errors as cnxcn_center_errors,
)
from src.cnxcn_geometry import (
    decode_spatial_argmax as decode_cnxcn_argmax,
)
from src.cnxcn_geometry import (
    gaussian_bump as cnxcn_gaussian_bump,
)
from src.cnxcn_geometry import (
    make_momentum_motion_sequence as make_cnxcn_motion_sequence,
)
from src.cnxcn_geometry import (
    transformed_center as transformed_cnxcn_center,
)
from src.discrete_se2_geometry import (
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
from src.discrete_se3_geometry import (
    align_rotation_slice as align_rotation_volume,
)
from src.discrete_se3_geometry import (
    decode_pose,
    gaussian_landmark,
    orientation_energy,
    peaked_orientation_weights,
    rotation_error,
    spatial_energy,
    transformed_pose,
)
from src.discrete_se3_geometry import (
    periodic_distance_squared as periodic_distance_squared_3d,
)
from src.finite_group_rnn import (
    build_finite_group_rnn,
    hidden_width,
    random_invertible_encoding,
    rollout,
    select_irreps_by_power,
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
    params = build_finite_group_rnn(group, x_ego, materialize_mix=False)
    sequence = [
        group.encode(1, 0, 0),
        group.encode(0, 1, 1),
        group.encode(-1, 0, -1),
    ]

    result = rollout(params, x_allo, sequence)

    np.testing.assert_allclose(
        result["predicted_outputs"],
        result["true_outputs"],
        atol=1e-12,
    )


def test_complete_cnxcn_construction_reproduces_group_action():
    group = ProductCyclicGroup(3, 3)
    x_ego = random_invertible_encoding(group, group.irreps(), seed=20)
    x_allo = np.random.default_rng(21).standard_normal(group.order)
    params = build_finite_group_rnn(group, x_ego, materialize_mix=False)
    sequence = [
        group.encode(1, 0),
        group.encode(0, -1),
        group.encode(1, 1),
    ]

    result = rollout(params, x_allo, sequence)

    np.testing.assert_allclose(
        result["predicted_outputs"],
        result["true_outputs"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        group.left_action(sequence[0], x_allo),
        group.regular_rep()[sequence[0]] @ x_allo,
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
        cumulative = group.compose(int(element), cumulative)
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
        cumulative = group.compose(int(element), cumulative)
        x, y, rotation = group.decode(cumulative)
        assert 1 <= x <= 6
        assert 1 <= y <= 6
        assert rotation == 0


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


def test_complete_se3_construction_reproduces_group_action():
    group = DiscreteSE3Group(n=2)
    x_ego = random_invertible_encoding(group, group.irreps(), seed=10)
    x_allo = np.random.default_rng(11).standard_normal(group.order)
    params = build_finite_group_rnn(group, x_ego, materialize_mix=False)
    sequence = [
        group.encode(1, 0, 0, 0),
        group.encode(0, 0, 0, 7),
        group.encode(0, 1, 0, 0),
    ]

    result = rollout(params, x_allo, sequence)

    assert params.hidden_dim == 9_312
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
