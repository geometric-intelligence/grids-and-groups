"""Tests for module-restricted neural manifold analysis."""

import numpy as np

from src.finite_group_rnn import (
    build_finite_group_rnn,
    probe_hidden_states,
    random_invertible_encoding,
)
from src.groups.cnxcn import ProductCyclicGroup
from src.neural_manifold import (
    analyze_module_orbit,
    build_module_orbits,
    conjugate_irrep_groups,
    coordinate_colors,
    fixed_point_embedding,
)


def _exact_cnxcn_model():
    group = ProductCyclicGroup(3, 3)
    irreps = group.irreps()
    x_ego = random_invertible_encoding(group, irreps, seed=30)
    params = build_finite_group_rnn(group, x_ego, materialize_mix=False)
    x_allo = np.random.default_rng(31).standard_normal(group.order)
    return group, params, x_allo


def test_fixed_point_embedding_converges_for_exact_model():
    group, params, x_allo = _exact_cnxcn_model()
    initial_states = probe_hidden_states(params, x_allo)

    fixed = fixed_point_embedding(params, initial_states)

    assert fixed.converged
    assert fixed.iterations == 1
    assert np.max(fixed.residuals) < 1e-10
    assert fixed.states.shape == (group.order, params.hidden_dim)


def test_conjugate_irreps_are_combined_into_real_modules():
    _, params, _ = _exact_cnxcn_model()

    groups = conjugate_irrep_groups(params)

    assert groups == [(1, 2), (3, 6), (4, 8), (5, 7)]


def test_module_orbits_and_analysis_have_expected_shapes():
    group, params, x_allo = _exact_cnxcn_model()
    hidden = probe_hidden_states(params, x_allo)
    fixed = fixed_point_embedding(params, hidden)
    modules = build_module_orbits(params, fixed.states)

    analysis = analyze_module_orbit(
        modules[0],
        max_persistence_points=20,
        max_homology_dimension=1,
    )
    colors = coordinate_colors(
        np.asarray([group.decode(element) for element in group.elements()]),
        (group.p1, group.p2),
    )

    assert len(modules) == 4
    assert analysis.embedding.shape == (group.order, 2)
    assert len(analysis.persistence_diagrams) == 2
    assert colors.shape == (group.order, 3)
