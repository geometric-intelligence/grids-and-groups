"""Irrep tests for DiscreteSE3A4Group (Z_n^3 ⋊ A4)."""

from collections import Counter

import numpy as np
import pytest

from src import template
from src.groups import (
    DiscreteSE3A4Group,
    LazyIrreducibleRepresentation,
    make_group,
)
from src.groups.znxznxzn_a4 import _element_orders

_TETRAHEDRON_VERTICES = {
    (1, 1, 1),
    (1, -1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
}


class TestTetrahedralRotations:
    def test_rotations_form_a4(self):
        group = DiscreteSE3A4Group(2)
        matrices = [group.rotation_matrix(rotation) for rotation in range(group.num_rotations)]

        assert len({tuple(matrix.ravel()) for matrix in matrices}) == 12
        np.testing.assert_array_equal(matrices[0], np.eye(3, dtype=int))
        assert all(round(np.linalg.det(matrix)) == 1 for matrix in matrices)
        assert Counter(_element_orders(group._rot_cayley)) == {
            1: 1,
            2: 3,
            3: 8,
        }

        matrix_keys = {tuple(matrix.ravel()) for matrix in matrices}
        for matrix_a in matrices:
            for matrix_b in matrices:
                assert tuple((matrix_a @ matrix_b).ravel()) in matrix_keys

    def test_rotations_preserve_tetrahedron(self):
        group = DiscreteSE3A4Group(2)
        for rotation in range(group.num_rotations):
            matrix = group.rotation_matrix(rotation)
            images = {
                tuple((matrix @ np.asarray(vertex)).tolist()) for vertex in _TETRAHEDRON_VERTICES
            }
            assert images == _TETRAHEDRON_VERTICES


class TestConstructionAndOrbits:
    def test_make_group(self):
        group = make_group("znxznxzn_a4", {"data": {"p": 2}})
        assert isinstance(group, DiscreteSE3A4Group)
        assert group.order == 12 * 2**3

    def test_irreps_are_lazy(self):
        group = DiscreteSE3A4Group(2)
        assert all(isinstance(irrep, LazyIrreducibleRepresentation) for irrep in group.irreps())

    def test_public_encode_decode_roundtrip(self):
        group = DiscreteSE3A4Group(2)
        for element in group.elements():
            assert group.encode(*group.decode(element)) == element

    def test_public_dimensions(self):
        group = DiscreteSE3A4Group(3)
        assert group.n == 3
        assert group.num_rotations == 12
        assert group.order == 12 * 3**3

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_orbit_stabilizer(self, n):
        group = DiscreteSE3A4Group(n)
        for data in group.orbit_data():
            assert len(data["orbit"]) * len(data["stabilizer"]) == 12
            assert len(data["coset_reps"]) == len(data["orbit"])
            assert len(set(data["orbit_labels"])) == len(data["orbit"])


class TestLittleGroups:
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_little_group_irreps(self, n):
        group = DiscreteSE3A4Group(n)
        for data in group.orbit_data():
            stabilizer = data["stabilizer"]
            cayley = group._subgroup_cayley(stabilizer)
            little_irreps = group._little_irreps(stabilizer)
            assert sum(irrep.dim**2 for irrep in little_irreps) == len(stabilizer)

            for irrep in little_irreps:
                np.testing.assert_allclose(irrep(0), np.eye(irrep.dim), atol=1e-8)
                for a in range(len(stabilizer)):
                    for b in range(len(stabilizer)):
                        np.testing.assert_allclose(
                            irrep(cayley[a, b]),
                            irrep(a) @ irrep(b),
                            atol=1e-7,
                            err_msg=f"{irrep.name} failed at {a}, {b}",
                        )


class TestInducedIrreps:
    def test_peter_weyl_dimension_sum(self):
        group = DiscreteSE3A4Group(2)
        assert sum(irrep.dim**2 for irrep in group.irreps()) == group.order

    def test_identity_is_identity_matrix(self):
        group = DiscreteSE3A4Group(2)
        for irrep in group.irreps():
            np.testing.assert_allclose(irrep(group.identity()), np.eye(irrep.dim), atol=1e-8)

    def test_unitarity_sample(self):
        group = DiscreteSE3A4Group(2)
        elements = np.random.default_rng(0).integers(0, group.order, size=12)
        for irrep in group.irreps():
            for element in elements:
                matrix = irrep(int(element))
                np.testing.assert_allclose(
                    matrix.conj().T @ matrix,
                    np.eye(irrep.dim),
                    atol=1e-7,
                )

    def test_homomorphism_sample(self):
        group = DiscreteSE3A4Group(2)
        pairs = np.random.default_rng(1).integers(0, group.order, size=(16, 2))
        for irrep in group.irreps():
            for a, b in pairs:
                a, b = int(a), int(b)
                np.testing.assert_allclose(
                    irrep(group.compose(a, b)),
                    irrep(a) @ irrep(b),
                    atol=1e-7,
                    err_msg=f"{irrep} failed at {a}, {b}",
                )


class TestActions:
    def test_inverse(self):
        group = DiscreteSE3A4Group(2)
        for element in group.elements():
            inverse = group.inverse(element)
            assert group.compose(element, inverse) == group.identity()
            assert group.compose(inverse, element) == group.identity()

    def test_left_action_matches_regular_rep(self):
        group = DiscreteSE3A4Group(2)
        signal = np.random.default_rng(4).standard_normal(group.order)
        for element in (0, 1, 17, group.order - 1):
            np.testing.assert_allclose(
                group.left_action(element, signal),
                group.regular_rep()[element] @ signal,
            )

    def test_left_actions_compose(self):
        group = DiscreteSE3A4Group(2)
        signal = np.random.default_rng(5).standard_normal(group.order)
        first = group.encode(1, 0, 0, 0)
        second = group.encode(0, 0, 0, 7)
        np.testing.assert_allclose(
            group.left_action(second, group.left_action(first, signal)),
            group.left_action(group.compose(second, first), signal),
        )

    def test_cumulative_product_uses_body_frame_order(self):
        group = DiscreteSE3A4Group(2)
        sequence = [
            group.encode(1, 0, 0, 0),
            group.encode(0, 0, 0, 3),
            group.encode(0, 1, 0, 0),
        ]
        expected = group.identity()
        for element in sequence:
            expected = group.compose(expected, element)
        assert group.cumulative_product(sequence) == expected


class TestFourier:
    def test_fourier_roundtrip(self):
        group = DiscreteSE3A4Group(2)
        signal = np.random.default_rng(2).standard_normal(group.order)
        reconstructed = group.inverse_fourier(group.fourier(signal))
        np.testing.assert_allclose(reconstructed.real, signal, atol=1e-10)
        np.testing.assert_allclose(reconstructed.imag, np.zeros(group.order), atol=1e-10)

    def test_parseval(self):
        group = DiscreteSE3A4Group(2)
        signal = np.random.default_rng(3).standard_normal(group.order)
        coefficients = group.fourier(signal)
        lhs = sum(
            irrep.dim * np.real(np.trace(coefficient.conj().T @ coefficient))
            for irrep, coefficient in zip(group.irreps(), coefficients)
        )
        rhs = group.order * float(signal @ signal)
        assert lhs == pytest.approx(rhs, rel=1e-8)

    def test_custom_fourier_real_for_conjugate_symmetric_powers(self):
        group = DiscreteSE3A4Group(2)
        powers = np.ones(len(group.irreps()))
        constructed_template = template.custom_fourier(group, powers)
        assert constructed_template.shape == (group.order,)
        assert np.isrealobj(constructed_template)


class TestConjugates:
    def test_conjugate_pairs_cover_irreps(self):
        group = DiscreteSE3A4Group(2)
        pairs = group.conjugate_pairs()
        covered = sorted(index for pair in pairs for index in pair)
        assert covered == list(range(len(group.irreps())))

    def test_conjugate_pairs_match_characters(self):
        group = DiscreteSE3A4Group(2)
        irreps = group.irreps()
        for pair in group.conjugate_pairs():
            if len(pair) == 1:
                i = pair[0]
                for element in group.elements():
                    character = np.trace(irreps[i](element))
                    assert character == pytest.approx(character.conjugate(), abs=1e-8)
            else:
                i, j = pair
                for element in group.elements():
                    np.testing.assert_allclose(
                        np.trace(irreps[j](element)),
                        np.trace(irreps[i](element)).conjugate(),
                        atol=1e-8,
                    )


class TestGuards:
    def test_regular_rep_guard_for_large_group(self):
        group = DiscreteSE3A4Group(3)
        with pytest.raises(MemoryError, match="left_action"):
            group.regular_rep()
