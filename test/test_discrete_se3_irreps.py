"""Irrep tests for DiscreteSE3Group (Z_n^3 ⋊ O)."""

import numpy as np
import pytest

from src import template
from src.groups import DiscreteSE3Group, LazyIrreducibleRepresentation, make_group


def _identity(group: DiscreteSE3Group) -> int:
    return group.identity()


class TestConstructionAndOrbits:
    def test_make_group(self):
        group = make_group("znxznxzn_oh", {"data": {"p": 2}})
        assert isinstance(group, DiscreteSE3Group)
        assert group.order == 24 * 2**3

    def test_irreps_are_lazy(self):
        group = DiscreteSE3Group(2)
        assert all(isinstance(ir, LazyIrreducibleRepresentation) for ir in group.irreps())

    def test_public_encode_decode_roundtrip(self):
        group = DiscreteSE3Group(2)
        for element in group.elements():
            assert group.encode(*group.decode(element)) == element

    def test_public_dimensions(self):
        group = DiscreteSE3Group(3)
        assert group.n == 3
        assert group.num_rotations == 24

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_orbit_stabilizer(self, n):
        group = DiscreteSE3Group(n)
        for data in group.orbit_data():
            assert len(data["orbit"]) * len(data["stabilizer"]) == 24
            assert len(data["coset_reps"]) == len(data["orbit"])
            assert len(set(data["orbit_labels"])) == len(data["orbit"])


class TestLittleGroups:
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_little_group_irreps(self, n):
        group = DiscreteSE3Group(n)
        for data in group.orbit_data():
            stabilizer = data["stabilizer"]
            cayley = group._subgroup_cayley(stabilizer)
            little_irreps = group._little_irreps(stabilizer)
            assert sum(ir.dim**2 for ir in little_irreps) == len(stabilizer)

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
        group = DiscreteSE3Group(2)
        assert sum(ir.dim**2 for ir in group.irreps()) == group.order

    def test_identity_is_identity_matrix(self):
        group = DiscreteSE3Group(2)
        e = _identity(group)
        for irrep in group.irreps():
            np.testing.assert_allclose(irrep(e), np.eye(irrep.dim), atol=1e-8)

    def test_unitarity_sample(self):
        group = DiscreteSE3Group(2)
        rng = np.random.default_rng(0)
        elems = rng.integers(0, group.order, size=12)
        for irrep in group.irreps():
            for elem in elems:
                mat = irrep(int(elem))
                np.testing.assert_allclose(mat.conj().T @ mat, np.eye(irrep.dim), atol=1e-7)

    def test_homomorphism_sample(self):
        group = DiscreteSE3Group(2)
        rng = np.random.default_rng(1)
        pairs = rng.integers(0, group.order, size=(16, 2))
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
        group = DiscreteSE3Group(2)
        for element in group.elements():
            inverse = group.inverse(element)
            assert group.compose(element, inverse) == group.identity()
            assert group.compose(inverse, element) == group.identity()

    def test_left_action_matches_regular_rep(self):
        group = DiscreteSE3Group(2)
        signal = np.random.default_rng(4).standard_normal(group.order)
        for element in (0, 1, 17, group.order - 1):
            np.testing.assert_allclose(
                group.left_action(element, signal),
                group.regular_rep()[element] @ signal,
            )

    def test_left_actions_compose(self):
        group = DiscreteSE3Group(2)
        signal = np.random.default_rng(5).standard_normal(group.order)
        first = group.encode(1, 0, 0, 0)
        second = group.encode(0, 0, 0, 7)
        np.testing.assert_allclose(
            group.left_action(second, group.left_action(first, signal)),
            group.left_action(group.compose(second, first), signal),
        )

    def test_cumulative_product_uses_left_action_order(self):
        group = DiscreteSE3Group(2)
        sequence = [
            group.encode(1, 0, 0, 0),
            group.encode(0, 0, 0, 3),
            group.encode(0, 1, 0, 0),
        ]
        expected = group.identity()
        for element in sequence:
            expected = group.compose(element, expected)
        assert group.cumulative_product(sequence) == expected


class TestFourier:
    def test_fourier_roundtrip(self):
        group = DiscreteSE3Group(2)
        rng = np.random.default_rng(2)
        signal = rng.standard_normal(group.order)
        reconstructed = group.inverse_fourier(group.fourier(signal))
        np.testing.assert_allclose(reconstructed.real, signal, atol=1e-10)
        np.testing.assert_allclose(reconstructed.imag, np.zeros(group.order), atol=1e-10)

    def test_parseval(self):
        group = DiscreteSE3Group(2)
        rng = np.random.default_rng(3)
        signal = rng.standard_normal(group.order)
        coefs = group.fourier(signal)
        lhs = sum(
            ir.dim * np.real(np.trace(coef.conj().T @ coef))
            for ir, coef in zip(group.irreps(), coefs)
        )
        rhs = group.order * float(signal @ signal)
        assert lhs == pytest.approx(rhs, rel=1e-8)

    def test_custom_fourier_real_for_conjugate_symmetric_powers(self):
        group = DiscreteSE3Group(2)
        powers = np.ones(len(group.irreps()))
        tpl = template.custom_fourier(group, powers)
        assert tpl.shape == (group.order,)
        assert np.isrealobj(tpl)


class TestConjugates:
    def test_conjugate_pairs_cover_irreps(self):
        group = DiscreteSE3Group(2)
        pairs = group.conjugate_pairs()
        flat = sorted(idx for pair in pairs for idx in pair)
        assert flat == list(range(len(group.irreps())))

    def test_conjugate_pairs_match_characters(self):
        group = DiscreteSE3Group(2)
        irreps = group.irreps()
        for pair in group.conjugate_pairs():
            if len(pair) == 1:
                i = pair[0]
                for elem in group.elements():
                    chi = np.trace(irreps[i](elem))
                    assert chi == pytest.approx(chi.conjugate(), abs=1e-8)
            else:
                i, j = pair
                for elem in group.elements():
                    np.testing.assert_allclose(
                        np.trace(irreps[j](elem)),
                        np.trace(irreps[i](elem)).conjugate(),
                        atol=1e-8,
                    )


class TestGuards:
    def test_regular_rep_guard_for_large_group(self):
        group = DiscreteSE3Group(3)
        with pytest.raises(MemoryError, match="left_action"):
            group.regular_rep()
