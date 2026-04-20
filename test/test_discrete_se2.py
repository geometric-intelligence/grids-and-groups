"""Unit tests for DiscreteSE2Group (Z_n² ⋊ C_m)."""

import numpy as np
import pytest

from src.groups.znxzn_cm import DiscreteSE2Group

# Small (n, m) pairs used across most tests.  Chosen to keep group order low
# while covering every valid m value.
SMALL_PARAMS = [
    (2, 1),   # order  4  – trivial rotation
    (2, 2),   # order  8  – 180° half-turn
    (2, 3),   # order 12  – 120°
    (2, 4),   # order 16  – 90°
    (3, 3),   # order 27  – 120°
    (2, 6),   # order 24  – 60°
]

IDS = [f"n{n}_m{m}" for n, m in SMALL_PARAMS]


@pytest.fixture(params=SMALL_PARAMS, ids=IDS)
def group(request):
    n, m = request.param
    return DiscreteSE2Group(n=n, m=m)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:
    @pytest.mark.parametrize("n,m", SMALL_PARAMS, ids=IDS)
    def test_order(self, n, m):
        g = DiscreteSE2Group(n=n, m=m)
        assert g.order == m * n * n

    def test_invalid_n_too_small(self):
        with pytest.raises(ValueError, match="n must be >= 2"):
            DiscreteSE2Group(n=1, m=3)

    def test_invalid_m(self):
        with pytest.raises(ValueError, match="m must be in"):
            DiscreteSE2Group(n=3, m=5)

    def test_elements_length(self, group):
        assert len(group.elements()) == group.order

    def test_elements_are_contiguous(self, group):
        assert group.elements() == list(range(group.order))


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

class TestEncodeDecode:
    def test_roundtrip_all_elements(self, group):
        for idx in range(group.order):
            x, y, r = group._decode(idx)
            assert group._encode(x, y, r) == idx

    def test_decode_bounds(self, group):
        n, m = group._n, group._m
        for idx in range(group.order):
            x, y, r = group._decode(idx)
            assert 0 <= x < n
            assert 0 <= y < n
            assert 0 <= r < m


# ---------------------------------------------------------------------------
# Group axioms
# ---------------------------------------------------------------------------

class TestGroupAxioms:
    def _identity(self, group):
        """Index of the identity element (x=0, y=0, r=0)."""
        return group._encode(0, 0, 0)

    def test_identity_is_zero(self, group):
        assert self._identity(group) == 0

    def test_left_identity(self, group):
        e = self._identity(group)
        for g in group.elements():
            assert group.compose(e, g) == g

    def test_right_identity(self, group):
        e = self._identity(group)
        for g in group.elements():
            assert group.compose(g, e) == g

    def test_every_element_has_inverse(self, group):
        e = self._identity(group)
        for g in group.elements():
            found = any(group.compose(g, h) == e for h in group.elements())
            assert found, f"No inverse found for element {g}"

    def test_associativity_sample(self, group):
        """Check (a*b)*c == a*(b*c) for a dense sample of triples."""
        rng = np.random.default_rng(0)
        elems = group.elements()
        triples = rng.choice(elems, size=(min(200, len(elems) ** 2), 3), replace=True)
        for a, b, c in triples:
            lhs = group.compose(group.compose(int(a), int(b)), int(c))
            rhs = group.compose(int(a), group.compose(int(b), int(c)))
            assert lhs == rhs


# ---------------------------------------------------------------------------
# Regular representation
# ---------------------------------------------------------------------------

class TestRegularRep:
    def test_shape(self, group):
        reg = group.regular_rep()
        n = group.order
        assert reg.shape == (n, n, n)

    def test_each_slice_is_permutation_matrix(self, group):
        reg = group.regular_rep()
        for g in range(group.order):
            mat = reg[g]
            np.testing.assert_array_equal(mat.sum(axis=0), np.ones(group.order))
            np.testing.assert_array_equal(mat.sum(axis=1), np.ones(group.order))
            assert set(np.unique(mat)) == {0.0, 1.0}

    def test_regular_rep_encodes_composition(self, group):
        """reg[g] @ e_h == e_{g*h} for a sample of (g, h) pairs."""
        reg = group.regular_rep()
        rng = np.random.default_rng(1)
        elems = group.elements()
        pairs = rng.choice(elems, size=(min(50, group.order), 2), replace=True)
        for g, h in pairs:
            g, h = int(g), int(h)
            gh = group.compose(g, h)
            e_h = np.zeros(group.order)
            e_h[h] = 1.0
            result = reg[g] @ e_h
            assert result[gh] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Irreducible representations
# ---------------------------------------------------------------------------

class TestIrreps:
    def test_peter_weyl_dimension_sum(self, group):
        """Sum of dim² must equal |G| (Peter–Weyl theorem)."""
        irreps = group.irreps()
        assert sum(ir.dim ** 2 for ir in irreps) == group.order

    def test_irrep_matrices_are_unitary(self, group):
        for irrep in group.irreps():
            for g in range(group.order):
                mat = irrep(g)
                eye = np.eye(irrep.dim)
                np.testing.assert_allclose(
                    mat.conj().T @ mat, eye, atol=1e-10,
                    err_msg=f"irrep {irrep._name} not unitary at g={g}",
                )

    def test_irrep_homomorphism_sample(self, group):
        """ρ(g·h) == ρ(g) @ ρ(h) for a random sample of pairs."""
        rng = np.random.default_rng(2)
        elems = group.elements()
        pairs = rng.choice(elems, size=(min(30, group.order), 2), replace=True)
        for irrep in group.irreps():
            for g, h in pairs:
                g, h = int(g), int(h)
                gh = group.compose(g, h)
                np.testing.assert_allclose(
                    irrep(gh), irrep(g) @ irrep(h), atol=1e-10,
                    err_msg=f"irrep {irrep._name} homomorphism failed at g={g}, h={h}",
                )

    def test_irrep_identity_is_identity_matrix(self, group):
        """ρ(e) must be the identity matrix for every irrep."""
        e = group._encode(0, 0, 0)
        for irrep in group.irreps():
            np.testing.assert_allclose(
                irrep(e), np.eye(irrep.dim), atol=1e-10,
                err_msg=f"irrep {irrep._name} ρ(e) != I",
            )


# ---------------------------------------------------------------------------
# Fourier analysis
# ---------------------------------------------------------------------------

class TestFourier:
    def _signal(self, group, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(group.order)

    def test_fourier_roundtrip(self, group):
        signal = self._signal(group)
        coefs = group.fourier(signal)
        reconstructed = group.inverse_fourier(coefs)
        np.testing.assert_allclose(
            np.real(reconstructed), signal, atol=1e-10,
            err_msg="Fourier roundtrip failed",
        )

    def test_fourier_coef_shapes(self, group):
        signal = self._signal(group)
        coefs = group.fourier(signal)
        assert len(coefs) == len(group.irreps())
        for coef, irrep in zip(coefs, group.irreps()):
            assert coef.shape == (irrep.dim, irrep.dim)

    def test_power_spectrum_shape(self, group):
        signal = self._signal(group)
        ps = group.power_spectrum(signal)
        assert ps.shape == (len(group.irreps()),)

    def test_power_spectrum_nonnegative(self, group):
        signal = self._signal(group)
        ps = group.power_spectrum(signal)
        assert np.all(ps >= -1e-12)

    def test_parseval(self, group):
        """Parseval: sum_rho dim(rho) * ||hat_x(rho)||_F^2 == |G| * ||x||^2."""
        signal = self._signal(group)
        coefs = group.fourier(signal)
        lhs = sum(
            ir.dim * np.real(np.trace(c.conj().T @ c))
            for ir, c in zip(group.irreps(), coefs)
        )
        rhs = group.order * float(signal @ signal)
        assert lhs == pytest.approx(rhs, rel=1e-8)
