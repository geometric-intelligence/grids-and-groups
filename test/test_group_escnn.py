"""Parity tests: verify our Group implementations match escnn exactly.

This file deliberately imports escnn and is intended to be run *before*
the escnn dependency is removed.  It will be skipped automatically if
escnn is not installed.
"""

import numpy as np
import pytest

escnn = pytest.importorskip("escnn", reason="escnn not installed; parity tests skipped")

from escnn.group import DihedralGroup as ESCNNDihedral  # noqa: E402
from escnn.group import Icosahedral as ESCNNIcosahedral  # noqa: E402
from escnn.group import Octahedral as ESCNNOctahedral  # noqa: E402

from src.groups.a5 import IcosahedralGroup  # noqa: E402
from src.groups.dn import DihedralGroup  # noqa: E402
from src.groups.oh import OctahedralGroup  # noqa: E402


def _escnn_irrep_matrices(escnn_group):
    """Extract all irrep matrices from an escnn group as a list of (|G|, d, d) arrays."""
    irreps = escnn_group.irreps()
    elements = list(escnn_group.elements)
    result = []
    for irrep in irreps:
        mats = np.array([irrep(g) for g in elements])
        result.append(mats)
    return result


def _escnn_regular_rep_matrices(escnn_group):
    """Extract all regular representation matrices from an escnn group."""
    regular_rep = escnn_group.representations["regular"]
    elements = list(escnn_group.elements)
    return np.array([regular_rep(g) for g in elements])


@pytest.fixture(params=[3, 5], ids=["D3", "D5"])
def dihedral_pair(request):
    n = request.param
    return DihedralGroup(N=n), ESCNNDihedral(N=n)


@pytest.fixture()
def octahedral_pair():
    return OctahedralGroup(), ESCNNOctahedral()


@pytest.fixture()
def a5_pair():
    return IcosahedralGroup(), ESCNNIcosahedral()


@pytest.fixture(
    params=["dihedral_3", "dihedral_5", "octahedral", "a5"],
    ids=["D3", "D5", "Oh", "A5"],
)
def group_pair(request):
    """Yield (our_group, escnn_group) for every group under test."""
    name = request.param
    if name == "dihedral_3":
        return DihedralGroup(N=3), ESCNNDihedral(N=3)
    if name == "dihedral_5":
        return DihedralGroup(N=5), ESCNNDihedral(N=5)
    if name == "octahedral":
        return OctahedralGroup(), ESCNNOctahedral()
    if name == "a5":
        return IcosahedralGroup(), ESCNNIcosahedral()
    raise ValueError(name)


# ===================================================================
# Tests
# ===================================================================


class TestOrder:
    def test_order(self, group_pair):
        ours, escnn_g = group_pair
        assert ours.order == escnn_g.order()


class TestElements:
    def test_num_elements(self, group_pair):
        ours, escnn_g = group_pair
        assert len(ours.elements()) == escnn_g.order()


class TestIrrepDimsOrdering:
    """Exact check that irrep dimensions appear in the expected order."""

    def test_octahedral_dims(self):
        g = OctahedralGroup()
        dims = [ir.size for ir in g.irreps()]
        assert dims == [1, 3, 3, 2, 1]

    def test_a5_dims(self):
        g = IcosahedralGroup()
        dims = [ir.size for ir in g.irreps()]
        assert dims == [1, 3, 5, 3, 4]


class TestIrrepCountAndDims:
    def test_same_count_and_dims(self, group_pair):
        ours, escnn_g = group_pair
        our_dims = [ir.size for ir in ours.irreps()]
        escnn_dims = [ir.size for ir in escnn_g.irreps()]
        assert our_dims == escnn_dims


class TestIrrepMatrices:
    """Irrep matrices must match exactly (not up to unitary equivalence)."""

    def test_irrep_matrices_exact(self, group_pair):
        ours, escnn_g = group_pair
        escnn_mats_list = _escnn_irrep_matrices(escnn_g)

        for i, (our_irrep, escnn_mats) in enumerate(zip(ours.irreps(), escnn_mats_list)):
            for g in range(ours.order):
                np.testing.assert_allclose(
                    our_irrep(g),
                    escnn_mats[g],
                    atol=1e-12,
                    err_msg=f"irrep {i} element {g}",
                )


class TestRegularRep:
    def test_regular_rep(self, group_pair):
        ours, escnn_g = group_pair
        our_reps = ours.regular_rep()
        escnn_reps = _escnn_regular_rep_matrices(escnn_g)
        assert our_reps.shape == escnn_reps.shape
        np.testing.assert_allclose(our_reps, escnn_reps, atol=1e-12)


class TestFourier:
    def _random_signal(self, n, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n)

    def test_fourier_roundtrip(self, group_pair):
        ours, _ = group_pair
        signal = self._random_signal(ours.order)
        coefs = ours.fourier(signal)
        reconstructed = ours.inverse_fourier(coefs)
        np.testing.assert_allclose(signal, reconstructed, atol=1e-10)

    def test_fourier_vs_escnn(self, group_pair):
        ours, escnn_g = group_pair
        signal = self._random_signal(ours.order)
        our_coefs = ours.fourier(signal)
        escnn_irreps = escnn_g.irreps()
        escnn_elements = list(escnn_g.elements)
        escnn_coefs = []
        for irrep in escnn_irreps:
            coef = sum(signal[i_g] * irrep(g).conj().T for i_g, g in enumerate(escnn_elements))
            escnn_coefs.append(coef)
        assert len(our_coefs) == len(escnn_coefs)
        for i, (oc, ec) in enumerate(zip(our_coefs, escnn_coefs)):
            np.testing.assert_allclose(oc, ec, atol=1e-10, err_msg=f"fourier coef {i}")


class TestPowerSpectrum:
    def _random_signal(self, n, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n)

    def test_power_spectrum_vs_escnn(self, group_pair):
        ours, escnn_g = group_pair
        signal = self._random_signal(ours.order)
        our_ps = ours.power_spectrum(signal)
        escnn_irreps = escnn_g.irreps()
        escnn_elements = list(escnn_g.elements)
        escnn_coefs = []
        for irrep in escnn_irreps:
            coef = sum(signal[i_g] * irrep(g).conj().T for i_g, g in enumerate(escnn_elements))
            escnn_coefs.append(coef)
        escnn_ps = np.zeros(len(escnn_irreps))
        for i, irrep in enumerate(escnn_irreps):
            fc = escnn_coefs[i]
            escnn_ps[i] = irrep.size * np.trace(fc.conj().T @ fc)
        np.testing.assert_allclose(our_ps, escnn_ps / ours.order, atol=1e-10)
