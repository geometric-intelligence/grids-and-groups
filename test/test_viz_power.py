"""Tests for power-related functions in src.viz module."""

import numpy as np

import src.template as template
import src.viz as viz
from src.groups import DihedralGroup, OctahedralGroup


class TestTopkTemplateFreqs:
    """Tests for viz.topk_template_freqs function (2D)."""

    def test_returns_top_k(self):
        """Test that function returns exactly K frequency pairs."""
        p1, p2 = 8, 8
        K = 3
        tpl = np.random.randn(p1, p2)

        top_freqs = viz.topk_template_freqs(tpl, K)

        assert len(top_freqs) == K, f"Expected {K} frequency pairs, got {len(top_freqs)}"

    def test_returns_tuples(self):
        """Test that returned values are (kx, ky) tuples."""
        p1, p2 = 8, 8
        K = 3
        tpl = np.random.randn(p1, p2)

        top_freqs = viz.topk_template_freqs(tpl, K)

        for freq in top_freqs:
            assert isinstance(freq, tuple), f"Expected tuple, got {type(freq)}"
            assert len(freq) == 2, f"Expected 2-tuple, got {len(freq)}-tuple"

    def test_empty_for_zero_signal(self):
        """Test that zero signal returns empty list."""
        p1, p2 = 6, 6
        tpl = np.zeros((p1, p2))

        top_freqs = viz.topk_template_freqs(tpl, K=3, min_power=1e-10)

        assert top_freqs == []


class TestPowersPerNeuronRowsCyclicGroup:
    """Tests for viz.powers_per_neuron_rows with CyclicGroup."""

    def test_shape_cn(self):
        """Each row must produce the correct power shape for CyclicGroup."""
        from src.groups.cn import CyclicGroup

        p = 10
        h = 3
        group = CyclicGroup(N=p)
        W = np.random.RandomState(1).randn(h, p)
        out = viz.powers_per_neuron_rows(W, group)
        n_irreps = len(group.irreps())
        assert out.shape == (h, n_irreps)
        for i in range(h):
            np.testing.assert_allclose(out[i], group.power_spectrum(W[i]), rtol=1e-10)

    def test_shape_cnxcn(self):
        """Each row must produce the correct power shape for ProductCyclicGroup."""
        from src.groups.cnxcn import ProductCyclicGroup

        p1, p2 = 2, 3
        h = 2
        group = ProductCyclicGroup(p1=p1, p2=p2)
        W = np.random.RandomState(2).randn(h, p1 * p2)
        out = viz.powers_per_neuron_rows(W, group)
        n_irreps = len(group.irreps())
        assert out.shape == (h, n_irreps)
        for i in range(h):
            np.testing.assert_allclose(out[i], group.power_spectrum(W[i]), rtol=1e-10)


class TestPowersPerNeuronRows:
    """Tests for viz.powers_per_neuron_rows."""

    def test_matches_rowwise_group_power_spectrum(self):
        """Each row must match group.power_spectrum on that row."""
        group = DihedralGroup(3)
        h, n = 5, group.order
        W = np.random.RandomState(0).randn(h, n)
        out = viz.powers_per_neuron_rows(W, group)
        assert out.shape == (h, len(group.irreps()))
        for i in range(h):
            np.testing.assert_allclose(out[i], group.power_spectrum(W[i]), rtol=1e-10)


class TestGroupPowerSpectrum:
    """Tests for group.power_spectrum via Group objects."""

    def test_group_power_spectrum(self):
        """Test that group.power_spectrum returns the config powers exactly.

        power_spectrum normalises by |G|, so for a template built with
        ``custom_fourier(group, powers)`` the returned spectrum must match
        ``powers`` directly.
        """
        group = OctahedralGroup()
        powers = [0.0, 20.0, 20.0, 100.0, 0.0]
        tpl = template.custom_fourier(group, powers=powers)

        spectrum = group.power_spectrum(tpl)

        np.testing.assert_allclose(
            spectrum,
            powers,
            atol=1e-4,
            err_msg=f"Power spectrum mismatch: {spectrum} vs {powers}",
        )
