"""Tests for src.power module."""

import numpy as np

import src.power as power
import src.template as template
from src.groups import DihedralGroup, OctahedralGroup


class TestGetPower1D:
    """Tests for power.get_power_1d function."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        p = 10
        signal = np.random.randn(p)

        pwr, freqs = power.get_power_1d(signal)

        expected_len = p // 2 + 1
        assert pwr.shape == (expected_len,), f"power shape mismatch: {pwr.shape}"
        assert freqs.shape == (expected_len,), f"freqs shape mismatch: {freqs.shape}"

    def test_parseval_theorem(self):
        """Test that Parseval's theorem holds (total power ~ norm squared)."""
        p = 16
        signal = np.random.randn(p)

        pwr, _ = power.get_power_1d(signal)
        total_power = np.sum(pwr)
        norm_squared = np.linalg.norm(signal) ** 2

        np.testing.assert_allclose(
            total_power, norm_squared, rtol=1e-6, err_msg="Parseval's theorem violated"
        )

    def test_parseval_theorem_odd_length(self):
        """Test Parseval's theorem for odd-length signals."""
        p = 15
        signal = np.random.randn(p)

        pwr, _ = power.get_power_1d(signal)
        total_power = np.sum(pwr)
        norm_squared = np.linalg.norm(signal) ** 2

        np.testing.assert_allclose(
            total_power,
            norm_squared,
            rtol=1e-6,
            err_msg="Parseval's theorem violated for odd length",
        )

    def test_dc_component(self):
        """Test that DC component power is correct for constant signal."""
        p = 8
        constant_value = 3.0
        signal = np.full(p, constant_value)

        pwr, freqs = power.get_power_1d(signal)

        expected_dc_power = constant_value**2 * p
        np.testing.assert_allclose(pwr[0], expected_dc_power, rtol=1e-6)

        assert np.allclose(pwr[1:], 0, atol=1e-10)


class TestGetPower2D:
    """Tests for power.get_power_2d function."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        M, N = 8, 10
        signal = np.random.randn(M, N)

        freqs_u, freqs_v, pwr = power.get_power_2d(signal)

        expected_power_shape = (M, N // 2 + 1)
        assert pwr.shape == expected_power_shape, f"power shape mismatch: {pwr.shape}"
        assert freqs_u.shape == (M,), f"freqs_u shape mismatch: {freqs_u.shape}"
        assert freqs_v.shape == (N // 2 + 1,), f"freqs_v shape mismatch: {freqs_v.shape}"

    def test_output_shape_no_freq(self):
        """Test output when no_freq=True."""
        M, N = 8, 10
        signal = np.random.randn(M, N)

        result = power.get_power_2d(signal, no_freq=True)

        expected_shape = (M, N // 2 + 1)
        assert result.shape == expected_shape

    def test_parseval_theorem(self):
        """Test that Parseval's theorem holds."""
        M, N = 12, 12
        signal = np.random.randn(M, N)

        pwr = power.get_power_2d(signal, no_freq=True)
        total_power = np.sum(pwr)
        norm_squared = np.linalg.norm(signal) ** 2

        np.testing.assert_allclose(
            total_power, norm_squared, rtol=1e-6, err_msg="Parseval's theorem violated for 2D"
        )

    def test_parseval_theorem_rectangular(self):
        """Test Parseval's theorem for rectangular arrays."""
        M, N = 7, 11
        signal = np.random.randn(M, N)

        pwr = power.get_power_2d(signal, no_freq=True)
        total_power = np.sum(pwr)
        norm_squared = np.linalg.norm(signal) ** 2

        np.testing.assert_allclose(
            total_power,
            norm_squared,
            rtol=1e-6,
            err_msg="Parseval's theorem violated for rectangular array",
        )


class TestTopkTemplateFreqs1D:
    """Tests for power.topk_template_freqs_1d function."""

    def test_returns_top_k(self):
        """Test that function returns exactly K frequencies."""
        p = 16
        K = 3
        tpl = np.random.randn(p)

        top_freqs = power.topk_template_freqs_1d(tpl, K)

        assert len(top_freqs) == K, f"Expected {K} frequencies, got {len(top_freqs)}"

    def test_returns_sorted_by_power(self):
        """Test that frequencies are sorted by descending power."""
        p = 16
        K = 5
        tpl = np.random.randn(p)

        top_freqs = power.topk_template_freqs_1d(tpl, K)
        pwr, _ = power.get_power_1d(tpl)

        returned_powers = [pwr[f] for f in top_freqs]

        assert returned_powers == sorted(returned_powers, reverse=True)

    def test_empty_for_zero_signal(self):
        """Test that zero signal with high min_power returns empty list."""
        p = 8
        tpl = np.zeros(p)

        top_freqs = power.topk_template_freqs_1d(tpl, K=3, min_power=1e-10)

        assert top_freqs == []

    def test_handles_k_larger_than_freqs(self):
        """Test behavior when K is larger than available frequencies."""
        p = 6
        K = 10
        tpl = np.random.randn(p)

        top_freqs = power.topk_template_freqs_1d(tpl, K)

        assert len(top_freqs) <= p // 2 + 1


class TestTopkTemplateFreqs:
    """Tests for power.topk_template_freqs function (2D)."""

    def test_returns_top_k(self):
        """Test that function returns exactly K frequency pairs."""
        p1, p2 = 8, 8
        K = 3
        tpl = np.random.randn(p1, p2)

        top_freqs = power.topk_template_freqs(tpl, K)

        assert len(top_freqs) == K, f"Expected {K} frequency pairs, got {len(top_freqs)}"

    def test_returns_tuples(self):
        """Test that returned values are (kx, ky) tuples."""
        p1, p2 = 8, 8
        K = 3
        tpl = np.random.randn(p1, p2)

        top_freqs = power.topk_template_freqs(tpl, K)

        for freq in top_freqs:
            assert isinstance(freq, tuple), f"Expected tuple, got {type(freq)}"
            assert len(freq) == 2, f"Expected 2-tuple, got {len(freq)}-tuple"

    def test_empty_for_zero_signal(self):
        """Test that zero signal returns empty list."""
        p1, p2 = 6, 6
        tpl = np.zeros((p1, p2))

        top_freqs = power.topk_template_freqs(tpl, K=3, min_power=1e-10)

        assert top_freqs == []


class TestPowersPerNeuronRowsCyclicGroup:
    """Tests for power.powers_per_neuron_rows with CyclicGroup."""

    def test_shape_cn(self):
        """Each row must produce the correct power shape for CyclicGroup."""
        from src.groups.cn import CyclicGroup

        p = 10
        h = 3
        group = CyclicGroup(N=p)
        W = np.random.RandomState(1).randn(h, p)
        out = power.powers_per_neuron_rows(W, group)
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
        out = power.powers_per_neuron_rows(W, group)
        n_irreps = len(group.irreps())
        assert out.shape == (h, n_irreps)
        for i in range(h):
            np.testing.assert_allclose(out[i], group.power_spectrum(W[i]), rtol=1e-10)


class TestPowersPerNeuronRows:
    """Tests for power.powers_per_neuron_rows."""

    def test_matches_rowwise_group_power_spectrum(self):
        """Each row must match group.power_spectrum on that row."""
        group = DihedralGroup(3)
        h, n = 5, group.order
        W = np.random.RandomState(0).randn(h, n)
        out = power.powers_per_neuron_rows(W, group)
        assert out.shape == (h, len(group.irreps()))
        for i in range(h):
            np.testing.assert_allclose(out[i], group.power_spectrum(W[i]), rtol=1e-10)


class TestGroupPowerSpectrum:
    """Tests for group.power_spectrum via Group objects."""

    def test_group_power_spectrum(self):
        """Test that group.power_spectrum computes correct power from a template.

        power_spectrum returns dim(rho) * Tr(hat_x^H @ hat_x) without dividing
        by |G|, so the expected values are the input powers scaled by group.order.
        """
        group = OctahedralGroup()
        powers = [0.0, 20.0, 20.0, 100.0, 0.0]
        tpl = template.fixed_group(group, powers=powers)

        spectrum = group.power_spectrum(tpl)

        expected = [p * group.order for p in powers]
        np.testing.assert_allclose(
            spectrum,
            expected,
            atol=1e-4,
            err_msg=f"Power spectrum mismatch: {spectrum} vs {expected}",
        )
