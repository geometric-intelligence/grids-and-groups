"""Tests for the unified group-generic code paths.

Verifies that the unified functions (loss_plateau_predictions,
powers_per_neuron_rows, fixed_group) work correctly for CyclicGroup
and ProductCyclicGroup instances.

KEY FINDING: Group.power_spectrum returns |F[k]|^2 (unnormalized), while the
cyclic FFT code in power.py uses |F[k]|^2 / N with Hermitian folding.  The
difference is exactly a factor of |G| in total power.  The refactored code
uses group.power_spectrum everywhere and applies 1/|G| normalization where
needed (e.g. loss plateau predictions).

Run with: pytest test/test_refactor_equivalence.py -v
"""

import numpy as np
import pytest
import torch

from src.groups.cn import CyclicGroup
from src.groups.cnxcn import ProductCyclicGroup

# ── dataset.py ────────────────────────────────────────────────────────────

from src.dataset import GroupCompositionDataset


class TestDatasetGroupPath:
    """_build_group works correctly for CyclicGroup and ProductCyclicGroup."""

    def test_build_group_cn(self):
        n = 7
        group = CyclicGroup(N=n)
        np.random.seed(0)
        template = np.random.randn(n).astype(np.float32)

        np.random.seed(123)
        X, Y, seq = GroupCompositionDataset._build_group(
            template=template, k=2, group=group, mode="exhaustive"
        )

        assert X.shape == (n**2, 2, n)
        assert Y.shape == (n**2, n)
        assert seq.shape == (n**2, 2)

    def test_build_group_cnxcn(self):
        p1, p2 = 3, 3
        group = ProductCyclicGroup(p1=p1, p2=p2)
        np.random.seed(0)
        template = np.random.randn(p1 * p2).astype(np.float32)

        np.random.seed(123)
        X, Y, seq = GroupCompositionDataset._build_group(
            template=template, k=2, group=group, mode="exhaustive"
        )

        group_size = p1 * p2
        assert X.shape == (group_size**2, 2, group_size)
        assert Y.shape == (group_size**2, group_size)
        assert seq.shape == (group_size**2, 2)


# ── power.py normalization ───────────────────────────────────────────────

from src.power import (
    loss_plateau_predictions,
    powers_per_neuron_rows,
)


class TestPowerNormalization:
    """Document the |G| normalization difference between cyclic FFT and group power spectrum."""

    def test_power_spectrum_normalization_cn(self):
        """CyclicGroup.power_spectrum returns |G| times the normalized cyclic power."""
        n = 11
        group = CyclicGroup(N=n)
        np.random.seed(0)
        template = np.random.randn(n).astype(np.float32)

        group_power = group.power_spectrum(template)
        group_total = group_power.sum()

        norm_sq = np.sum(template ** 2)
        expected_total = n * norm_sq  # Parseval: sum |F[k]|^2 = N * ||x||^2
        assert np.isclose(group_total, expected_total, rtol=1e-4)

    def test_power_spectrum_normalization_cnxcn(self):
        """ProductCyclicGroup.power_spectrum returns |G| times the normalized power."""
        p1, p2 = 5, 5
        group = ProductCyclicGroup(p1=p1, p2=p2)
        np.random.seed(0)
        template = np.random.randn(p1 * p2).astype(np.float32)

        group_power = group.power_spectrum(template)
        group_total = group_power.sum()

        norm_sq = np.sum(template ** 2)
        expected_total = (p1 * p2) * norm_sq
        assert np.isclose(group_total, expected_total, rtol=1e-4)

    def test_loss_plateau_predictions_cn(self):
        """loss_plateau_predictions works correctly for CyclicGroup."""
        n = 11
        group = CyclicGroup(N=n)
        np.random.seed(0)
        template = np.random.randn(n).astype(np.float32)
        template -= template.mean()

        levels = loss_plateau_predictions(template, group)

        assert len(levels) > 0
        assert all(levels[i] >= levels[i + 1] for i in range(len(levels) - 1))

        # First level should approximate MSE of zero-prediction baseline
        norm_sq = np.sum(template ** 2)
        expected_initial = norm_sq / n
        assert np.isclose(levels[0], expected_initial, rtol=1e-3), (
            f"Initial plateau {levels[0]:.6f} vs expected {expected_initial:.6f}"
        )

    def test_powers_per_neuron_rows_cn(self):
        """powers_per_neuron_rows works correctly for CyclicGroup."""
        n = 11
        group = CyclicGroup(N=n)
        np.random.seed(0)
        W = np.random.randn(4, n)

        pw_group = powers_per_neuron_rows(W, group)

        assert pw_group.shape[0] == 4
        assert pw_group.shape[1] == len(group.irreps())
        # Each row's total power should be |G| * ||row||^2
        for i in range(4):
            expected_total = n * np.sum(W[i] ** 2)
            assert np.isclose(pw_group[i].sum(), expected_total, rtol=1e-4)


# ── template.py ──────────────────────────────────────────────────────────

from src.template import fixed_group


class TestTemplateFixedGroup:
    """fixed_group produces valid templates for CyclicGroup and ProductCyclicGroup."""

    @pytest.mark.parametrize("n", [5, 7, 11])
    def test_fixed_group_cn_produces_valid_template(self, n):
        """fixed_group produces a mean-centered template of the right shape for CyclicGroup."""
        group = CyclicGroup(N=n)
        powers = [0.0] + [1.0] * (len(group.irreps()) - 1)
        tpl = fixed_group(group, powers)
        assert tpl.shape == (n,)
        assert np.abs(tpl.mean()) < 1e-5

    @pytest.mark.parametrize("p1,p2", [(3, 3), (4, 5)])
    def test_fixed_group_cnxcn_produces_valid_template(self, p1, p2):
        """fixed_group produces a mean-centered template for ProductCyclicGroup."""
        group = ProductCyclicGroup(p1=p1, p2=p2)
        n_irreps = len(group.irreps())
        powers = [0.0] + [1.0] * (n_irreps - 1)
        tpl = fixed_group(group, powers)
        assert tpl.shape == (p1 * p2,)
        assert np.abs(tpl.mean()) < 1e-5
