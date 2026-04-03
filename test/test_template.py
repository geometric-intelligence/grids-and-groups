"""Tests for src.template module."""

import numpy as np
import pytest

import src.template as template


class TestOneHot:
    """Tests for template.one_hot function."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        p = 7
        tpl = template.one_hot(p)

        assert tpl.shape == (p,), f"Expected shape ({p},), got {tpl.shape}"

    def test_mean_centered(self):
        """Test that the template is mean-centered."""
        p = 10
        tpl = template.one_hot(p)

        np.testing.assert_allclose(tpl.mean(), 0, atol=1e-10)

    def test_has_spike(self):
        """Test that template has a spike at index 1."""
        p = 5
        tpl = template.one_hot(p)

        zeroth_freq = 10 / p
        expected_spike_value = 10 - zeroth_freq

        np.testing.assert_allclose(tpl[1], expected_spike_value, rtol=1e-5)


class TestFixedCn:
    """Tests for template.fixed_cn function."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        group_size = 8
        powers = [0.0, 50.0, 30.0, 20.0, 10.0]

        tpl = template.fixed_cn(group_size, powers)

        assert tpl.shape == (group_size,), f"Expected shape ({group_size},), got {tpl.shape}"

    def test_mean_centered(self):
        """Test that the template is mean-centered."""
        group_size = 10
        n_modes = group_size // 2 + 1
        powers = [0.0] + [5.0] * (n_modes - 1)

        tpl = template.fixed_cn(group_size, powers)

        np.testing.assert_allclose(tpl.mean(), 0, atol=1e-10)

    def test_real_valued(self):
        """Test that the template is real-valued."""
        group_size = 8
        powers = [0.0, 50.0, 30.0, 20.0, 10.0]

        tpl = template.fixed_cn(group_size, powers)

        assert np.isreal(tpl).all()


class TestFixedCnxcn:
    """Tests for template.fixed_cnxcn function."""

    def test_output_shape(self):
        """Test that output shape is correct (flattened)."""
        p1 = 6
        p2 = 6
        powers = [50.0, 30.0, 20.0]

        tpl = template.fixed_cnxcn(p1, p2, powers)

        expected_size = p1 * p2
        assert tpl.shape == (expected_size,), f"Expected shape ({expected_size},), got {tpl.shape}"

    def test_mean_centered(self):
        """Test that the template is mean-centered."""
        p1 = 5
        p2 = 5
        powers = [50.0, 30.0]

        tpl = template.fixed_cnxcn(p1, p2, powers)

        np.testing.assert_allclose(tpl.mean(), 0, atol=1e-6)

    def test_real_valued(self):
        """Test that the template is real-valued."""
        p1 = 4
        p2 = 4
        powers = [50.0]

        tpl = template.fixed_cnxcn(p1, p2, powers)

        assert np.isreal(tpl).all()


class TestFixedGroup:
    """Tests for template.fixed_group function."""

    @pytest.fixture
    def dihedral_group(self):
        """Create a DihedralGroup for testing."""
        from src.groups import DihedralGroup

        return DihedralGroup(N=3)

    def test_output_shape(self, dihedral_group):
        """Test that output shape matches group order."""
        group_order = dihedral_group.order
        num_irreps = len(list(dihedral_group.irreps()))
        powers = [1.0] * num_irreps

        tpl = template.fixed_group(dihedral_group, powers)

        assert tpl.shape == (group_order,), f"Expected shape ({group_order},), got {tpl.shape}"

    def test_mean_centered(self, dihedral_group):
        """Test that the template is mean-centered."""
        num_irreps = len(list(dihedral_group.irreps()))
        powers = [1.0] * num_irreps

        tpl = template.fixed_group(dihedral_group, powers)

        np.testing.assert_allclose(tpl.mean(), 0, atol=1e-10)

    def test_wrong_num_powers_error(self, dihedral_group):
        """Test that mismatched number of powers raises error."""
        wrong_num_powers = [1.0, 2.0]

        with pytest.raises(AssertionError):
            template.fixed_group(dihedral_group, wrong_num_powers)

    def test_real_valued(self, dihedral_group):
        """Test that the template is real-valued."""
        num_irreps = len(list(dihedral_group.irreps()))
        powers = [1.0] * num_irreps

        tpl = template.fixed_group(dihedral_group, powers)

        assert np.isreal(tpl).all()
