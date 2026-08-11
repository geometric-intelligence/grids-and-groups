"""Tests for discrete-SE(2) geometry helpers."""

import numpy as np
import pytest

from src.geometry.discrete_se2 import periodic_spatial_autocorrelation


def test_periodic_spatial_autocorrelation_is_translation_invariant():
    rng = np.random.default_rng(0)
    values = rng.standard_normal((7, 7))

    original = periodic_spatial_autocorrelation(values)
    shifted = periodic_spatial_autocorrelation(np.roll(values, (2, -3), axis=(0, 1)))

    np.testing.assert_allclose(shifted, original, atol=1e-12)
    assert original[values.shape[0] // 2, values.shape[1] // 2] == pytest.approx(1.0)


def test_periodic_spatial_autocorrelation_of_constant_map_is_zero():
    result = periodic_spatial_autocorrelation(np.ones((5, 5)))
    np.testing.assert_array_equal(result, np.zeros((5, 5)))


def test_periodic_spatial_autocorrelation_requires_square_map():
    with pytest.raises(ValueError, match="square two-dimensional"):
        periodic_spatial_autocorrelation(np.ones((3, 4)))
