"""Tests for discrete-SE(2) geometry helpers."""

import numpy as np
import pytest

from src.geometry.discrete_se2.core import (
    lattice_path_segments,
    periodic_spatial_autocorrelation,
)


def test_lattice_path_segments_splits_wrapped_offset_seam():
    points = np.asarray([(7, 2), (8, 2), (9, 2), (0, 2), (1, 2)])

    segments = lattice_path_segments(points, 10, mode="offset")

    assert [len(segment) for segment in segments] == [2, 3]
    for segment in segments:
        if len(segment) > 1:
            assert np.all(np.linalg.norm(np.diff(segment, axis=0), axis=1) <= 1.5)


def test_lattice_path_segments_validates_step_length():
    with pytest.raises(ValueError, match="finite and positive"):
        lattice_path_segments(np.asarray([(0, 0)]), 5, maximum_step_length=0)


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
