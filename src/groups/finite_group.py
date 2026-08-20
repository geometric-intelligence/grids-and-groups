"""Utilities for constructing representations of small finite groups."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def regular_representation(cayley: np.ndarray) -> np.ndarray:
    """Build the left-regular representation from a Cayley table."""
    cayley = np.asarray(cayley, dtype=np.int64)
    _validate_cayley_table(cayley)
    order = cayley.shape[0]
    matrices = np.zeros((order, order, order), dtype=np.float64)
    for element in range(order):
        matrices[element, cayley[element], np.arange(order)] = 1.0
    return matrices


def irreps_from_regular(
    cayley: np.ndarray,
    dimension_order: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """Decompose the regular representation into inequivalent unitary irreps.

    A generic Hermitian element of the right group algebra commutes with the
    left-regular action. Its eigenspaces therefore isolate irreducible copies
    of the left action. Repeated copies are removed by comparing characters.
    """
    cayley = np.asarray(cayley, dtype=np.int64)
    _validate_cayley_table(cayley)
    order = cayley.shape[0]
    inverse = _inverses(cayley)

    left = np.zeros((order, order, order), dtype=np.complex128)
    right = np.zeros_like(left)
    columns = np.arange(order)
    for element in range(order):
        left[element, cayley[element], columns] = 1.0
        right[element, cayley[:, inverse[element]], columns] = 1.0

    rng = np.random.default_rng(1729 + order)
    operator = np.zeros((order, order), dtype=np.complex128)
    for element in range(order):
        coefficient = rng.normal() + 1j * rng.normal()
        operator += coefficient * right[element] + coefficient.conjugate() * right[element].conj().T
    operator = 0.5 * (operator + operator.conj().T)

    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    clusters: list[list[int]] = []
    for index, value in enumerate(eigenvalues):
        if not clusters or abs(value - eigenvalues[clusters[-1][-1]]) > 1e-8:
            clusters.append([index])
        else:
            clusters[-1].append(index)

    irreps: list[np.ndarray] = []
    characters: list[np.ndarray] = []
    for cluster in clusters:
        basis = eigenvectors[:, cluster]
        matrices = np.asarray([basis.conj().T @ left[element] @ basis for element in range(order)])
        character = np.trace(matrices, axis1=1, axis2=2)
        if any(np.allclose(character, seen, atol=1e-7) for seen in characters):
            continue
        characters.append(character)
        irreps.append(matrices)

    if sum(matrices.shape[1] ** 2 for matrices in irreps) != order:
        raise ValueError("Regular-representation decomposition is incomplete")

    if dimension_order is not None:
        irreps = _order_by_dimensions(irreps, dimension_order)
    return irreps


def _validate_cayley_table(cayley: np.ndarray) -> None:
    if cayley.ndim != 2 or cayley.shape[0] != cayley.shape[1]:
        raise ValueError(f"Cayley table must be square, got {cayley.shape}")
    order = cayley.shape[0]
    expected = np.arange(order)
    if not np.array_equal(cayley[0], expected) or not np.array_equal(cayley[:, 0], expected):
        raise ValueError("Cayley table must use index 0 for the identity")
    if np.any(cayley < 0) or np.any(cayley >= order):
        raise ValueError("Cayley table contains an invalid element index")


def _inverses(cayley: np.ndarray) -> np.ndarray:
    order = cayley.shape[0]
    inverse = np.empty(order, dtype=np.int64)
    for element in range(order):
        matches = np.flatnonzero((cayley[element] == 0) & (cayley[:, element] == 0))
        if len(matches) != 1:
            raise ValueError(f"Element {element} does not have a unique inverse")
        inverse[element] = matches[0]
    return inverse


def _order_by_dimensions(
    irreps: list[np.ndarray], dimension_order: Sequence[int]
) -> list[np.ndarray]:
    remaining = list(irreps)
    ordered = []
    for dimension in dimension_order:
        match = next(
            (index for index, matrices in enumerate(remaining) if matrices.shape[1] == dimension),
            None,
        )
        if match is None:
            actual = [matrices.shape[1] for matrices in irreps]
            raise ValueError(f"Expected irrep dimensions {list(dimension_order)}, got {actual}")
        ordered.append(remaining.pop(match))
    if remaining:
        actual = [matrices.shape[1] for matrices in irreps]
        raise ValueError(f"Expected irrep dimensions {list(dimension_order)}, got {actual}")
    return ordered
