"""Icosahedral rotation group, represented as A5 (order 60, 5 irreps)."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from src.groups.finite_group import irreps_from_regular, regular_representation
from src.groups.group import Group
from src.groups.irrep import IrreducibleRepresentation

_IRREP_NAMES = [
    "Icosahedral|[irrep_0]:1",
    "Icosahedral|[irrep_1]:3",
    "Icosahedral|[irrep_2]:5",
    "Icosahedral|[irrep_3]:3",
    "Icosahedral|[irrep_4]:4",
]

_ORDER = 60


class IcosahedralGroup(Group):
    """The icosahedral rotation group, isomorphic to A5 (order 60)."""

    def __init__(self):
        self._permutations = _generate_even_permutations()
        self._cayley = _build_cayley_table(self._permutations)
        matrices = irreps_from_regular(
            self._cayley,
            dimension_order=(1, 3, 5, 3, 4),
        )
        self._irreps = [
            IrreducibleRepresentation(name, irrep_matrices)
            for name, irrep_matrices in zip(_IRREP_NAMES, matrices)
        ]
        self._regular: np.ndarray | None = None

    @property
    def order(self) -> int:
        return _ORDER

    def elements(self) -> list[int]:
        return list(range(_ORDER))

    def irreps(self) -> list[IrreducibleRepresentation]:
        return list(self._irreps)

    def regular_rep(self) -> np.ndarray:
        if self._regular is None:
            self._regular = regular_representation(self._cayley)
        return self._regular

    def compose(self, left: int, right: int) -> int:
        """Return the index of the product ``left * right``."""
        return int(self._cayley[int(left), int(right)])

    def inverse(self, element: int) -> int:
        """Return the inverse element index."""
        matches = np.flatnonzero(
            (self._cayley[int(element)] == 0) & (self._cayley[:, int(element)] == 0)
        )
        return int(matches[0])

    def permutation(self, element: int) -> tuple[int, ...]:
        """Return an element as an even permutation of five points."""
        return self._permutations[int(element)]


def _generate_even_permutations() -> list[tuple[int, ...]]:
    elements = [
        permutation
        for permutation in permutations(range(5))
        if _permutation_parity(permutation) == 0
    ]
    identity = tuple(range(5))
    result = [identity, *sorted(element for element in elements if element != identity)]
    if len(result) != _ORDER:
        raise RuntimeError(f"Expected {_ORDER} even permutations, got {len(result)}")
    return result


def _permutation_parity(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return inversions % 2


def _build_cayley_table(elements: list[tuple[int, ...]]) -> np.ndarray:
    index = {element: i for i, element in enumerate(elements)}
    return np.asarray(
        [
            [index[tuple(left[right[position]] for position in range(5))] for right in elements]
            for left in elements
        ],
        dtype=np.int64,
    )
