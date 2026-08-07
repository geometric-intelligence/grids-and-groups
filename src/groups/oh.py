"""Proper octahedral rotation group (order 24, 5 irreps)."""

from __future__ import annotations

from itertools import permutations, product

import numpy as np

from src.groups.finite_group import irreps_from_regular, regular_representation
from src.groups.group import Group
from src.groups.irrep import IrreducibleRepresentation

_IRREP_NAMES = [
    "Octahedral|[irrep_0]:1",
    "Octahedral|[irrep_1]:3",
    "Octahedral|[irrep_-1]:3",
    "Octahedral|[irrep_2]:2",
    "Octahedral|[irrep_3]:1",
]

_ORDER = 24


class OctahedralGroup(Group):
    """The chiral octahedral rotation group (order 24)."""

    def __init__(self):
        self._rotation_matrices = _generate_rotations()
        self._cayley = _build_cayley_table(self._rotation_matrices)
        matrices = irreps_from_regular(
            self._cayley,
            dimension_order=(1, 3, 3, 2, 1),
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

    def rotation_matrix(self, element: int) -> np.ndarray:
        """Return the 3D rotation matrix represented by an element."""
        return self._rotation_matrices[int(element)].copy()


def _generate_rotations() -> list[np.ndarray]:
    """Generate all orientation-preserving signed permutation matrices."""
    rotations = []
    for permutation in permutations(range(3)):
        permutation_matrix = np.zeros((3, 3), dtype=np.int64)
        for row, column in enumerate(permutation):
            permutation_matrix[row, column] = 1
        for signs in product((-1, 1), repeat=3):
            matrix = np.diag(signs) @ permutation_matrix
            if round(np.linalg.det(matrix)) == 1:
                rotations.append(matrix)

    identity = np.eye(3, dtype=np.int64)
    unique = {tuple(matrix.ravel()): matrix for matrix in rotations}
    rest = [
        matrix
        for _, matrix in sorted(unique.items())
        if not np.array_equal(matrix, identity)
    ]
    result = [identity, *rest]
    if len(result) != _ORDER:
        raise RuntimeError(f"Expected {_ORDER} rotations, got {len(result)}")
    return result


def _build_cayley_table(rotations: list[np.ndarray]) -> np.ndarray:
    index = {tuple(matrix.ravel()): i for i, matrix in enumerate(rotations)}
    return np.asarray(
        [
            [index[tuple((left @ right).ravel())] for right in rotations]
            for left in rotations
        ],
        dtype=np.int64,
    )
