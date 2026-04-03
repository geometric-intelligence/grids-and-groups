"""Icosahedral / A5 group (order 60, 5 irreps).

All irrep and regular-representation matrices are loaded from pre-computed
``.npy`` files in ``src/groups/data/``.
"""

from pathlib import Path

import numpy as np

from src.groups.group import Group
from src.groups.irrep import IrreducibleRepresentation

_DATA = Path(__file__).resolve().parent / "data"

_IRREP_NAMES = [
    "Icosahedral|[irrep_0]:1",
    "Icosahedral|[irrep_1]:3",
    "Icosahedral|[irrep_2]:5",
    "Icosahedral|[irrep_3]:3",
    "Icosahedral|[irrep_4]:4",
]

_N_IRREPS = 5
_ORDER = 60


class IcosahedralGroup(Group):
    """The icosahedral rotation group, isomorphic to A5 (order 60)."""

    def __init__(self):
        self._irreps = []
        for i in range(_N_IRREPS):
            mats = np.load(_DATA / f"a5_irrep_{i}.npy")
            self._irreps.append(IrreducibleRepresentation(_IRREP_NAMES[i], mats))
        self._regular = np.load(_DATA / "a5_regular_rep.npy")

    @property
    def order(self) -> int:
        return _ORDER

    def elements(self) -> list[int]:
        return list(range(_ORDER))

    def irreps(self) -> list[IrreducibleRepresentation]:
        return list(self._irreps)

    def regular_rep(self) -> np.ndarray:
        return self._regular
