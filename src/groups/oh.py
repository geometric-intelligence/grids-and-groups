"""Octahedral group (order 24, 5 irreps).

All irrep and regular-representation matrices are loaded from pre-computed
``.npy`` files in ``src/groups/data/``.
"""

from pathlib import Path

import numpy as np

from src.groups.group import Group
from src.groups.irrep import IrreducibleRepresentation

_DATA = Path(__file__).resolve().parent / "data"

_IRREP_NAMES = [
    "Octahedral|[irrep_0]:1",
    "Octahedral|[irrep_1]:3",
    "Octahedral|[irrep_-1]:3",
    "Octahedral|[irrep_2]:2",
    "Octahedral|[irrep_3]:1",
]

_N_IRREPS = 5
_ORDER = 24


class OctahedralGroup(Group):
    """The chiral octahedral rotation group (order 24)."""

    def __init__(self):
        self._irreps = []
        for i in range(_N_IRREPS):
            mats = np.load(_DATA / f"oh_irrep_{i}.npy")
            self._irreps.append(IrreducibleRepresentation(_IRREP_NAMES[i], mats))
        self._regular = np.load(_DATA / "oh_regular_rep.npy")

    @property
    def order(self) -> int:
        return _ORDER

    def elements(self) -> list[int]:
        return list(range(_ORDER))

    def irreps(self) -> list[IrreducibleRepresentation]:
        return list(self._irreps)

    def regular_rep(self) -> np.ndarray:
        return self._regular
