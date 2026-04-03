"""Dihedral group D_N (order 2N).

Elements are indexed 0..2N-1:
  - 0..N-1   : rotations r^0, r^1, ..., r^{N-1}
  - N..2N-1  : r^0*s, r^1*s, ..., r^{N-1}*s  (reflection composed with rotation)

Irrep ordering matches escnn.group.DihedralGroup exactly.
"""

import numpy as np

from src.groups.group import Group
from src.groups.irrep import IrreducibleRepresentation


class DihedralGroup(Group):
    """Dihedral group D_N (order 2N), parameterised by N >= 2."""

    def __init__(self, N: int):
        if N < 2:
            raise ValueError(f"N must be >= 2, got {N}")
        self._N = N
        self._order = 2 * N
        self._irreps = self._build_irreps()
        self._regular = self._build_regular_rep()

    @property
    def order(self) -> int:
        return self._order

    def elements(self) -> list[int]:
        return list(range(self._order))

    def irreps(self) -> list[IrreducibleRepresentation]:
        return list(self._irreps)

    def regular_rep(self) -> np.ndarray:
        return self._regular

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_irreps(self) -> list[IrreducibleRepresentation]:
        N = self._N
        n_elems = self._order
        irreps = []

        trivial = np.ones((n_elems, 1, 1))
        irreps.append(IrreducibleRepresentation(
            f"D{N}|[irrep_0,0]:1", trivial
        ))

        sign = np.ones((n_elems, 1, 1))
        sign[N:] = -1.0
        irreps.append(IrreducibleRepresentation(
            f"D{N}|[irrep_1,0]:1", sign
        ))

        n_2d = (N - 1) // 2
        for j in range(1, n_2d + 1):
            mats = np.empty((n_elems, 2, 2))
            for k in range(N):
                angle = 2.0 * np.pi * j * k / N
                c, s = np.cos(angle), np.sin(angle)
                mats[k] = [[c, -s], [s, c]]
                mats[N + k] = [[c, s], [s, -c]]
            irreps.append(IrreducibleRepresentation(
                f"D{N}|[irrep_1,{j}]:2", mats
            ))

        if N % 2 == 0:
            half = N // 2

            rep_1_half = np.empty((n_elems, 1, 1))
            for k in range(N):
                rep_1_half[k, 0, 0] = (-1.0) ** k
                rep_1_half[N + k, 0, 0] = -((-1.0) ** k)
            irreps.append(IrreducibleRepresentation(
                f"D{N}|[irrep_1,{half}]:1", rep_1_half
            ))

            rep_0_half = np.empty((n_elems, 1, 1))
            for k in range(N):
                rep_0_half[k, 0, 0] = (-1.0) ** k
                rep_0_half[N + k, 0, 0] = (-1.0) ** k
            irreps.append(IrreducibleRepresentation(
                f"D{N}|[irrep_0,{half}]:1", rep_0_half
            ))

        return irreps

    def _cayley(self, g: int, h: int) -> int:
        """Index of the product g * h in the group."""
        N = self._N
        g_rot = g < N
        h_rot = h < N
        a = g if g_rot else g - N
        b = h if h_rot else h - N

        if g_rot and h_rot:
            return (a + b) % N
        if g_rot and not h_rot:
            return N + (a + b) % N
        if not g_rot and h_rot:
            return N + (a - b) % N
        return (a - b) % N

    def _build_regular_rep(self) -> np.ndarray:
        n = self._order
        reg = np.zeros((n, n, n))
        for g in range(n):
            for h in range(n):
                i = self._cayley(g, h)
                reg[g, i, h] = 1.0
        return reg
