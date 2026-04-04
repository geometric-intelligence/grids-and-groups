"""Cyclic group C_N (order N).

The Fourier transform and power spectrum override the base-class
implementations to use ``np.fft``, which is equivalent but faster.
"""

import numpy as np

from src.groups.group import Group
from src.groups.irrep import IrreducibleRepresentation


class CyclicGroup(Group):
    """Cyclic group C_N of order N.

    Parameters
    ----------
    N : int
        Order of the cyclic group (N >= 1).
    """

    def __init__(self, N: int):
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        self._N = N
        self._irreps = self._build_irreps()
        self._regular = self._build_regular_rep()

    @property
    def order(self) -> int:
        return self._N

    def elements(self) -> list[int]:
        return list(range(self._N))

    def irreps(self) -> list[IrreducibleRepresentation]:
        return list(self._irreps)

    def regular_rep(self) -> np.ndarray:
        return self._regular

    def fourier(self, signal: np.ndarray) -> list[np.ndarray]:
        """DFT-based group Fourier transform for cyclic groups."""
        return [irrep_mat for irrep_mat in [np.array([[coef]]) for coef in np.fft.fft(signal)]]

    def inverse_fourier(self, fourier_coefs: list[np.ndarray]) -> np.ndarray:
        """IDFT-based inverse group Fourier transform for cyclic groups."""
        spectrum = np.array([fc[0, 0] for fc in fourier_coefs])
        return np.fft.ifft(spectrum).real

    def power_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """Power spectrum via FFT for cyclic groups.

        Returns one power value per frequency bin (same count as irreps = N).
        """
        ft = np.fft.fft(signal)
        return np.abs(ft) ** 2 / self._N

    def _build_irreps(self) -> list[IrreducibleRepresentation]:
        N = self._N
        irreps = []
        for j in range(N):
            mats = np.empty((N, 1, 1), dtype=np.complex128)
            for k in range(N):
                mats[k, 0, 0] = np.exp(2j * np.pi * j * k / N)
            irreps.append(IrreducibleRepresentation(f"C{N}|[irrep_{j}]:1", mats))
        return irreps

    def _build_regular_rep(self) -> np.ndarray:
        N = self._N
        reg = np.zeros((N, N, N))
        for g in range(N):
            for h in range(N):
                i = (g + h) % N
                reg[g, i, h] = 1.0
        return reg
