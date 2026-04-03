"""Product cyclic group C_{p1} x C_{p2} (order p1*p2).

The Fourier transform and power spectrum override the base-class
implementations to use ``np.fft.fft2`` / ``np.fft.rfft2``, which is
equivalent but faster.
"""

import numpy as np

from src.groups.group import Group
from src.groups.irrep import IrreducibleRepresentation


class ProductCyclicGroup(Group):
    """Product cyclic group C_{p1} x C_{p2} (order p1*p2).

    Parameters
    ----------
    p1, p2 : int
        Orders of the two cyclic factors.
    """

    def __init__(self, p1: int, p2: int):
        if p1 < 1 or p2 < 1:
            raise ValueError(f"p1, p2 must be >= 1, got ({p1}, {p2})")
        self._p1 = p1
        self._p2 = p2
        self._order = p1 * p2

    @property
    def order(self) -> int:
        return self._order

    def elements(self) -> list[int]:
        return list(range(self._order))

    def irreps(self) -> list[IrreducibleRepresentation]:
        p1, p2 = self._p1, self._p2
        n = self._order
        irreps = []
        for j1 in range(p1):
            for j2 in range(p2):
                mats = np.empty((n, 1, 1), dtype=np.complex128)
                for k1 in range(p1):
                    for k2 in range(p2):
                        idx = k1 * p2 + k2
                        phase = 2j * np.pi * (j1 * k1 / p1 + j2 * k2 / p2)
                        mats[idx, 0, 0] = np.exp(phase)
                name = f"C{p1}xC{p2}|[irrep_{j1},{j2}]:1"
                irreps.append(IrreducibleRepresentation(name, mats))
        return irreps

    def regular_rep(self) -> np.ndarray:
        p1, p2, n = self._p1, self._p2, self._order
        reg = np.zeros((n, n, n))
        for g in range(n):
            g1, g2 = divmod(g, p2)
            for h in range(n):
                h1, h2 = divmod(h, p2)
                i1, i2 = (g1 + h1) % p1, (g2 + h2) % p2
                i = i1 * p2 + i2
                reg[g, i, h] = 1.0
        return reg

    # ------------------------------------------------------------------
    # Fast FFT overrides
    # ------------------------------------------------------------------

    def fourier_2d(self, signal_2d: np.ndarray) -> np.ndarray:
        """2D DFT-based Fourier transform returning the full spectrum array."""
        return np.fft.fft2(signal_2d)

    def power_spectrum_2d(self, signal_2d: np.ndarray) -> np.ndarray:
        """2D power spectrum (full, not rfft2-reduced).

        Parameters
        ----------
        signal_2d : np.ndarray, shape (p1, p2)

        Returns
        -------
        np.ndarray, shape (p1, p2)
            Normalised by p1*p2.
        """
        ft = np.fft.fft2(signal_2d)
        return np.abs(ft) ** 2 / self._order

    def fourier(self, signal: np.ndarray) -> list[np.ndarray]:
        """Flat-signal group Fourier transform."""
        signal_2d = signal.reshape(self._p1, self._p2)
        ft = np.fft.fft2(signal_2d)
        return [np.array([[ft[j1, j2]]]) for j1 in range(self._p1) for j2 in range(self._p2)]

    def inverse_fourier(self, fourier_coefs: list[np.ndarray]) -> np.ndarray:
        """Inverse group Fourier transform."""
        spectrum = np.array([fc[0, 0] for fc in fourier_coefs]).reshape(self._p1, self._p2)
        return np.fft.ifft2(spectrum).real.ravel()

    def power_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """Flat-signal power spectrum (one value per irrep)."""
        signal_2d = signal.reshape(self._p1, self._p2)
        ft = np.fft.fft2(signal_2d)
        return (np.abs(ft) ** 2).ravel()
