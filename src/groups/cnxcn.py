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

    @property
    def p1(self) -> int:
        """Order of the first cyclic factor."""
        return self._p1

    @property
    def p2(self) -> int:
        """Order of the second cyclic factor."""
        return self._p2

    def elements(self) -> list[int]:
        return list(range(self._order))

    def encode(self, x: int, y: int) -> int:
        """Encode a pair of cyclic coordinates as one group index."""
        return (int(x) % self._p1) * self._p2 + (int(y) % self._p2)

    def decode(self, element: int) -> tuple[int, int]:
        """Decode a group index into its two cyclic coordinates."""
        element = int(element)
        if not 0 <= element < self._order:
            raise ValueError(f"element must lie in [0, {self._order}), got {element}")
        return divmod(element, self._p2)

    def identity(self) -> int:
        """Return the identity element index."""
        return self.encode(0, 0)

    def compose(self, left: int, right: int) -> int:
        """Return the product ``left * right``."""
        left_x, left_y = self.decode(left)
        right_x, right_y = self.decode(right)
        return self.encode(left_x + right_x, left_y + right_y)

    def inverse(self, element: int) -> int:
        """Return the inverse element index."""
        x, y = self.decode(element)
        return self.encode(-x, -y)

    def action_permutation(self, element: int) -> np.ndarray:
        """Return source indices for the regular left action by ``element``."""
        shift_x, shift_y = self.decode(element)
        x, y = np.meshgrid(
            np.arange(self._p1),
            np.arange(self._p2),
            indexing="ij",
        )
        return (((x - shift_x) % self._p1) * self._p2 + (y - shift_y) % self._p2).ravel()

    def left_action(self, element: int, signal: np.ndarray) -> np.ndarray:
        """Apply the regular left action without materializing permutation matrices."""
        signal = np.asarray(signal)
        if signal.shape != (self._order,):
            raise ValueError(
                f"signal must have shape ({self._order},), got {signal.shape}"
            )
        return signal[self.action_permutation(element)]

    def cumulative_product(self, sequence) -> int:
        """Compose a sequence in the order used by recurrent left actions."""
        cumulative = self.identity()
        for element in sequence:
            cumulative = self.compose(int(element), cumulative)
        return cumulative

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
        return (np.abs(ft) ** 2 / self._order).ravel()
