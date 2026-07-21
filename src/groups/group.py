from abc import ABC, abstractmethod

import numpy as np


class Group(ABC):
    """Abstract Base Class for Finite Groups in Group-AGF.

    Subclasses must implement ``order``, ``elements``, ``irreps``, and
    ``regular_rep``.  Irreps may be dense objects or lazy objects, as long
    as they expose ``.dim`` and ``__call__(element_index)``.  The Fourier
    analysis methods (``fourier``, ``inverse_fourier``, ``power_spectrum``)
    are provided as concrete implementations that rely solely on that small
    interface.
    """

    @property
    @abstractmethod
    def order(self) -> int:
        """Cardinality |G| of the group."""

    @abstractmethod
    def elements(self) -> list[int]:
        """List of element indices ``[0, 1, ..., |G|-1]``."""

    @abstractmethod
    def irreps(self) -> list:
        """List of :class:`IrreducibleRepresentation` objects."""

    @abstractmethod
    def regular_rep(self) -> np.ndarray:
        """All regular-representation (permutation) matrices.

        Returns
        -------
        np.ndarray, shape (|G|, |G|, |G|)
            ``result[i]`` is the |G| x |G| permutation matrix for element *i*.
        """

    def fourier(self, signal: np.ndarray) -> list[np.ndarray]:
        """Group Fourier Transform.

        For each irrep rho, compute the Fourier coefficient:
            hat_x[rho] = sum_{g in G} x[g] * rho(g).conj().T

        Parameters
        ----------
        signal : np.ndarray, shape (|G|,)

        Returns
        -------
        list of np.ndarray, each of shape (d_rho, d_rho)
        """
        irreps = self.irreps()
        fourier_coefs = []
        for irrep in irreps:
            coef = sum(signal[i_g] * irrep(i_g).conj().T for i_g in range(self.order))
            fourier_coefs.append(coef)
        return fourier_coefs

    def inverse_fourier(self, fourier_coefs: list[np.ndarray]) -> np.ndarray:
        """Inverse Group Fourier Transform.

        x(g) = 1/|G| * sum_{rho} dim(rho) * Tr(rho(g) @ hat_x[rho])

        Parameters
        ----------
        fourier_coefs : list of np.ndarray, each of shape (d_rho, d_rho)

        Returns
        -------
        np.ndarray, shape (|G|,)
        """
        irreps = self.irreps()
        n = self.order

        def _at_element(g):
            return (1.0 / n) * sum(
                irrep.dim * np.trace(irrep(g) @ fourier_coefs[i]) for i, irrep in enumerate(irreps)
            )

        return np.array([_at_element(g) for g in range(n)])

    def power_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """Group power spectrum (normalised by ``|G|``).

        For each irrep rho the power is::

            P(rho) = dim(rho) / |G| * Tr( hat_x(rho)^H  @  hat_x(rho) )

        where ``hat_x`` uses the un-normalised Fourier convention
        ``hat_x = sum_g x(g) rho(g)^H``.  The ``1/|G|`` factor ensures
        that the returned values match the per-irrep powers passed to
        :func:`template.custom_fourier`.

        Parameters
        ----------
        signal : np.ndarray, shape (|G|,)

        Returns
        -------
        np.ndarray, shape (n_irreps,)
        """
        fourier_coefs = self.fourier(signal)
        irreps = self.irreps()
        ps = np.zeros(len(irreps))
        for i, irrep in enumerate(irreps):
            fc = fourier_coefs[i]
            ps[i] = np.real(irrep.dim * np.trace(fc.conj().T @ fc)) / self.order
        return ps
