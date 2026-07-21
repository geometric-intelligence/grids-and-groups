import numpy as np


class IrreducibleRepresentation:
    """Stores the matrices of a single irreducible representation for every group element.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"trivial"``, ``"standard_2d"``).
    matrices : np.ndarray, shape (|G|, d, d)
        Representation matrices indexed by element index.
    """

    def __init__(self, name: str, matrices: np.ndarray):
        if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
            raise ValueError(f"matrices must have shape (n_elements, d, d), got {matrices.shape}")
        self._name = name
        self._matrices = np.asarray(matrices)
        self._dim = int(matrices.shape[1])

    @property
    def dim(self) -> int:
        """Dimension of the irrep (d)."""
        return self._dim

    def __call__(self, element_index: int) -> np.ndarray:
        """Return the representation matrix for the given element index."""
        return self._matrices[element_index]

    def __repr__(self) -> str:
        return f"IrreducibleRepresentation(name={self._name!r}, dim={self._dim})"

    def __str__(self) -> str:
        return self._name


class LazyIrreducibleRepresentation:
    """Irrep whose matrices are computed on demand.

    This implements the same small interface as :class:`IrreducibleRepresentation`
    without storing a dense ``(|G|, d, d)`` array.  It is useful for induced
    representations of larger semidirect-product groups.
    """

    def __init__(self, name: str, dim: int, matrix_fn, cache_size: int = 0):
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self._name = name
        self._dim = int(dim)
        self._matrix_fn = matrix_fn
        self._cache_size = int(cache_size)
        self._cache: dict[int, np.ndarray] = {}

    @property
    def dim(self) -> int:
        """Dimension of the irrep (d)."""
        return self._dim

    def __call__(self, element_index: int) -> np.ndarray:
        """Return the representation matrix for the given element index."""
        element_index = int(element_index)
        if self._cache_size <= 0:
            return self._matrix_fn(element_index)

        if element_index not in self._cache:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[element_index] = self._matrix_fn(element_index)
        return self._cache[element_index]

    def __repr__(self) -> str:
        return f"LazyIrreducibleRepresentation(name={self._name!r}, dim={self._dim})"

    def __str__(self) -> str:
        return self._name
