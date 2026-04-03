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
            raise ValueError(
                f"matrices must have shape (n_elements, d, d), got {matrices.shape}"
            )
        self._name = name
        self._matrices = np.asarray(matrices, dtype=np.float64)
        self._size = int(matrices.shape[1])

    @property
    def size(self) -> int:
        """Dimension of the irrep (d)."""
        return self._size

    def __call__(self, element_index: int) -> np.ndarray:
        """Return the representation matrix for the given element index."""
        return self._matrices[element_index]

    def __repr__(self) -> str:
        return f"IrreducibleRepresentation(name={self._name!r}, size={self._size})"

    def __str__(self) -> str:
        return self._name
