"""DiscreteSE2Group: Z_n² ⋊ C_m for m ∈ {1, 2, 3, 4, 6} (order mn²).

The semidirect product of the translation group Z_n × Z_n with the cyclic
rotation group C_m, where the semidirect product is defined by the group homomorphism
φ: C_m → Aut(Z_n²) given by φ(r^k) = A^k, where A is planar rotation matrix such that A^m = I
and A is modulo n.

Standard actions (crystallographic restriction):
    m = 1: identity               A = [[ 1,  0], [ 0,  1]]
    m = 2: 180° half-turn         A = [[-1,  0], [ 0, -1]]
    m = 3: 120° (triangular)      A = [[-1, -1], [ 1,  0]]
    m = 4: 90°  (square)          A = [[ 0, -1], [ 1,  0]]
    m = 6: 60°  (triangular)      A = [[ 0, -1], [ 1,  1]]

Composition rule:
    ((x1,y1), r^k_1) * ((x2,y2), r^k_2)
      = ((x1,y1) + A^{k_1}(x2,y2) mod n,  k_1+k_2 mod m)

Element indexing (0 .. mn²-1):
    index = k * n² + x * n + y
    where (x, y) ∈ Z_n² and k ∈ {0, 1, ..., m-1}.
"""

import math
import warnings

import numpy as np

from src.groups.group import Group
from src.groups.irrep import IrreducibleRepresentation

_REGULAR_REP_MAX_BYTES = 1_000_000_000

# Specify the standard action matrices for each m = 1, 2, 3, 4, 6.
_STANDARD_ACTIONS: dict[int, np.ndarray] = {
    1: np.array([[1, 0], [0, 1]], dtype=int),
    2: np.array([[-1, 0], [0, -1]], dtype=int),
    3: np.array([[-1, -1], [1, 0]], dtype=int),
    4: np.array([[0, -1], [1, 0]], dtype=int),
    6: np.array([[0, -1], [1, 1]], dtype=int),
}

_VALID_M = frozenset(_STANDARD_ACTIONS)


class DiscreteSE2Group(Group):
    """Z_n² ⋊ C_m: translations and rotations on a discrete lattice.

    Parameters
    ----------
    n : int
        Order of each cyclic translation factor (n >= 2).
    m : int
        Rotation order; must be one of {1, 2, 3, 4, 6}.
    """

    def __init__(self, n: int, m: int):
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        if m not in _VALID_M:
            raise ValueError(f"m must be in {sorted(_VALID_M)}, got {m}")

        self._n = n
        self._m = m
        self._order_val = m * n * n

        self._A: np.ndarray = _STANDARD_ACTIONS[m] % n
        self._rot_mats: list[np.ndarray] = (
            self._precompute_rotations()
        )  # A^k mod n for k = 0, ..., m-1

        actual_order = self._matrix_order()
        if actual_order is None:
            raise ValueError(
                f"Action matrix for m={m} does not satisfy A^m = I mod n={n}. "
                f"The semidirect product is not well-defined."
            )
        if actual_order != m:
            warnings.warn(
                f"Action matrix for m={m} has actual order {actual_order} mod n={n}. "
                f"The group is still well-defined but the C_{m} action is not faithful.",
                stacklevel=2,
            )

        self._regular: np.ndarray | None = None
        self._action_permutations: dict[int, np.ndarray] = {}
        self._A_dual: np.ndarray = self._compute_dual_action_matrix()
        self._irreps: list[IrreducibleRepresentation] = self._build_irreps()
        self._conjugate_pairs: list[list[int]] | None = None

    # ------------------------------------------------------------------
    # Group ABC interface
    # ------------------------------------------------------------------

    @property
    def order(self) -> int:
        return self._order_val

    @property
    def n(self) -> int:
        """Order of each cyclic translation factor."""
        return self._n

    @property
    def m(self) -> int:
        """Order of the cyclic rotation factor."""
        return self._m

    def elements(self) -> list[int]:
        return list(range(self._order_val))

    def irreps(self) -> list[IrreducibleRepresentation]:
        return list(self._irreps)

    def orbit_dict(self) -> dict[int, list[list[tuple[int, int]]]]:
        """Return translation-character orbits grouped by orbit size."""
        return self._compute_char_orbits()

    def conjugate_pairs(self) -> list[list[int]]:
        """Return indices of real irreps and complex-conjugate irrep pairs."""
        if self._conjugate_pairs is None:
            characters = [
                np.asarray([np.trace(irrep(g)) for g in self.elements()])
                for irrep in self._irreps
            ]
            processed: set[int] = set()
            pairs = []
            for index, character in enumerate(characters):
                if index in processed:
                    continue
                match = next(
                    (
                        candidate
                        for candidate, candidate_character in enumerate(characters)
                        if candidate not in processed
                        and np.allclose(
                            candidate_character,
                            character.conjugate(),
                            atol=1e-9,
                        )
                    ),
                    None,
                )
                if match is None:
                    raise ValueError(f"could not find conjugate irrep for index {index}")
                processed.add(index)
                processed.add(match)
                pairs.append([index] if index == match else sorted([index, match]))
            self._conjugate_pairs = pairs
        return [list(pair) for pair in self._conjugate_pairs]

    def regular_rep(self) -> np.ndarray:
        """Return dense left-regular representation matrices, built lazily.

        Most operations should use :meth:`left_action`, which applies the same
        permutation without allocating an ``(|G|, |G|, |G|)`` tensor.
        """
        required_bytes = self.order**3 * np.dtype(np.float32).itemsize
        if required_bytes > _REGULAR_REP_MAX_BYTES:
            required_gib = required_bytes / 1024**3
            raise MemoryError(
                f"regular_rep() would require {required_gib:.1f} GiB for shape "
                f"({self.order}, {self.order}, {self.order}). Use left_action() instead."
            )
        if self._regular is None:
            self._regular = self._build_regular_rep()
        return self._regular

    # ------------------------------------------------------------------
    # Element encoding
    # ------------------------------------------------------------------

    def _encode(self, x: int, y: int, r: int) -> int:
        """Bijective encoding: (x, y, r) → flat index."""
        n = self._n
        return r * n * n + x * n + y

    def _decode(self, idx: int) -> tuple[int, int, int]:
        """Bijective decoding: flat index → (x, y, r)."""
        n = self._n
        r, rem = divmod(idx, n * n)
        x, y = divmod(rem, n)
        return x, y, r

    def encode(self, x: int, y: int, r: int) -> int:
        """Encode a group element, reducing each coordinate modulo its order."""
        return self._encode(int(x) % self.n, int(y) % self.n, int(r) % self.m)

    def decode(self, idx: int) -> tuple[int, int, int]:
        """Decode a flat element index as ``(x, y, r)``."""
        idx = int(idx)
        if not 0 <= idx < self.order:
            raise ValueError(f"element index must be in [0, {self.order}), got {idx}")
        return self._decode(idx)

    def identity(self) -> int:
        """Return the identity element index."""
        return self.encode(0, 0, 0)

    def inverse(self, g: int) -> int:
        """Return the inverse of element ``g``."""
        x, y, r = self.decode(g)
        x_inv, y_inv = self._apply_rotation(-r, -x, -y)
        return self.encode(x_inv, y_inv, -r)

    def rotation_matrix(self, r: int) -> np.ndarray:
        """Return the lattice action matrix for rotation ``r``."""
        return self._rot_mats[int(r) % self.m].copy()

    def apply_rotation(self, r: int, x: int, y: int) -> tuple[int, int]:
        """Apply rotation ``r`` to a translation coordinate."""
        return self._apply_rotation(r, x, y)

    # ------------------------------------------------------------------
    # Group law
    # ------------------------------------------------------------------

    def _apply_rotation(self, k: int, x: int, y: int) -> tuple[int, int]:
        """Apply A^k to (x, y) mod n, using precomputed rotation matrices."""
        A_k = self._rot_mats[k % self._m]
        x_rotated = (A_k[0, 0] * x + A_k[0, 1] * y) % self._n
        y_rotated = (A_k[1, 0] * x + A_k[1, 1] * y) % self._n
        return x_rotated, y_rotated

    def compose(self, g: int, h: int) -> int:
        """Product g * h in the group."""
        x1, y1, r1 = self.decode(g)
        x2, y2, r2 = self.decode(h)
        x2r, y2r = self._apply_rotation(r1, x2, y2)
        return self._encode((x1 + x2r) % self._n, (y1 + y2r) % self._n, (r1 + r2) % self._m)

    def action_permutation(self, g: int) -> np.ndarray:
        """Indices implementing ``(g · x)[h] = x[g⁻¹h]``.

        The returned integer array has shape ``(|G|,)`` and is much smaller
        than a dense regular-representation matrix.
        """
        g = int(g)
        if g not in self._action_permutations:
            inv_g = self.inverse(g)
            self._action_permutations[g] = np.fromiter(
                (self.compose(inv_g, h) for h in self.elements()),
                dtype=np.int64,
                count=self.order,
            )
        return self._action_permutations[g]

    def left_action(self, g: int, signal: np.ndarray) -> np.ndarray:
        """Apply the left action to signals whose final axis indexes the group."""
        signal = np.asarray(signal)
        if signal.shape[-1] != self.order:
            raise ValueError(
                f"signal final axis must have length {self.order}, got {signal.shape}"
            )
        return np.take(signal, self.action_permutation(g), axis=-1)

    def cumulative_product(self, sequence) -> int:
        """Return ``g_T ... g_1`` for drives ``(g_1, ..., g_T)``."""
        total = self.identity()
        for g in sequence:
            total = self.compose(int(g), total)
        return total

    # ------------------------------------------------------------------
    # Regular representation
    # ------------------------------------------------------------------

    def _build_regular_rep(self) -> np.ndarray:
        """Regular representation of DiscreteSE2Group (Z_n² ⋊ C_m).

        Permutation matrix for each group element.
        Shape: (n_elem, n_elem, n_elem)., one n_elem by n_elem permutation matrix for each group element."""
        n_elem = self._order_val
        reg = np.zeros((n_elem, n_elem, n_elem), dtype=np.float32)
        for g in range(n_elem):
            for h in range(n_elem):
                i = self.compose(g, h)
                reg[g, i, h] = 1.0
        return reg

    # ------------------------------------------------------------------
    # Precomputation helpers
    # ------------------------------------------------------------------

    def _precompute_rotations(self) -> list[np.ndarray]:
        """Compute A^k mod n for k = 0, ..., m-1."""
        n, m = self._n, self._m
        mats = [np.eye(2, dtype=int) % n]
        for _ in range(1, m):
            mats.append((mats[-1] @ self._A) % n)
        return mats

    def _matrix_order(self) -> int | None:
        """Actual multiplicative order of A in GL(2, Z/nZ)."""
        identity = np.eye(2, dtype=int) % self._n
        for order in range(1, self._m):
            if np.array_equal(self._rot_mats[order], identity):
                return order
        A_m = (self._rot_mats[-1] @ self._A) % self._n
        if np.array_equal(A_m, identity):
            return self._m
        else:
            return None

    def _compute_dual_action_matrix(self) -> np.ndarray:
        """Compute (A^{-1})^T mod n for the dual action on characters.

        Characters transform by k ↦ ((A^{-1})^T mod n) k under the C_m action.
        """
        n = self._n
        a, b = int(self._A[0, 0]), int(self._A[0, 1])
        c, d = int(self._A[1, 0]), int(self._A[1, 1])
        det = (a * d - b * c) % n
        if math.gcd(det, n) != 1:
            raise ValueError(f"Action matrix is not invertible mod {n}: det = {det}.")
        det_inv = pow(det, -1, n)
        A_inv = (det_inv * np.array([[d, -b], [-c, a]], dtype=int)) % n
        return A_inv.T % n

    # ------------------------------------------------------------------
    # Dual action on characters
    # ------------------------------------------------------------------

    def _dual_action_once(self, j1: int, j2: int) -> tuple[int, int]:
        """One step of the C_m action on character labels via (A^{-1})^T mod n."""
        n = self._n
        Ad = self._A_dual
        return (Ad[0, 0] * j1 + Ad[0, 1] * j2) % n, (Ad[1, 0] * j1 + Ad[1, 1] * j2) % n

    def _compute_char_orbits(self) -> dict[int, list[list[tuple[int, int]]]]:
        """Partition character labels into C_m-orbits under the dual action.

        Returns a dict mapping orbit_size → list of orbits, where each orbit
        is a list of (j1, j2) tuples in cyclic order.
        """
        n, m = self._n, self._m

        visited: set[tuple[int, int]] = set()  # set of visited characters
        orbits_by_size: dict[int, list[list[tuple[int, int]]]] = {}

        for j1 in range(n):
            for j2 in range(n):
                if (j1, j2) in visited:
                    continue

                orbit: list[tuple[int, int]] = []
                a, b = j1, j2
                for _ in range(m):
                    if (a, b) in orbit:
                        break
                    orbit.append((a, b))
                    a, b = self._dual_action_once(a, b)

                for pt in orbit:
                    visited.add(pt)

                orbits_by_size.setdefault(len(orbit), []).append(orbit)

        return orbits_by_size

    # ------------------------------------------------------------------
    # Irreps via "Mackey's little group method" (Induced Representations of G by )
    # ------------------------------------------------------------------

    def _build_irreps(self) -> list[IrreducibleRepresentation]:
        """Construct all irreps of DiscreteSE2Group (Z_n² ⋊ C_m) via Clifford-Mackey theory: orbits in dual group and induction from stabilizers.

        For an orbit of size t (a divisor of m), the stabilizer is C_h with
        h = m / t.  Each of the h characters of C_h yields one irrep of
        dimension t.  The representation matrix for element (x, y, r) has
        a single nonzero entry per column: ρ(x,y,r)_{i,j} = δ_{i, (r+j) mod t} · χ_{k_i}(x, y) · exp(2πi · s · q / h)
        where q = (r+j) ÷ t counts wrap-arounds and s indexes the stabilizer character.
        """
        n, m = self._n, self._m
        order = self._order_val
        omega = np.exp(2j * np.pi / n)  # nth root of unity

        orbit_dict = self._compute_char_orbits()
        irreps: list[IrreducibleRepresentation] = []

        for orbit_size in sorted(orbit_dict):
            for orb_idx, orbit in enumerate(orbit_dict[orbit_size]):
                h = m // orbit_size  # order of stabilizer subgroup of C_m on elements of the orbit;
                # h = number of irreps contributed by this orbit, each of dimension orbit_size * (dimension of stabilizer irreps).
                # since stabilizers here are all cyclic groups, all their irreps have dimension 1, so dim = orbit_size * 1.
                dim = orbit_size * 1

                for s in range(h):
                    mats = np.zeros(
                        (order, dim, dim), dtype=np.complex128
                    )  # one dim by dim matrix for each element in G

                    for idx in range(order):  # loop over elements in G
                        x, y, r = self._decode(idx)

                        for j in range(dim):
                            # j is the index of the character in the orbit
                            total = r + j  # shift by r
                            i = total % dim  # wrap around
                            q = total // dim  # number of wrap-arounds

                            a_i, b_i = orbit[i]
                            char_val = omega ** (a_i * x + b_i * y)
                            stab_phase = np.exp(2j * np.pi * s * q / h)

                            mats[idx, i, j] = char_val * stab_phase

                    name = f"ZnZnCm_n{n}_m{m}|{dim}d_orb{orb_idx}_s{s}"
                    irreps.append(IrreducibleRepresentation(name, mats))

        return irreps
