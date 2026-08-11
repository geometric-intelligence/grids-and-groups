"""DiscreteSE3A4Group: Z_n^3 ⋊ A4 for tetrahedral rotations.

Irreps are built lazily from Clifford/Mackey theory. Characters of the
translation subgroup are indexed by ``k in Z_n^3``. For each A4-orbit of such
characters and each irrep of the stabilizer subgroup, we induce to the full
semidirect product.
"""

from dataclasses import dataclass
from itertools import permutations, product
from typing import Any

import numpy as np

from src.groups.group import Group
from src.groups.irrep import LazyIrreducibleRepresentation

_ROTATION_ORDER = 12
_REGULAR_REP_MAX_ORDER = 256
_TETRAHEDRON_VERTICES = {
    (1, 1, 1),
    (1, -1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
}


@dataclass(frozen=True)
class LittleIrrep:
    """Irrep of a stabilizer subgroup, indexed by local subgroup elements."""

    name: str
    matrices: np.ndarray
    classification: str

    @property
    def dim(self) -> int:
        return int(self.matrices.shape[1])

    def __call__(self, idx: int) -> np.ndarray:
        return self.matrices[int(idx)]


def _generate_rotational_tetrahedral_matrices() -> list[np.ndarray]:
    """Generate the 12 proper rotations preserving a fixed tetrahedron."""
    mats = []
    for perm in permutations(range(3)):
        perm_mat = np.zeros((3, 3), dtype=int)
        for row, col in enumerate(perm):
            perm_mat[row, col] = 1
        for signs in product((-1, 1), repeat=3):
            mat = np.diag(signs) @ perm_mat
            if round(np.linalg.det(mat)) != 1:
                continue
            images = {
                tuple((mat @ np.asarray(vertex, dtype=int)).tolist())
                for vertex in _TETRAHEDRON_VERTICES
            }
            if images == _TETRAHEDRON_VERTICES:
                mats.append(mat)

    identity = np.eye(3, dtype=int)
    keyed = {tuple(mat.ravel()): mat for mat in mats}
    if len(keyed) != _ROTATION_ORDER:
        raise RuntimeError(f"Expected 12 tetrahedral rotations, got {len(keyed)}")

    rest = [mat for _, mat in sorted(keyed.items()) if not np.array_equal(mat, identity)]
    return [identity, *rest]


def _element_orders(cayley: np.ndarray) -> list[int]:
    """Return element orders for a finite group with identity index zero."""
    n = cayley.shape[0]
    orders = []
    for g in range(n):
        current = 0
        for power in range(1, n + 1):
            current = int(cayley[current, g])
            if current == 0:
                orders.append(power)
                break
    return orders


def _classify_subgroup(cayley: np.ndarray) -> str:
    """Classify stabilizer subgroups of the rotational tetrahedral group."""
    orders = _element_orders(cayley)
    counts = {order: orders.count(order) for order in sorted(set(orders))}
    size = cayley.shape[0]

    if size == 1:
        return "C1"
    if size in {2, 3} and max(orders) == size:
        return f"C{size}"
    if size == 4 and counts == {1: 1, 2: 3}:
        return "V4"
    if size == 12 and counts == {1: 1, 2: 3, 3: 8}:
        return "A4"
    return f"subgroup_order_{size}_orders_{counts}"


def _regular_irreps_from_cayley(cayley: np.ndarray, classification: str) -> list[LittleIrrep]:
    """Build subgroup irreps by decomposing the regular representation."""
    size = cayley.shape[0]
    inverse = np.zeros(size, dtype=np.int64)
    for g in range(size):
        hits = np.where((cayley[g] == 0) & (cayley[:, g] == 0))[0]
        if len(hits) != 1:
            raise ValueError(f"Could not find inverse for subgroup element {g}")
        inverse[g] = int(hits[0])

    left = np.zeros((size, size, size), dtype=np.complex128)
    right = np.zeros_like(left)
    for h in range(size):
        for g in range(size):
            left[h, cayley[h, g], g] = 1.0
            right[h, cayley[g, inverse[h]], g] = 1.0

    rng = np.random.default_rng(1729 + size)
    operator = np.zeros((size, size), dtype=np.complex128)
    for h in range(size):
        coefficient = rng.normal() + 1j * rng.normal()
        operator += coefficient * right[h] + coefficient.conjugate() * right[h].conj().T
    operator = 0.5 * (operator + operator.conj().T)

    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    clusters: list[list[int]] = []
    for idx, value in enumerate(eigenvalues):
        if not clusters or abs(value - eigenvalues[clusters[-1][-1]]) > 1e-8:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)

    irreps: list[LittleIrrep] = []
    characters_seen: list[np.ndarray] = []
    for cluster in clusters:
        basis = eigenvectors[:, cluster]
        dim = basis.shape[1]
        matrices = np.empty((size, dim, dim), dtype=np.complex128)
        for h in range(size):
            matrices[h] = basis.conj().T @ left[h] @ basis
        characters = np.trace(matrices, axis1=1, axis2=2)
        if any(np.allclose(characters, seen, atol=1e-7) for seen in characters_seen):
            continue
        characters_seen.append(characters)
        irreps.append(
            LittleIrrep(
                name=f"{classification}|irrep_{len(irreps)}:{dim}",
                matrices=matrices,
                classification=classification,
            )
        )

    dim_sum = sum(irrep.dim**2 for irrep in irreps)
    if dim_sum != size:
        raise ValueError(
            f"Failed to decompose {classification}: sum dim^2={dim_sum}, subgroup size={size}"
        )
    return sorted(irreps, key=lambda irrep: (irrep.dim, irrep.name))


class DiscreteSE3A4Group(Group):
    """Z_n^3 ⋊ A4: translations and rotational tetrahedral symmetries."""

    def __init__(self, n: int):
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")

        self._n = int(n)
        self._rot_mats = _generate_rotational_tetrahedral_matrices()
        self._rot_cayley = self._build_rotation_cayley()
        self._rot_inverse = self._build_rotation_inverses()
        self._order_val = _ROTATION_ORDER * self._n**3
        self._regular: np.ndarray | None = None
        self._action_permutations: dict[int, np.ndarray] = {}

        self._orbit_data = self._compute_orbit_data()
        self._little_irrep_cache: dict[tuple[int, ...], list[LittleIrrep]] = {}
        self._irreps = self._build_irreps()
        self._conjugate_pairs = self._build_conjugate_pairs()

    @property
    def order(self) -> int:
        return self._order_val

    @property
    def n(self) -> int:
        """Order of each cyclic translation factor."""
        return self._n

    @property
    def num_rotations(self) -> int:
        """Number of proper tetrahedral rotations."""
        return _ROTATION_ORDER

    def elements(self) -> list[int]:
        return list(range(self._order_val))

    def irreps(self) -> list[LazyIrreducibleRepresentation]:
        return list(self._irreps)

    def regular_rep(self) -> np.ndarray:
        """Return dense regular-representation matrices for small groups only."""
        if self.order > _REGULAR_REP_MAX_ORDER:
            raise MemoryError(
                f"regular_rep() would allocate a dense ({self.order}, {self.order}, "
                f"{self.order}) tensor. Use left_action() instead."
            )
        if self._regular is None:
            regular = np.zeros((self.order, self.order, self.order), dtype=np.float32)
            for g in range(self.order):
                for h in range(self.order):
                    regular[g, self.compose(g, h), h] = 1.0
            self._regular = regular
        return self._regular

    def conjugate_pairs(self) -> list[list[int]]:
        return list(self._conjugate_pairs)

    def orbit_data(self) -> list[dict[str, Any]]:
        return list(self._orbit_data)

    def _encode(self, x: int, y: int, z: int, rotation: int) -> int:
        n = self._n
        return rotation * n**3 + x * n**2 + y * n + z

    def _decode(self, idx: int) -> tuple[int, int, int, int]:
        n = self._n
        rotation, remainder = divmod(int(idx), n**3)
        x, remainder = divmod(remainder, n**2)
        y, z = divmod(remainder, n)
        return x, y, z, rotation

    def encode(self, x: int, y: int, z: int, rotation: int) -> int:
        """Encode a group element, reducing coordinates to valid ranges."""
        return self._encode(
            int(x) % self.n,
            int(y) % self.n,
            int(z) % self.n,
            int(rotation) % self.num_rotations,
        )

    def decode(self, idx: int) -> tuple[int, int, int, int]:
        """Decode an element index as ``(x, y, z, rotation)``."""
        idx = int(idx)
        if not 0 <= idx < self.order:
            raise ValueError(f"element index must be in [0, {self.order}), got {idx}")
        return self._decode(idx)

    def identity(self) -> int:
        """Return the identity element index."""
        return self.encode(0, 0, 0, 0)

    def rotation_matrix(self, rotation: int) -> np.ndarray:
        """Return the integer matrix for a rotation index."""
        return self._rot_mats[int(rotation) % self.num_rotations].copy()

    def apply_rotation(self, rotation: int, x: int, y: int, z: int) -> tuple[int, int, int]:
        """Apply a tetrahedral rotation to a translation coordinate."""
        return self._apply_rotation(int(rotation) % self.num_rotations, x, y, z)

    def inverse(self, g: int) -> int:
        """Return the inverse of element ``g``."""
        x, y, z, rotation = self.decode(g)
        rotation_inverse = int(self._rot_inverse[rotation])
        x_inverse, y_inverse, z_inverse = self._apply_rotation(rotation_inverse, -x, -y, -z)
        return self.encode(x_inverse, y_inverse, z_inverse, rotation_inverse)

    def _apply_rotation(self, rotation: int, x: int, y: int, z: int) -> tuple[int, int, int]:
        vector = np.array([x, y, z], dtype=int)
        rotated = self._rot_mats[int(rotation)] @ vector
        return tuple((rotated % self._n).tolist())

    def compose(self, g: int, h: int) -> int:
        x1, y1, z1, rotation1 = self.decode(g)
        x2, y2, z2, rotation2 = self.decode(h)
        x2_rotated, y2_rotated, z2_rotated = self._apply_rotation(rotation1, x2, y2, z2)
        rotation12 = int(self._rot_cayley[rotation1, rotation2])
        return self._encode(
            (x1 + x2_rotated) % self._n,
            (y1 + y2_rotated) % self._n,
            (z1 + z2_rotated) % self._n,
            rotation12,
        )

    def action_permutation(self, g: int) -> np.ndarray:
        """Return indices implementing ``(g · x)[h] = x[g⁻¹h]``."""
        g = int(g)
        if g not in self._action_permutations:
            inverse = self.inverse(g)
            self._action_permutations[g] = np.fromiter(
                (self.compose(inverse, h) for h in self.elements()),
                dtype=np.int64,
                count=self.order,
            )
        return self._action_permutations[g]

    def left_action(self, g: int, signal: np.ndarray) -> np.ndarray:
        """Apply the left action to signals whose final axis indexes the group."""
        signal = np.asarray(signal)
        if signal.shape[-1] != self.order:
            raise ValueError(f"signal final axis must have length {self.order}, got {signal.shape}")
        return np.take(signal, self.action_permutation(g), axis=-1)

    def cumulative_product(self, sequence) -> int:
        """Return the body-frame product ``g_1 ... g_T``."""
        total = self.identity()
        for element in sequence:
            total = self.compose(total, int(element))
        return total

    def _build_rotation_cayley(self) -> np.ndarray:
        index = {tuple(matrix.ravel()): idx for idx, matrix in enumerate(self._rot_mats)}
        cayley = np.empty((_ROTATION_ORDER, _ROTATION_ORDER), dtype=np.int64)
        for a, matrix_a in enumerate(self._rot_mats):
            for b, matrix_b in enumerate(self._rot_mats):
                cayley[a, b] = index[tuple((matrix_a @ matrix_b).ravel())]
        return cayley

    def _build_rotation_inverses(self) -> np.ndarray:
        inverses = np.empty(_ROTATION_ORDER, dtype=np.int64)
        for rotation in range(_ROTATION_ORDER):
            matches = np.where(
                (self._rot_cayley[rotation] == 0) & (self._rot_cayley[:, rotation] == 0)
            )[0]
            if len(matches) != 1:
                raise ValueError(f"could not find inverse for rotation {rotation}")
            inverses[rotation] = int(matches[0])
        return inverses

    def _dual_action(self, rotation: int, character: tuple[int, int, int]) -> tuple[int, int, int]:
        vector = np.array(character, dtype=int)
        return tuple((self._rot_mats[int(rotation)] @ vector % self._n).tolist())

    def _compute_orbit_data(self) -> list[dict[str, Any]]:
        visited: set[tuple[int, int, int]] = set()
        data = []
        for character in product(range(self._n), repeat=3):
            if character in visited:
                continue
            orbit = sorted(
                {self._dual_action(rotation, character) for rotation in range(_ROTATION_ORDER)}
            )
            visited.update(orbit)
            representative = min(orbit)
            stabilizer = tuple(
                rotation
                for rotation in range(_ROTATION_ORDER)
                if self._dual_action(rotation, representative) == representative
            )
            if len(orbit) * len(stabilizer) != _ROTATION_ORDER:
                raise ValueError(
                    f"Orbit-stabilizer failed for {representative}: "
                    f"|orbit|={len(orbit)}, |stabilizer|={len(stabilizer)}"
                )
            coset_reps, orbit_labels, transition = self._coset_data(representative, stabilizer)
            data.append(
                {
                    "representative": representative,
                    "orbit": orbit,
                    "stabilizer": stabilizer,
                    "coset_reps": coset_reps,
                    "orbit_labels": orbit_labels,
                    "transition": transition,
                }
            )
        return sorted(
            data,
            key=lambda item: (
                len(item["orbit"]),
                item["representative"],
            ),
        )

    def _coset_data(
        self,
        representative: tuple[int, int, int],
        stabilizer: tuple[int, ...],
    ) -> tuple[tuple[int, ...], list[tuple[int, int, int]], np.ndarray]:
        covered: set[int] = set()
        coset_reps = []
        for rotation in range(_ROTATION_ORDER):
            if rotation in covered:
                continue
            coset_reps.append(rotation)
            covered.update(
                int(self._rot_cayley[rotation, stabilizer_element])
                for stabilizer_element in stabilizer
            )

        orbit_labels = [self._dual_action(coset_rep, representative) for coset_rep in coset_reps]
        if len(set(orbit_labels)) != len(orbit_labels):
            raise ValueError(f"Coset reps do not map bijectively onto orbit for {representative}")

        coset_lookup = {}
        stabilizer_lookup = {
            stabilizer_element: idx for idx, stabilizer_element in enumerate(stabilizer)
        }
        for i, coset_rep in enumerate(coset_reps):
            for stabilizer_element in stabilizer:
                coset_lookup[int(self._rot_cayley[coset_rep, stabilizer_element])] = (
                    i,
                    stabilizer_lookup[stabilizer_element],
                )

        transition = np.empty((_ROTATION_ORDER, len(coset_reps), 2), dtype=np.int64)
        for rotation in range(_ROTATION_ORDER):
            for j, coset_rep in enumerate(coset_reps):
                transition[rotation, j] = coset_lookup[int(self._rot_cayley[rotation, coset_rep])]

        return tuple(coset_reps), orbit_labels, transition

    def _subgroup_cayley(self, stabilizer: tuple[int, ...]) -> np.ndarray:
        local = {rotation: idx for idx, rotation in enumerate(stabilizer)}
        cayley = np.empty((len(stabilizer), len(stabilizer)), dtype=np.int64)
        for i, a in enumerate(stabilizer):
            for j, b in enumerate(stabilizer):
                cayley[i, j] = local[int(self._rot_cayley[a, b])]
        return cayley

    def _little_irreps(self, stabilizer: tuple[int, ...]) -> list[LittleIrrep]:
        if stabilizer not in self._little_irrep_cache:
            cayley = self._subgroup_cayley(stabilizer)
            classification = _classify_subgroup(cayley)
            self._little_irrep_cache[stabilizer] = _regular_irreps_from_cayley(
                cayley, classification
            )
        return self._little_irrep_cache[stabilizer]

    def _build_irreps(self) -> list[LazyIrreducibleRepresentation]:
        irreps = []
        for orbit_idx, data in enumerate(self._orbit_data):
            little_irreps = self._little_irreps(data["stabilizer"])
            for sigma_idx, sigma in enumerate(little_irreps):
                irreps.append(self._make_induced_irrep(orbit_idx, data, sigma_idx, sigma))
        dim_sum = sum(irrep.dim**2 for irrep in irreps)
        if dim_sum != self.order:
            raise ValueError(f"Peter-Weyl dimension sum failed: {dim_sum} != {self.order}")
        return irreps

    def _make_induced_irrep(
        self,
        orbit_idx: int,
        data: dict[str, Any],
        sigma_idx: int,
        sigma: LittleIrrep,
    ) -> LazyIrreducibleRepresentation:
        orbit_labels = np.array(data["orbit_labels"], dtype=int)
        transition = data["transition"]
        orbit_size = len(orbit_labels)
        sigma_dim = sigma.dim
        dim = orbit_size * sigma_dim

        def matrix_fn(element_index: int) -> np.ndarray:
            x, y, z, rotation = self._decode(element_index)
            translation = np.array([x, y, z], dtype=int)
            matrix = np.zeros((dim, dim), dtype=np.complex128)
            for j in range(orbit_size):
                i, stabilizer_idx = transition[rotation, j]
                phase_arg = int(np.dot(orbit_labels[i], translation)) % self._n
                character = np.exp(2j * np.pi * phase_arg / self._n)
                row = slice(i * sigma_dim, (i + 1) * sigma_dim)
                column = slice(j * sigma_dim, (j + 1) * sigma_dim)
                matrix[row, column] = character * sigma(int(stabilizer_idx))
            return matrix

        name = (
            f"Zn3A4_n{self._n}|orb{orbit_idx}_size{orbit_size}_"
            f"{sigma.classification}_s{sigma_idx}_d{dim}"
        )
        irrep = LazyIrreducibleRepresentation(name, dim, matrix_fn, cache_size=128)
        irrep._metadata = {
            "orbit_idx": orbit_idx,
            "orbit": tuple(data["orbit"]),
            "representative": data["representative"],
            "stabilizer": data["stabilizer"],
            "little_irrep_idx": sigma_idx,
            "little_irrep_name": sigma.name,
            "little_irrep_classification": sigma.classification,
        }
        return irrep

    def _character_vector(self, irrep, elements: list[int]) -> np.ndarray:
        return np.array([np.trace(irrep(element)) for element in elements])

    def _build_conjugate_pairs(self) -> list[list[int]]:
        irreps = self._irreps
        elements = (
            self.elements()
            if self.order <= 5000
            else self.elements()[:: max(1, self.order // 4096)]
        )
        characters = [self._character_vector(irrep, elements) for irrep in irreps]
        processed: set[int] = set()
        pairs = []
        for i, character in enumerate(characters):
            if i in processed:
                continue
            matches = [
                j
                for j, candidate in enumerate(characters)
                if j not in processed and np.allclose(candidate, character.conjugate(), atol=1e-7)
            ]
            if not matches:
                raise ValueError(f"Could not find conjugate irrep for index {i}")
            j = matches[0]
            processed.add(i)
            processed.add(j)
            pairs.append([i] if i == j else sorted([i, j]))
        return pairs
