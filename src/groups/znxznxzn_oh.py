"""DiscreteSE3Group: Z_n^3 ⋊ O for rotational octahedral group O.

Irreps are built lazily from Clifford/Mackey theory.  Characters of the
translation subgroup are indexed by ``k in Z_n^3``.  For each O-orbit of such
characters and each irrep of the stabilizer subgroup, we induce to the full
semidirect product.
"""

from dataclasses import dataclass
from itertools import permutations, product
from typing import Any

import numpy as np

from src.groups.group import Group
from src.groups.irrep import LazyIrreducibleRepresentation

_ROTATION_ORDER = 24
_REGULAR_REP_MAX_ORDER = 256


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


def _generate_rotational_octahedral_matrices() -> list[np.ndarray]:
    """Generate all 24 orientation-preserving signed permutation matrices."""
    mats = []
    for perm in permutations(range(3)):
        perm_mat = np.zeros((3, 3), dtype=int)
        for row, col in enumerate(perm):
            perm_mat[row, col] = 1
        for signs in product((-1, 1), repeat=3):
            mat = np.diag(signs) @ perm_mat
            if round(np.linalg.det(mat)) == 1:
                mats.append(mat)

    identity = np.eye(3, dtype=int)
    keyed = {tuple(mat.ravel()): mat for mat in mats}
    if len(keyed) != _ROTATION_ORDER:
        raise RuntimeError(f"Expected 24 rotations, got {len(keyed)}")

    rest = [mat for key, mat in sorted(keyed.items()) if not np.array_equal(mat, identity)]
    return [identity, *rest]


def _element_orders(cayley: np.ndarray) -> list[int]:
    """Element orders for a finite group with identity index 0."""
    n = cayley.shape[0]
    orders = []
    for g in range(n):
        cur = 0
        for power in range(1, n + 1):
            cur = cayley[cur, g]
            if cur == 0:
                orders.append(power)
                break
    return orders


def _classify_subgroup(cayley: np.ndarray) -> str:
    """Classify small stabilizer subgroups of the rotational octahedral group."""
    orders = _element_orders(cayley)
    counts = {order: orders.count(order) for order in sorted(set(orders))}
    size = cayley.shape[0]

    if size == 1:
        return "C1"
    if size in {2, 3, 4} and max(orders) == size:
        return f"C{size}"
    if size == 4 and counts == {1: 1, 2: 3}:
        return "V4"
    if size == 6 and counts == {1: 1, 2: 3, 3: 2}:
        return "D3"
    if size == 8 and counts == {1: 1, 2: 5, 4: 2}:
        return "D4"
    if size == 12 and counts == {1: 1, 2: 3, 3: 8}:
        return "A4"
    if size == 24 and counts == {1: 1, 2: 9, 3: 8, 4: 6}:
        return "O"
    return f"subgroup_order_{size}_orders_{counts}"


def _regular_irreps_from_cayley(cayley: np.ndarray, classification: str) -> list[LittleIrrep]:
    """Build subgroup irreps by decomposing the regular representation.

    The subgroup is first classified explicitly for labels/debuggability.  The
    small matrices are then obtained from invariant subspaces of the regular
    representation using a deterministic right-regular Hermitian operator.
    """
    size = cayley.shape[0]
    inv = np.zeros(size, dtype=np.int64)
    for g in range(size):
        hits = np.where((cayley[g] == 0) & (cayley[:, g] == 0))[0]
        if len(hits) != 1:
            raise ValueError(f"Could not find inverse for subgroup element {g}")
        inv[g] = int(hits[0])

    left = np.zeros((size, size, size), dtype=np.complex128)
    right = np.zeros_like(left)
    for h in range(size):
        for g in range(size):
            left[h, cayley[h, g], g] = 1.0
            right[h, cayley[g, inv[h]], g] = 1.0

    rng = np.random.default_rng(1729 + size)
    operator = np.zeros((size, size), dtype=np.complex128)
    for h in range(size):
        coeff = rng.normal() + 1j * rng.normal()
        operator += coeff * right[h] + coeff.conjugate() * right[h].conj().T
    operator = 0.5 * (operator + operator.conj().T)

    evals, evecs = np.linalg.eigh(operator)
    clusters: list[list[int]] = []
    for idx, val in enumerate(evals):
        if not clusters or abs(val - evals[clusters[-1][-1]]) > 1e-8:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)

    irreps: list[LittleIrrep] = []
    chars_seen: list[np.ndarray] = []
    for cluster in clusters:
        basis = evecs[:, cluster]
        dim = basis.shape[1]
        mats = np.empty((size, dim, dim), dtype=np.complex128)
        for h in range(size):
            mats[h] = basis.conj().T @ left[h] @ basis
        chars = np.trace(mats, axis1=1, axis2=2)
        if any(np.allclose(chars, seen, atol=1e-7) for seen in chars_seen):
            continue
        chars_seen.append(chars)
        irreps.append(
            LittleIrrep(
                name=f"{classification}|irrep_{len(irreps)}:{dim}",
                matrices=mats,
                classification=classification,
            )
        )

    dim_sum = sum(ir.dim**2 for ir in irreps)
    if dim_sum != size:
        raise ValueError(
            f"Failed to decompose {classification}: sum dim^2={dim_sum}, subgroup size={size}"
        )
    return sorted(irreps, key=lambda ir: (ir.dim, ir.name))


class DiscreteSE3Group(Group):
    """Z_n^3 ⋊ O: translations and rotational octahedral symmetries."""

    def __init__(self, n: int):
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")

        self._n = int(n)
        self._rot_mats = _generate_rotational_octahedral_matrices()
        self._rot_cayley = self._build_rotation_cayley()
        self._order_val = _ROTATION_ORDER * self._n**3
        self._regular: np.ndarray | None = None

        self._orbit_data = self._compute_orbit_data()
        self._little_irrep_cache: dict[tuple[int, ...], list[LittleIrrep]] = {}
        self._irreps = self._build_irreps()
        self._conjugate_pairs = self._build_conjugate_pairs()

    @property
    def order(self) -> int:
        return self._order_val

    def elements(self) -> list[int]:
        return list(range(self._order_val))

    def irreps(self) -> list[LazyIrreducibleRepresentation]:
        return list(self._irreps)

    def regular_rep(self) -> np.ndarray:
        if self.order > _REGULAR_REP_MAX_ORDER:
            raise MemoryError(
                f"regular_rep() would allocate a dense ({self.order}, {self.order}, "
                f"{self.order}) tensor. Use GroupCompositionDataset(..., online=True)."
            )
        if self._regular is None:
            reg = np.zeros((self.order, self.order, self.order), dtype=np.float32)
            for g in range(self.order):
                for h in range(self.order):
                    reg[g, self.compose(g, h), h] = 1.0
            self._regular = reg
        return self._regular

    def conjugate_pairs(self) -> list[list[int]]:
        return list(self._conjugate_pairs)

    def orbit_data(self) -> list[dict[str, Any]]:
        return list(self._orbit_data)

    def _encode(self, x: int, y: int, z: int, r: int) -> int:
        n = self._n
        return r * n**3 + x * n**2 + y * n + z

    def _decode(self, idx: int) -> tuple[int, int, int, int]:
        n = self._n
        r, rem = divmod(int(idx), n**3)
        x, rem = divmod(rem, n**2)
        y, z = divmod(rem, n)
        return x, y, z, r

    def _apply_rotation(self, r: int, x: int, y: int, z: int) -> tuple[int, int, int]:
        vec = np.array([x, y, z], dtype=int)
        rotated = self._rot_mats[int(r)] @ vec
        return tuple((rotated % self._n).tolist())

    def compose(self, g: int, h: int) -> int:
        x1, y1, z1, r1 = self._decode(g)
        x2, y2, z2, r2 = self._decode(h)
        x2r, y2r, z2r = self._apply_rotation(r1, x2, y2, z2)
        r12 = int(self._rot_cayley[r1, r2])
        return self._encode(
            (x1 + x2r) % self._n,
            (y1 + y2r) % self._n,
            (z1 + z2r) % self._n,
            r12,
        )

    def _build_rotation_cayley(self) -> np.ndarray:
        index = {tuple(mat.ravel()): i for i, mat in enumerate(self._rot_mats)}
        cayley = np.empty((_ROTATION_ORDER, _ROTATION_ORDER), dtype=np.int64)
        for a, mat_a in enumerate(self._rot_mats):
            for b, mat_b in enumerate(self._rot_mats):
                cayley[a, b] = index[tuple((mat_a @ mat_b).ravel())]
        return cayley

    def _dual_action(self, r: int, k: tuple[int, int, int]) -> tuple[int, int, int]:
        vec = np.array(k, dtype=int)
        return tuple((self._rot_mats[int(r)] @ vec % self._n).tolist())

    def _compute_orbit_data(self) -> list[dict[str, Any]]:
        visited: set[tuple[int, int, int]] = set()
        data = []
        for k in product(range(self._n), repeat=3):
            if k in visited:
                continue
            orbit = sorted({self._dual_action(r, k) for r in range(_ROTATION_ORDER)})
            visited.update(orbit)
            rep = min(orbit)
            stabilizer = tuple(
                r for r in range(_ROTATION_ORDER) if self._dual_action(r, rep) == rep
            )
            if len(orbit) * len(stabilizer) != _ROTATION_ORDER:
                raise ValueError(
                    f"Orbit-stabilizer failed for {rep}: |O|={len(orbit)}, |S|={len(stabilizer)}"
                )
            coset_reps, orbit_labels, transition = self._coset_data(rep, stabilizer)
            data.append(
                {
                    "representative": rep,
                    "orbit": orbit,
                    "stabilizer": stabilizer,
                    "coset_reps": coset_reps,
                    "orbit_labels": orbit_labels,
                    "transition": transition,
                }
            )
        return sorted(data, key=lambda item: (len(item["orbit"]), item["representative"]))

    def _coset_data(
        self, rep: tuple[int, int, int], stabilizer: tuple[int, ...]
    ) -> tuple[tuple[int, ...], list[tuple[int, int, int]], np.ndarray]:
        covered: set[int] = set()
        coset_reps = []
        for r in range(_ROTATION_ORDER):
            if r in covered:
                continue
            coset_reps.append(r)
            covered.update(int(self._rot_cayley[r, s]) for s in stabilizer)

        orbit_labels = [self._dual_action(t, rep) for t in coset_reps]
        if len(set(orbit_labels)) != len(orbit_labels):
            raise ValueError(f"Coset reps do not map bijectively onto orbit for {rep}")

        coset_lookup = {}
        stab_lookup = {s: i for i, s in enumerate(stabilizer)}
        for i, t_i in enumerate(coset_reps):
            for s in stabilizer:
                coset_lookup[int(self._rot_cayley[t_i, s])] = (i, stab_lookup[s])

        transition = np.empty((_ROTATION_ORDER, len(coset_reps), 2), dtype=np.int64)
        for r in range(_ROTATION_ORDER):
            for j, t_j in enumerate(coset_reps):
                transition[r, j] = coset_lookup[int(self._rot_cayley[r, t_j])]

        return tuple(coset_reps), orbit_labels, transition

    def _subgroup_cayley(self, stabilizer: tuple[int, ...]) -> np.ndarray:
        local = {r: i for i, r in enumerate(stabilizer)}
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
            for sigma_idx, sigma in enumerate(self._little_irreps(data["stabilizer"])):
                irreps.append(self._make_induced_irrep(orbit_idx, data, sigma_idx, sigma))
        dim_sum = sum(ir.dim**2 for ir in irreps)
        if dim_sum != self.order:
            raise ValueError(f"Peter-Weyl dimension sum failed: {dim_sum} != {self.order}")
        return irreps

    def _make_induced_irrep(
        self, orbit_idx: int, data: dict[str, Any], sigma_idx: int, sigma: LittleIrrep
    ) -> LazyIrreducibleRepresentation:
        orbit_labels = np.array(data["orbit_labels"], dtype=int)
        transition = data["transition"]
        orbit_size = len(orbit_labels)
        sigma_dim = sigma.dim
        dim = orbit_size * sigma_dim

        def matrix_fn(element_index: int) -> np.ndarray:
            x, y, z, r = self._decode(element_index)
            v = np.array([x, y, z], dtype=int)
            mat = np.zeros((dim, dim), dtype=np.complex128)
            for j in range(orbit_size):
                i, s_idx = transition[r, j]
                phase_arg = int(np.dot(orbit_labels[i], v)) % self._n
                chi = np.exp(2j * np.pi * phase_arg / self._n)
                row = slice(i * sigma_dim, (i + 1) * sigma_dim)
                col = slice(j * sigma_dim, (j + 1) * sigma_dim)
                mat[row, col] = chi * sigma(int(s_idx))
            return mat

        name = (
            f"Zn3Oh_n{self._n}|orb{orbit_idx}_size{orbit_size}_"
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
        return np.array([np.trace(irrep(g)) for g in elements])

    def _build_conjugate_pairs(self) -> list[list[int]]:
        irreps = self._irreps
        elements = (
            self.elements()
            if self.order <= 5000
            else self.elements()[:: max(1, self.order // 4096)]
        )
        chars = [self._character_vector(irrep, elements) for irrep in irreps]
        processed: set[int] = set()
        pairs = []
        for i, chi in enumerate(chars):
            if i in processed:
                continue
            matches = [
                j
                for j, chi_j in enumerate(chars)
                if j not in processed and np.allclose(chi_j, chi.conjugate(), atol=1e-7)
            ]
            if not matches:
                raise ValueError(f"Could not find conjugate irrep for index {i}")
            j = matches[0]
            processed.add(i)
            processed.add(j)
            pairs.append([i] if i == j else sorted([i, j]))
        return pairs
