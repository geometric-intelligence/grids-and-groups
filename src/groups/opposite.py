"""Opposite-group views for right-regular, body-frame dynamics."""

from __future__ import annotations

import numpy as np

from src.groups.irrep import (
    IrreducibleRepresentation,
    LazyIrreducibleRepresentation,
)

_REGULAR_REP_MAX_BYTES = 1_000_000_000


def opposite_irrep(irrep):
    """Return the corresponding irrep of the opposite group."""
    name = f"op({irrep})"
    if hasattr(irrep, "_matrices"):
        matrices = np.asarray(irrep._matrices).transpose(0, 2, 1)
        return IrreducibleRepresentation(name, matrices)
    return LazyIrreducibleRepresentation(
        name,
        irrep.dim,
        lambda element, source=irrep: source(element).T,
    )


class OppositeGroup:
    """A view of ``group`` whose multiplication order is reversed.

    If the original product is written ``g * h``, this view defines
    ``g ∘ h = h * g``.  Its left-regular action is therefore the original
    group's right-regular action,

    ``(g ·_right x)[h] = x[h * g⁻¹]``.

    Acting on a one-hot state at ``s`` moves it to ``s * g``, which is the
    body-frame update convention used for path integration.
    """

    def __init__(self, group):
        if isinstance(group, OppositeGroup):
            raise ValueError("group is already an OppositeGroup")
        self.base_group = group
        self._irreps = [opposite_irrep(irrep) for irrep in group.irreps()]
        self._action_permutations: dict[int, np.ndarray] = {}
        self._regular: np.ndarray | None = None
        self._cayley_table: np.ndarray | None = None
        self._identity: int | None = None
        self._inverses: np.ndarray | None = None

    @property
    def order(self) -> int:
        return self.base_group.order

    def elements(self) -> list[int]:
        return self.base_group.elements()

    def irreps(self) -> list:
        return list(self._irreps)

    def _base_cayley(self) -> np.ndarray:
        if self._cayley_table is None:
            regular = np.asarray(self.base_group.regular_rep())
            self._cayley_table = np.argmax(regular, axis=1).astype(np.int64)
        return self._cayley_table

    def _base_compose(self, g: int, h: int) -> int:
        if hasattr(self.base_group, "compose"):
            return int(self.base_group.compose(int(g), int(h)))
        return int(self._base_cayley()[int(g), int(h)])

    def identity(self) -> int:
        if hasattr(self.base_group, "identity"):
            return int(self.base_group.identity())
        if self._identity is None:
            cayley = self._base_cayley()
            elements = np.arange(self.order)
            matches = np.flatnonzero(
                np.all(cayley == elements[None, :], axis=1)
                & np.all(cayley == elements[:, None], axis=0)
            )
            if matches.size != 1:
                raise ValueError("could not infer a unique identity element")
            self._identity = int(matches[0])
        return self._identity

    def inverse(self, g: int) -> int:
        if hasattr(self.base_group, "inverse"):
            return int(self.base_group.inverse(int(g)))
        if self._inverses is None:
            cayley = self._base_cayley()
            identity = self.identity()
            inverses = np.empty(self.order, dtype=np.int64)
            for element in self.elements():
                matches = np.flatnonzero(
                    (cayley[element] == identity)
                    & (cayley[:, element] == identity)
                )
                if matches.size != 1:
                    raise ValueError(f"could not infer inverse of element {element}")
                inverses[element] = matches[0]
            self._inverses = inverses
        return int(self._inverses[int(g)])

    def compose(self, g: int, h: int) -> int:
        """Return ``g ∘ h = h * g``."""
        return self._base_compose(h, g)

    def action_permutation(self, g: int) -> np.ndarray:
        """Source indices implementing ``x[h] -> x[h * g⁻¹]``."""
        g = int(g)
        if g not in self._action_permutations:
            inverse = self.inverse(g)
            self._action_permutations[g] = np.fromiter(
                (self._base_compose(h, inverse) for h in self.elements()),
                dtype=np.int64,
                count=self.order,
            )
        return self._action_permutations[g]

    def left_action(self, g: int, signal: np.ndarray) -> np.ndarray:
        """Apply the original group's right-regular action."""
        signal = np.asarray(signal)
        if signal.shape[-1] != self.order:
            raise ValueError(
                f"signal final axis must have length {self.order}, got {signal.shape}"
            )
        return np.take(signal, self.action_permutation(g), axis=-1)

    right_action = left_action

    def regular_rep(self) -> np.ndarray:
        """Return right-regular matrices for the original group."""
        required_bytes = self.order**3 * np.dtype(np.float32).itemsize
        if required_bytes > _REGULAR_REP_MAX_BYTES:
            required_gib = required_bytes / 1024**3
            raise MemoryError(
                f"regular_rep() would require {required_gib:.1f} GiB for shape "
                f"({self.order}, {self.order}, {self.order}). Use left_action() instead."
            )
        if self._regular is None:
            self._regular = np.zeros(
                (self.order, self.order, self.order), dtype=np.float32
            )
            for g in self.elements():
                permutation = self.action_permutation(g)
                self._regular[g, np.arange(self.order), permutation] = 1.0
        return self._regular

    def cumulative_product(self, sequence) -> int:
        """Return ``g_1 * ... * g_T`` in the original group."""
        total = self.identity()
        for element in sequence:
            total = self.compose(int(element), total)
        return total

    def __getattr__(self, name):
        return getattr(self.base_group, name)


def as_action_group(group, action_side: str = "right"):
    """Return a group view implementing the requested action convention."""
    if action_side == "left":
        return group
    if action_side == "right":
        return group if isinstance(group, OppositeGroup) else OppositeGroup(group)
    raise ValueError("action_side must be 'left' or 'right'")
