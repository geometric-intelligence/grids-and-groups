"""Closed-form, fixed-weight PyTorch RNNs for finite-group actions.

The construction is group-agnostic.  A group must expose ``order``,
``elements()``, ``irreps()``, and ``regular_rep()``.  Groups that additionally
provide ``identity()``, ``inverse()``, and ``compose()`` avoid reconstructing
those operations from the regular representation.  Irreps may be dense or lazy
as long as they expose ``dim`` and are callable on an element index.

The analytical weights are registered as PyTorch buffers, not parameters.
Consequently, constructing and evaluating :class:`FiniteGroupRNN` never
performs training or optimization.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.groups.opposite import as_action_group, opposite_irrep


def squared_relu(values: torch.Tensor) -> torch.Tensor:
    """Elementwise squared ReLU."""
    return torch.relu(values).square()


def _irrep_matrices(irrep, group_order: int) -> np.ndarray:
    """Materialize one irrep once, supporting both dense and lazy objects."""
    if hasattr(irrep, "_matrices"):
        return irrep._matrices
    return np.stack([irrep(g) for g in range(group_order)])


def fourier_hat(
    signal: np.ndarray,
    irrep,
    *,
    matrices: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``sum_g signal[g] rho(g)^H`` for one irrep."""
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got {signal.shape}")
    if matrices is None:
        matrices = _irrep_matrices(irrep, signal.size)
    if matrices.shape[0] != signal.size:
        raise ValueError("signal length does not match the irrep's group order")
    return np.einsum("gba,g->ab", matrices.conj(), signal)


def fourier_power(signal: np.ndarray, irrep, *, normalize_by_dim: bool = True) -> float:
    """Return squared Frobenius power of ``signal`` at one irrep."""
    power = float(np.linalg.norm(fourier_hat(signal, irrep), ord="fro") ** 2)
    return power / irrep.dim if normalize_by_dim else power


def minimum_fourier_singular_value(signal: np.ndarray, irreps) -> float:
    """Return the smallest singular value over the supplied Fourier blocks."""
    return min(
        float(np.linalg.svd(fourier_hat(signal, irrep), compute_uv=False).min())
        for irrep in irreps
    )


def random_invertible_encoding(
    group,
    irreps,
    *,
    seed: int = 0,
    min_singular_value: float = 1e-5,
    max_tries: int = 10_000,
) -> np.ndarray:
    """Sample a real group signal with invertible selected Fourier blocks."""
    irreps = list(irreps)
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        signal = rng.normal(size=group.order)
        signal += 0.5 * rng.normal(size=group.order) ** 2
        if minimum_fourier_singular_value(signal, irreps) > min_singular_value:
            return signal
    raise RuntimeError("failed to sample an encoding with invertible Fourier matrices")


def hidden_width(irrep, *, q_rho: int = 3) -> int:
    """Number of hidden units contributed by one irrep."""
    _validate_q_rho(q_rho)
    return 4 * q_rho * irrep.dim**3


def _validate_q_rho(q_rho: int) -> None:
    if isinstance(q_rho, bool) or not isinstance(q_rho, (int, np.integer)) or q_rho < 3:
        raise ValueError("q_rho must be an integer greater than or equal to 3")


def select_irreps_by_power(
    irreps,
    signal: np.ndarray,
    *,
    num_irreps: int | None = None,
    max_hidden_width: int | None = None,
    q_rho: int = 3,
    normalize_by_dim: bool = True,
    always_include_trivial: bool = True,
    ranking: str = "power",
) -> tuple[list, list[int]]:
    """Select high-power irreps subject to count and hidden-width budgets.

    Irreps are ranked by Fourier power.  The returned list follows the original
    irrep ordering so metadata and Fourier blocks remain easy to compare.
    """
    irreps = list(irreps)
    if num_irreps is None:
        num_irreps = len(irreps)
    if num_irreps < 1:
        raise ValueError("num_irreps must be positive")

    if ranking not in {"power", "power_per_hidden"}:
        raise ValueError("ranking must be 'power' or 'power_per_hidden'")
    scored = []
    for index, irrep in enumerate(irreps):
        score = fourier_power(signal, irrep, normalize_by_dim=normalize_by_dim)
        if ranking == "power_per_hidden":
            score /= hidden_width(irrep, q_rho=q_rho)
        scored.append((score, index))
    ranked = sorted(scored, reverse=True)
    candidates = [index for _, index in ranked]
    if always_include_trivial and 0 in candidates:
        candidates.remove(0)
        candidates.insert(0, 0)

    selected = []
    width = 0
    for index in candidates:
        contribution = hidden_width(irreps[index], q_rho=q_rho)
        if max_hidden_width is not None and width + contribution > max_hidden_width:
            continue
        selected.append(index)
        width += contribution
        if len(selected) >= num_irreps:
            break

    if not selected:
        raise ValueError("hidden-width budget excludes every irrep")
    selected.sort()
    return [irreps[index] for index in selected], selected


def _amplitude_factors(
    irrep_dim: int,
    q_rho: int,
    group_order: int,
    mode: str,
    multipliers: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, float, float]:
    product = irrep_dim / (q_rho * group_order)
    if mode == "balanced":
        amplitude = product ** (1 / 3)
        baseline = np.asarray((amplitude, amplitude, amplitude))
    elif mode == "put_on_drive":
        baseline = np.asarray((1.0, product, 1.0))
    else:
        raise ValueError("amplitude_mode must be 'balanced' or 'put_on_drive'")

    multipliers_array = np.asarray(multipliers, dtype=float)
    if multipliers_array.shape != (3,):
        raise ValueError("amplitude_multipliers must contain exactly three values")
    if not np.all(np.isfinite(multipliers_array)) or np.any(multipliers_array <= 0):
        raise ValueError("amplitude_multipliers must be finite and positive")
    if not np.isclose(np.prod(multipliers_array), 1.0):
        raise ValueError(
            "amplitude_multipliers must have product one to preserve the "
            "closed-form RNN identity"
        )
    return tuple(baseline * multipliers_array)


def _matrix_unit(dim: int, row: int, column: int) -> np.ndarray:
    result = np.zeros((dim, dim), dtype=np.complex128)
    result[row, column] = 1.0
    return result


def _trace_features(irrep_matrices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("gab,ba->g", irrep_matrices, matrix))


class FiniteGroupRNN(nn.Module):
    """Analytically constructed recurrent network with fixed PyTorch weights.

    ``W_in``, ``W_drive``, ``W_out``, and the optional dense ``W_mix`` are
    buffers rather than :class:`torch.nn.Parameter` objects.  They therefore
    move with :meth:`~torch.nn.Module.to` and appear in ``state_dict()``, but
    are excluded from ``parameters()`` and are not trainable.

    The default recurrence keeps mixing factored as
    ``W_in @ (W_out @ hidden)`` to avoid allocating a hidden-by-hidden matrix.
    """

    def __init__(
        self,
        *,
        group,
        physical_group,
        action_side: str,
        irreps: list,
        all_irreps: list,
        selected_irrep_indices: list[int],
        q_rho: int,
        x_ego: np.ndarray,
        W_in: np.ndarray,
        W_drive: np.ndarray,
        W_out: np.ndarray,
        metadata: list[dict],
        amplitude_mode: str,
        amplitude_multipliers: tuple[float, float, float],
        W_mix: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.group = group
        self.physical_group = physical_group
        self.action_side = action_side
        self.irreps = list(irreps)
        self.all_irreps = list(all_irreps)
        self.selected_irrep_indices = list(selected_irrep_indices)
        self.q_rho = q_rho
        self.metadata = list(metadata)
        self.amplitude_mode = amplitude_mode
        self.amplitude_multipliers = amplitude_multipliers

        self.register_buffer("x_ego", torch.as_tensor(x_ego, dtype=torch.float64))
        self.register_buffer("W_in", torch.as_tensor(W_in, dtype=torch.float64))
        self.register_buffer("W_drive", torch.as_tensor(W_drive, dtype=torch.float64))
        self.register_buffer("W_out", torch.as_tensor(W_out, dtype=torch.float64))
        self.register_buffer(
            "W_mix",
            None if W_mix is None else torch.as_tensor(W_mix, dtype=torch.float64),
        )

    @property
    def hidden_dim(self) -> int:
        return self.W_in.shape[0]

    @property
    def group_size(self) -> int:
        return self.W_in.shape[1]

    def _as_tensor(self, values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=self.W_in.dtype, device=self.W_in.device)

    def apply_mix(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply ``W_in W_out`` without requiring a dense hidden-by-hidden matrix."""
        hidden = self._as_tensor(hidden)
        if self.W_mix is not None:
            return F.linear(hidden, self.W_mix)
        return F.linear(F.linear(hidden, self.W_out), self.W_in)

    def _run(
        self,
        x_allo,
        drives,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all outputs and hidden states for encoded drive signals."""
        x_allo = self._as_tensor(x_allo)
        drives = self._as_tensor(drives)

        if x_allo.ndim == 1:
            x_allo = x_allo.unsqueeze(0)
        if drives.ndim == 2:
            drives = drives.unsqueeze(0)
        if x_allo.ndim != 2 or x_allo.shape[-1] != self.group_size:
            raise ValueError(
                f"x_allo must have shape ({self.group_size},) or "
                f"(batch, {self.group_size}), got {tuple(x_allo.shape)}"
            )
        if drives.ndim != 3 or drives.shape[-1] != self.group_size:
            raise ValueError(
                f"drives must have shape (steps, {self.group_size}) or "
                f"(batch, steps, {self.group_size}), got {tuple(drives.shape)}"
            )
        if drives.shape[1] == 0:
            raise ValueError("drives must contain at least one step")
        if x_allo.shape[0] != drives.shape[0]:
            if x_allo.shape[0] == 1:
                x_allo = x_allo.expand(drives.shape[0], -1)
            elif drives.shape[0] == 1:
                drives = drives.expand(x_allo.shape[0], -1, -1)
            else:
                raise ValueError("x_allo and drives batch dimensions must match")

        hidden = squared_relu(
            F.linear(x_allo, self.W_in)
            + F.linear(drives[:, 0], self.W_drive)
        )
        hidden_states = [hidden]
        outputs = [F.linear(hidden, self.W_out)]
        for step in range(1, drives.shape[1]):
            hidden = squared_relu(
                self.apply_mix(hidden)
                + F.linear(drives[:, step], self.W_drive)
            )
            hidden_states.append(hidden)
            outputs.append(F.linear(hidden, self.W_out))
        return torch.stack(outputs, dim=1), torch.stack(hidden_states, dim=1)

    def forward(
        self,
        x_allo,
        drives,
        *,
        return_all_outputs: bool = False,
    ) -> torch.Tensor:
        """Evaluate encoded drive signals with optional batched leading axes.

        Args:
            x_allo: Allocentric signal with shape ``(group_size,)`` or
                ``(batch, group_size)``.
            drives: Egocentric drive signals with shape
                ``(steps, group_size)`` or ``(batch, steps, group_size)``.
            return_all_outputs: Return one output per drive rather than only
                the final output.
        """
        unbatched = torch.as_tensor(x_allo).ndim == 1 and torch.as_tensor(drives).ndim == 2
        outputs, _ = self._run(x_allo, drives)
        result = outputs if return_all_outputs else outputs[:, -1]
        return result.squeeze(0) if unbatched else result

    def encode_drives(self, sequence) -> np.ndarray:
        """Encode group elements using this model's action convention."""
        x_ego = self.x_ego.detach().cpu().numpy()
        sequence = [int(element) for element in sequence]
        if not sequence:
            raise ValueError("sequence must contain at least one group element")
        return np.stack(
            [self.group.left_action(element, x_ego) for element in sequence]
        )

    def rollout(self, x_allo, sequence) -> dict[str, torch.Tensor]:
        """Evaluate group elements and return outputs, targets, and hidden states."""
        sequence = [int(element) for element in sequence]
        if not sequence:
            raise ValueError("sequence must contain at least one group element")

        drives = self.encode_drives(sequence)
        outputs, hidden_states = self._run(x_allo, drives)

        cumulative = self.group.identity()
        cumulative_states = []
        true_outputs = []
        x_allo_array = (
            x_allo.detach().cpu().numpy()
            if isinstance(x_allo, torch.Tensor)
            else np.asarray(x_allo)
        )
        if x_allo_array.ndim != 1:
            raise ValueError("rollout currently expects one unbatched allocentric signal")
        for element in sequence:
            cumulative = self.group.compose(element, cumulative)
            cumulative_states.append(cumulative)
            true_outputs.append(self.group.left_action(cumulative, x_allo_array))

        return {
            "cumulative_states": torch.as_tensor(
                cumulative_states, dtype=torch.long, device=self.W_in.device
            ),
            "true_outputs": self._as_tensor(np.asarray(true_outputs)),
            "predicted_outputs": outputs.squeeze(0),
            "hidden_states": hidden_states.squeeze(0),
        }

    def probe_hidden_states(
        self,
        x_allo,
        *,
        drive_element: int | None = None,
    ) -> torch.Tensor:
        """Evaluate static input tuning over all transformed allocentric signals."""
        if drive_element is None:
            drive_element = self.group.identity()
        x_allo = (
            x_allo.detach().cpu().numpy()
            if isinstance(x_allo, torch.Tensor)
            else np.asarray(x_allo)
        )
        allocentric_orbit = np.stack(
            [
                self.group.left_action(element, x_allo)
                for element in self.group.elements()
            ]
        )
        drive = self.group.left_action(
            drive_element, self.x_ego.detach().cpu().numpy()
        )
        return squared_relu(
            F.linear(self._as_tensor(allocentric_orbit), self.W_in)
            + F.linear(self._as_tensor(drive), self.W_drive)
        )


def build_finite_group_rnn(
    group,
    x_ego: np.ndarray,
    *,
    irreps=None,
    x_allo: np.ndarray | None = None,
    q_rho: int = 3,
    amplitude_mode: str = "balanced",
    amplitude_multipliers: tuple[float, float, float] = (1.0, 1.0, 1.0),
    irrep_selection: str = "all",
    num_irreps: int | None = None,
    max_hidden_width: int | None = None,
    normalize_power_by_dim: bool = True,
    always_include_trivial: bool = True,
    power_ranking: str = "power",
    materialize_mix: bool = False,
    action_side: str = "right",
) -> FiniteGroupRNN:
    """Build closed-form RNN weights from finite-group irreps.

    By default, the construction uses ``group.irreps()`` as the complete irrep
    list.  Passing ``irreps`` overrides that list, which is useful when irreps
    have already been materialized or prefiltered.

    The ``irrep_selection`` argument controls which irreps from this list are
    retained:

    - ``"all"`` uses every available irrep and gives the complete construction.
    - ``"first"`` keeps the first ``num_irreps`` irreps in list order.
    - ``"power"`` ranks irreps by Fourier power in ``x_allo`` and keeps a
      truncated subset.  In this mode, ``x_allo`` is required.

    ``num_irreps`` is a count limit for the ``"first"`` and ``"power"`` modes.
    ``max_hidden_width`` can further restrict ``"power"`` selection by skipping
    irreps whose hidden-width contribution would exceed the budget.

    ``action_side="right"`` uses body-frame updates: a state ``s`` followed by
    a drive ``g`` becomes ``s * g``.  Internally this is the left action of the
    opposite group.  Set ``action_side="left"`` to recover the spatial update
    convention ``g * s``.

    ``q_rho`` is the per-irrep phase/multiplicity parameter in the closed-form
    construction.  An irrep of dimension ``d`` contributes
    ``4 * q_rho * d**3`` hidden units; with the default ``q_rho=3``, this is
    ``12 * d**3`` units per retained irrep.

    ``amplitude_multipliers`` rescales the baseline
    ``(A_u, A_v, A_w)`` factors. Its three entries must be positive and have
    product one so the closed-form reconstruction identity remains unchanged.
    """
    _validate_q_rho(q_rho)
    physical_group = group
    group = as_action_group(group, action_side)
    if irreps is None:
        all_irreps = list(group.irreps())
    elif action_side == "right":
        all_irreps = [opposite_irrep(irrep) for irrep in irreps]
    else:
        all_irreps = list(irreps)
    x_ego = np.asarray(x_ego)
    if x_ego.shape != (group.order,):
        raise ValueError(f"x_ego must have shape ({group.order},), got {x_ego.shape}")

    if irrep_selection == "all":
        selected_irreps = all_irreps
        selected_indices = list(range(len(all_irreps)))
    elif irrep_selection == "first":
        count = len(all_irreps) if num_irreps is None else num_irreps
        selected_indices = list(range(min(count, len(all_irreps))))
        selected_irreps = [all_irreps[index] for index in selected_indices]
    elif irrep_selection == "power":
        if x_allo is None:
            raise ValueError("x_allo is required for power-based irrep selection")
        selected_irreps, selected_indices = select_irreps_by_power(
            all_irreps,
            x_allo,
            num_irreps=num_irreps,
            max_hidden_width=max_hidden_width,
            q_rho=q_rho,
            normalize_by_dim=normalize_power_by_dim,
            always_include_trivial=always_include_trivial,
            ranking=power_ranking,
        )
    else:
        raise ValueError("irrep_selection must be 'all', 'first', or 'power'")

    rows_in = []
    rows_drive = []
    columns_out = []
    metadata = []
    sign_pairs = ((1, 1), (-1, 1), (-1, -1), (1, -1))

    for local_index, irrep in enumerate(selected_irreps):
        global_index = selected_indices[local_index]
        dim = irrep.dim
        matrices = _irrep_matrices(irrep, group.order)
        xhat = fourier_hat(x_ego, irrep, matrices=matrices)
        min_singular_value = float(np.linalg.svd(xhat, compute_uv=False).min())
        if min_singular_value < 1e-10:
            raise ValueError(
                f"x_ego Fourier block is nearly singular for {irrep}: "
                f"minimum singular value={min_singular_value}"
            )
        xhat_inv_dagger = np.linalg.inv(xhat.conj().T)
        amplitude_in, amplitude_drive, amplitude_out = _amplitude_factors(
            dim,
            q_rho,
            group.order,
            amplitude_mode,
            amplitude_multipliers,
        )

        for eps1, eps2 in sign_pairs:
            for delta in range(q_rho):
                phase_in = np.exp(1j * np.pi * delta / q_rho)
                phase_drive = phase_in
                phase_out = np.exp(2j * np.pi * delta / q_rho)
                for k0 in range(dim):
                    for k1 in range(dim):
                        for k2 in range(dim):
                            matrix_in = (
                                eps1
                                * amplitude_in
                                * phase_in
                                * _matrix_unit(dim, k0, k2)
                            )
                            matrix_drive = (
                                eps1
                                * eps2
                                * amplitude_drive
                                * phase_drive
                                * (xhat_inv_dagger @ _matrix_unit(dim, k2, k1))
                            )
                            matrix_out = (
                                eps2
                                * amplitude_out
                                * phase_out
                                * _matrix_unit(dim, k0, k1)
                            )
                            rows_in.append(_trace_features(matrices, matrix_in))
                            rows_drive.append(_trace_features(matrices, matrix_drive))
                            columns_out.append(_trace_features(matrices, matrix_out))
                            metadata.append(
                                {
                                    "irrep_index": global_index,
                                    "irrep_name": str(irrep),
                                    "irrep_dim": dim,
                                    "eps1": eps1,
                                    "eps2": eps2,
                                    "delta": delta,
                                    "k0": k0,
                                    "k1": k1,
                                    "k2": k2,
                                }
                            )

    W_in = np.asarray(rows_in)
    W_drive = np.asarray(rows_drive)
    W_out = np.asarray(columns_out).T
    W_mix = W_in @ W_out if materialize_mix else None
    return FiniteGroupRNN(
        group=group,
        physical_group=physical_group,
        action_side=action_side,
        irreps=selected_irreps,
        all_irreps=all_irreps,
        selected_irrep_indices=selected_indices,
        q_rho=q_rho,
        x_ego=x_ego,
        W_in=W_in,
        W_drive=W_drive,
        W_out=W_out,
        W_mix=W_mix,
        metadata=metadata,
        amplitude_mode=amplitude_mode,
        amplitude_multipliers=tuple(float(value) for value in amplitude_multipliers),
    )
