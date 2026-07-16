"""Closed-form QuadraticRNN construction for finite-group actions.

The construction is group-agnostic.  A group must expose ``order``,
``elements()``, ``irreps()``, ``left_action()``, ``identity()``, and
``compose()``.  Irreps may be dense or lazy as long as they expose ``dim`` and
are callable on an element index.
"""

from dataclasses import dataclass

import numpy as np


def squared_relu(values: np.ndarray) -> np.ndarray:
    """Elementwise squared ReLU."""
    return np.maximum(0, values) ** 2


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
    return 4 * q_rho * irrep.dim**3


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
) -> tuple[float, float, float]:
    product = irrep_dim / (q_rho * group_order)
    if mode == "balanced":
        amplitude = product ** (1 / 3)
        return amplitude, amplitude, amplitude
    if mode == "put_on_drive":
        return 1.0, product, 1.0
    raise ValueError("amplitude_mode must be 'balanced' or 'put_on_drive'")


def _matrix_unit(dim: int, row: int, column: int) -> np.ndarray:
    result = np.zeros((dim, dim), dtype=np.complex128)
    result[row, column] = 1.0
    return result


def _trace_features(irrep_matrices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("gab,ba->g", irrep_matrices, matrix))


@dataclass
class FiniteGroupRNNParams:
    """Weights and irrep metadata for a closed-form finite-group RNN."""

    group: object
    irreps: list
    all_irreps: list
    selected_irrep_indices: list[int]
    q_rho: int
    x_ego: np.ndarray
    W_in: np.ndarray
    W_drive: np.ndarray
    W_out: np.ndarray
    metadata: list[dict]
    amplitude_mode: str
    W_mix: np.ndarray | None = None

    @property
    def hidden_dim(self) -> int:
        return self.W_in.shape[0]

    def apply_mix(self, hidden: np.ndarray) -> np.ndarray:
        """Apply ``W_in W_out`` without requiring a dense hidden-by-hidden matrix."""
        if self.W_mix is not None:
            return self.W_mix @ hidden
        return self.W_in @ (self.W_out @ hidden)


def build_finite_group_rnn(
    group,
    x_ego: np.ndarray,
    *,
    irreps=None,
    x_allo: np.ndarray | None = None,
    q_rho: int = 3,
    amplitude_mode: str = "balanced",
    irrep_selection: str = "all",
    num_irreps: int | None = None,
    max_hidden_width: int | None = None,
    normalize_power_by_dim: bool = True,
    always_include_trivial: bool = True,
    power_ranking: str = "power",
    materialize_mix: bool = False,
) -> FiniteGroupRNNParams:
    """Build closed-form RNN weights from finite-group irreps.

    ``irrep_selection='all'`` gives the complete construction.
    ``irrep_selection='power'`` keeps high-power blocks of ``x_allo`` and is a
    truncated approximation.
    """
    all_irreps = list(group.irreps() if irreps is None else irreps)
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
            dim, q_rho, group.order, amplitude_mode
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
    return FiniteGroupRNNParams(
        group=group,
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
    )


def forward_sequence(
    params: FiniteGroupRNNParams,
    x_allo: np.ndarray,
    sequence,
) -> tuple[np.ndarray, np.ndarray]:
    """Return final output and hidden state for a nonempty drive sequence."""
    sequence = list(sequence)
    if not sequence:
        raise ValueError("sequence must contain at least one group element")
    group = params.group
    hidden = squared_relu(
        params.W_in @ x_allo
        + params.W_drive @ group.left_action(sequence[0], params.x_ego)
    )
    for element in sequence[1:]:
        hidden = squared_relu(
            params.apply_mix(hidden)
            + params.W_drive @ group.left_action(element, params.x_ego)
        )
    return params.W_out @ hidden, hidden


def rollout(params: FiniteGroupRNNParams, x_allo: np.ndarray, sequence) -> dict[str, np.ndarray]:
    """Return predictions, exact action targets, hidden states, and group states."""
    group = params.group
    cumulative = group.identity()
    hidden = None
    cumulative_states = []
    true_outputs = []
    predicted_outputs = []
    hidden_states = []

    for step, element in enumerate(sequence):
        element = int(element)
        cumulative = group.compose(element, cumulative)
        drive = params.W_drive @ group.left_action(element, params.x_ego)
        if step == 0:
            hidden = squared_relu(params.W_in @ x_allo + drive)
        else:
            hidden = squared_relu(params.apply_mix(hidden) + drive)
        predicted = params.W_out @ hidden
        cumulative_states.append(cumulative)
        true_outputs.append(group.left_action(cumulative, x_allo))
        predicted_outputs.append(predicted)
        hidden_states.append(hidden.copy())

    return {
        "cumulative_states": np.asarray(cumulative_states),
        "true_outputs": np.asarray(true_outputs),
        "predicted_outputs": np.asarray(predicted_outputs),
        "hidden_states": np.asarray(hidden_states),
    }


def probe_hidden_states(
    params: FiniteGroupRNNParams,
    x_allo: np.ndarray,
    *,
    drive_element: int | None = None,
) -> np.ndarray:
    """Evaluate static input tuning over all transformed allocentric signals."""
    group = params.group
    if drive_element is None:
        drive_element = group.identity()
    drive = params.W_drive @ group.left_action(drive_element, params.x_ego)
    return np.asarray(
        [
            squared_relu(params.W_in @ group.left_action(g, x_allo) + drive)
            for g in group.elements()
        ]
    )
