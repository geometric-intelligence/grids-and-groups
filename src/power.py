import numpy as np
import torch

# ---------------------------------------------------------------------------
# Loss plateau prediction
# ---------------------------------------------------------------------------


def loss_plateau_predictions(template, group):
    """Compute theoretical MSE loss plateau predictions for a group template.

    Uses ``group.power_spectrum`` to obtain per-irrep power, normalizes by
    ``|G|`` (so that the total matches ``||template||^2``, i.e. the Parseval
    convention used by ``nn.MSELoss``), then returns cumulative sums in
    descending power order.

    Replaces the former ``loss_plateau_predictions_cyclic`` (which used
    ``np.fft.rfft`` / ``np.fft.rfft2`` with built-in ``1/N`` normalization)
    and ``loss_plateau_predictions_group``.  Equivalence verified in
    ``test/test_refactor_equivalence.py``: the ``1/|G|`` normalization of
    ``group.power_spectrum`` is the only difference; after correction the
    plateaus match the legacy cyclic code to machine precision.

    Parameters
    ----------
    template : np.ndarray, shape (group.order,)
        The template array (mean-centered).
    group : Group
        A group instance with a ``power_spectrum`` method and an ``order``
        property.

    Returns
    -------
    list of float
        Theoretical loss plateau predictions (one per non-zero power
        component, in descending-power order).  The first element is the
        initial MSE loss (before any learning).
    """
    template = np.asarray(template).ravel()
    p = group.order
    power = group.power_spectrum(template) / p

    nonzero_mask = power > 1e-20
    power = power[nonzero_mask]
    power = np.sort(power)[::-1]

    coef = 1 / p
    return [coef * np.sum(power[k:]) for k in range(len(power))]


# ---------------------------------------------------------------------------
# Per-neuron power helpers
# ---------------------------------------------------------------------------


def powers_per_neuron_rows(W: np.ndarray, group) -> np.ndarray:
    """Irrep power spectrum for each row of ``W`` using ``group.power_spectrum``.

    Each row is treated as a real signal on ``group`` (length ``group.order``).

    Replaces the former ``powers_per_neuron_rows_cyclic`` which used
    ``np.fft.rfft`` / ``np.fft.rfft2``.  The absolute scale differs by a
    factor of ``|G|`` (group convention vs normalized FFT), but all
    downstream consumers (dominant-mode fraction plots) use *ratios*, so
    this is transparent.

    Parameters
    ----------
    W : ndarray, shape (hidden_dim, group.order)
    group : Group
        Must match the group structure of the weight rows.

    Returns
    -------
    ndarray, shape (hidden_dim, len(group.irreps()))
        ``out[h, i]`` is the irrep power at index ``i`` for hidden unit ``h``.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D, got shape {W.shape}")
    if W.shape[1] != group.order:
        raise ValueError(f"W.shape[1] ({W.shape[1]}) must equal group.order ({group.order})")
    hidden = W.shape[0]
    n_irreps = len(group.irreps())
    out = np.empty((hidden, n_irreps))
    for h in range(hidden):
        out[h] = group.power_spectrum(W[h])
    return out


def model_power_over_time(group, model, param_history, model_inputs):
    """Compute the power spectrum of the model's learned outputs over time.

    Replaces the former version that branched on ``group_name`` (``"cn"`` /
    ``"cnxcn"`` / else).  Now uses ``group.power_spectrum`` uniformly.

    Parameters
    ----------
    group : Group
        The group object.
    model : nn.Module
        The trained model.
    param_history : list of dict
        State-dict snapshots at each saved training step.
    model_inputs : torch.Tensor
        Input data tensor (a small evaluation batch).

    Returns
    -------
    powers_over_time : ndarray, shape (num_steps, n_irreps)
        Average output power spectrum at each sampled step.
    steps : ndarray of int
        The param_history indices that were sampled.
    """
    n_irreps = len(group.irreps())

    model.eval()
    with torch.no_grad():
        test_output = model(model_inputs[:1])
    output_dim = test_output.shape[-1]

    num_points = 200
    max_step = len(param_history) - 1
    num_inputs = max(1, len(model_inputs) // 50)
    X_tensor = model_inputs[:num_inputs]

    if max_step <= 1:
        steps = np.arange(max_step + 1)
    else:
        steps = np.unique(np.logspace(1, np.log10(max_step), num_points, dtype=int))
        steps = steps[steps > 50]
        steps = np.hstack([np.linspace(1, min(50, max_step), 5).astype(int), steps])
    steps = np.unique(steps)
    steps = steps[steps <= max_step]

    powers_over_time = np.zeros([len(steps), n_irreps])

    for i_step, step in enumerate(steps):
        model.load_state_dict(param_history[step])
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            outputs_arr = outputs.detach().cpu().numpy().reshape(-1, output_dim)

            if i_step % 10 == 0:
                print("Computing power at step", step, "with output shape", outputs_arr.shape)

            powers = []
            for out in outputs_arr:
                one_power = group.power_spectrum(out.flatten())
                powers.append(one_power.flatten())
            powers = np.array(powers)

            average_power = np.mean(powers, axis=0)
            powers_over_time[i_step, :] = average_power

    powers_over_time = np.array(powers_over_time)
    powers_over_time[powers_over_time < 1e-20] = 0

    return powers_over_time, steps


# ---------------------------------------------------------------------------
# Legacy 1D / 2D FFT power helpers (kept for notebooks & non-group contexts)
# ---------------------------------------------------------------------------


def get_power_1d(points_1d):
    """Compute 1D power spectrum using rfft (for real-valued inputs).

    Parameters
    ----------
    points_1d : array, shape (p,)

    Returns
    -------
    power : array, shape (p//2+1,)
    freqs : array
    """
    p = len(points_1d)

    ft = np.fft.rfft(points_1d)
    power = np.abs(ft) ** 2 / p

    power = 2 * power.copy()
    power[0] = power[0] / 2
    if p % 2 == 0:
        power[-1] = power[-1] / 2

    freqs = np.fft.rfftfreq(p, 1.0) * p
    return power, freqs


def topk_template_freqs_1d(template_1d: np.ndarray, K: int, min_power: float = 1e-20):
    """Return top-K frequency indices by power for 1D template."""
    power, _ = get_power_1d(template_1d)
    mask = power > min_power
    if not np.any(mask):
        return []
    valid_power = power[mask]
    valid_indices = np.flatnonzero(mask)
    top_idx = valid_indices[np.argsort(valid_power)[::-1]][:K]
    return top_idx.tolist()


def topk_template_freqs(template_2d: np.ndarray, K: int, min_power: float = 1e-20):
    """Return top-K (kx, ky) rFFT2 bins by power from get_power_2d(template_2d)."""
    freqs_u, freqs_v, power = get_power_2d(template_2d)
    shp = power.shape
    flat = power.ravel()
    mask = flat > min_power
    if not np.any(mask):
        return []
    top_idx = np.flatnonzero(mask)[np.argsort(flat[mask])[::-1]][:K]
    kx, ky = np.unravel_index(top_idx, shp)
    return list(zip(kx.tolist(), ky.tolist()))


def get_power_2d(points, no_freq=False):
    """Compute 2D power spectrum using rfft2 with proper symmetry handling.

    Parameters
    ----------
    points : array, shape (M, N)
    no_freq : bool
        If True, only return power (no frequency arrays).

    Returns
    -------
    power : array, shape (M, N//2+1)
        (plus freqs_u, freqs_v when ``no_freq`` is False)
    """
    M, N = points.shape

    ft = np.fft.rfft2(points)
    power = np.abs(ft) ** 2 / (M * N)

    weight = 2 * np.ones((M, N // 2 + 1))
    weight[0, 0] = 1
    weight[(M // 2 + 1) :, 0] = 0
    if M % 2 == 0:
        weight[M // 2, 0] = 1
    if N % 2 == 0:
        weight[(M // 2 + 1) :, N // 2] = 0
        weight[0, N // 2] = 1
    if (M % 2 == 0) and (N % 2 == 0):
        weight[M // 2, N // 2] = 1

    power = weight * power

    total_power = np.sum(power)
    norm_squared = np.linalg.norm(points) ** 2
    if not np.isclose(total_power, norm_squared, rtol=1e-6):
        print(
            f"Warning: Total power {total_power:.3f} does not match norm squared {norm_squared:.3f}"
        )

    if no_freq:
        return power

    freqs_u = np.fft.fftfreq(M)
    freqs_v = np.fft.rfftfreq(N)
    return freqs_u, freqs_v, power


def _tracked_power_from_fft2(power2d, kx, ky, p1, p2):
    """Sum power at (kx, ky) and its real-signal mirror (-kx, -ky)."""
    i0, j0 = kx % p1, ky % p2
    i1, j1 = (-kx) % p1, (-ky) % p2
    if (i0, j0) == (i1, j1):
        return float(power2d[i0, j0])
    return float(power2d[i0, j0] + power2d[i1, j1])


def theoretical_loss_levels_2d(template_2d):
    """Compute theoretical MSE loss levels based on 2D template power spectrum."""
    p1, p2 = template_2d.shape
    power = get_power_2d(template_2d, no_freq=True)

    power_flat = power.flatten()
    power_flat = np.sort(power_flat[power_flat > 1e-20])[::-1]

    coef = 1.0 / (p1 * p2)
    levels = [coef * np.sum(power_flat[k:]) for k in range(len(power_flat) + 1)]

    return {
        "initial": levels[0] if levels else 0.0,
        "final": 0.0,
        "levels": levels,
    }


def theoretical_loss_levels_1d(template_1d):
    """Compute theoretical MSE loss levels based on 1D template power spectrum."""
    p = len(template_1d)
    power, _ = get_power_1d(template_1d)

    power = np.sort(power[power > 1e-20])[::-1]

    coef = 1.0 / p
    levels = [coef * np.sum(power[k:]) for k in range(len(power) + 1)]

    return {
        "initial": levels[0] if levels else 0.0,
        "final": 0.0,
        "levels": levels,
    }


def theoretical_final_loss_2d(template_2d):
    """Returns expected initial loss (for setting convergence targets)."""
    return theoretical_loss_levels_2d(template_2d)["initial"]


def theoretical_final_loss_1d(template_1d):
    """Returns expected initial loss (for setting convergence targets)."""
    return theoretical_loss_levels_1d(template_1d)["initial"]
