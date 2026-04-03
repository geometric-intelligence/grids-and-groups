import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loss plateau prediction functions (extracted from former CyclicPower /
# GroupPower classes)
# ---------------------------------------------------------------------------


def loss_plateau_predictions_cyclic(template, template_dim, p1=None, p2=None):
    """Compute theoretical loss plateau predictions for a cyclic template.

    Parameters
    ----------
    template : np.ndarray
        The template array.  For ``template_dim == 1``, shape ``(group_size,)``.
        For ``template_dim == 2``, shape ``(p1, p2)`` or flattened ``(p1*p2,)``.
    template_dim : int
        ``1`` for C_n (1D cyclic), ``2`` for C_n x C_m (2D).
    p1, p2 : int, optional
        Required when ``template_dim == 2`` to specify the grid shape.

    Returns
    -------
    list of float
        Theoretical loss plateau predictions for each nonzero power, in descending order.
    """
    template = np.asarray(template)

    if template_dim == 2:
        if p1 is not None and p2 is not None:
            template_2d = template.reshape((p1, p2))
            group_size = p1 * p2
            print("Computing loss plateau predictions for template of shape:", (p1, p2))
        else:
            flat = template.ravel()
            g = int(np.sqrt(len(flat)))
            if g * g != len(flat):
                raise ValueError(
                    "2D cyclic template must be square or pass p1, p2 for a rectangular grid"
                )
            template_2d = flat.reshape((g, g))
            group_size = g * g
            print("Computing loss plateau predictions for template of shape:", (g, g))

        ft = np.fft.rfft2(template_2d)
        M, N = template_2d.shape
        power = np.abs(ft) ** 2 / (M * N)
        power[(N // 2 + 1):, 0] = 0
        power *= 2
        power[0, 0] /= 2
        if N % 2 == 0:
            power[N // 2, 0] /= 2
        power = power.flatten()
    else:
        template = template.ravel()
        group_size = len(template)
        num_coefficients = (group_size // 2) + 1
        ft = np.fft.fft(template)
        power = np.abs(ft[:num_coefficients]) ** 2 / group_size
        if group_size % 2 == 0:
            power[1: num_coefficients - 1] *= 2
        else:
            power[1:] *= 2

    nonzero_power_mask = power > 1e-20
    power = power[nonzero_power_mask]
    i_power_descending_order = np.argsort(power)[::-1]
    power = power[i_power_descending_order]

    coef = 1 / group_size
    return [coef * np.sum(power[k:]) for k in range(len(power))]


def loss_plateau_predictions_group(template, group):
    """Compute theoretical loss plateau predictions for a generic group template.

    Parameters
    ----------
    template : np.ndarray, shape (group.order,)
        The template array.
    group : Group
        A group instance with a ``power_spectrum`` method.

    Returns
    -------
    list of float
        Theoretical loss plateau predictions for each nonzero power, in descending order.
    """
    p = len(template)
    print("Computing loss plateau predictions for template of shape:", (p,))
    power = group.power_spectrum(template)
    nonzero_power_mask = power > 1e-20
    power = power[nonzero_power_mask]
    print("Found ", len(power), "non-zero power coefficients.")
    i_power_descending_order = np.argsort(power)[::-1]
    power = power[i_power_descending_order]
    coef = 1 / p
    return [coef * np.sum(power[k:]) for k in range(len(power))]


# ---------------------------------------------------------------------------
# Per-neuron power helpers
# ---------------------------------------------------------------------------


def powers_per_neuron_rows(W: np.ndarray, group) -> np.ndarray:
    """Irrep power spectrum for each row of ``W`` using ``group.power_spectrum``.

    Each row is treated as a real signal on ``group`` (length ``group.order``).

    Parameters
    ----------
    W : ndarray, shape (hidden_dim, group.order)
    group : Group
        Must match the group structure of the weight rows.

    Returns
    -------
    ndarray, shape (hidden_dim, len(group.irreps()))
        ``out[h, i]`` is the normalized irrep power at index ``i`` for hidden unit ``h``.
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


def powers_per_neuron_rows_cyclic(
    W: np.ndarray,
    *,
    template_dim: int,
    p1: int | None = None,
    p2: int | None = None,
) -> np.ndarray:
    """Cyclic power spectrum for each row of ``W``.

    Parameters
    ----------
    W : ndarray, shape (hidden_dim, group_size) or (hidden_dim, p1 * p2)
    template_dim : int
        ``1`` for C_n (1D), ``2`` for CnxCn grid flattened row-major.
    p1, p2 : int, optional
        Required when ``template_dim == 2`` (rectangular or explicit grid shape).

    Returns
    -------
    ndarray
        For 1D, shape ``(hidden_dim, group_size // 2 + 1)``.
        For 2D, shape ``(hidden_dim, M * (N//2 + 1))``
        with ``M, N = p1, p2``.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D, got shape {W.shape}")
    hidden = W.shape[0]
    if template_dim == 1:
        p = W.shape[1]
        num_coeffs = (p // 2) + 1
        out = np.empty((hidden, num_coeffs))
        for h in range(hidden):
            pw, _ = get_power_1d(W[h])
            out[h] = pw
        return out
    if template_dim != 2:
        raise ValueError(f"template_dim must be 1 or 2, got {template_dim}")
    if p1 is None or p2 is None:
        raise ValueError("p1 and p2 are required for cyclic 2D power (template_dim=2)")
    if W.shape[1] != p1 * p2:
        raise ValueError(f"W.shape[1] ({W.shape[1]}) must equal p1*p2 ({p1 * p2})")
    pw0 = get_power_2d(W[0].reshape(p1, p2), no_freq=True)
    n_bins = pw0.size
    out = np.empty((hidden, n_bins))
    out[0] = pw0.ravel()
    for h in range(1, hidden):
        out[h] = get_power_2d(W[h].reshape(p1, p2), no_freq=True).ravel()
    return out


def model_power_over_time(group_name, model, param_history, model_inputs, group=None):
    """Compute the power spectrum of the model's learned weights over time.

    Parameters
    ----------
    group_name : str
        Group type (e.g., 'cnxcn').
    model : nn.Module
        The trained model (TwoLayerMLP or QuadraticRNN).
    param_history : list of dict
        List of model parameters at each training step.
    model_inputs : torch.Tensor
        Input data tensor.
    group : Group, optional
        Required for non-cyclic groups.

    Returns
    -------
    avg_power_history : list of ndarray (num_steps, num_freqs)
        List of average power spectra at each step.
    """
    model.eval()
    with torch.no_grad():
        test_output = model(model_inputs[:1])
    output_shape = test_output.shape[1:]

    if group_name == "cnxcn":
        p1 = int(np.sqrt(output_shape[0]))
        p2 = p1
        template_power_length = p1 * (p2 // 2 + 1)
        reshape_dims = (-1, p1, p2)
    elif group_name == "cn":
        template_power_length = (output_shape[0] // 2) + 1
        p1 = output_shape[0]
        reshape_dims = (-1, p1)
    else:
        template_power_length = len(group.irreps())
        p1 = output_shape[0]
        reshape_dims = (-1, p1)

    num_points = 200
    max_step = len(param_history) - 1
    num_inputs_to_compute_power = max(1, len(model_inputs) // 50)
    X_tensor = model_inputs[:num_inputs_to_compute_power]
    if max_step <= 1:
        steps = np.arange(max_step + 1)
    else:
        steps = np.unique(np.logspace(1, np.log10(max_step), num_points, dtype=int))
        steps = steps[steps > 50]
        steps = np.hstack([np.linspace(1, min(50, max_step), 5).astype(int), steps])
    steps = np.unique(steps)
    steps = steps[steps <= max_step]
    powers_over_time = np.zeros([len(steps), template_power_length])

    for i_step, step in enumerate(steps):
        model.load_state_dict(param_history[step])

        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            outputs_arr = outputs.detach().cpu().numpy().reshape(reshape_dims)

            if i_step % 10 == 0:
                print("Computing power at step", step, "with output shape", outputs_arr.shape)

            powers = []
            for out in outputs_arr:
                if group_name == "cnxcn":
                    one_power = get_power_2d(out, no_freq=True).flatten()
                elif group_name == "cn":
                    one_power, _ = get_power_1d(out.flatten())
                else:
                    one_power = group.power_spectrum(out.flatten())
                powers.append(one_power.flatten())
            powers = np.array(powers)

            average_power = np.mean(powers, axis=0)
            powers_over_time[i_step, :] = average_power

    powers_over_time = np.array(powers_over_time)
    powers_over_time[powers_over_time < 1e-20] = 0

    return powers_over_time, steps


# ---------------------------------------------------------------------------
# Power spectrum computation functions
# ---------------------------------------------------------------------------


def get_power_1d(points_1d):
    """Compute 1D power spectrum using rfft (for real-valued inputs).

    Args:
        points_1d: (p,) array

    Returns:
        power: (p//2+1,) array of power values
        freqs: frequency indices
    """
    p = len(points_1d)

    ft = np.fft.rfft(points_1d)
    power = np.abs(ft) ** 2 / p

    power = 2 * power.copy()
    power[0] = power[0] / 2  # DC component
    if p % 2 == 0:
        power[-1] = power[-1] / 2  # Nyquist frequency

    freqs = np.fft.rfftfreq(p, 1.0) * p

    return power, freqs


def topk_template_freqs_1d(template_1d: np.ndarray, K: int, min_power: float = 1e-20):
    """Return top-K frequency indices by power for 1D template.

    Args:
        template_1d: 1D template array (p,)
        K: Number of top frequencies to return
        min_power: Minimum power threshold

    Returns:
        List of frequency indices (as integers)
    """
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

    Args:
        points: (M, N) array, the 2D signal
        no_freq: if True, only return power (no frequency arrays)

    Returns:
        freqs_u: frequency bins for rows (if no_freq=False)
        freqs_v: frequency bins for columns (if no_freq=False)
        power: 2D power spectrum (M, N//2 + 1)
    """
    M, N = points.shape

    ft = np.fft.rfft2(points)
    power = np.abs(ft) ** 2 / (M * N)

    weight = 2 * np.ones((M, N // 2 + 1))
    weight[0, 0] = 1
    weight[(M // 2 + 1):, 0] = 0
    if M % 2 == 0:
        weight[M // 2, 0] = 1
    if N % 2 == 0:
        weight[(M // 2 + 1):, N // 2] = 0
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
    """Sum power at (kx, ky) and its real-signal mirror (-kx, -ky).

    Args:
        power2d: 2D power spectrum from fft2 (shape: p1, p2)
        kx, ky: Frequency indices
        p1, p2: Dimensions of the signal

    Returns:
        float: Total power at this frequency (including mirror)
    """
    i0, j0 = kx % p1, ky % p2
    i1, j1 = (-kx) % p1, (-ky) % p2
    if (i0, j0) == (i1, j1):
        return float(power2d[i0, j0])
    return float(power2d[i0, j0] + power2d[i1, j1])


def theoretical_loss_levels_2d(template_2d):
    """Compute theoretical MSE loss levels based on 2D template power spectrum.

    Args:
        template_2d: 2D template array (p1, p2)

    Returns:
        dict with 'initial', 'final', and 'levels' keys
    """
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
    """Compute theoretical MSE loss levels based on 1D template power spectrum.

    Args:
        template_1d: 1D template array (p,)

    Returns:
        dict with 'initial', 'final', and 'levels' keys
    """
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


# Backward compatibility aliases
def theoretical_final_loss_2d(template_2d):
    """Returns expected initial loss (for setting convergence targets)."""
    return theoretical_loss_levels_2d(template_2d)["initial"]


def theoretical_final_loss_1d(template_1d):
    """Returns expected initial loss (for setting convergence targets)."""
    return theoretical_loss_levels_1d(template_1d)["initial"]
