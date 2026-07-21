from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from src.groups.cn import CyclicGroup
from src.groups.cnxcn import ProductCyclicGroup


def _independent_irrep_indices(group):
    """Return irrep indices that are not conjugate duplicates.

    For ``CyclicGroup(N)`` the independent modes are ``0, 1, …, N//2``
    (the rest satisfy ``power[k] = power[N-k]``).

    For ``ProductCyclicGroup(p1, p2)`` one representative is kept from
    each ``{(j1, j2), ((-j1)%p1, (-j2)%p2)}`` pair (the one with the
    smaller row-major index).

    For all other groups every irrep is independent and all indices are
    returned.
    """
    if isinstance(group, CyclicGroup):
        N = group.order
        return list(range(N // 2 + 1))

    if isinstance(group, ProductCyclicGroup):
        p1, p2 = group._p1, group._p2
        seen = set()
        result = []
        for j1 in range(p1):
            for j2 in range(p2):
                idx = j1 * p2 + j2
                conj_idx = ((-j1) % p1) * p2 + ((-j2) % p2)
                if conj_idx not in seen:
                    seen.add(idx)
                    result.append(idx)
        return result

    return list(range(len(group.irreps())))


def loss_plateau_predictions(template, group):
    """Compute theoretical MSE loss plateau predictions for a group template.

    Uses ``group.power_spectrum`` (which already normalises by ``|G|``) to
    obtain per-irrep power, then returns cumulative sums in descending power
    order scaled by ``1/|G|`` (the MSE averaging factor).

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
    power = group.power_spectrum(template)

    nonzero_mask = power > 1e-20
    power = power[nonzero_mask]
    power = np.sort(power)[::-1]

    coef = 1 / p
    return [coef * np.sum(power[k:]) for k in range(len(power))]


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


def topk_template_freqs(template_2d: np.ndarray, K: int, min_power: float = 1e-20):
    """Return top-K (kx, ky) FFT2 bins by power for a 2D template.

    Parameters
    ----------
    template_2d : np.ndarray, shape (p1, p2)
        The 2D template array.
    K : int
        Number of top frequency bins to return.
    min_power : float
        Minimum power threshold for a bin to be considered.

    Returns
    -------
    list of tuple
        Top-K (kx, ky) frequency index pairs sorted by descending power.
    """
    F = np.fft.fft2(template_2d)
    pwr = (F.conj() * F).real
    shp = pwr.shape
    flat = pwr.ravel()
    mask = flat > min_power
    if not np.any(mask):
        return []
    top_idx = np.flatnonzero(mask)[np.argsort(flat[mask])[::-1]][:K]
    kx, ky = np.unravel_index(top_idx, shp)
    return list(zip(kx.tolist(), ky.tolist()))


def style_axes(ax, numyticks=5, numxticks=5, labelsize=24):
    # Y-axis ticks
    ax.tick_params(
        axis="y",
        which="both",
        bottom=True,
        top=False,
        labelbottom=True,
        left=True,
        right=False,
        labelleft=True,
        direction="out",
        length=7,
        width=1.5,
        pad=8,
        labelsize=labelsize,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=numyticks))

    # X-axis ticks
    ax.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        labelbottom=True,
        left=True,
        right=False,
        labelleft=True,
        direction="out",
        length=7,
        width=1.5,
        pad=8,
        labelsize=labelsize,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=numxticks))

    ax.xaxis.offsetText.set_fontsize(20)
    ax.grid()

    # Customize spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(3)


def _permutation_from_groups_with_dead(
    dom_idx, phase, dom_power, l2, *, within="phase", dead_l2_thresh=1e-1
):
    """Create neuron permutation grouped by dominant frequency.

    Args:
        dom_idx: Dominant frequency index for each neuron
        phase: Phase at dominant frequency for each neuron
        dom_power: Power at dominant frequency for each neuron
        l2: L2 norm of each neuron's weights
        within: How to order within groups ('phase', 'power', 'phase_power', 'none')
        dead_l2_thresh: L2 threshold below which neurons are "dead"

    Returns:
        perm: Permutation indices
        ordered_keys: Ordered list of group keys (-1 for dead)
        boundaries: Cumulative indices where groups end
    """
    dead_mask = l2 < float(dead_l2_thresh)
    groups = {}
    for i, f in enumerate(dom_idx):
        key = -1 if dead_mask[i] else int(f)
        groups.setdefault(key, []).append(i)

    freq_keys = sorted([k for k in groups.keys() if k >= 0])
    ordered_keys = freq_keys + ([-1] if -1 in groups else [])

    perm, boundaries = [], []
    for f in ordered_keys:
        idxs = groups[f]
        if f == -1:
            idxs = sorted(idxs, key=lambda i: l2[i])
        else:
            if within == "phase" and phase is not None:
                idxs = sorted(idxs, key=lambda i: (phase[i] + 2 * np.pi) % (2 * np.pi))
            elif within == "power" and dom_power is not None:
                idxs = sorted(idxs, key=lambda i: -dom_power[i])
            elif within == "phase_power":
                idxs = sorted(
                    idxs, key=lambda i: ((phase[i] + 2 * np.pi) % (2 * np.pi), -dom_power[i])
                )
        perm.extend(idxs)
        boundaries.append(len(perm))

    return np.array(perm, dtype=int), ordered_keys, boundaries


def _training_loss_log_y_floor(ax):
    """Set log-scale training-loss y-axis lower limit (fixed in code; tweak here as needed)."""
    ax.set_ylim(bottom=10e-2)


def _theory_loss_y_levels_from_run(run_dir: Path, cfg: dict) -> list[float] | None:
    """Template MSE plateau levels via ``loss_plateau_predictions``."""
    from src.groups import make_group

    tpl_path = run_dir / "template.npy"
    if not tpl_path.exists():
        return None

    template_np = np.load(tpl_path)
    gn = cfg["data"]["group_name"]

    try:
        group = make_group(gn, cfg)
    except (ValueError, KeyError):
        return None

    t = np.asarray(template_np).ravel()
    out = loss_plateau_predictions(t, group)
    return list(out) if out else None


def _draw_theory_loss_hlines(ax, theory_y_levels: list[float] | None) -> None:
    """Black dashed horizontal lines at MSE plateaus (combined top row)."""
    if not theory_y_levels:
        return
    for y in theory_y_levels:
        if y > 0 and np.isfinite(y):
            ax.axhline(
                y=y,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                zorder=1,
            )


def _add_line_labels(ax, lines_info, fontsize=12, min_frac_sep: float = 0.07):
    """Add right-aligned colored text labels above (or below) each line.

    Each label is anchored at the rightmost data-point of its line and
    right-aligned so the text extends leftward inside the axes box.
    When two labels would overlap vertically the lower one is placed
    below its line instead.

    Args:
        min_frac_sep: Minimum axis-fraction separation before flipping placement (larger ⇒ fewer overlaps).
    """
    if not lines_info:
        return

    label_data = []
    for info in lines_info:
        x_arr = np.asarray(info["x"])
        y_arr = np.asarray(info["y"])
        if len(x_arr) == 0:
            continue
        label_data.append(
            {
                "x_pos": x_arr[-1],
                "y_pos": y_arr[-1],
                "label": info["label"],
                "color": info["color"],
            }
        )

    if not label_data:
        return

    # Sort descending by y so highest line comes first
    label_data.sort(key=lambda d: -d["y_pos"])

    # Convert y-positions to axis-fraction for overlap detection
    y_lo, y_hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        log_lo = np.log10(max(y_lo, 1e-30))
        log_hi = np.log10(max(y_hi, 1e-30))
        span = max(log_hi - log_lo, 1e-30)
        fracs = [(np.log10(max(d["y_pos"], 1e-30)) - log_lo) / span for d in label_data]
    else:
        span = max(y_hi - y_lo, 1e-30)
        fracs = [(d["y_pos"] - y_lo) / span for d in label_data]

    placements = ["above"] * len(label_data)
    for i in range(1, len(label_data)):
        if abs(fracs[i - 1] - fracs[i]) < min_frac_sep:
            placements[i] = "below"

    # Keep labels inside the axes: flip if too close to the top/bottom edge
    for i in range(len(label_data)):
        if placements[i] == "above" and fracs[i] > 0.88:
            placements[i] = "below"
        elif placements[i] == "below" and fracs[i] < 0.12:
            placements[i] = "above"

    for d, placement in zip(label_data, placements):
        y_off = 3 if placement == "above" else -3
        va = "bottom" if placement == "above" else "top"
        ax.annotate(
            d["label"],
            xy=(d["x_pos"], d["y_pos"]),
            xytext=(-4, y_off),
            textcoords="offset points",
            color=d["color"],
            fontsize=fontsize,
            fontweight="bold",
            va=va,
            ha="right",
        )


def analyze_wout_frequency_dominance(state_dict, tracked_freqs, p1, p2):
    """Analyze W_out to find dominant frequency for each neuron.

    Args:
        state_dict: Model parameters (expects 'W_out' key)
        tracked_freqs: List of (kx, ky) tuples
        p1, p2: Template dimensions

    Returns:
        dom_idx: Dominant frequency index for each neuron
        phase: Phase at dominant frequency for each neuron
        dom_power: Power at dominant frequency for each neuron
        l2: L2 norm of each neuron's weights
    """
    Wo = state_dict["W_out"].detach().cpu().numpy()  # (p, H)
    W = Wo.T  # (H, p)
    H, D = W.shape
    assert D == p1 * p2

    dom_idx = np.empty(H, dtype=int)
    dom_pow = np.empty(H, dtype=float)
    phase = np.empty(H, dtype=float)
    l2 = np.linalg.norm(W, axis=1)

    for j in range(H):
        m = W[j].reshape(p1, p2)
        F = np.fft.fft2(m)
        P = (F.conj() * F).real
        tp = [P[kx % p1, ky % p2] for (kx, ky) in tracked_freqs]
        jj = int(np.argmax(tp))
        dom_idx[j] = jj
        i0, j0 = tracked_freqs[jj][0] % p1, tracked_freqs[jj][1] % p2
        phase[j] = np.angle(F[i0, j0])
        dom_pow[j] = tp[jj]

    return dom_idx, phase, dom_pow, l2


def plot_signal_2d(
    signal_2d,
    title="",
    cmap="RdBu_r",
    colorbar=True,
):
    """Plot a 2D signal as a heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(signal_2d, cmap=cmap, aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("y", fontsize=12)
    ax.set_ylabel("x", fontsize=12)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, ax


def plot_train_loss_with_theory(
    loss_history, template, group, x_values=None, x_label="Step", save_path=None, show=True
):
    """Plot training loss with theoretical power spectrum lines.

    Parameters
    ----------
    loss_history : list of float
    template : np.ndarray
        Template array (any shape; will be flattened).
    group : Group
        Group object used for plateau prediction.
    x_values, x_label, save_path, show : optional
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if x_values is None:
        x_values = np.arange(len(loss_history))

    ax.plot(x_values, loss_history, lw=4, color="#1f77b4", label="Training Loss")

    for y in loss_plateau_predictions(template.ravel(), group):
        ax.axhline(y=y, color="black", linestyle="--", linewidth=2, zorder=-2)

    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel("Train Loss", fontsize=24)

    style_axes(ax)
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved loss plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_predictions_group(
    model,
    param_hist,
    X_eval,
    Y_eval,
    group_size: int,
    checkpoint_indices: list,
    save_path: str = None,
    num_samples: int = 5,
    group_label: str = "Group",
):
    """Plot model predictions vs targets at different training checkpoints.

    Replaces the former ``plot_predictions_1d``, ``plot_predictions_2d``,
    and ``plot_predictions_group``.  Uses a bar-chart representation that
    works for any group order.
    """
    import torch

    n_checkpoints = len(checkpoint_indices)

    fig, axes = plt.subplots(
        num_samples, n_checkpoints, figsize=(4 * n_checkpoints, 3 * num_samples)
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if n_checkpoints == 1:
        axes = axes.reshape(-1, 1)

    sample_indices = np.random.choice(
        len(X_eval), size=min(num_samples, len(X_eval)), replace=False
    )

    for col, ckpt_idx in enumerate(checkpoint_indices):
        model.load_state_dict(param_hist[ckpt_idx])
        model.eval()

        with torch.no_grad():
            outputs = model(X_eval[sample_indices])
            outputs_np = outputs.cpu().numpy()
            targets_np = Y_eval[sample_indices].cpu().numpy()

        for row, (output, target) in enumerate(zip(outputs_np, targets_np)):
            ax = axes[row, col]
            x_axis = np.arange(group_size)

            ax.bar(x_axis - 0.15, target, width=0.3, label="Target", alpha=0.7, color="#2ecc71")
            ax.bar(x_axis + 0.15, output, width=0.3, label="Output", alpha=0.7, color="#e74c3c")

            if row == 0:
                ax.set_title(f"Checkpoint {ckpt_idx}")
            if col == 0:
                ax.set_ylabel(f"Sample {sample_indices[row]}")
            if row == num_samples - 1:
                ax.set_xlabel("Group element")
            if row == 0 and col == n_checkpoints - 1:
                ax.legend(loc="upper right", fontsize=8)

            ax.set_xticks(x_axis)
            ax.grid(True, alpha=0.3)

    plt.suptitle(f"{group_label} Predictions vs Targets Over Training", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def _hidden_by_group_weights_from_state_dict(sd: dict) -> np.ndarray | None:
    """Return weight matrix of shape (hidden, group_dim) for TwoLayerMLP ``W`` or RNN ``W_out.T``."""
    if "W" in sd:
        return sd["W"].detach().cpu().numpy()
    if "W_out" in sd:
        return sd["W_out"].detach().cpu().numpy().T
    return None


def _w_dominant_mode_colors(n_modes: int) -> list:
    """Colors for spectral modes / irreps (matches standalone ``plot_w_dominant_irrep_fraction``)."""
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(n_modes, 1))
    manual_colors = {
        0: "tab:blue",
        1: "tab:orange",
        2: "tab:red",
        3: "tab:green",
        4: "tab:brown",
        5: "tab:purple",
    }
    return [manual_colors.get(i, cmap(i)) for i in range(n_modes)]


def mode_colors_aligned_with_power_plot(
    n_modes: int,
    top_irrep_indices,
    colors_line,
) -> list:
    """One color per mode index 0..n_modes-1, matching :func:`plot_power_cn` / ``plot_power_group``.

    For each irrep/mode index that appears in the power-over-time plot, use that plot's line color.
    Other modes keep :func:`_w_dominant_mode_colors` so the W-row can still show unracked neurons.
    """
    base = _w_dominant_mode_colors(n_modes)
    out = list(base)
    top_irrep_indices = np.asarray(top_irrep_indices).ravel()
    colors_line = list(colors_line)
    for i, idx in enumerate(top_irrep_indices):
        idx = int(idx)
        if 0 <= idx < n_modes and i < len(colors_line):
            out[idx] = colors_line[i]
    return out


def compute_w_dominant_irrep_fraction_data(
    param_hist: list,
    param_save_indices: list | np.ndarray,
    group,
    num_points: int = 1000,
    active_energy_thresh: float = 1e-4,
) -> dict | None:
    """Compute per-neuron dominant-mode fraction curves.

    Uses ``powers_per_neuron_rows(W, group)`` uniformly for all groups.
    Replaces the former version that branched on ``group_name in ("cn","cnxcn")``.

    Returns a dict suitable for :func:`save_w_dominant_irrep_fraction_npz` and
    :func:`draw_w_dominant_irrep_fraction_ax`, or ``None`` if not applicable.
    """
    if not param_hist or group is None:
        return None

    group_size = group.order
    W0 = _hidden_by_group_weights_from_state_dict(param_hist[0])
    if W0 is None or W0.shape[1] != group_size:
        return None

    hidden_dim = W0.shape[0]
    n_modes = len(group.irreps())
    ylabel = r"Fraction in final dominant irrep ($E_{\mathrm{dom}}/\max_t E_{\mathrm{tot}}$)"

    n_snap = len(param_hist)
    max_idx = max(0, n_snap - 1)
    if max_idx == 0:
        steps = np.array([0], dtype=int)
    else:
        steps = np.unique(np.logspace(0, np.log10(max_idx), num_points, dtype=int))
        steps = np.sort(np.unique(np.concatenate([[0], steps])))

    W_power_over_time = []
    for step in steps:
        W = _hidden_by_group_weights_from_state_dict(param_hist[step])
        if W is None or W.shape != (hidden_dim, group_size):
            return None
        row_powers = powers_per_neuron_rows(W, group)
        W_power_over_time.append(row_powers)

    W_power_over_time = np.array(W_power_over_time)
    final_power = W_power_over_time[-1]
    dominant_idx = np.argmax(final_power, axis=1)

    dominant_fraction_over_time = np.zeros((len(steps), hidden_dim))
    for h in range(hidden_dim):
        k = dominant_idx[h]
        dominant_energy = W_power_over_time[:, h, k]
        total_energy = W_power_over_time[:, h, :].sum(axis=1)
        max_tot = np.max(total_energy)
        if max_tot >= active_energy_thresh:
            dominant_fraction_over_time[:, h] = dominant_energy / max_tot
        else:
            dominant_fraction_over_time[:, h] = np.nan

    psi = np.asarray(param_save_indices, dtype=float)
    x_plot = psi[steps]

    return {
        "x_plot": x_plot,
        "dominant_fraction_over_time": dominant_fraction_over_time,
        "dominant_idx": dominant_idx.astype(np.int64),
        "n_modes": int(n_modes),
        "ylabel": ylabel,
    }


def save_w_dominant_irrep_fraction_npz(path: str | Path, data: dict) -> None:
    """Save dominant-fraction curves to ``w_dominant_irrep_fraction.npz`` (used by combined plot / runs_data)."""
    path = Path(path)
    np.savez_compressed(
        path,
        x_plot=np.asarray(data["x_plot"], dtype=np.float64),
        dominant_fraction_over_time=np.asarray(
            data["dominant_fraction_over_time"], dtype=np.float64
        ),
        dominant_idx=np.asarray(data["dominant_idx"], dtype=np.int64),
        n_modes=np.int32(data["n_modes"]),
        ylabel=np.array(data["ylabel"], dtype=object),
    )


def load_w_dominant_irrep_fraction_npz(path: str | Path) -> dict:
    """Load dict returned by :func:`save_w_dominant_irrep_fraction_npz`."""
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    try:
        return {
            "x_plot": np.asarray(z["x_plot"], dtype=np.float64),
            "dominant_fraction_over_time": np.asarray(
                z["dominant_fraction_over_time"], dtype=np.float64
            ),
            "dominant_idx": np.asarray(z["dominant_idx"], dtype=np.int64),
            "n_modes": int(z["n_modes"]),
            "ylabel": str(z["ylabel"].item()),
        }
    finally:
        z.close()


def maybe_save_w_dominant_irrep_fraction_npz(
    run_dir: str | Path,
    param_hist: list,
    param_save_indices: list | np.ndarray,
    group,
) -> bool:
    """Compute and save ``w_dominant_irrep_fraction.npz`` when the model has group readout weights."""
    run_dir = Path(run_dir)
    if group is None:
        return False
    data = compute_w_dominant_irrep_fraction_data(param_hist, param_save_indices, group)
    if data is None:
        return False
    out = run_dir / "w_dominant_irrep_fraction.npz"
    save_w_dominant_irrep_fraction_npz(out, data)
    print(f"  ✓ Saved {out.name} (dominant-mode fraction curves for combined plot / runs_data)")
    return True


def load_w_dominant_irrep_fraction_for_run_dir(run_dir: str | Path) -> dict | None:
    """Load ``w_dominant_irrep_fraction.npz`` from *run_dir*, or compute from ``param_history.pt`` + config."""
    import torch
    import yaml

    from src.groups import make_group

    run_dir = Path(run_dir)
    npz = run_dir / "w_dominant_irrep_fraction.npz"
    if npz.exists():
        return load_w_dominant_irrep_fraction_npz(npz)

    ph_path = run_dir / "param_history.pt"
    cfg_path = run_dir / "config.yaml"
    psi_path = run_dir / "param_save_indices.npy"
    if not ph_path.exists() or not cfg_path.exists() or not psi_path.exists():
        return None

    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    param_hist = torch.load(ph_path, map_location="cpu", weights_only=False)
    param_save_indices = np.load(psi_path).tolist()

    try:
        group = make_group(config["data"]["group_name"], config)
    except (ValueError, KeyError):
        return None

    return compute_w_dominant_irrep_fraction_data(param_hist, param_save_indices, group)


def draw_w_dominant_irrep_fraction_ax(
    ax,
    data: dict,
    x_label: str,
    *,
    apply_style_axes: bool = False,
    xlabel_fontsize: float = 11,
    ylabel_fontsize: float = 11,
    tick_labelsize: float | None = None,
    lw: float = 1.5,
    alpha: float = 0.5,
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    show_ylabel: bool = True,
    mode_colors: list | None = None,
    aggregate_by_dominant_mode: bool = False,
    aggregate_lw: float | None = None,
    aggregate_std_fill_alpha: float = 0.22,
):
    """Draw dominant-fraction curves on *ax* (shared helpers for standalone and combined figures).

    When ``apply_style_axes`` is True, uses :func:`style_axes` and disables grid (standalone PDF).
    Set ``show_ylabel`` to False for non-leading columns in multi-column layouts.
    Pass ``mode_colors`` (length ``n_modes``) to match line colors in the power panel, e.g. from
    :func:`mode_colors_aligned_with_power_plot`.

    When ``aggregate_by_dominant_mode`` is True, instead of one faint line per neuron, plot one
    curve per mode/irrep color: mean fraction over neurons whose **final** dominant mode is that
    index, with a light ``mean ± std`` band behind a thicker mean line.
    """
    n_modes = int(data["n_modes"])
    if mode_colors is not None and len(mode_colors) == n_modes:
        colors = list(mode_colors)
    else:
        colors = _w_dominant_mode_colors(n_modes)
    x_plot = np.asarray(data["x_plot"])
    frac = np.asarray(data["dominant_fraction_over_time"])
    dom_idx = np.asarray(data["dominant_idx"], dtype=int)
    ylabel = data["ylabel"]
    hidden_dim = frac.shape[1]

    if aggregate_by_dominant_mode:
        thick = float(aggregate_lw) if aggregate_lw is not None else max(2.8, lw + 1.2)
        for k in range(n_modes):
            cols = [
                h
                for h in range(hidden_dim)
                if int(dom_idx[h]) == k and np.any(np.isfinite(frac[:, h]))
            ]
            if not cols:
                continue
            block = frac[:, cols]
            mean_v = np.nanmean(block, axis=1)
            std_v = np.nanstd(block, axis=1)
            low = mean_v - std_v
            high = mean_v + std_v
            c = colors[k]
            ax.fill_between(
                x_plot,
                low,
                high,
                color=c,
                alpha=aggregate_std_fill_alpha,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                x_plot,
                mean_v,
                color=c,
                lw=thick,
                solid_capstyle="round",
                zorder=2,
            )
    else:
        for h in range(hidden_dim):
            if not np.any(np.isfinite(frac[:, h])):
                continue
            k = int(dom_idx[h])
            ax.plot(
                x_plot,
                frac[:, h],
                color=colors[k],
                lw=lw,
                alpha=alpha,
            )

    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(x_label, fontsize=xlabel_fontsize)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if tick_labelsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    if apply_style_axes:
        style_axes(ax)
        ax.grid(False)
    elif show_grid:
        ax.grid(True, alpha=grid_alpha)


def plot_w_dominant_irrep_fraction(
    param_hist: list,
    param_save_indices: list[int],
    group,
    x_label: str,
    save_path: str | None = None,
    show: bool = False,
    num_points: int = 1000,
    active_energy_thresh: float = 1e-4,
):
    """Plot how each hidden unit's output weights concentrate on one Fourier / group mode over training.

    **What you see**

    - **Each colored line is one hidden neuron** (one row of ``W`` for :class:`~src.model.TwoLayerMLP`,
      or one column of ``W_out`` turned into a row for RNN/MLP models). Lines use the color of the
      **mode or irrep** that neuron ends up favoring (see below).
    - **Inactive** neurons (their total spectral power never exceeds ``active_energy_thresh``) are
      omitted so the plot is not flooded with flat zeros.

    **What “power” and “fraction” mean**

    Each neuron's weights form a vector on the group (length ``group_size``). That vector is decomposed into a
    **power spectrum**: either cyclic bins via FFT (for ``cn`` / ``cnxcn``)
    or irrep-wise power via ``group.power_spectrum`` (for generic groups). **Total power** is the
    sum of that spectrum—how much “energy” the row has in Fourier / group space. **Fraction of power**
    here means: at time ``t``, take the power in **one** chosen mode and divide by a **single**
    normalization: the **maximum** of total power that neuron ever had over training. So the y-axis
    is between 0 and 1 and answers: “At this time, how much of this neuron's *peak* total spectral
    weight sits in this one mode?”

    **What “final-dominant mode” means**

    First we look at the **last** saved snapshot. For each neuron we find which mode / irrep has the
    **most** power there—that index is fixed for that neuron for the whole plot. So we are **not**
    tracking “whatever is dominant at each time step”; we ask: “The mode this neuron **ends up**
    specializing in—how did the fraction of *peak* total power in that mode evolve from initialization
    to the end?” A curve that rises toward 1 means that neuron **converged** toward putting almost all
    of its (peak-normalized) spectral weight into that final preferred mode. A curve that stays low
    means that mode was not yet dominant at that time (relative to the neuron's own best total power
    later).

    **How to read the figure**

    - Many lines near **1** late in training ⇒ most active neurons' rows are **almost entirely** one
      Fourier / irrep mode (the one they favor at the end).
    - **Spread** of line colors ⇒ different neurons specialize to **different** modes.
    - **Rising** curves ⇒ **increasing** alignment of that neuron's weights with the mode it will
      eventually dominate.

    Args:
        param_hist: Parameter snapshots from training
        param_save_indices: Step or epoch index for each snapshot (same length as ``param_hist``)
        group_size: Flattened group dimension (second axis of ``W`` / ``W_out.T``)
        x_label: X-axis label (e.g. ``\"Epoch\"`` or ``\"Step\"``)
        group_name: ``\"cn\"``, ``\"cnxcn\"``, or a group name (e.g. ``\"dihedral\"``)
        group: ``Group`` (required unless ``group_name`` is ``cn`` or ``cnxcn``)
        p1, p2: Grid shape for ``cnxcn`` (required when ``group_name == \"cnxcn\"``)
        save_path: If set, save figure to this path
        show: If True, display the figure
        num_points: Number of log-spaced snapshots along training to use
        active_energy_thresh: Neurons with max total power below this are skipped

    Returns:
        ``matplotlib.figure.Figure`` or ``None`` if the model has no ``W`` / ``W_out`` parameters

    See Also:
        :func:`compute_w_dominant_irrep_fraction_data`,
        :func:`maybe_save_w_dominant_irrep_fraction_npz`,
        :func:`plot_combined_loss_and_power`
    """
    data = compute_w_dominant_irrep_fraction_data(
        param_hist,
        param_save_indices,
        group,
        num_points=num_points,
        active_energy_thresh=active_energy_thresh,
    )
    if data is None:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_w_dominant_irrep_fraction_ax(
        ax,
        data,
        x_label,
        apply_style_axes=True,
        xlabel_fontsize=24,
        ylabel_fontsize=19,
        lw=2.0,
        alpha=0.5,
        show_grid=False,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Saved W dominant-irrep fraction plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_wmix_structure(
    param_history,
    tracked_freqs,
    colors,
    p1,
    p2,
    steps=None,
    within_group_order="phase",
    dead_l2_thresh=0.1,
    save_path=None,
    show=False,
):
    """Visualize W_mix structure grouped by W_out frequency specialization.

    Args:
        param_history: List of parameter snapshots
        tracked_freqs: List of (kx, ky) frequency tuples
        colors: Array of colors for each frequency
        p1, p2: Template dimensions
        steps: List of epoch indices to plot
        within_group_order: How to order neurons within each frequency group
        dead_l2_thresh: L2 threshold for dead neurons
        save_path: Path to save figure
        show: Whether to display plot
    """
    from matplotlib.patches import Rectangle

    if steps is None:
        final_step = len(param_history) - 1
        steps = [1, min(5, final_step), final_step]
        steps = sorted(list(set(steps)))

    tracked_labels = [
        ("DC" if (kx, ky) == (0, 0) else f"({kx},{ky})") for (kx, ky) in tracked_freqs
    ]

    Wmix_perm_list = []
    group_info_list = []

    for s in steps:
        sd = param_history[s]
        dom_idx, phase, dom_power, l2 = analyze_wout_frequency_dominance(sd, tracked_freqs, p1, p2)

        if "W_mix" in sd:
            M = sd["W_mix"].detach().cpu().numpy()
        elif "W_h" in sd:
            M = sd["W_h"].detach().cpu().numpy()
        else:
            raise KeyError("Neither 'W_mix' nor 'W_h' found in state dict.")

        perm, group_keys, boundaries = _permutation_from_groups_with_dead(
            dom_idx, phase, dom_power, l2, within=within_group_order, dead_l2_thresh=dead_l2_thresh
        )

        M_perm = M[perm][:, perm]
        Wmix_perm_list.append(M_perm)
        group_info_list.append((group_keys, boundaries))

    vmax = max(np.max(np.abs(M)) for M in Wmix_perm_list)
    vmin = -vmax if vmax > 0 else 0.0

    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.8), constrained_layout=True)
    if n == 1:
        axes = [axes]

    cmap = "RdBu_r"
    dead_gray = "0.35"

    im = None
    for j, (s, M_perm) in enumerate(zip(steps, Wmix_perm_list)):
        ax = axes[j]
        im = ax.imshow(
            M_perm, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest"
        )

        ax.set_yticks([])
        ax.tick_params(axis="x", bottom=False)

        group_keys, boundaries = group_info_list[j]

        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color="k", lw=0.9, alpha=0.65)
            ax.axvline(b - 0.5, color="k", lw=0.9, alpha=0.65)

        starts = [0] + boundaries[:-1]
        ends = [b - 1 for b in boundaries]
        for kk, s0, e0 in zip(group_keys, starts, ends):
            if kk == -1:
                continue
            size = e0 - s0 + 1
            rect = Rectangle(
                (s0 - 0.5, s0 - 0.5),
                width=size,
                height=size,
                fill=False,
                linewidth=2.0,
                edgecolor=colors[kk],
                alpha=0.95,
                joinstyle="miter",
            )
            ax.add_patch(rect)

        centers = [(s + e) / 2.0 for s, e in zip(starts, ends)]
        sizes = [e - s + 1 for s, e in zip(starts, ends)]

        labels = []
        label_colors = []
        for kk, nn in zip(group_keys, sizes):
            if kk == -1:
                labels.append(f"DEAD\n(n={nn})")
                label_colors.append(dead_gray)
            else:
                labels.append(f"{tracked_labels[kk]}\n(n={nn})")
                label_colors.append(colors[kk])

        ax.set_xticks(centers)
        ax.set_xticklabels(labels, fontsize=11, ha="center")
        ax.tick_params(
            axis="x", bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=11
        )
        for lbl, clr in zip(ax.get_xticklabels(), label_colors):
            lbl.set_color(clr)

        ax.set_xlabel(f"Epoch {s}", fontsize=18, labelpad=8)

    cbar = fig.colorbar(im, ax=axes, shrink=1.0, pad=0.012, aspect=18)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Weight value", fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print("  ✓ Saved W_mix structure plot")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


def plot_power_group(
    model,
    param_hist,
    param_save_indices,
    X_eval,
    template: np.ndarray,
    group,
    k: int,
    optimizer: str,
    init_scale: float,
    save_path: str = None,
    group_label: str = "Group",
    learning_rate: float = None,
    hidden_dim: int = None,
):
    """Plot power spectrum of model outputs vs template over training.

    Uses group.power_spectrum for template power and model_power_over_time
    for model output power over training checkpoints.

    Args:
        model: Trained model
        param_hist: List of parameter snapshots
        param_save_indices: List mapping param_hist index to epoch number
        X_eval: Input evaluation tensor
        template: Template array (group_size,)
        group: Group object
        k: Sequence length
        optimizer: Optimizer name
        init_scale: Initialization scale
        save_path: Path to save the plot
        group_label: Human-readable label for the group
    """
    irreps = group.irreps()
    n_irreps = len(irreps)

    template_power = group.power_spectrum(template)

    print(f"  Template power spectrum: {template_power}")

    indep = _independent_irrep_indices(group)
    if len(indep) < n_irreps:
        print(
            f"  Showing {len(indep)}/{n_irreps} independent irreps "
            f"(conjugate pairs share power for real templates)"
        )

    model_powers, steps = model_power_over_time(group, model, param_hist, X_eval)
    epoch_numbers = [param_save_indices[min(s, len(param_save_indices) - 1)] for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    top_k = min(5, len(indep))
    indep_powers = np.array([template_power[i] for i in indep])
    ranked = np.argsort(indep_powers)[::-1]
    top_irrep_indices = np.array([indep[r] for r in ranked[:top_k]])
    top_irrep_indices = top_irrep_indices[top_irrep_indices != 0]

    _group_power_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors_line = _group_power_colors[: len(top_irrep_indices)]

    valid_mask = np.array(epoch_numbers) > 0
    valid_epochs = np.array(epoch_numbers)[valid_mask]
    valid_model_powers = model_powers[valid_mask, :]

    def _irrep_label(idx, irreps):
        dim = irreps[idx].dim
        dim_str = f"{dim}D"
        return rf"$\rho_{{{idx}}}$ ({dim_str})"

    # Plot 1: Linear scales
    ax = axes[0]
    lines_info = []
    for i, irrep_idx in enumerate(top_irrep_indices):
        power_values = model_powers[:, irrep_idx]
        ax.plot(epoch_numbers, power_values, "-", lw=2, color=colors_line[i])
        ax.axhline(template_power[irrep_idx], linestyle="--", alpha=0.5, color=colors_line[i])
        lines_info.append(
            {
                "x": epoch_numbers,
                "y": power_values,
                "label": _irrep_label(irrep_idx, irreps),
                "color": colors_line[i],
            }
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Power")
    ax.set_title("Linear Scales", fontsize=12)
    _add_line_labels(ax, lines_info)
    ax.grid(True, alpha=0.3)

    # Plot 2: Log x-axis only
    ax = axes[1]
    lines_info = []
    for i, irrep_idx in enumerate(top_irrep_indices):
        power_values = valid_model_powers[:, irrep_idx]
        ax.plot(valid_epochs, power_values, "-", lw=2, color=colors_line[i])
        ax.axhline(template_power[irrep_idx], linestyle="--", alpha=0.5, color=colors_line[i])
        lines_info.append(
            {
                "x": valid_epochs,
                "y": power_values,
                "label": _irrep_label(irrep_idx, irreps),
                "color": colors_line[i],
            }
        )
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Power")
    ax.set_title("Log X-axis", fontsize=12)
    _add_line_labels(ax, lines_info)
    ax.grid(True, alpha=0.3)

    # Plot 3: Log-log scales
    ax = axes[2]
    lines_info = []
    for i, irrep_idx in enumerate(top_irrep_indices):
        power_values = valid_model_powers[:, irrep_idx]
        power_mask = power_values > 0
        if np.any(power_mask):
            x_data = valid_epochs[power_mask]
            y_data = power_values[power_mask]
            ax.plot(x_data, y_data, "-", lw=2, color=colors_line[i])
            lines_info.append(
                {
                    "x": x_data,
                    "y": y_data,
                    "label": _irrep_label(irrep_idx, irreps),
                    "color": colors_line[i],
                }
            )
        if template_power[irrep_idx] > 0:
            ax.axhline(template_power[irrep_idx], linestyle="--", alpha=0.5, color=colors_line[i])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("Log-Log Scales", fontsize=12)
    _add_line_labels(ax, lines_info)
    ax.grid(True, alpha=0.3)

    title_parts = [
        f"{group_label} Power Evolution Over Training (k={k}, {optimizer}, init={init_scale:.0e}"
    ]
    if learning_rate is not None:
        title_parts.append(f", lr={learning_rate}")
    if hidden_dim is not None:
        title_parts.append(f", h={hidden_dim}")
    title_parts.append(")")
    fig.suptitle("".join(title_parts), fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

    labels = [_irrep_label(idx, irreps) for idx in top_irrep_indices]
    return {
        "valid_epochs": valid_epochs,
        "valid_model_powers": valid_model_powers,
        "model_powers": model_powers,
        "epoch_numbers": epoch_numbers,
        "template_power": template_power,
        "top_irrep_indices": top_irrep_indices,
        "colors_line": colors_line,
        "labels": labels,
    }


def plot_loss_power_and_weight_power(
    x_values,
    train_loss_hist,
    x_label,
    power_data,
    save_path=None,
    title=None,
    **weight_kw,
):
    """Create a 3-row plot: training loss, output power, and W dominant-mode fraction (optional).

    The first two rows show training loss and power spectrum. The third row is drawn when
    ``weight_kw`` includes ``param_hist`` and group fields (same as
    :func:`plot_w_dominant_irrep_fraction`).

    Returns:
        True if the third row was drawn, else False.
    """
    param_hist = weight_kw.get("param_hist")
    nrows = 3 if param_hist is not None else 2
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(4, 4 * nrows),
        sharex=True,
        gridspec_kw={"hspace": 0.10},
    )
    if nrows == 2:
        ax_loss, ax_power = axes
    else:
        ax_loss, ax_power, ax_w = axes

    x_arr = np.asarray(x_values)
    loss_arr = np.asarray(train_loss_hist)
    pos_mask = x_arr > 0
    ax_loss.plot(x_arr[pos_mask], loss_arr[pos_mask], lw=2, color="#1f77b4")
    ax_loss.set_xscale("log")
    ax_loss.set_yscale("log")
    _training_loss_log_y_floor(ax_loss)
    ax_loss.set_ylabel("Training Loss", fontsize=11)
    ax_loss.grid(True, alpha=0.3)
    if nrows == 3:
        ax_loss.tick_params(labelbottom=False)

    valid_epochs = power_data["valid_epochs"]
    valid_model_powers = power_data["valid_model_powers"]
    template_power = power_data["template_power"]
    top_indices = power_data["top_irrep_indices"]
    colors_line = power_data["colors_line"]
    labels = power_data["labels"]

    lines_info = []
    for i, idx in enumerate(top_indices):
        pv = valid_model_powers[:, idx]
        ax_power.plot(valid_epochs, pv, "-", lw=2, color=colors_line[i])
        ax_power.axhline(template_power[idx], linestyle="--", alpha=0.5, color=colors_line[i])
        lines_info.append(
            {
                "x": valid_epochs,
                "y": pv,
                "label": labels[i],
                "color": colors_line[i],
            }
        )
    _add_line_labels(ax_power, lines_info, fontsize=10)

    ax_power.set_xscale("log")
    ax_power.set_ylabel("Power", fontsize=11)
    ax_power.grid(True, alpha=0.3)
    if nrows == 3:
        ax_power.tick_params(labelbottom=False)
    else:
        ax_power.set_xlabel(x_label, fontsize=11)

    weight_row = False
    if param_hist is not None:
        wdata = compute_w_dominant_irrep_fraction_data(
            param_hist,
            weight_kw["param_save_indices"],
            weight_kw["group"],
        )
        if wdata is not None:
            mc = mode_colors_aligned_with_power_plot(
                int(wdata["n_modes"]),
                top_indices,
                colors_line,
            )
            draw_w_dominant_irrep_fraction_ax(
                ax_w,
                wdata,
                x_label,
                apply_style_axes=False,
                xlabel_fontsize=11,
                ylabel_fontsize=11,
                lw=2.0,
                alpha=0.5,
                show_grid=True,
                grid_alpha=0.3,
                mode_colors=mc,
            )
            ax_w.set_xlabel(x_label, fontsize=11)
            weight_row = True
        else:
            ax_w.set_xscale("log")
            ax_w.text(
                0.5,
                0.5,
                "W dominant fraction\n(not available for this model)",
                ha="center",
                va="center",
                transform=ax_w.transAxes,
                fontsize=10,
            )
            ax_w.set_xlabel(x_label, fontsize=11)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved {save_path}")
    plt.close()
    return weight_row


def plot_combined_loss_and_power(
    run_dirs,
    group_labels,
    save_path=None,
    w_dominant_aggregate_by_mode: bool = False,
):
    """Create a 3-row x N-column combined figure from multiple run directories.

    Rows (same fonts/grid/line weights across panels): (1) Log-Log training loss with
    theoretical MSE plateaus from ``template.npy``, (2) Log-X output power vs template
    with inline labels, (3) per-neuron fraction of spectral power in the **final-dominant**
    mode/irrep.      Axes are built with ``GridSpec`` (not ``plt.subplots(sharex='col')``) so the loss and
    power rows never share a y-axis; x-limits are matched per column after plotting.
    Rows 0–1 show y tick labels on every column (independent scales); axis titles are on
    the left column only. Row 3 **y** (0–1)
    is shared across columns only. Mode/irrep line colors in row 3 match row 2 (see
    :func:`mode_colors_aligned_with_power_plot`).

    If ``w_dominant_aggregate_by_mode`` is True, row 3 draws one mean ± std band per mode color
    (see :func:`draw_w_dominant_irrep_fraction_ax`) instead of one semi-transparent line per neuron.

    Each run directory must contain ``train_loss_history.npy``, ``power_data.npz``,
    ``config.yaml``, and ``template.npy`` (for theory lines). For the third row, prefer
    ``w_dominant_irrep_fraction.npz``; otherwise ``param_history.pt`` and
    ``param_save_indices.npy`` are used to recompute curves.
    """
    import yaml

    n_cols = len(run_dirs)
    # Build axes with GridSpec so *no* y-sharing between rows (``plt.subplots(sharex='col')``
    # can still link y-limits between stacked panels).  X alignment per column is applied
    # manually after plotting.
    fig = plt.figure(figsize=(5.8 * n_cols, 9.0))
    gs = gridspec.GridSpec(
        3,
        n_cols,
        figure=fig,
        hspace=0.32,
        wspace=0.38,
        top=0.90,
    )
    axes = np.empty((3, n_cols), dtype=object)
    for r in range(3):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])
    if n_cols > 1:
        for c in range(1, n_cols):
            axes[2, c].sharey(axes[2, 0])

    for col, (rd, label) in enumerate(zip(run_dirs, group_labels)):
        rd = Path(rd)
        loss_hist = np.load(rd / "train_loss_history.npy")
        pd = np.load(rd / "power_data.npz", allow_pickle=True)
        with open(rd / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        training_mode = cfg["training"]["mode"]
        if training_mode == "online":
            x_all = np.arange(len(loss_hist))
            x_label = "Step"
        else:
            x_all = np.arange(len(loss_hist))
            x_label = "Epoch"

        # -- Row 0: Log-Log training loss + theory plateaus --
        ax = axes[0, col]
        theory_ys = _theory_loss_y_levels_from_run(rd, cfg)
        _draw_theory_loss_hlines(ax, theory_ys)
        pos = x_all > 0
        loss_arr = np.asarray(loss_hist)[pos]
        ax.plot(
            x_all[pos],
            loss_arr,
            lw=1.5,
            color="#1f77b4",
            zorder=3,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        _training_loss_log_y_floor(ax)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Training Loss", fontsize=10)

        hp_parts = [f"k={cfg['data']['k']}"]
        hp_parts.append(f"lr={cfg['training']['learning_rate']}")
        hp_parts.append(f"init={cfg['model']['init_scale']:.0e}")
        hp_parts.append(f"h={cfg['model']['hidden_dim']}")
        hp_parts.append(cfg["training"]["optimizer"])
        ax.set_title(f"{label}\n({', '.join(hp_parts)})", fontsize=9)

        # -- Row 1: Log-X power spectrum --
        ax = axes[1, col]
        valid_epochs = np.asarray(pd["valid_epochs"])
        valid_model_powers = np.asarray(pd["valid_model_powers"])
        template_power = np.asarray(pd["template_power"])
        top_indices = np.asarray(pd["top_irrep_indices"])
        colors_line = list(pd["colors_line"])
        labels_list = list(pd["labels"])

        lines_info = []
        for i, idx in enumerate(top_indices):
            pv = valid_model_powers[:, idx]
            ax.plot(valid_epochs, pv, "-", lw=1.5, color=colors_line[i])
            ax.axhline(template_power[idx], linestyle="--", alpha=0.4, color=colors_line[i])
            lines_info.append(
                {
                    "x": valid_epochs,
                    "y": pv,
                    "label": labels_list[i],
                    "color": colors_line[i],
                }
            )
        _add_line_labels(ax, lines_info, fontsize=7, min_frac_sep=0.14)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Power", fontsize=10)

        # -- Row 2: W dominant-mode / irrep fraction --
        ax = axes[2, col]
        wdata = load_w_dominant_irrep_fraction_for_run_dir(rd)
        if wdata is not None:
            mode_colors = mode_colors_aligned_with_power_plot(
                int(wdata["n_modes"]),
                top_indices,
                colors_line,
            )
            draw_w_dominant_irrep_fraction_ax(
                ax,
                wdata,
                x_label,
                apply_style_axes=False,
                xlabel_fontsize=9,
                ylabel_fontsize=10,
                tick_labelsize=8,
                lw=1.5,
                alpha=0.5,
                show_grid=True,
                grid_alpha=0.3,
                show_ylabel=(col == 0),
                mode_colors=mode_colors,
                aggregate_by_dominant_mode=w_dominant_aggregate_by_mode,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No dominant-fraction data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
        ax.set_xlabel(x_label, fontsize=9)
        if col != 0:
            ax.tick_params(labelleft=False)

    for col in range(n_cols):
        axes[0, col].tick_params(labelbottom=False)
        axes[1, col].tick_params(labelbottom=False)

    # Align log-x epoch/step range within each column (independent axes; no sharex).
    for col in range(n_cols):
        x0 = min(axes[r, col].get_xlim()[0] for r in range(3))
        x1 = max(axes[r, col].get_xlim()[1] for r in range(3))
        for r in range(3):
            axes[r, col].set_xlim(x0, x1)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  \u2713 Saved {save_path}")
    plt.close(fig)
    return fig


def plot_irreps(group, show=False):
    """Plot the irreducible representations (irreps) of the group.

    Parameters
    ----------
    group : class instance
        The group for which the irreps are being plotted.
    show : bool, optional
        Whether to display the plot immediately.
    """
    FONT_SIZES = {"title": 30, "axes_label": 30, "tick_label": 30, "legend": 15}

    irreps = group.irreps()
    group_elements = group.elements()

    num_irreps = len(irreps)
    fig, axs = plt.subplots(1, num_irreps, figsize=(3 * num_irreps, 4), squeeze=False)
    axs = axs[0]

    for i, irrep in enumerate(irreps):
        matrices = [irrep(g) for g in group_elements]
        matrices = np.array(matrices)

        if matrices.ndim == 1 or (matrices.ndim == 2 and matrices.shape[1] == 1):
            axs[i].plot(range(len(group_elements)), matrices.real, marker="o", label="Re")
            if np.any(np.abs(matrices.imag) > 1e-10):
                axs[i].plot(range(len(group_elements)), matrices.imag, marker="x", label="Im")
            axs[i].set_title(f"Irrep {i}: {str(irrep)} (dim=1)")
            axs[i].set_xlabel("Group element idx")
            axs[i].set_ylabel("Irrep value")
            axs[i].legend()
        else:
            d = matrices.shape[1]
            num_group_elements = len(group_elements)
            num_irrep_entries = d * d
            irrep_matrix_entries = matrices.real.reshape(num_group_elements, num_irrep_entries)
            im = axs[i].imshow(irrep_matrix_entries, aspect="auto", cmap="viridis")
            axs[i].set_title(f"Irrep {i}: {str(irrep)} (size={d}x{d})")
            axs[i].set_xlabel("Flattened Irreps")
            axs[i].set_ylabel("Irrep(g)")
            plt.colorbar(im, ax=axs[i])
    fig.suptitle(
        "Irreducible Representations (matrix values for all group elements)",
        fontsize=FONT_SIZES["title"],
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig
