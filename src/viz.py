from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """After setting training loss to log-y, match main.py and floor the axis at 1.0."""
    ax.set_ylim(bottom=1e-1)


def _add_line_labels(ax, lines_info, fontsize=12):
    """Add right-aligned colored text labels above (or below) each line.

    Each label is anchored at the rightmost data-point of its line and
    right-aligned so the text extends leftward inside the axes box.
    When two labels would overlap vertically the lower one is placed
    below its line instead.
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
        if abs(fracs[i - 1] - fracs[i]) < 0.07:
            placements[i] = "below"

    # Keep labels inside the axes: flip if too close to the top/bottom edge
    for i in range(len(label_data)):
        if placements[i] == "above" and fracs[i] > 0.88:
            placements[i] = "below"
        elif placements[i] == "below" and fracs[i] < 0.12:
            placements[i] = "above"

    for d, placement in zip(label_data, placements):
        y_off = 7 if placement == "above" else -7
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
    import src.power as power

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
        tp = [power._tracked_power_from_fft2(P, kx, ky, p1, p2) for (kx, ky) in tracked_freqs]
        jj = int(np.argmax(tp))
        dom_idx[j] = jj
        i0, j0 = tracked_freqs[jj][0] % p1, tracked_freqs[jj][1] % p2
        phase[j] = np.angle(F[i0, j0])
        dom_pow[j] = tp[jj]

    return dom_idx, phase, dom_pow, l2


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


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
    loss_history, template_2d, p1, p2, x_values=None, x_label="Step", save_path=None, show=True
):
    """Plot training loss with theoretical power spectrum lines.

    Args:
        loss_history: List of loss values
        template_2d: The 2D template array (p1, p2)
        p1, p2: Dimensions
        x_values: X-axis values (if None, uses indices 0, 1, 2, ...)
        x_label: Label for x-axis (e.g., "Samples Seen", "Fraction of Space")
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    import src.power as power

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if x_values is None:
        x_values = np.arange(len(loss_history))

    ax.plot(x_values, loss_history, lw=4, color="#1f77b4", label="Training Loss")

    x_freq, y_freq, pwr = power.get_power_2d(template_2d)
    pwr = pwr.flatten()
    valid = pwr > 1e-20
    pwr = pwr[valid]
    pwr = np.sort(pwr)[::-1]

    alpha_values = [np.sum(pwr[k:]) for k in range(len(pwr))]
    coef = 1 / (p1 * p2)
    for k, alpha in enumerate(alpha_values):
        ax.axhline(y=coef * alpha, color="black", linestyle="--", linewidth=2, zorder=-2)

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


def plot_predictions_2d(
    model,
    param_history,
    X_data,
    Y_data,
    p1,
    p2,
    steps=None,
    example_idx=None,
    cmap="gray",
    save_path=None,
    show=False,
):
    """Plot model predictions at different training steps vs ground truth (2D).

    Args:
        model: The trained model
        param_history: List of parameter snapshots from training
        X_data: Input tensor (N, k, p1*p2)
        Y_data: Target tensor (N, p1*p2)
        p1, p2: Dimensions
        steps: List of epoch indices to plot
        example_idx: Index of example to visualize
        cmap: Colormap to use
        save_path: Path to save figure
        show: Whether to display the plot
    """
    import torch

    if steps is None:
        final_step = len(param_history) - 1
        steps = [1, min(5, final_step), min(10, final_step), final_step]
        steps = sorted(list(set(steps)))

    if example_idx is None:
        example_idx = int(np.random.randint(len(Y_data)))

    device = next(model.parameters()).device
    model.to(device).eval()

    if Y_data.dim() == 3:
        Y_data = Y_data[:, -1, :]
    with torch.no_grad():
        truth_2d = Y_data[example_idx].reshape(p1, p2).cpu().numpy()

    preds = []
    for step in steps:
        model.load_state_dict(param_history[step], strict=True)
        with torch.no_grad():
            x = X_data[example_idx : example_idx + 1].to(device)
            pred_2d = model(x)
            if pred_2d.dim() == 3:
                pred_2d = pred_2d[:, -1, :]
            pred_2d = pred_2d.reshape(p1, p2).detach().cpu().numpy()
            preds.append(pred_2d)

    vmin = np.min(truth_2d)
    vmax = np.max(truth_2d)

    fig, axes = plt.subplots(2, len(steps), figsize=(3.5 * len(steps), 6), layout="constrained")
    if len(steps) == 1:
        axes = axes.reshape(2, 1)

    for col, (step, pred_2d) in enumerate(zip(steps, preds)):
        im = axes[0, col].imshow(pred_2d, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
        axes[0, col].set_title(f"Epoch {step}", fontsize=12)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        axes[1, col].imshow(truth_2d, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel("Prediction", fontsize=14)
    axes[1, 0].set_ylabel("Target", fontsize=14)

    fig.colorbar(im, ax=axes, location="right", shrink=0.9, pad=0.02).set_label(
        "Value", fontsize=12
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved predictions plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


def plot_predictions_1d(
    model,
    param_history,
    X_data,
    Y_data,
    group_size,
    steps=None,
    example_idx=None,
    save_path=None,
    show=False,
):
    """Plot model predictions at different training steps vs ground truth (1D).

    Args:
        model: The trained model
        param_history: List of parameter snapshots from training
        X_data: Input tensor (N, k, group_size)
        Y_data: Target tensor (N, group_size)
        group_size: Dimension of the group
        steps: List of epoch indices to plot
        example_idx: Index of example to visualize
        save_path: Path to save figure
        show: Whether to display the plot
    """
    import torch

    if steps is None:
        final_step = len(param_history) - 1
        steps = [1, min(5, final_step), min(10, final_step), final_step]
        steps = sorted(list(set(steps)))

    if example_idx is None:
        example_idx = int(np.random.randint(len(Y_data)))

    device = next(model.parameters()).device
    model.to(device).eval()

    if Y_data.dim() == 3:
        Y_data = Y_data[:, -1, :]
    with torch.no_grad():
        truth_1d = Y_data[example_idx].cpu().numpy()

    preds = []
    for step in steps:
        model.load_state_dict(param_history[step], strict=True)
        with torch.no_grad():
            x = X_data[example_idx : example_idx + 1].to(device)
            pred = model(x)
            if pred.dim() == 3:
                pred = pred[:, -1, :]
            pred_1d = pred.squeeze().detach().cpu().numpy()
            preds.append(pred_1d)

    fig, axes = plt.subplots(2, len(steps), figsize=(3.5 * len(steps), 4), layout="constrained")
    if len(steps) == 1:
        axes = axes.reshape(2, 1)

    x = np.arange(group_size)

    for col, (step, pred_1d) in enumerate(zip(steps, preds)):
        axes[0, col].plot(x, pred_1d, "b-", lw=2)
        axes[0, col].set_title(f"Epoch {step}", fontsize=12)
        axes[0, col].set_ylim(
            truth_1d.min() - 0.1 * np.abs(truth_1d.min()),
            truth_1d.max() + 0.1 * np.abs(truth_1d.max()),
        )
        axes[0, col].set_xticks([])
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(x, truth_1d, "k-", lw=2)
        axes[1, col].set_ylim(
            truth_1d.min() - 0.1 * np.abs(truth_1d.min()),
            truth_1d.max() + 0.1 * np.abs(truth_1d.max()),
        )
        axes[1, col].set_xticks([])
        axes[1, col].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Prediction", fontsize=14)
    axes[1, 0].set_ylabel("Target", fontsize=14)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved predictions plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, axes


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
    group_size: int,
    group_name: str,
    group=None,
    p1: int | None = None,
    p2: int | None = None,
    num_points: int = 1000,
    active_energy_thresh: float = 1e-4,
) -> dict | None:
    """Compute per-neuron dominant-mode fraction curves (same math as ``plot_w_dominant_irrep_fraction``).

    Returns a dict suitable for :func:`save_w_dominant_irrep_fraction_npz` and
    :func:`draw_w_dominant_irrep_fraction_ax`, or ``None`` if not applicable.
    """
    import src.power as power

    if not param_hist:
        return None

    use_cyclic = group_name in ("cn", "cnxcn")
    if not use_cyclic and group is None:
        return None
    if group_name == "cnxcn" and (p1 is None or p2 is None):
        return None

    W0 = _hidden_by_group_weights_from_state_dict(param_hist[0])
    if W0 is None:
        return None
    if W0.shape[1] != group_size:
        return None
    if use_cyclic:
        if group_name == "cnxcn" and group_size != p1 * p2:
            return None
    elif group_size != group.order:
        return None

    hidden_dim = W0.shape[0]

    if use_cyclic:
        if group_name == "cn":
            probe = power.powers_per_neuron_rows_cyclic(W0[:1], template_dim=1)
        else:
            probe = power.powers_per_neuron_rows_cyclic(W0[:1], template_dim=2, p1=p1, p2=p2)
        n_modes = probe.shape[1]
        ylabel = r"Fraction in final dominant mode ($E_{\mathrm{dom}}/\max_t E_{\mathrm{tot}}$)"
    else:
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
        if use_cyclic:
            if group_name == "cn":
                row_powers = power.powers_per_neuron_rows_cyclic(W, template_dim=1)
            else:
                row_powers = power.powers_per_neuron_rows_cyclic(W, template_dim=2, p1=p1, p2=p2)
        else:
            row_powers = power.powers_per_neuron_rows(W, group)
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
    config: dict,
    group=None,
) -> bool:
    """Compute and save ``w_dominant_irrep_fraction.npz`` when the model has group readout weights."""
    run_dir = Path(run_dir)
    gn = config["data"]["group_name"]
    if gn == "cn":
        group_size = config["data"]["p"]
        data = compute_w_dominant_irrep_fraction_data(
            param_hist, param_save_indices, group_size, "cn"
        )
    elif gn == "cnxcn":
        p1, p2 = config["data"]["p1"], config["data"]["p2"]
        data = compute_w_dominant_irrep_fraction_data(
            param_hist,
            param_save_indices,
            p1 * p2,
            "cnxcn",
            p1=p1,
            p2=p2,
        )
    elif gn in ("dihedral", "octahedral", "A5"):
        if group is None:
            return False
        data = compute_w_dominant_irrep_fraction_data(
            param_hist,
            param_save_indices,
            group.order,
            gn,
            group=group,
        )
    else:
        return False
    if data is None:
        return False
    out = run_dir / "w_dominant_irrep_fraction.npz"
    save_w_dominant_irrep_fraction_npz(out, data)
    print(f"  ✓ Saved {out.name} (dominant-mode fraction curves for combined plot / runs_data)")
    return True


def load_w_dominant_irrep_fraction_for_run_dir(run_dir: str | Path) -> dict | None:
    """Load ``w_dominant_irrep_fraction.npz`` from *run_dir*, or compute from ``param_history.pt`` + config.

    Used by :func:`plot_combined_loss_and_power`. Requires ``param_save_indices.npy`` when falling
    back to raw checkpoints (same as power plots).
    """
    import torch
    import yaml

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

    gn = config["data"]["group_name"]
    if gn == "cn":
        group_size = config["data"]["p"]
        return compute_w_dominant_irrep_fraction_data(
            param_hist, param_save_indices, group_size, "cn"
        )
    if gn == "cnxcn":
        p1, p2 = config["data"]["p1"], config["data"]["p2"]
        return compute_w_dominant_irrep_fraction_data(
            param_hist,
            param_save_indices,
            p1 * p2,
            "cnxcn",
            p1=p1,
            p2=p2,
        )
    if gn == "dihedral":
        from src.groups import DihedralGroup

        group = DihedralGroup(N=config["data"].get("group_n", 3))
    elif gn == "octahedral":
        from src.groups import OctahedralGroup

        group = OctahedralGroup()
    elif gn == "A5":
        from src.groups import IcosahedralGroup

        group = IcosahedralGroup()
    else:
        return None

    return compute_w_dominant_irrep_fraction_data(
        param_hist,
        param_save_indices,
        group.order,
        gn,
        group=group,
    )


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
):
    """Draw dominant-fraction curves on *ax* (shared helpers for standalone and combined figures).

    When ``apply_style_axes`` is True, uses :func:`style_axes` and disables grid (standalone PDF).
    Set ``show_ylabel`` to False for non-leading columns in multi-column layouts.
    Pass ``mode_colors`` (length ``n_modes``) to match line colors in the power panel, e.g. from
    :func:`mode_colors_aligned_with_power_plot`.
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
    group_size: int,
    x_label: str,
    group_name: str,
    group=None,
    p1: int | None = None,
    p2: int | None = None,
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
        group_size,
        group_name,
        group=group,
        p1=p1,
        p2=p2,
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


def plot_power_1d(
    model,
    param_history,
    X_data,
    Y_data,
    template_1d,
    group_size,
    loss_history,
    param_save_indices=None,
    num_freqs_to_track=10,
    checkpoint_indices=None,
    num_samples=100,
    save_path=None,
    show=False,
):
    """Plot training loss with power spectrum analysis of predictions over time (1D).

    Creates a two-panel plot:
    - Top: Training loss with colored bands for theory lines
    - Bottom: Power in tracked frequencies over time

    Args:
        model: The trained model
        param_history: List of parameter snapshots
        X_data: Input tensor (N, k, group_size)
        Y_data: Target tensor (N, group_size)
        template_1d: The 1D template array (group_size,)
        group_size: Dimension of the group
        loss_history: List of loss values
        param_save_indices: List of step/epoch numbers where params were saved
        num_freqs_to_track: Number of top frequencies to track
        checkpoint_indices: (deprecated/unused)
        num_samples: Number of samples to average for power computation
        save_path: Path to save figure
        show: Whether to display the plot
    """
    import torch
    from matplotlib.ticker import FormatStrFormatter
    from tqdm import tqdm

    import src.power as power

    device = next(model.parameters()).device

    tracked_freqs = power.topk_template_freqs_1d(template_1d, K=num_freqs_to_track)
    template_power, _ = power.get_power_1d(template_1d)
    target_powers = {k: template_power[k] for k in tracked_freqs}

    T = len(param_history)
    steps_analysis = list(range(len(param_history)))

    if param_save_indices is not None:
        actual_steps = param_save_indices
    else:
        actual_steps = list(range(len(param_history)))

    powers_over_time = {freq: [] for freq in tracked_freqs}

    print(f"  Analyzing {len(steps_analysis)} checkpoints for power spectrum (1D)...")

    with torch.no_grad():
        for step in tqdm(steps_analysis, desc="  Computing power spectra", leave=False):
            model.load_state_dict(param_history[step], strict=True)
            model.eval()

            outputs_flat = model(X_data[:num_samples].to(device)).detach().cpu().numpy()

            powers_batch = []
            for i in range(outputs_flat.shape[0]):
                if outputs_flat.ndim == 3:
                    out_1d = outputs_flat[i, -1, :]
                else:
                    out_1d = outputs_flat[i]
                power_i, _ = power.get_power_1d(out_1d)
                powers_batch.append(power_i)
            avg_power = np.mean(powers_batch, axis=0)

            for k in tracked_freqs:
                powers_over_time[k].append(avg_power[k])

    for freq in tracked_freqs:
        powers_over_time[freq] = np.array(powers_over_time[freq])

    if param_save_indices is None:
        loss_epochs = np.arange(len(param_history))
        loss_history_subset = loss_history
    else:
        loss_epochs = np.array(param_save_indices)
        loss_history_subset = [loss_history[i] for i in param_save_indices]

    colors = plt.cm.tab10(np.linspace(0, 1, len(tracked_freqs)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.10, hspace=0.12)

    ax1.plot(loss_epochs, loss_history_subset, lw=4, color="#1f77b4", label="Training Loss")

    pwr, _ = power.get_power_1d(template_1d)
    power_sorted = np.sort(pwr[pwr > 1e-20])[::-1]

    alpha_values = np.array([np.sum(power_sorted[k:]) for k in range(len(power_sorted))])
    coef = 1.0 / group_size
    y_levels = coef * alpha_values

    n_bands = min(len(tracked_freqs), len(y_levels) - 1)
    for i in range(n_bands):
        y_top = y_levels[i]
        y_bot = y_levels[i + 1]
        ax1.axhspan(y_bot, y_top, facecolor=colors[i], alpha=0.15, zorder=-3)

    for y in y_levels[: n_bands + 1]:
        ax1.axhline(y=y, color="black", linestyle="--", linewidth=2, zorder=-2)

    ax1.set_ylabel("Theory Loss Levels", fontsize=20)
    ax1.set_ylim(y_levels[n_bands], y_levels[0] * 1.1)
    style_axes(ax1)
    ax1.grid(False)
    ax1.tick_params(labelbottom=False)

    for i, k in enumerate(tracked_freqs):
        ax2.plot(actual_steps, powers_over_time[k], color=colors[i], lw=3, label=f"k={k}")
        ax2.axhline(
            target_powers[k],
            color=colors[i],
            linestyle="dotted",
            linewidth=2,
            alpha=0.5,
        )

    ax2.set_xlabel("Steps", fontsize=20)
    ax2.set_ylabel("Power in Prediction", fontsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="best", ncol=2)
    style_axes(ax2)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved power spectrum plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, (ax1, ax2), powers_over_time, tracked_freqs


def plot_power_cn(
    model,
    param_hist,
    param_save_indices,
    X_eval,
    template_1d: np.ndarray,
    group_size: int,
    k: int,
    optimizer: str,
    init_scale: float,
    save_path: str = None,
    group_label: str = "Group",
    learning_rate: float = None,
    hidden_dim: int = None,
):
    """Plot power spectrum of model outputs vs template for cyclic group Cn.

    Mirrors plot_power_group but uses 1D FFT power.
    Each frequency mode is treated as a 1D irrep.
    """
    import src.power as power

    template_power, _ = power.get_power_1d(template_1d)
    n_modes = len(template_power)

    print(f"  Template power spectrum (cn): {template_power}")

    model_powers, steps = power.model_power_over_time("cn", model, param_hist, X_eval)
    epoch_numbers = [param_save_indices[min(s, len(param_save_indices) - 1)] for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    top_k = min(5, n_modes)
    top_mode_indices = np.argsort(template_power)[::-1][:top_k]
    top_mode_indices = top_mode_indices[top_mode_indices != 0]

    _cn_power_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors_line = _cn_power_colors[: len(top_mode_indices)]

    valid_mask = np.array(epoch_numbers) > 0
    valid_epochs = np.array(epoch_numbers)[valid_mask]
    valid_model_powers = model_powers[valid_mask, :]

    def _mode_label(idx):
        return rf"$\rho_{{{idx}}}$ (1D)"

    # Plot 1: Linear scales
    ax = axes[0]
    lines_info = []
    for i, mode_idx in enumerate(top_mode_indices):
        power_values = model_powers[:, mode_idx]
        ax.plot(epoch_numbers, power_values, "-", lw=2, color=colors_line[i])
        ax.axhline(template_power[mode_idx], linestyle="dotted", alpha=0.5, color=colors_line[i])
        lines_info.append(
            {
                "x": epoch_numbers,
                "y": power_values,
                "label": _mode_label(mode_idx),
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
    for i, mode_idx in enumerate(top_mode_indices):
        power_values = valid_model_powers[:, mode_idx]
        ax.plot(valid_epochs, power_values, "-", lw=2, color=colors_line[i])
        ax.axhline(template_power[mode_idx], linestyle="dotted", alpha=0.5, color=colors_line[i])
        lines_info.append(
            {
                "x": valid_epochs,
                "y": power_values,
                "label": _mode_label(mode_idx),
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
    for i, mode_idx in enumerate(top_mode_indices):
        power_values = valid_model_powers[:, mode_idx]
        power_mask = power_values > 0
        if np.any(power_mask):
            x_data = valid_epochs[power_mask]
            y_data = power_values[power_mask]
            ax.plot(x_data, y_data, "-", lw=2, color=colors_line[i])
            lines_info.append(
                {
                    "x": x_data,
                    "y": y_data,
                    "label": _mode_label(mode_idx),
                    "color": colors_line[i],
                }
            )
        if template_power[mode_idx] > 0:
            ax.axhline(
                template_power[mode_idx], linestyle="dotted", alpha=0.5, color=colors_line[i]
            )
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
        print(f"  ✓ Saved {save_path}")
    plt.close()

    labels = [_mode_label(idx) for idx in top_mode_indices]
    return {
        "valid_epochs": valid_epochs,
        "valid_model_powers": valid_model_powers,
        "model_powers": model_powers,
        "epoch_numbers": epoch_numbers,
        "template_power": template_power,
        "top_irrep_indices": top_mode_indices,
        "colors_line": colors_line,
        "labels": labels,
    }


def plot_power_cnxcn(
    model,
    param_hist,
    param_save_indices,
    X_eval,
    template_2d: np.ndarray,
    p1: int,
    p2: int,
    k: int,
    optimizer: str,
    init_scale: float,
    save_path: str = None,
    group_label: str = "Group",
    learning_rate: float = None,
    hidden_dim: int = None,
):
    """Plot power spectrum of model outputs vs template for CnxCn group.

    Mirrors plot_power_cn but uses 2D rfft2 power.
    Each 2D frequency mode (u, v) is tracked separately.
    """
    import src.power as power

    template_power_2d = power.get_power_2d(template_2d, no_freq=True)  # (p1, p2//2+1)
    template_power = template_power_2d.flatten()
    n_modes = len(template_power)
    n_cols = p2 // 2 + 1

    print(f"  Template 2D power spectrum shape: {template_power_2d.shape}")

    model_powers, steps = power.model_power_over_time("cnxcn", model, param_hist, X_eval)
    epoch_numbers = [param_save_indices[min(s, len(param_save_indices) - 1)] for s in steps]

    top_k = min(5, n_modes)
    top_mode_indices = np.argsort(template_power)[::-1][:top_k]
    top_mode_indices = top_mode_indices[top_mode_indices != 0]

    _cnxcn_power_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors_line = _cnxcn_power_colors[: len(top_mode_indices)]

    valid_mask = np.array(epoch_numbers) > 0
    valid_epochs = np.array(epoch_numbers)[valid_mask]
    valid_model_powers = model_powers[valid_mask, :]

    def _mode_label(idx):
        u = idx // n_cols
        v = idx % n_cols
        return rf"$({u},\,{v})$"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Linear scales
    ax = axes[0]
    lines_info = []
    for i, mode_idx in enumerate(top_mode_indices):
        power_values = model_powers[:, mode_idx]
        ax.plot(epoch_numbers, power_values, "-", lw=2, color=colors_line[i])
        ax.axhline(template_power[mode_idx], linestyle="dotted", alpha=0.5, color=colors_line[i])
        lines_info.append(
            {
                "x": epoch_numbers,
                "y": power_values,
                "label": _mode_label(mode_idx),
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
    for i, mode_idx in enumerate(top_mode_indices):
        power_values = valid_model_powers[:, mode_idx]
        ax.plot(valid_epochs, power_values, "-", lw=2, color=colors_line[i])
        ax.axhline(template_power[mode_idx], linestyle="dotted", alpha=0.5, color=colors_line[i])
        lines_info.append(
            {
                "x": valid_epochs,
                "y": power_values,
                "label": _mode_label(mode_idx),
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
    for i, mode_idx in enumerate(top_mode_indices):
        power_values = valid_model_powers[:, mode_idx]
        power_mask = power_values > 0
        if np.any(power_mask):
            x_data = valid_epochs[power_mask]
            y_data = power_values[power_mask]
            ax.plot(x_data, y_data, "-", lw=2, color=colors_line[i])
            lines_info.append(
                {
                    "x": x_data,
                    "y": y_data,
                    "label": _mode_label(mode_idx),
                    "color": colors_line[i],
                }
            )
        if template_power[mode_idx] > 0:
            ax.axhline(
                template_power[mode_idx], linestyle="dotted", alpha=0.5, color=colors_line[i]
            )
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
        print(f"  \u2713 Saved {save_path}")
    plt.close()

    labels = [_mode_label(idx) for idx in top_mode_indices]
    return {
        "valid_epochs": valid_epochs,
        "valid_model_powers": valid_model_powers,
        "model_powers": model_powers,
        "epoch_numbers": epoch_numbers,
        "template_power": template_power,
        "top_irrep_indices": top_mode_indices,
        "colors_line": colors_line,
        "labels": labels,
    }


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

    Args:
        model: Trained model
        param_hist: List of parameter snapshots
        X_eval: Input evaluation tensor (N, k, group_size)
        Y_eval: Target evaluation tensor (N, group_size)
        group_size: Order of the group
        checkpoint_indices: Indices into param_hist to visualize
        save_path: Path to save the plot
        num_samples: Number of samples to show
        group_label: Human-readable label for the group (used in plot title)
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
    import src.power as power

    group_name = "group"
    irreps = group.irreps()
    n_irreps = len(irreps)

    template_power = group.power_spectrum(template)

    print(f"  Template power spectrum: {template_power}")
    print("  (These are dim^2 * diag_value^2 / |G| for each irrep)")

    model_powers, steps = power.model_power_over_time(
        group_name, model, param_hist, X_eval, group=group
    )
    epoch_numbers = [param_save_indices[min(s, len(param_save_indices) - 1)] for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    top_k = min(5, n_irreps)
    top_irrep_indices = np.argsort(template_power)[::-1][:top_k]
    top_irrep_indices = top_irrep_indices[top_irrep_indices != 0]

    _group_power_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors_line = _group_power_colors[: len(top_irrep_indices)]

    valid_mask = np.array(epoch_numbers) > 0
    valid_epochs = np.array(epoch_numbers)[valid_mask]
    valid_model_powers = model_powers[valid_mask, :]

    def _irrep_label(idx, irreps):
        dim = irreps[idx].size
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


def plot_loss_and_power(
    x_values,
    train_loss_hist,
    x_label,
    power_data,
    save_path=None,
    title=None,
):
    """Create combined 2-row plot: Log-Log loss (top) and Log-X power spectrum (bottom).

    The two subplots share the x-axis (log-scale epochs/steps).

    Args:
        x_values: x-axis values for loss plot (epochs or steps)
        train_loss_hist: training loss history
        x_label: x-axis label (e.g., "Epoch")
        power_data: dict returned by plot_power_group or plot_power_cn
        save_path: path to save the figure
        title: optional suptitle
    """
    fig, (ax_loss, ax_power) = plt.subplots(
        2,
        1,
        figsize=(4, 8),
        sharex=True,
        gridspec_kw={"hspace": 0.10},
    )

    # --- Top row: Log-Log training loss ---
    x_arr = np.asarray(x_values)
    loss_arr = np.asarray(train_loss_hist)
    pos_mask = x_arr > 0
    ax_loss.plot(x_arr[pos_mask], loss_arr[pos_mask], lw=2, color="#1f77b4")
    ax_loss.set_xscale("log")
    ax_loss.set_yscale("log")
    _training_loss_log_y_floor(ax_loss)
    ax_loss.set_ylabel("Training Loss", fontsize=11)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.tick_params(labelbottom=False)

    # --- Bottom row: Log-X power spectrum with inline labels ---
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
    ax_power.set_xlabel(x_label, fontsize=11)
    ax_power.set_ylabel("Power", fontsize=11)
    ax_power.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved {save_path}")
    plt.close()


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

    The first two rows match :func:`plot_loss_and_power`. The third row is drawn when
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
            weight_kw["group_size"],
            weight_kw["group_name"],
            group=weight_kw.get("group"),
            p1=weight_kw.get("p1"),
            p2=weight_kw.get("p2"),
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
):
    """Create a 3-row x N-column combined figure from multiple run directories.

    Rows (same fonts/grid/line weights across panels): (1) Log-Log training loss,
    (2) Log-X output power vs template with inline labels, (3) per-neuron fraction of
    spectral power in the **final-dominant** mode/irrep — the same quantity as
    :func:`plot_w_dominant_irrep_fraction`.  Mode/irrep line colors in row 3 match row 2
    for modes that appear in the power plot (see :func:`mode_colors_aligned_with_power_plot`).

    Each run directory must contain ``train_loss_history.npy``, ``power_data.npz``, and
    ``config.yaml``. For the third row, prefer ``w_dominant_irrep_fraction.npz`` (saved during
    training via :func:`maybe_save_w_dominant_irrep_fraction_npz`); otherwise
    ``param_history.pt`` and ``param_save_indices.npy`` are used to recompute curves.
    """
    import yaml

    n_cols = len(run_dirs)
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 12))
    if n_cols == 1:
        axes = axes.reshape(3, 1)

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

        # -- Row 0: Log-Log training loss --
        ax = axes[0, col]
        pos = x_all > 0
        ax.plot(x_all[pos], np.asarray(loss_hist)[pos], lw=1.5, color="#1f77b4")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _training_loss_log_y_floor(ax)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(x_label, fontsize=9)
        if col == 0:
            ax.set_ylabel("Training Loss", fontsize=10)
        else:
            ax.tick_params(labelleft=False)

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
        _add_line_labels(ax, lines_info, fontsize=8)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(x_label, fontsize=9)
        if col == 0:
            ax.set_ylabel("Power", fontsize=10)
        else:
            ax.tick_params(labelleft=False)

        # -- Row 2: W dominant-mode / irrep fraction (same as plot_w_dominant_irrep_fraction) --
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

    fig.subplots_adjust(hspace=0.35, wspace=0.15, top=0.88)
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
    group_elements = group.elements

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
