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
    p,
    steps=None,
    example_idx=None,
    save_path=None,
    show=False,
):
    """Plot model predictions at different training steps vs ground truth (1D).

    Args:
        model: The trained model
        param_history: List of parameter snapshots from training
        X_data: Input tensor (N, k, p)
        Y_data: Target tensor (N, p)
        p: Dimension
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

    x = np.arange(p)

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


def plot_power_1d(
    model,
    param_history,
    X_data,
    Y_data,
    template_1d,
    p,
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
        X_data: Input tensor (N, k, p)
        Y_data: Target tensor (N, p)
        template_1d: The 1D template array (p,)
        p: Dimension of the template
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
    coef = 1.0 / p
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
    p: int,
    k: int,
    optimizer: str,
    init_scale: float,
    save_path: str = None,
    group_label: str = "Group",
    learning_rate: float = None,
    hidden_dim: int = None,
):
    """Plot power spectrum of model outputs vs template for cyclic group Cn.

    Mirrors plot_power_group but uses CyclicPower (no escnn).
    Each frequency mode is treated as a 1D irrep.
    """
    import src.power as power

    template_power_obj = power.CyclicPower(template_1d, template_dim=1)
    template_power = template_power_obj.power
    n_modes = len(template_power)

    print(f"  Template power spectrum (cn): {template_power}")

    model_powers, steps = power.model_power_over_time("cn", model, param_hist, X_eval)
    epoch_numbers = [param_save_indices[min(s, len(param_save_indices) - 1)] for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    top_k = min(5, n_modes)
    top_mode_indices = np.argsort(template_power)[::-1][:top_k]

    _cn_power_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors_line = _cn_power_colors[:top_k]

    valid_mask = np.array(epoch_numbers) > 0
    valid_epochs = np.array(epoch_numbers)[valid_mask]
    valid_model_powers = model_powers[valid_mask, :]

    def _mode_label(idx):
        return rf"$\rho_{{{idx}}}$ (1D)"

    # Plot 1: Linear scales
    ax = axes[0]
    for i, mode_idx in enumerate(top_mode_indices):
        power_values = model_powers[:, mode_idx]
        ax.plot(
            epoch_numbers,
            power_values,
            "-",
            lw=2,
            color=colors_line[i],
            label=_mode_label(mode_idx),
        )
        ax.axhline(template_power[mode_idx], linestyle="dotted", alpha=0.5, color=colors_line[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Power")
    ax.set_title("Linear Scales", fontsize=12)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 2: Log x-axis only
    ax = axes[1]
    for i, mode_idx in enumerate(top_mode_indices):
        power_values = valid_model_powers[:, mode_idx]
        ax.plot(
            valid_epochs, power_values, "-", lw=2, color=colors_line[i], label=_mode_label(mode_idx)
        )
        ax.axhline(template_power[mode_idx], linestyle="dotted", alpha=0.5, color=colors_line[i])
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Power")
    ax.set_title("Log X-axis", fontsize=12)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 3: Log-log scales
    ax = axes[2]
    for i, mode_idx in enumerate(top_mode_indices):
        power_values = valid_model_powers[:, mode_idx]
        power_mask = power_values > 0
        if np.any(power_mask):
            ax.plot(
                valid_epochs[power_mask],
                power_values[power_mask],
                "-",
                lw=2,
                color=colors_line[i],
                label=_mode_label(mode_idx),
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
    ax.legend(loc="upper left", fontsize=7)
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
    group_order: int,
    checkpoint_indices: list,
    save_path: str = None,
    num_samples: int = 5,
    group_label: str = "Group",
):
    """Plot model predictions vs targets at different training checkpoints.

    Args:
        model: Trained model
        param_hist: List of parameter snapshots
        X_eval: Input evaluation tensor (N, k, group_order)
        Y_eval: Target evaluation tensor (N, group_order)
        group_order: Order of the group
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
            x_axis = np.arange(group_order)

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

    Uses GroupPower from src/power.py for template power and model_power_over_time
    for model output power over training checkpoints.

    Args:
        model: Trained model
        param_hist: List of parameter snapshots
        param_save_indices: List mapping param_hist index to epoch number
        X_eval: Input evaluation tensor
        template: Template array (group_order,)
        group: escnn group object
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

    template_power_obj = power.GroupPower(template, group=group)
    template_power = template_power_obj.power

    print(f"  Template power spectrum: {template_power}")
    print("  (These are dim^2 * diag_value^2 / |G| for each irrep)")

    model_powers, steps = power.model_power_over_time(
        group_name, model, param_hist, X_eval, group=group
    )
    epoch_numbers = [param_save_indices[min(s, len(param_save_indices) - 1)] for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    top_k = min(5, n_irreps)
    top_irrep_indices = np.argsort(template_power)[::-1][:top_k]

    _group_power_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors_line = _group_power_colors[:top_k]

    valid_mask = np.array(epoch_numbers) > 0
    valid_epochs = np.array(epoch_numbers)[valid_mask]
    valid_model_powers = model_powers[valid_mask, :]

    def _irrep_label(idx, irreps):
        dim = irreps[idx].size
        dim_str = f"{dim}D"
        return rf"$\rho_{{{idx}}}$ ({dim_str})"

    # Plot 1: Linear scales
    ax = axes[0]
    for i, irrep_idx in enumerate(top_irrep_indices):
        power_values = model_powers[:, irrep_idx]
        ax.plot(
            epoch_numbers,
            power_values,
            "-",
            lw=2,
            color=colors_line[i],
            label=_irrep_label(irrep_idx, irreps),
        )
        ax.axhline(template_power[irrep_idx], linestyle="--", alpha=0.5, color=colors_line[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Power")
    ax.set_title("Linear Scales", fontsize=12)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 2: Log x-axis only
    ax = axes[1]
    for i, irrep_idx in enumerate(top_irrep_indices):
        power_values = valid_model_powers[:, irrep_idx]
        ax.plot(
            valid_epochs,
            power_values,
            "-",
            lw=2,
            color=colors_line[i],
            label=_irrep_label(irrep_idx, irreps),
        )
        ax.axhline(template_power[irrep_idx], linestyle="--", alpha=0.5, color=colors_line[i])
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Power")
    ax.set_title("Log X-axis", fontsize=12)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 3: Log-log scales
    ax = axes[2]
    for i, irrep_idx in enumerate(top_irrep_indices):
        power_values = valid_model_powers[:, irrep_idx]
        power_mask = power_values > 0
        if np.any(power_mask):
            ax.plot(
                valid_epochs[power_mask],
                power_values[power_mask],
                "-",
                lw=2,
                color=colors_line[i],
                label=_irrep_label(irrep_idx, irreps),
            )
        if template_power[irrep_idx] > 0:
            ax.axhline(template_power[irrep_idx], linestyle="--", alpha=0.5, color=colors_line[i])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("Log-Log Scales", fontsize=12)
    ax.legend(loc="upper left", fontsize=7)
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
