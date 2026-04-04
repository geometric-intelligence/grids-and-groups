import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

import src.dataset as dataset
import src.model as model
import src.optimizer as optimizer
import src.power as power
import src.template as template
import src.viz as viz
from src.groups import make_group
from src.groups.cnxcn import ProductCyclicGroup

matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts for PDF viewer compatibility
matplotlib.rcParams["ps.fonttype"] = 42


def _save_param_history_pt(param_hist: list, path: Path) -> None:
    """Write ``param_history.pt``; legacy (non-zip) format avoids zip-writer failures on huge files."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(
            param_hist,
            tmp,
            pickle_protocol=4,
            _use_new_zipfile_serialization=False,
        )
    except TypeError:
        try:
            torch.save(param_hist, tmp, _use_new_zipfile_serialization=False)
        except TypeError:
            torch.save(param_hist, tmp)
    os.replace(tmp, path)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _try_load_resume_training_state(run_dir: Path) -> dict | None:
    """Load prior loss/param histories from *run_dir* for continued plotting.

    Returns ``None`` if any expected file is missing (caller may still resume
    weights-only from ``final_model.pt``).
    """
    tl = run_dir / "train_loss_history.npy"
    vl = run_dir / "val_loss_history.npy"
    ph = run_dir / "param_history.pt"
    psi = run_dir / "param_save_indices.npy"
    if not all(p.is_file() for p in (tl, vl, ph, psi)):
        return None

    train_loss_history = np.load(tl).tolist()
    val_loss_history = np.load(vl).tolist()
    param_history = torch.load(ph, map_location="cpu", weights_only=False)
    param_save_indices = [int(x) for x in np.load(psi).tolist()]

    if len(train_loss_history) != len(val_loss_history):
        raise ValueError(
            f"resume run {run_dir}: train_loss_history and val_loss_history length mismatch"
        )
    if len(param_history) != len(param_save_indices):
        raise ValueError(
            f"resume run {run_dir}: param_history and param_save_indices length mismatch"
        )

    initial_loss = float(val_loss_history[0])
    return {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "param_history": param_history,
        "param_save_indices": param_save_indices,
        "initial_loss": initial_loss,
    }


def _apply_training_resume(model: nn.Module, config: dict, device: str) -> dict | None:
    """Load ``training.resume_from`` weights; optionally full train/param history.

    * Path ending in ``.pt`` → load that ``state_dict`` only (no history merge).
    * Run directory → load ``checkpoints/final_model.pt``; if history files exist,
      return a ``resume_state`` dict for :func:`train.train` / :func:`train.train_online`.

    Optimizer state is never restored. ``--regenerate`` is unrelated (plots only).
    """
    raw = config.get("training", {}).get("resume_from")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None

    path = Path(raw).expanduser()
    resume_state: dict | None = None

    if path.suffix.lower() == ".pt":
        ckpt = path
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"training.resume_from={raw!r}: checkpoint file not found: {ckpt}"
            )
    else:
        ckpt = path / "checkpoints" / "final_model.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"training.resume_from={raw!r}: expected {ckpt} (run directory or .pt path)"
            )
        resume_state = _try_load_resume_training_state(path)
        if resume_state is None:
            print(
                "  Note: prior run directory missing some of train_loss_history.npy, "
                "val_loss_history.npy, param_history.pt, param_save_indices.npy — "
                "resuming weights only (plots will not include earlier segment)."
            )

    if not config.get("training", {}).get("save_param_snapshots", True):
        if resume_state is not None:
            print(
                "  Note: save_param_snapshots=false — loading weights only "
                "(not merging loss/param history from prior run)."
            )
        resume_state = None

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    print(f"  ✓ Loaded weights from {ckpt}")
    if resume_state is not None:
        n_loss = len(resume_state["train_loss_history"])
        n_snap = len(resume_state["param_history"])
        print(
            f"  ✓ Merging training history: {n_loss} loss points, {n_snap} param snapshots "
            f"(next index {n_loss})"
        )
    return resume_state


def setup_run_directory(base_dir: str = "runs") -> Path:
    """Create timestamped run directory."""
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    return Path(run_dir)


def save_results(
    run_dir: Path,
    config: dict,
    model,
    train_loss_hist,
    val_loss_hist,
    param_hist,
    param_save_indices,
    template: np.ndarray,
    training_time: float,
    device: str,
    save_param_snapshots: bool = True,
) -> dict:
    """Save all experiment results (histories, checkpoints, config).

    Plot-specific arrays such as ``power_data.npz`` and
    ``w_dominant_irrep_fraction.npz`` are written later by ``produce_plots_*``
    (not by this function).

    If ``save_param_snapshots`` is False, ``param_history.pt`` is omitted (empty
    ``param_save_indices.npy`` is still written). When the flag is omitted from
    config, snapshots are saved (default True).
    """
    print(f"Saving results to {run_dir}...")

    # Ensure checkpoints directory exists
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save template
    np.save(run_dir / "template.npy", template)

    # Save training history
    np.save(run_dir / "train_loss_history.npy", np.array(train_loss_hist))
    np.save(run_dir / "val_loss_history.npy", np.array(val_loss_hist))
    if save_param_snapshots:
        _save_param_history_pt(param_hist, run_dir / "param_history.pt")
        np.save(run_dir / "param_save_indices.npy", np.array(param_save_indices))
    else:
        print("  (skipping param_history.pt; save_param_snapshots=false)")
        np.save(run_dir / "param_save_indices.npy", np.array([], dtype=np.int64))

    # Save final model
    torch.save(model.state_dict(), checkpoints_dir / "final_model.pt")

    # Save metadata
    metadata = {
        "final_train_loss": float(train_loss_hist[-1]),
        "final_val_loss": float(val_loss_hist[-1]),
        "training_time_seconds": training_time,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": device,
        "description": config.get("description", ""),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("  ✓ All results saved")
    return metadata


def _reconstruct_param_save_indices(config, num_saved):
    """Reconstruct param_save_indices from config when not saved to disk."""
    dense = config["training"].get("dense_save_until", 0)
    interval = config["training"].get("save_param_interval", 1)
    epochs = config["training"]["epochs"]

    indices = [0]
    for epoch in range(1, epochs + 1):
        should_save = (
            epoch <= dense
            or (interval is not None and (epoch % interval == 0 or epoch == epochs))
            or (interval is None and epoch == epochs)
        )
        if should_save:
            indices.append(epoch)
    if len(indices) != num_saved:
        print(
            f"  Warning: reconstructed {len(indices)} indices "
            f"but have {num_saved} snapshots, using simple range"
        )
        indices = list(range(num_saved))
    return indices


def regenerate_plots(run_dir, device="cpu"):
    """Regenerate analysis plots from a saved run directory (no re-training).

    Loads config, template, param_history, and loss history from the run
    directory, rebuilds the model, and calls the appropriate produce_plots
    function.  Useful when plotting code has been updated and you want to
    refresh PDFs / save new artefacts (e.g. ``power_data.npz``,
    ``w_dominant_irrep_fraction.npz``) without repeating the expensive
    training step.
    """
    run_dir = Path(run_dir)
    print(f"\n=== Regenerating plots from {run_dir} ===")

    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    train_loss_hist = np.load(run_dir / "train_loss_history.npy").tolist()
    template = np.load(run_dir / "template.npy")
    param_hist = torch.load(run_dir / "param_history.pt", map_location=device, weights_only=False)

    psi_path = run_dir / "param_save_indices.npy"
    if psi_path.exists():
        param_save_indices = np.load(psi_path).tolist()
    else:
        param_save_indices = _reconstruct_param_save_indices(config, len(param_hist))

    group_name = config["data"]["group_name"]
    group = make_group(group_name, config)
    group_size = group.order
    model_type = config["model"]["model_type"]

    if model_type == "TwoLayerMLP":
        mdl = model.TwoLayerMLP(
            group_size=group_size,
            hidden_dim=config["model"]["hidden_dim"],
            k=config["data"]["k"],
            nonlinearity=config["model"].get("nonlinearity", "square"),
            init_scale=config["model"]["init_scale"],
            output_scale=config["model"].get("output_scale", 1.0),
        ).to(device)
    elif model_type == "QuadraticRNN":
        mdl = model.QuadraticRNN(
            group_size=group_size,
            hidden_dim=config["model"]["hidden_dim"],
            k=config["data"]["k"],
            init_scale=config["model"]["init_scale"],
            return_all_outputs=config["model"]["return_all_outputs"],
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    produce_plots(
        run_dir=run_dir,
        config=config,
        model=mdl,
        param_hist=param_hist,
        param_save_indices=param_save_indices,
        train_loss_hist=train_loss_hist,
        template=template,
        device=device,
        group=group,
    )

    sync_runs_data_cache(run_dir, config)

    print(f"\u2713 Plots regenerated for {run_dir}")


def produce_plots(
    run_dir: Path,
    config: dict,
    model,
    param_hist,
    param_save_indices,
    train_loss_hist,
    template: np.ndarray,
    device: str = "cpu",
    group=None,
):
    """Generate all analysis plots after training for any group.

    Unified replacement for the former ``produce_plots_cn``,
    ``produce_plots_cnxcn``, and ``produce_plots_group``.  See
    ``docs/refactoring_notes.md`` for details on the unification.
    """
    group_name = config["data"]["group_name"]
    group_size = group.order
    k = config["data"]["k"]
    batch_size = config["data"]["batch_size"]
    training_mode = config["training"]["mode"]

    # Human-readable label for plot titles
    _label_map = {
        "cn": lambda c: f"C{c['data']['p']}",
        "cnxcn": lambda c: f"C{c['data']['p1']}\u00d7C{c['data']['p2']}",
        "dihedral": lambda c: f"D{c['data'].get('group_n', 3)} (order {group.order})",
        "octahedral": lambda _: f"Octahedral (order {group.order})",
        "A5": lambda _: f"A5 / Icosahedral (order {group.order})",
    }
    group_label = _label_map.get(group_name, lambda _: group_name)(config)

    print(f"\n=== Generating Analysis Plots ({group_label}) ===")

    plots_bool_dict = config.get("analysis", {}).get("plots", {})
    plot_training_loss_bool = plots_bool_dict.get("training_loss", True)
    plot_predictions_bool = plots_bool_dict.get("predictions", True)
    plot_power_spectrum = plots_bool_dict.get("power_spectrum", True)
    plot_wmix_bool = plots_bool_dict.get("wmix", True)
    plot_w_dominant_irrep_fraction_bool = plots_bool_dict.get("w_dominant_irrep_fraction", True)

    # X-axis values
    total_space_size = group_size ** k
    if training_mode == "online":
        steps = np.arange(len(train_loss_hist))
        samples_seen = batch_size * steps
        fraction_of_space = samples_seen / total_space_size
        x_label = "Step"
        x_values = steps
    else:
        epochs = np.arange(len(train_loss_hist))
        samples_seen = config["data"]["num_samples"] * epochs
        fraction_of_space = samples_seen / total_space_size
        x_label = "Epoch"
        x_values = epochs

    np.save(run_dir / "samples_seen.npy", samples_seen)
    np.save(run_dir / "fraction_of_space_seen.npy", fraction_of_space)

    print(f"Total data space: {total_space_size:,} sequences")
    if len(samples_seen) > 0:
        print(f"Samples seen: {samples_seen[-1]:,} ({fraction_of_space[-1] * 100:.4f}% of space)")

    # Evaluation dataset
    print("\nGenerating evaluation data for visualization...")
    tpl_flat = np.asarray(template).ravel()
    eval_ds = dataset.GroupCompositionDataset(
        group,
        template=tpl_flat,
        k=k,
        mode="sampled",
        num_samples=min(config["data"].get("num_samples", 1000), 1000),
        return_all_outputs=config["model"]["return_all_outputs"],
    )
    X_eval_t = eval_ds.X.to(device)
    Y_eval_t = eval_ds.Y.to(device)
    print(f"  Generated {X_eval_t.shape[0]} samples for visualization")

    # Checkpoint indices
    total_checkpoints = len(param_hist)
    checkpoint_fractions = config["analysis"]["checkpoints"]
    checkpoint_indices = [int(f * (total_checkpoints - 1)) for f in checkpoint_fractions]
    print(f"Analysis checkpoints: {checkpoint_indices} (out of {total_checkpoints})")

    # Training loss plots
    if plot_training_loss_bool:
        print("\nPlotting training loss...")

        viz.plot_train_loss_with_theory(
            loss_history=train_loss_hist,
            template=template,
            group=group,
            x_values=None,
            x_label=x_label,
            save_path=os.path.join(run_dir, "training_loss_vs_steps.pdf"),
            show=False,
        )
        viz.plot_train_loss_with_theory(
            loss_history=train_loss_hist,
            template=template,
            group=group,
            x_values=samples_seen,
            x_label="Samples Seen",
            save_path=os.path.join(run_dir, "training_loss_vs_samples.pdf"),
            show=False,
        )
        viz.plot_train_loss_with_theory(
            loss_history=train_loss_hist,
            template=template,
            group=group,
            x_values=fraction_of_space,
            x_label="Samples Seen / Data Space Size",
            save_path=os.path.join(run_dir, "training_loss_vs_fraction.pdf"),
            show=False,
        )

        # 2x2 grid (Linear, Log Y, Log X, Log-Log)
        _, axes = plt.subplots(2, 2, figsize=(12, 10))
        scale_configs = [
            ("linear", "linear", "Linear Scale"),
            ("linear", "log", "Log Y"),
            ("log", "linear", "Log X"),
            ("log", "log", "Log-Log"),
        ]
        for ax, (xscale, yscale, title) in zip(axes.flat, scale_configs):
            ax.plot(x_values, train_loss_hist, lw=2, color="#1f77b4")
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Training Loss")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if yscale == "log":
                ax.set_ylim(bottom=1.0)

        lr = config["training"]["learning_rate"]
        hidden_dim = config["model"]["hidden_dim"]
        init_scale = config["model"]["init_scale"]
        plt.suptitle(
            f"{group_label} Composition (k={k}, lr={lr}, h={hidden_dim}, init={init_scale:.0e})",
            fontsize=14,
        )
        plt.tight_layout()
        training_loss_path = os.path.join(run_dir, "training_loss.pdf")
        plt.savefig(training_loss_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  \u2713 Saved {training_loss_path}")

    # Predictions
    if plot_predictions_bool:
        print("\nPlotting model predictions over time...")
        viz.plot_predictions_group(
            model=model,
            param_hist=param_hist,
            X_eval=X_eval_t,
            Y_eval=Y_eval_t,
            group_size=group_size,
            checkpoint_indices=checkpoint_indices,
            save_path=os.path.join(run_dir, "predictions_over_time.pdf"),
            group_label=group_label,
        )
        print(f"  \u2713 Saved predictions_over_time.pdf")

    # Power spectrum
    power_data = None
    if plot_power_spectrum:
        print("\nPlotting power spectrum over time...")
        optimizer_name = config["training"]["optimizer"]
        init_scale = config["model"]["init_scale"]
        power_data = viz.plot_power_group(
            model=model,
            param_hist=param_hist,
            param_save_indices=param_save_indices,
            X_eval=X_eval_t,
            template=tpl_flat,
            group=group,
            k=k,
            optimizer=optimizer_name,
            init_scale=init_scale,
            save_path=os.path.join(run_dir, "power_spectrum_analysis.pdf"),
            group_label=group_label,
            learning_rate=config["training"]["learning_rate"],
            hidden_dim=config["model"]["hidden_dim"],
        )
        print(f"  \u2713 Saved power_spectrum_analysis.pdf")
        np.savez(run_dir / "power_data.npz", **power_data)

    # Combined loss + power + weight power
    weight_power_in_combined = False
    if plot_training_loss_bool and power_data is not None:
        print("\nPlotting combined loss, power, and weight power...")
        weight_kw = (
            {
                "param_hist": param_hist,
                "param_save_indices": param_save_indices,
                "group": group,
            }
            if plot_w_dominant_irrep_fraction_bool
            else {}
        )
        weight_power_in_combined = viz.plot_loss_power_and_weight_power(
            x_values=x_values,
            train_loss_hist=train_loss_hist,
            x_label=x_label,
            power_data=power_data,
            save_path=os.path.join(run_dir, "loss_power_and_weight_power.pdf"),
            title=(
                f"{group_label} Training"
                f" (k={k}, lr={config['training']['learning_rate']},"
                f" init={config['model']['init_scale']:.0e},"
                f" h={config['model']['hidden_dim']}, {config['training']['optimizer']})"
            ),
            **weight_kw,
        )

    # W_mix frequency structure (QuadraticRNN + cnxcn only)
    model_type = config["model"]["model_type"]
    if plot_wmix_bool and model_type == "QuadraticRNN" and isinstance(group, ProductCyclicGroup):
        print("Visualizing W_mix frequency structure...")
        tracked_freqs = power.topk_template_freqs(
            np.asarray(template).reshape(group._p1, group._p2), K=10
        )
        colors = plt.cm.tab10(np.linspace(0, 1, len(tracked_freqs)))
        viz.plot_wmix_structure(
            param_hist,
            tracked_freqs,
            colors,
            group._p1,
            group._p2,
            steps=checkpoint_indices,
            within_group_size="phase",
            dead_l2_thresh=0.1,
            save_path=os.path.join(run_dir, "wmix_frequency_structure.pdf"),
            show=False,
        )

    # W-row dominant irrep fraction
    if plot_w_dominant_irrep_fraction_bool and not weight_power_in_combined:
        print("Plotting W-row dominant irrep fraction over time...")
        fig_w = viz.plot_w_dominant_irrep_fraction(
            param_hist=param_hist,
            param_save_indices=param_save_indices,
            group=group,
            x_label=x_label,
            save_path=os.path.join(run_dir, "w_dominant_irrep_fraction.pdf"),
            show=False,
        )
        if fig_w is None:
            print(
                f"  (skipped w_dominant_irrep_fraction: need W or W_out with second dim {group_size})"
            )

    viz.maybe_save_w_dominant_irrep_fraction_npz(
        run_dir, param_hist, param_save_indices, group
    )
    print(f"\n\u2713 All {group_label} plots generated successfully!")


def train_single_run(config: dict, run_dir: Path = None) -> dict:
    """Train a model on group composition for a single configuration.

    Args:
        config: Configuration dictionary. Must include 'model.model_type' to specify
                'TwoLayerMLP' or 'QuadraticRNN'.
        run_dir: Optional run directory. If None, will create a timestamped directory.

    Returns:
        dict: Training results including final losses and metadata.
    """
    # Setup run directory if not provided
    if run_dir is None:
        run_dir = setup_run_directory(base_dir="runs")
    print(f"Experiment directory: {run_dir}")

    # Set seed
    np.random.seed(config["data"]["seed"])
    torch.manual_seed(config["data"]["seed"])

    # Determine device
    device = config["device"] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ### ----- INSTANTIATE GROUP & GENERATE TEMPLATE ----- ###
    print("Generating data...")

    group_name = config["data"]["group_name"]
    template_type = config["data"]["template_type"]
    group = make_group(group_name, config)
    group_size = group.order

    print(f"Group: {group_name}, order {group_size}")
    print(f"Irrep dimensions: {[ir.size for ir in group.irreps()]}")

    # Template generation -- special-purpose templates first, then custom_fourier via group
    if template_type == "mnist" and group_name == "cn":
        tpl = template.mnist_1d(config["data"]["p"], config["data"]["mnist_label"], root="data")
    elif template_type == "mnist" and group_name == "cnxcn":
        p1, p2 = config["data"]["p1"], config["data"]["p2"]
        tpl = template.mnist_2d(p1, p2, config["data"]["mnist_label"], root="data")
    elif template_type == "gaussian":
        tpl = template.gaussian_1d(
            config["data"]["p"], n_gaussians=3, seed=config["data"]["seed"]
        )
    elif template_type == "onehot":
        tpl = template.one_hot(group_size)
    elif template_type == "custom_fourier":
        tpl = template.fixed_group(group, config["data"]["powers"])
        print(f"Template type: custom_fourier, powers={config['data']['powers']}")
    else:
        raise ValueError(f"Unknown template_type: {template_type}")

    tpl = np.asarray(tpl, dtype=np.float32)
    tpl = tpl - np.mean(tpl)

    # Visualize template
    if config.get("analysis", {}).get("plots", {}).get("template", True):
        print("Visualizing template...")
        if isinstance(group, ProductCyclicGroup):
            fig, ax = viz.plot_signal_2d(
                tpl.reshape(group._p1, group._p2), title="Template", cmap="gray"
            )
        else:
            fig, ax = plt.subplots(figsize=(max(8, group_size // 5), 4))
            if group_size <= 30:
                ax.bar(range(group_size), tpl.ravel())
                ax.set_xticks(range(group_size))
            else:
                ax.plot(tpl.ravel())
            ax.set_xlabel("Group element index")
            ax.set_ylabel("Value")
            ax.set_title(f"{group_name} Template (order={group_size})")
            ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(run_dir, "template.pdf"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print("  \u2713 Saved template")

    ### ----- SETUP TRAINING ----- ###
    print("Setting up model and training...")

    model_type = config["model"]["model_type"]
    print(f"Using model type: {model_type}")

    if model_type == "TwoLayerMLP":
        net = model.TwoLayerMLP(
            group_size=group_size,
            hidden_dim=config["model"]["hidden_dim"],
            k=config["data"]["k"],
            nonlinearity=config["model"].get("nonlinearity", "square"),
            init_scale=config["model"]["init_scale"],
            output_scale=config["model"].get("output_scale", 1.0),
        ).to(device)
    elif model_type == "QuadraticRNN":
        net = model.QuadraticRNN(
            group_size=group_size,
            hidden_dim=config["model"]["hidden_dim"],
            k=config["data"]["k"],
            init_scale=config["model"]["init_scale"],
            return_all_outputs=config["model"]["return_all_outputs"],
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    resume_state = _apply_training_resume(net, config, device)

    criterion = nn.MSELoss()

    optimizer_name = config["training"]["optimizer"]

    if optimizer_name == "adam":
        opt = optim.Adam(
            net.parameters(),
            lr=config["training"]["learning_rate"],
            betas=tuple(config["training"]["betas"]),
            weight_decay=config["training"]["weight_decay"],
        )
    elif optimizer_name == "per_neuron":
        opt = optimizer.PerNeuronScaledSGD(
            net,
            lr=config["training"]["learning_rate"],
            degree=config["training"]["degree"],
        )
    elif optimizer_name == "hybrid":
        if model_type != "QuadraticRNN":
            raise ValueError(
                f"'hybrid' optimizer is only supported for QuadraticRNN, got {model_type}"
            )
        opt = optimizer.HybridRNNOptimizer(
            net,
            lr=1,
            scaling_factor=config["training"]["scaling_factor"],
            adam_lr=config["training"]["learning_rate"],
            adam_betas=tuple(config["training"]["betas"]),
            adam_eps=1e-8,
        )
    else:
        raise ValueError(
            f"Invalid optimizer: {optimizer_name}. Must be 'adam', 'hybrid', or 'per_neuron'"
        )

    ### ----- CREATE DATA LOADERS ----- ###
    training_mode = config["training"]["mode"]
    tpl_flat = tpl.ravel()

    if training_mode == "online":
        print("Using ONLINE data generation...")

        online_kwargs = dict(
            template=tpl_flat,
            k=config["data"]["k"],
            batch_size=config["data"]["batch_size"],
            device=device,
            return_all_outputs=config["model"]["return_all_outputs"],
        )

        train_dataset = dataset.GroupCompositionDataset(group, online=True, **online_kwargs)
        val_dataset = dataset.GroupCompositionDataset(group, online=True, **online_kwargs)

        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

        num_steps = config["training"]["num_steps"]
        print(f"  Training for {num_steps} steps")

    elif training_mode == "offline":
        print("Using OFFLINE pre-generated dataset...")
        from torch.utils.data import TensorDataset

        ds_kwargs = dict(
            template=tpl_flat,
            k=config["data"]["k"],
            mode=config["data"]["mode"],
            num_samples=config["data"]["num_samples"],
            return_all_outputs=config["model"]["return_all_outputs"],
        )

        train_ds = dataset.GroupCompositionDataset(group, **ds_kwargs)

        val_samples = max(1000, config["data"]["num_samples"] // 10)
        val_kwargs = {**ds_kwargs, "mode": "sampled", "num_samples": val_samples}
        val_ds = dataset.GroupCompositionDataset(group, **val_kwargs)

        X_train_t = train_ds.X.to(device)
        Y_train_t = train_ds.Y.to(device)
        X_val_t = val_ds.X.to(device)
        Y_val_t = val_ds.Y.to(device)

        dataset_fraction = config["data"].get("dataset_fraction", 1.0)
        if dataset_fraction < 1.0:
            N = X_train_t.shape[0]
            n_sample = int(np.ceil(N * dataset_fraction))
            indices = np.random.choice(N, size=n_sample, replace=False)
            X_train_t = X_train_t[indices]
            Y_train_t = Y_train_t[indices]

        train_dataset = TensorDataset(X_train_t, Y_train_t)
        val_dataset = TensorDataset(X_val_t, Y_val_t)

        train_loader = DataLoader(
            train_dataset, batch_size=config["data"]["batch_size"], shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)

        epochs = config["training"]["epochs"]
        print(
            f"  Training for {epochs} epochs with {len(train_dataset)} samples (leaving {len(val_dataset)} samples for validation)"
        )

    else:
        raise ValueError(f"Invalid training mode: {training_mode}. Must be 'online' or 'offline'")

    ### ----- TRAIN MODEL ----- ###
    print(f"Starting training in {training_mode} mode...")

    # Get optional early stopping threshold
    reduction_threshold = config["training"].get("reduction_threshold")
    if reduction_threshold is not None:
        print(f"Early stopping enabled at {reduction_threshold * 100:.1f}% reduction")

    start_time = time.time()

    save_param_snapshots = config["training"].get("save_param_snapshots", True)

    if training_mode == "online":
        from src import train as train_mod

        train_loss_hist, val_loss_hist, param_hist, param_save_indices, final_step = (
            train_mod.train_online(
                net,
                train_loader,
                criterion,
                opt,
                num_steps=num_steps,
                verbose_interval=config["training"]["verbose_interval"],
                grad_clip=config["training"]["grad_clip"],
                eval_dataloader=val_loader,
                save_param_interval=config["training"]["save_param_interval"],
                save_param_snapshots=save_param_snapshots,
                reduction_threshold=reduction_threshold,
                resume_state=resume_state,
            )
        )
    else:  # offline
        from src import train as train_mod

        train_loss_hist, val_loss_hist, param_hist, param_save_indices, final_step = (
            train_mod.train(
                net,
                train_loader,
                criterion,
                opt,
                epochs=epochs,
                verbose_interval=config["training"]["verbose_interval"],
                grad_clip=config["training"]["grad_clip"],
                eval_dataloader=val_loader,
                save_param_interval=config["training"]["save_param_interval"],
                save_param_snapshots=save_param_snapshots,
                reduction_threshold=reduction_threshold,
                dense_save_until=config["training"].get("dense_save_until", 0),
                resume_state=resume_state,
            )
        )

    training_time = time.time() - start_time

    if not save_param_snapshots:
        with torch.no_grad():
            param_hist = [{n: p.detach().cpu().clone() for n, p in net.named_parameters()}]
        param_save_indices = [int(final_step)]

    print("\nTraining complete!")
    print(f"  Final train loss: {train_loss_hist[-1]:.6f}")
    print(f"  Final val loss: {val_loss_hist[-1]:.6f}")
    print(f"  Training time: {training_time:.2f}s")
    if reduction_threshold is not None:
        max_steps_or_epochs = num_steps if training_mode == "online" else epochs
        stopped_early = final_step < max_steps_or_epochs
        status = "CONVERGED" if stopped_early else "DID NOT CONVERGE"
        print(f"  Status: {status} at step/epoch {final_step}")

    ### ----- SAVE RESULTS ----- ###
    metadata = save_results(
        run_dir,
        config,
        net,
        train_loss_hist,
        val_loss_hist,
        param_hist,
        param_save_indices,
        tpl,
        training_time,
        device,
        save_param_snapshots=save_param_snapshots,
    )

    ### ----- PRODUCE ALL PLOTS ----- ###
    produce_plots(
        run_dir=run_dir,
        config=config,
        model=net,
        param_hist=param_hist,
        param_save_indices=param_save_indices,
        train_loss_hist=train_loss_hist,
        template=tpl,
        device=device,
        group=group,
    )

    # Return results dictionary
    results = {
        "final_train_loss": float(train_loss_hist[-1]),
        "final_val_loss": float(val_loss_hist[-1]),
        "training_time": training_time,
        "metadata": metadata,
        "run_dir": str(run_dir),
        "final_step": final_step,
    }

    # Add early stopping info if enabled
    if reduction_threshold is not None:
        max_steps_or_epochs = num_steps if training_mode == "online" else epochs
        results["converged"] = final_step < max_steps_or_epochs

    sync_runs_data_cache(run_dir, config)

    return results


def main(config: dict):
    """
    Main entry point for single training run.

    Args:
        config: Configuration dictionary.
    """
    train_single_run(config)


GROUP_CONFIG_MAP = {
    "C11": "src/configs/config_c11.yaml",
    "C5xC5": "src/configs/config_c5xc5.yaml",
    "D5": "src/configs/config_d5.yaml",
    "Oh": "src/configs/config_oh.yaml",
    "A5": "src/configs/config_a5.yaml",
}


def _group_key_for_combined_plot(config: dict) -> str | None:
    """Return GROUP_CONFIG_MAP key if *config* matches that group's reference YAML, else None."""
    for key, path in GROUP_CONFIG_MAP.items():
        try:
            ref = load_config(path)
        except FileNotFoundError:
            continue
        if _group_matches(config, ref):
            return key
    return None


def sync_runs_data_cache(run_dir: Path, config: dict) -> None:
    """Copy key plot artefacts into ``runs_data/<GROUP_KEY>/`` for :func:`make_combined_plot`.

    Copies ``train_loss_history.npy``, ``power_data.npz``, ``config.yaml``,
    ``w_dominant_irrep_fraction.npz``, and ``param_save_indices.npy`` when present.
    """
    key = _group_key_for_combined_plot(config)
    if key is None:
        return
    dest = Path("runs_data") / key
    dest.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)
    for name in (
        "train_loss_history.npy",
        "power_data.npz",
        "config.yaml",
        "w_dominant_irrep_fraction.npz",
        "param_save_indices.npy",
    ):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
    print(f"  ✓ Updated runs_data cache: {dest}")


def _auto_device():
    """Return 'cuda:0' if CUDA is available, else 'cpu'."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _group_matches(cfg, ref_config):
    """Check whether a run config matches the reference config's group."""
    ref_gn = ref_config["data"]["group_name"]
    if cfg["data"]["group_name"] != ref_gn:
        return False
    if ref_gn == "dihedral" and cfg["data"].get("group_n") != ref_config["data"].get("group_n"):
        return False
    if ref_gn == "cn" and cfg["data"].get("p") != ref_config["data"].get("p"):
        return False
    if ref_gn == "cnxcn":
        if cfg["data"].get("p1") != ref_config["data"].get("p1"):
            return False
        if cfg["data"].get("p2") != ref_config["data"].get("p2"):
            return False
    return True


def _find_latest_run_with(artifact, group_key, runs_root="runs", min_epochs=None):
    """Return the most recent run directory containing *artifact* whose
    config matches the group and has at least *min_epochs* epochs."""
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return None

    ref_config = load_config(GROUP_CONFIG_MAP[group_key])

    for d in sorted(runs_root.iterdir(), reverse=True):
        if not d.is_dir() or not (d / artifact).exists():
            continue
        cfg_path = d / "config.yaml"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        if not _group_matches(cfg, ref_config):
            continue
        if min_epochs is not None and cfg["training"].get("epochs", 0) < min_epochs:
            continue
        return d
    return None


def _estimate_training_time(group_key, target_epochs, runs_root="runs"):
    """Estimate training time in seconds by extrapolating from past runs."""
    ref_config = load_config(GROUP_CONFIG_MAP[group_key])
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return None

    for d in sorted(runs_root.iterdir(), reverse=True):
        cfg_path = d / "config.yaml"
        meta_path = d / "metadata.json"
        if not cfg_path.exists() or not meta_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        if not _group_matches(cfg, ref_config):
            continue
        past_epochs = cfg["training"].get("epochs", 0)
        if past_epochs <= 0:
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        past_time = meta.get("training_time_seconds", 0)
        return past_time / past_epochs * target_epochs
    return None


def make_combined_plot(groups=None):
    """Orchestrate: find or produce runs for each group, then combine.

    Writes ``combined_loss_and_power.pdf`` (three rows per group: loss, power,
    W dominant-mode fraction) via :func:`viz.plot_combined_loss_and_power`.
    Cached runs under ``runs_data/<GROUP>/`` should include
    ``w_dominant_irrep_fraction.npz`` (written when you train a matching config;
    see :func:`sync_runs_data_cache`).

    Automatically uses CUDA when available.  Before executing any work,
    prints a plan listing which groups are ready, which will be
    regenerated, and which require full training, together with an
    estimated total wall-clock time.
    """
    device = _auto_device()
    print(f"Device: {device}")
    if groups is None:
        groups = list(GROUP_CONFIG_MAP.keys())

    # --- Planning pass: classify each group ---
    RUNS_DATA = Path("runs_data")
    REGEN_SECONDS = 120
    PLOT_SECONDS = 10

    plan = []
    for g in groups:
        ref_cfg = load_config(GROUP_CONFIG_MAP[g])
        target_epochs = ref_cfg["training"]["epochs"]

        rd_ready = _find_latest_run_with("power_data.npz", g, min_epochs=target_epochs)
        if rd_ready is not None:
            plan.append({"group": g, "action": "ready", "run_dir": rd_ready, "est_sec": 0})
            continue

        cached = RUNS_DATA / g
        if (
            (cached / "power_data.npz").exists()
            and (cached / "train_loss_history.npy").exists()
            and (cached / "w_dominant_irrep_fraction.npz").exists()
        ):
            plan.append({"group": g, "action": "cached", "run_dir": cached, "est_sec": 0})
            continue

        rd_regen = _find_latest_run_with("param_history.pt", g, min_epochs=target_epochs)
        if rd_regen is not None:
            plan.append(
                {"group": g, "action": "regenerate", "run_dir": rd_regen, "est_sec": REGEN_SECONDS}
            )
            continue

        est = _estimate_training_time(g, target_epochs)
        if est is None:
            est = target_epochs * 0.05
        plan.append(
            {"group": g, "action": "train", "run_dir": None, "est_sec": est + REGEN_SECONDS}
        )

    # --- Print plan ---
    total_est = sum(p["est_sec"] for p in plan) + PLOT_SECONDS
    print("\n" + "=" * 60)
    print("COMBINED PLOT PLAN")
    print("=" * 60)
    for p in plan:
        g = p["group"]
        ref_cfg = load_config(GROUP_CONFIG_MAP[g])
        epochs = ref_cfg["training"]["epochs"]
        if p["action"] in ("ready", "cached"):
            status = f"READY  (using {p['run_dir']})"
        elif p["action"] == "regenerate":
            status = f"REGEN  (replot {p['run_dir']}, ~{p['est_sec']:.0f}s)"
        else:
            status = f"TRAIN  ({epochs:,} epochs, ~{p['est_sec']:.0f}s)"
        print(f"  {g:8s}  {status}")

    minutes = total_est / 60
    if minutes < 1:
        time_str = f"{total_est:.0f}s"
    else:
        time_str = f"{minutes:.1f} min"
    print(f"\nEstimated total time: {time_str}")
    print("=" * 60)

    # --- Execution pass ---
    run_dirs = []
    group_labels = []
    for p in plan:
        g = p["group"]
        print(f"\n--- {g} [{p['action'].upper()}] ---")

        if p["action"] in ("ready", "cached"):
            rd = p["run_dir"]
            print(f"  Using existing run: {rd}")
        elif p["action"] == "regenerate":
            rd = p["run_dir"]
            print(f"  Regenerating plots for: {rd}")
            regenerate_plots(rd, device=device)
        else:
            print(f"  Training from {GROUP_CONFIG_MAP[g]}")
            cfg = load_config(GROUP_CONFIG_MAP[g])
            results = train_single_run(cfg)
            rd = Path(results["run_dir"])

        run_dirs.append(rd)
        group_labels.append(g)

    save_path = "combined_loss_and_power.pdf"
    print(f"\n=== Creating combined plot: {save_path} ===")
    viz.plot_combined_loss_and_power(
        run_dirs=run_dirs,
        group_labels=group_labels,
        save_path=save_path,
    )
    print(f"\nDone. Output: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TwoLayerMLP or QuadraticRNN on group composition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/config.yaml",
        help="Path to config YAML file (default: src/configs/config.yaml)",
    )
    parser.add_argument(
        "--regenerate",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help="Regenerate plots from a saved run directory (no re-training)",
    )
    parser.add_argument(
        "--combined-plot",
        action="store_true",
        help="Produce combined_loss_and_power.pdf (3 rows x N groups: loss, power, W dominant fraction)",
    )

    args = parser.parse_args()

    if args.regenerate:
        regenerate_plots(args.regenerate, device=_auto_device())
    elif args.combined_plot:
        make_combined_plot()
    else:
        config = load_config(args.config)
        main(config)