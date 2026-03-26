import argparse
import json
import os
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

matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts for PDF viewer compatibility
matplotlib.rcParams["ps.fonttype"] = 42


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


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
) -> dict:
    """Save all experiment results."""
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
    torch.save(param_hist, run_dir / "param_history.pt")
    np.save(run_dir / "param_save_indices.npy", np.array(param_save_indices))

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


def _build_model_from_config(config, template_flat, device):
    """Reconstruct a model from config (no weights loaded)."""
    model_type = config["model"]["model_type"]
    p_flat = len(template_flat)
    template_torch = torch.tensor(template_flat, device=device, dtype=torch.float32)

    if model_type == "QuadraticRNN":
        m = model.QuadraticRNN(
            p=p_flat,
            d=config["model"]["hidden_dim"],
            template=template_torch,
            init_scale=config["model"]["init_scale"],
            return_all_outputs=config["model"]["return_all_outputs"],
            transform_type=config["model"]["transform_type"],
        )
    elif model_type == "SequentialMLP":
        m = model.SequentialMLP(
            p=p_flat,
            d=config["model"]["hidden_dim"],
            template=template_torch,
            k=config["data"]["k"],
            init_scale=config["model"]["init_scale"],
            return_all_outputs=config["model"]["return_all_outputs"],
        )
    elif model_type == "TwoLayerNet":
        m = model.TwoLayerNet(
            group_size=p_flat,
            hidden_size=config["model"]["hidden_dim"],
            nonlinearity=config["model"].get("nonlinearity", "square"),
            init_scale=config["model"]["init_scale"],
            output_scale=config["model"].get("output_scale", 1.0),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return m.to(device)


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
    refresh PDFs / save new artefacts (e.g. power_data.npz) without
    repeating the expensive training step.
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

    mdl = _build_model_from_config(config, template.flatten(), device)

    group_name = config["data"]["group_name"]
    training_mode = config["training"]["mode"]

    if group_name == "cnxcn":
        produce_plots_cnxcn(
            run_dir=run_dir,
            config=config,
            model=mdl,
            param_hist=param_hist,
            param_save_indices=param_save_indices,
            train_loss_hist=train_loss_hist,
            template_2d=template,
            training_mode=training_mode,
            device=device,
        )
    elif group_name == "cn":
        produce_plots_cn(
            run_dir=run_dir,
            config=config,
            model=mdl,
            param_hist=param_hist,
            param_save_indices=param_save_indices,
            train_loss_hist=train_loss_hist,
            template_1d=template,
            training_mode=training_mode,
            device=device,
        )
    elif group_name in ("dihedral", "octahedral", "A5"):
        if group_name == "dihedral":
            from escnn.group import DihedralGroup

            group = DihedralGroup(N=config["data"].get("group_n", 3))
        elif group_name == "octahedral":
            from escnn.group import Octahedral

            group = Octahedral()
        else:
            from escnn.group import Icosahedral

            group = Icosahedral()
        produce_plots_group(
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
    else:
        raise ValueError(f"Unknown group_name: {group_name}")

    print(f"\u2713 Plots regenerated for {run_dir}")


def produce_plots_cnxcn(
    run_dir: Path,
    config: dict,
    model,
    param_hist,
    param_save_indices,
    train_loss_hist,
    template_2d: np.ndarray,
    training_mode: str,
    device: str,
):
    """
    Generate all analysis plots after training (2D only).

    Note: This function currently only supports 2D templates with p1 and p2 dimensions.
    For 1D templates, basic plots are generated separately in train_single_run.

    Some plots are model-specific:
    - W_mix frequency structure: QuadraticRNN only (skipped for SequentialMLP)
    - W_out neuron specialization: All models
    - Power spectrum, predictions, loss curves: All models

    Args:
        run_dir: Directory to save plots
        config: Configuration dictionary (must have dimension=2)
        model: Trained model (QuadraticRNN or SequentialMLP)
        param_hist: List of parameter snapshots
        param_save_indices: Indices where params were saved
        train_loss_hist: Training loss history
        template_2d: 2D template array (p1, p2)
        training_mode: 'online' or 'offline'
        device: Device string ('cpu' or 'cuda')
    """
    print("\n=== Generating Analysis Plots ===")

    plots = config.get("analysis", {}).get("plots", {})
    plot_training_loss = plots.get("training_loss", True)
    plot_predictions = plots.get("predictions", True)
    plot_power_spectrum = plots.get("power_spectrum", True)
    plot_wmix = plots.get("wmix", True)

    ### ----- COMPUTE X-AXIS VALUES ----- ###
    group_name = config["data"]["group_name"]
    if group_name == "cn":
        p_flat = config["data"]["p"]
    else:
        p_flat = config["data"]["p1"] * config["data"]["p2"]

    k = config["data"]["k"]
    batch_size = config["data"]["batch_size"]
    total_space_size = p_flat**k

    # Calculate different x-axis values for plotting
    if training_mode == "online":
        steps = np.arange(len(train_loss_hist))
        samples_seen = batch_size * steps
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Step"
    else:  # offline
        epochs = np.arange(len(train_loss_hist))
        samples_seen = config["data"]["num_samples"] * epochs
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Epoch"

    # Save x-axis data
    np.save(run_dir / "samples_seen.npy", samples_seen)
    np.save(run_dir / "fraction_of_space_seen.npy", fraction_of_space)

    print(f"Total data space: {total_space_size:,} sequences")
    print(f"Samples seen: {samples_seen[-1]:,} ({fraction_of_space[-1] * 100:.4f}% of space)")

    ### ----- GENERATE EVALUATION DATA ----- ###
    print("Generating evaluation data for visualization...")
    model_type = config["model"]["model_type"]

    if model_type == "TwoLayerNet":
        eval_ds_2d, _ = dataset.OfflineModularCompositionDataset.from_cnxcn(
            config["data"]["p1"],
            config["data"]["p2"],
            template_2d,
            k=2,
            mode="exhaustive",
        )
        N_eval = len(eval_ds_2d)
        X_eval_2d_t = eval_ds_2d.X.reshape(N_eval, -1).to(device)
        Y_eval_2d_t = eval_ds_2d.Y.to(device)
    else:
        eval_ds_2d, _ = dataset.OfflineModularCompositionDataset.from_cnxcn(
            config["data"]["p1"],
            config["data"]["p2"],
            template_2d,
            config["data"]["k"],
            mode="sampled",
            num_samples=min(config["data"]["num_samples"], 1000),
            return_all_outputs=config["model"]["return_all_outputs"],
        )
        X_eval_2d_t = eval_ds_2d.X.to(device)
        Y_eval_2d_t = eval_ds_2d.Y.to(device)
    print(f"  Generated {len(eval_ds_2d)} samples for visualization")

    ### ----- COMPUTE CHECKPOINT INDICES ----- ###
    total_checkpoints = len(param_hist)
    checkpoint_fractions = config["analysis"]["checkpoints"]
    checkpoint_indices = [int(f * (total_checkpoints - 1)) for f in checkpoint_fractions]

    print(f"Analysis checkpoints: {checkpoint_indices} (out of {total_checkpoints})")
    print(
        f"  Corresponding to step/epoch indices: {[param_save_indices[i] for i in checkpoint_indices]}"
    )

    ### ----- PLOT TRAINING LOSS ----- ###
    if plot_training_loss:
        print("\nPlotting training loss...")

        # Plot 1: Loss vs Steps/Epochs
        viz.plot_train_loss_with_theory(
            loss_history=train_loss_hist,
            template_2d=template_2d,
            p1=config["data"]["p1"],
            p2=config["data"]["p2"],
            x_values=None,
            x_label=x_label_steps,
            save_path=os.path.join(run_dir, "training_loss_vs_steps.pdf"),
            show=False,
        )

        # Plot 2: Loss vs Samples Seen
        viz.plot_train_loss_with_theory(
            loss_history=train_loss_hist,
            template_2d=template_2d,
            p1=config["data"]["p1"],
            p2=config["data"]["p2"],
            x_values=samples_seen,
            x_label="Samples Seen",
            save_path=os.path.join(run_dir, "training_loss_vs_samples.pdf"),
            show=False,
        )

        # Plot 3: Loss vs Fraction of Space
        viz.plot_train_loss_with_theory(
            loss_history=train_loss_hist,
            template_2d=template_2d,
            p1=config["data"]["p1"],
            p2=config["data"]["p2"],
            x_values=fraction_of_space,
            x_label="Samples Seen / Data Space Size",
            save_path=os.path.join(run_dir, "training_loss_vs_fraction.pdf"),
            show=False,
        )

        # Plot 4: 2x2 grid (Linear, Log Y, Log X, Log-Log)
        x_values = steps if training_mode == "online" else epochs
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
            ax.set_xlabel(x_label_steps)
            ax.set_ylabel("Training Loss")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if yscale == "log":
                ax.set_ylim(bottom=1.0)

        p1 = config["data"]["p1"]
        p2 = config["data"]["p2"]
        group_label = f"C{p1}\u00d7C{p2}"
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

    ### ----- PLOT MODEL PREDICTIONS ----- ###
    if plot_predictions and model_type != "TwoLayerNet":
        print("Plotting model predictions over time...")
        viz.plot_predictions_2d(
            model,
            param_hist,
            X_eval_2d_t,
            Y_eval_2d_t,
            config["data"]["p1"],
            config["data"]["p2"],
            steps=checkpoint_indices,
            save_path=os.path.join(run_dir, "predictions_over_time.pdf"),
            show=False,
        )

    ### ----- PLOT POWER SPECTRUM ANALYSIS ----- ###
    power_data = None
    if plot_power_spectrum:
        print("Plotting power spectrum over time...")
        p1 = config["data"]["p1"]
        p2 = config["data"]["p2"]
        optimizer_name = config["training"]["optimizer"]
        init_scale = config["model"]["init_scale"]
        group_label = f"C{p1}\u00d7C{p2}"
        power_data = viz.plot_power_cnxcn(
            model=model,
            param_hist=param_hist,
            param_save_indices=param_save_indices,
            X_eval=X_eval_2d_t,
            template_2d=template_2d,
            p1=p1,
            p2=p2,
            k=k,
            optimizer=optimizer_name,
            init_scale=init_scale,
            save_path=os.path.join(run_dir, "power_spectrum_analysis.pdf"),
            group_label=group_label,
            learning_rate=config["training"]["learning_rate"],
            hidden_dim=config["model"]["hidden_dim"],
        )
        np.savez(run_dir / "power_data.npz", **power_data)

    ### ----- PLOT COMBINED LOSS AND POWER ----- ###
    if plot_training_loss and power_data is not None:
        print("\nPlotting combined loss and power...")
        x_values = steps if training_mode == "online" else epochs
        p1 = config["data"]["p1"]
        p2 = config["data"]["p2"]
        group_label = f"C{p1}\u00d7C{p2}"
        viz.plot_loss_and_power(
            x_values=x_values,
            train_loss_hist=train_loss_hist,
            x_label=x_label_steps,
            power_data=power_data,
            save_path=os.path.join(run_dir, "loss_and_power.pdf"),
            title=(
                f"{group_label} Training"
                f" (k={k}, lr={config['training']['learning_rate']},"
                f" init={config['model']['init_scale']:.0e},"
                f" h={config['model']['hidden_dim']}, {config['training']['optimizer']})"
            ),
        )

    ### ----- PLOT W_MIX FREQUENCY STRUCTURE (QuadraticRNN only) ----- ###
    if plot_wmix and model_type == "QuadraticRNN":
        print("Creating Fourier modes reference...")
        tracked_freqs = power.topk_template_freqs(template_2d, K=10)
        colors = plt.cm.tab10(np.linspace(0, 1, len(tracked_freqs)))
        print("Visualizing W_mix frequency structure...")
        viz.plot_wmix_structure(
            param_hist,
            tracked_freqs,
            colors,
            config["data"]["p1"],
            config["data"]["p2"],
            steps=checkpoint_indices,
            within_group_order="phase",
            dead_l2_thresh=0.1,
            save_path=os.path.join(run_dir, "wmix_frequency_structure.pdf"),
            show=False,
        )
    elif model_type != "QuadraticRNN":
        print("Skipping W_mix frequency structure plot (not applicable for SequentialMLP)")

    print("\n✓ All plots generated successfully!")


def produce_plots_cn(
    run_dir: Path,
    config: dict,
    model,
    param_hist,
    param_save_indices,
    train_loss_hist,
    template_1d: np.ndarray,
    training_mode: str,
    device: str,
):
    """
    Generate all analysis plots after training (1D version).

    Args:
        run_dir: Directory to save plots
        config: Configuration dictionary (must have dimension=1)
        model: Trained model (QuadraticRNN or SequentialMLP)
        param_hist: List of parameter snapshots
        param_save_indices: Indices where params were saved
        train_loss_hist: Training loss history
        template_1d: 1D template array (p,)
        training_mode: 'online' or 'offline'
        device: Device string ('cpu' or 'cuda')
    """
    print("\n=== Generating Analysis Plots (1D) ===")

    plots = config.get("analysis", {}).get("plots", {})
    plot_training_loss = plots.get("training_loss", True)
    plot_predictions = plots.get("predictions", True)
    plot_power_spectrum = plots.get("power_spectrum", True)

    ### ----- COMPUTE X-AXIS VALUES ----- ###
    p = config["data"]["p"]
    k = config["data"]["k"]
    batch_size = config["data"]["batch_size"]
    total_space_size = p**k

    # Calculate different x-axis values for plotting
    if training_mode == "online":
        steps = np.arange(len(train_loss_hist))
        samples_seen = batch_size * steps
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Step"
    else:  # offline
        epochs = np.arange(len(train_loss_hist))
        samples_seen = config["data"]["num_samples"] * epochs
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Epoch"

    # Save x-axis data
    np.save(run_dir / "samples_seen.npy", samples_seen)
    np.save(run_dir / "fraction_of_space_seen.npy", fraction_of_space)

    print(f"Total data space: {total_space_size:,} sequences")
    print(f"Samples seen: {samples_seen[-1]:,} ({fraction_of_space[-1] * 100:.4f}% of space)")

    model_type = config["model"]["model_type"]
    template_type = config["data"]["template_type"]
    use_group_style = model_type == "TwoLayerNet" and template_type == "custom_fourier"

    ### ----- GENERATE EVALUATION DATA ----- ###
    print("Generating evaluation data for visualization...")

    if use_group_style:
        eval_ds, _ = dataset.OfflineModularCompositionDataset.from_cn(
            config["data"]["p"],
            template_1d,
            k=2,
            mode="exhaustive",
        )
        N_eval = len(eval_ds)
        X_eval_t = eval_ds.X.reshape(N_eval, -1).to(device)
        Y_eval_t = eval_ds.Y.to(device)
        print(f"  Generated {N_eval} samples for visualization")
    else:
        eval_ds_1d, _ = dataset.OfflineModularCompositionDataset.from_cn(
            config["data"]["p"],
            template_1d,
            config["data"]["k"],
            mode="sampled",
            num_samples=min(config["data"]["num_samples"], 1000),
            return_all_outputs=config["model"]["return_all_outputs"],
        )
        X_seq_1d_t = eval_ds_1d.X.to(device)
        Y_seq_1d_t = eval_ds_1d.Y.to(device)
        print(f"  Generated {len(eval_ds_1d)} samples for visualization")

    ### ----- COMPUTE CHECKPOINT INDICES ----- ###
    total_checkpoints = len(param_hist)
    checkpoint_fractions = config["analysis"]["checkpoints"]
    checkpoint_indices = [int(f * (total_checkpoints - 1)) for f in checkpoint_fractions]

    print(f"Analysis checkpoints: {checkpoint_indices} (out of {total_checkpoints})")
    print(
        f"  Corresponding to step/epoch indices: {[param_save_indices[i] for i in checkpoint_indices]}"
    )

    ### ----- PLOT TRAINING LOSS ----- ###
    if plot_training_loss:
        print("\nPlotting training loss...")

        _, axes = plt.subplots(2, 2, figsize=(12, 10))
        x_values = steps if training_mode == "online" else epochs

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
            ax.set_xlabel(x_label_steps)
            ax.set_ylabel("Training Loss")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        if yscale == "log":
            ax.set_ylim(bottom=1.0)

    if use_group_style:
        lr = config["training"]["learning_rate"]
        hidden_dim = config["model"]["hidden_dim"]
        init_scale = config["model"]["init_scale"]
        group_label = f"C{p}"
        plt.suptitle(
            f"{group_label} Composition (k={k}, lr={lr}, h={hidden_dim}, init={init_scale:.0e})",
            fontsize=14,
        )

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "training_loss.pdf"), bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  ✓ Saved {os.path.join(run_dir, 'training_loss.pdf')}")

    ### ----- PLOT MODEL PREDICTIONS ----- ###
    if not use_group_style:
        if plot_predictions:
            print("Plotting model predictions over time...")
            viz.plot_predictions_1d(
                model,
                param_hist,
                X_seq_1d_t,
                Y_seq_1d_t,
                p,
                steps=checkpoint_indices,
                save_path=os.path.join(run_dir, "predictions_over_time.pdf"),
                show=False,
            )

    ### ----- PLOT POWER SPECTRUM ANALYSIS ----- ###
    power_data = None
    if plot_power_spectrum:
        print("Plotting power spectrum over time...")

        if use_group_style:
            optimizer_name = config["training"]["optimizer"]
            init_scale = config["model"]["init_scale"]
            group_label = f"C{p}"
            power_data = viz.plot_power_cn(
                model=model,
                param_hist=param_hist,
                param_save_indices=param_save_indices,
                X_eval=X_eval_t,
                template_1d=template_1d,
                p=p,
                k=k,
                optimizer=optimizer_name,
                init_scale=init_scale,
                save_path=os.path.join(run_dir, "power_spectrum_analysis.pdf"),
                group_label=group_label,
                learning_rate=config["training"]["learning_rate"],
                hidden_dim=config["model"]["hidden_dim"],
            )
            np.savez(run_dir / "power_data.npz", **power_data)
        else:
            viz.plot_power_1d(
                model,
                param_hist,
                X_seq_1d_t,
                Y_seq_1d_t,
                template_1d,
                p,
                loss_history=train_loss_hist,
                param_save_indices=param_save_indices,
                num_freqs_to_track=min(10, p // 4),
                checkpoint_indices=checkpoint_indices,
                num_samples=100,
                save_path=os.path.join(run_dir, "power_spectrum_analysis.pdf"),
                show=False,
            )

    ### ----- PLOT COMBINED LOSS AND POWER ----- ###
    if plot_training_loss and power_data is not None:
        print("\nPlotting combined loss and power...")
        x_values = steps if training_mode == "online" else epochs
        group_label = f"C{p}"
        viz.plot_loss_and_power(
            x_values=x_values,
            train_loss_hist=train_loss_hist,
            x_label=x_label_steps,
            power_data=power_data,
            save_path=os.path.join(run_dir, "loss_and_power.pdf"),
            title=(
                f"{group_label} Training"
                f" (k={k}, lr={config['training']['learning_rate']},"
                f" init={config['model']['init_scale']:.0e},"
                f" h={config['model']['hidden_dim']}, {config['training']['optimizer']})"
            ),
        )

    print(f"\n\u2713 All C{p} plots generated successfully!")


def produce_plots_group(
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
    """
    Generate all analysis plots after training for any escnn group.

    Args:
        run_dir: Directory to save plots
        config: Configuration dictionary
        model: Trained model
        param_hist: List of parameter snapshots
        param_save_indices: Indices where params were saved
        train_loss_hist: Training loss history
        template: 1D template array of shape (group_order,)
        device: Device string ('cpu' or 'cuda')
        group: escnn group object (required)
    """
    group_name = config["data"]["group_name"]

    # Build a human-readable label for plot titles
    if group_name == "dihedral":
        n = config["data"].get("group_n", 3)
        group_label = f"D{n} (Dihedral, order {group.order()})"
    elif group_name == "octahedral":
        group_label = f"Octahedral (order {group.order()})"
    elif group_name == "A5":
        group_label = f"A5 / Icosahedral (order {group.order()})"
    else:
        group_label = group_name

    print(f"\n=== Generating Analysis Plots ({group_label}) ===")

    plots = config.get("analysis", {}).get("plots", {})
    plot_training_loss = plots.get("training_loss", True)
    plot_predictions = plots.get("predictions", True)
    plot_power_spectrum = plots.get("power_spectrum", True)

    group_order = group.order()

    k = config["data"]["k"]
    batch_size = config["data"]["batch_size"]
    training_mode = config["training"]["mode"]

    # Total data space size with k compositions
    total_space_size = group_order**k

    # Calculate x-axis values
    if training_mode == "online":
        steps = np.arange(len(train_loss_hist))
        samples_seen = batch_size * steps
        fraction_of_space = samples_seen / total_space_size
        x_label = "Step"
        x_values = steps
    else:  # offline
        epochs = np.arange(len(train_loss_hist))
        samples_seen = config["data"]["num_samples"] * epochs
        fraction_of_space = samples_seen / total_space_size
        x_label = "Epoch"
        x_values = epochs

    # Save x-axis data
    samples_seen_path = run_dir / "samples_seen.npy"
    fraction_path = run_dir / "fraction_of_space_seen.npy"
    np.save(samples_seen_path, samples_seen)
    np.save(fraction_path, fraction_of_space)
    print(f"  ✓ Saved {samples_seen_path}")
    print(f"  ✓ Saved {fraction_path}")

    print(f"\n{group_name} group order: {group_order}")
    print(f"Sequence length k: {k}")
    print(f"Total data space: {total_space_size:,} sequences")
    if len(samples_seen) > 0:
        print(f"Samples seen: {samples_seen[-1]:,} ({fraction_of_space[-1] * 100:.4f}% of space)")

    ### ----- GENERATE EVALUATION DATA ----- ###
    print("\nGenerating evaluation data for visualization...")
    model_type = config["model"]["model_type"]

    if model_type == "TwoLayerNet":
        # TwoLayerNet expects flattened binary pair input: (N, 2*group_size)
        eval_ds, _ = dataset.OfflineModularCompositionDataset.from_group(
            template,
            k=2,
            group=group,
            mode="exhaustive",
        )
        N_eval = len(eval_ds)
        X_eval_t = eval_ds.X.reshape(N_eval, -1).to(device)
        Y_eval_t = eval_ds.Y.to(device)
        # Optionally subsample for visualization
        n_eval = min(len(X_eval_t), 1000)
        if n_eval < len(X_eval_t):
            indices = np.random.choice(len(X_eval_t), size=n_eval, replace=False)
            X_eval_t = X_eval_t[indices]
            Y_eval_t = Y_eval_t[indices]
    else:
        # Sequence models use the generic sequence dataset
        eval_ds, _ = dataset.OfflineModularCompositionDataset.from_group(
            template,
            k,
            group=group,
            mode="sampled",
            num_samples=min(config["data"]["num_samples"], 1000),
            return_all_outputs=config["model"]["return_all_outputs"],
        )
        X_eval_t = eval_ds.X.to(device)
        Y_eval_t = eval_ds.Y.to(device)

    print(f"  Generated {X_eval_t.shape[0]} samples for visualization")

    ### ----- COMPUTE CHECKPOINT INDICES ----- ###
    total_checkpoints = len(param_hist)
    checkpoint_fractions = config["analysis"]["checkpoints"]
    checkpoint_indices = [int(f * (total_checkpoints - 1)) for f in checkpoint_fractions]
    print(f"Analysis checkpoints: {checkpoint_indices} (out of {total_checkpoints})")

    ### ----- PLOT TRAINING LOSS ----- ###
    if plot_training_loss:
        print("\nPlotting training loss...")

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
    print(f"  ✓ Saved {training_loss_path}")

    ### ----- PLOT MODEL PREDICTIONS OVER TIME ----- ###
    if plot_predictions:
        print("\nPlotting model predictions over time...")
        viz.plot_predictions_group(
            model=model,
            param_hist=param_hist,
            X_eval=X_eval_t,
            Y_eval=Y_eval_t,
            group_order=group_order,
            checkpoint_indices=checkpoint_indices,
            save_path=os.path.join(run_dir, "predictions_over_time.pdf"),
            group_label=group_label,
        )
        print(f"  ✓ Saved {os.path.join(run_dir, 'predictions_over_time.pdf')}")

    ### ----- PLOT POWER SPECTRUM OVER TIME ----- ###
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
            template=template,
            group=group,
            k=k,
            optimizer=optimizer_name,
            init_scale=init_scale,
            save_path=os.path.join(run_dir, "power_spectrum_analysis.pdf"),
            group_label=group_label,
            learning_rate=config["training"]["learning_rate"],
            hidden_dim=config["model"]["hidden_dim"],
        )
        print(f"  ✓ Saved {os.path.join(run_dir, 'power_spectrum_analysis.pdf')}")
        np.savez(run_dir / "power_data.npz", **power_data)

    ### ----- PLOT COMBINED LOSS AND POWER ----- ###
    if plot_training_loss and power_data is not None:
        print("\nPlotting combined loss and power...")
        viz.plot_loss_and_power(
            x_values=x_values,
            train_loss_hist=train_loss_hist,
            x_label=x_label,
            power_data=power_data,
            save_path=os.path.join(run_dir, "loss_and_power.pdf"),
            title=(
                f"{group_label} Training"
                f" (k={k}, lr={config['training']['learning_rate']},"
                f" init={config['model']['init_scale']:.0e},"
                f" h={config['model']['hidden_dim']}, {config['training']['optimizer']})"
            ),
        )

    print(f"\n✓ All {group_label} plots generated successfully!")


def train_single_run(config: dict, run_dir: Path = None) -> dict:
    """
    Train a model (QuadraticRNN or SequentialMLP) on modular addition for a single configuration.

    Args:
        config: Configuration dictionary. Must include 'model.model_type' to specify
                'QuadraticRNN' or 'SequentialMLP'.
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

    ### ----- GENERATE DATA ----- ###
    print("Generating data...")

    group_name = config["data"]["group_name"]
    group_n = config["data"].get("group_n")  # For dihedral groups (D3, D4, etc.)
    template_type = config["data"]["template_type"]

    if group_name == "cn":
        # 1D template generation
        p = config["data"]["p"]
        p_flat = p

        if template_type == "mnist":
            template_1d = template.mnist_1d(p, config["data"]["mnist_label"], root="data")
        elif template_type == "gaussian":
            template_1d = template.gaussian_1d(p, n_gaussians=3, seed=config["data"]["seed"])
        elif template_type == "onehot":
            template_1d = template.onehot_1d(p)
        elif template_type == "custom_fourier":
            powers = config["data"]["powers"]
            print("Template type: custom_fourier")
            print(f"Desired powers (per freq mode): {powers}")
            template_1d = template.fixed_cn(p, powers)
        else:
            raise ValueError(f"Unknown template_type: {template_type}")

        template_1d = template_1d - np.mean(template_1d)
        tpl = template_1d  # For consistency in code below

        # Visualize 1D template
        if config.get("analysis", {}).get("plots", {}).get("template", True):
            print("Visualizing template...")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(template_1d)
            ax.set_xlabel("Position")
            ax.set_ylabel("Value")
            ax.set_title("1D Template")
            ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(run_dir, "template.pdf"), bbox_inches="tight", dpi=150)
            plt.close(fig)
            print("  ✓ Saved template")

    elif group_name == "cnxcn":
        # 2D template generation
        p1 = config["data"]["p1"]
        p2 = config["data"]["p2"]
        p_flat = p1 * p2

        if template_type == "mnist":
            template_2d = template.mnist_2d(p1, p2, config["data"]["mnist_label"], root="data")
        elif template_type == "custom_fourier":
            assert p1 == p2, f"custom_fourier for cnxcn requires p1 == p2, got p1={p1}, p2={p2}"
            powers = config["data"]["powers"]
            print("Template type: custom_fourier")
            print(f"Desired powers (per 2D mode): {powers}")
            tpl_flat = template.fixed_cnxcn(p1, p2, powers)
            template_2d = tpl_flat.reshape(p1, p2)
        else:
            raise ValueError(f"Unknown template_type for cnxcn: {template_type}")

        template_2d = template_2d - np.mean(template_2d)
        tpl = template_2d  # For consistency in code below

        # Visualize 2D template
        if config.get("analysis", {}).get("plots", {}).get("template", True):
            print("Visualizing template...")
            fig, ax = viz.plot_signal_2d(template_2d, title="Template", cmap="gray")
            fig.savefig(os.path.join(run_dir, "template.pdf"), bbox_inches="tight", dpi=150)
            plt.close(fig)
            print("  ✓ Saved template")
    elif group_name in ("dihedral", "octahedral", "A5"):
        # Construct the escnn group object
        if group_name == "dihedral":
            from escnn.group import DihedralGroup

            n = group_n if group_n is not None else 3
            group = DihedralGroup(N=n)
            group_label = f"Dihedral D{n}"
        elif group_name == "octahedral":
            from escnn.group import Octahedral

            group = Octahedral()
            group_label = "Octahedral"
        elif group_name == "A5":
            from escnn.group import Icosahedral

            group = Icosahedral()
            group_label = "Icosahedral (A5)"

        group_order = group.order()
        p_flat = group_order

        print(f"{group_label} group order: {group_order}")
        print(f"{group_label} irreps: {[irrep.size for irrep in group.irreps()]} (dimensions)")

        # Generate template
        if template_type == "onehot":
            tpl = np.zeros(group_order, dtype=np.float32)
            tpl[1] = 10.0
            tpl = tpl - np.mean(tpl)
            print("Template type: onehot")

        elif template_type == "custom_fourier":
            powers = config["data"]["powers"]
            print("Template type: custom_fourier")
            print(f"Desired powers (per irrep): {powers}")
            tpl = template.fixed_group(group, powers)
        else:
            raise ValueError(
                f"Unknown template_type for {group_name}: {template_type}. "
                "Must be 'onehot' or 'custom_fourier'"
            )

        print(f"Template shape: {tpl.shape}")

        # Visualize template
        if config.get("analysis", {}).get("plots", {}).get("template", True):
            print("Visualizing template...")
            fig, ax = plt.subplots(figsize=(max(8, group_order // 5), 4))
            ax.bar(range(group_order), tpl)
            ax.set_xlabel("Group element index")
            ax.set_ylabel("Value")
            title = f"{group_label} Template (order={group_order}, type={template_type})"
            if template_type == "custom_fourier":
                title += f"\npowers={powers}"
            ax.set_title(title)
            if group_order <= 30:
                ax.set_xticks(range(group_order))
            fig.savefig(os.path.join(run_dir, "template.pdf"), bbox_inches="tight", dpi=150)
            plt.close(fig)
            print("  ✓ Saved template")
    else:
        raise ValueError(
            f"group_name must be 'cn', 'cnxcn', 'dihedral', 'octahedral', or 'A5', got {group_name}"
        )

    ### ----- SETUP TRAINING ----- ###
    print("Setting up model and training...")

    # Flatten template for model (works for both 1D and 2D)
    template_torch = torch.tensor(tpl, device=device, dtype=torch.float32).flatten()

    # Determine which model to use
    model_type = config["model"]["model_type"]
    print(f"Using model type: {model_type}")

    if model_type == "QuadraticRNN":
        rnn_2d = model.QuadraticRNN(
            p=p_flat,
            d=config["model"]["hidden_dim"],
            template=template_torch,
            init_scale=config["model"]["init_scale"],
            return_all_outputs=config["model"]["return_all_outputs"],
            transform_type=config["model"]["transform_type"],
        ).to(device)
    elif model_type == "SequentialMLP":
        rnn_2d = model.SequentialMLP(
            p=p_flat,
            d=config["model"]["hidden_dim"],
            template=template_torch,
            k=config["data"]["k"],
            init_scale=config["model"]["init_scale"],
            return_all_outputs=config["model"]["return_all_outputs"],
        ).to(device)
    elif model_type == "TwoLayerNet":
        hidden_dim = config["model"]["hidden_dim"]
        nonlinearity = config["model"].get("nonlinearity", "square")
        output_scale = config["model"].get("output_scale", 1.0)
        rnn_2d = model.TwoLayerNet(
            group_size=p_flat,
            hidden_size=hidden_dim,
            nonlinearity=nonlinearity,
            init_scale=config["model"]["init_scale"],
            output_scale=output_scale,
        ).to(device)
    else:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be 'QuadraticRNN', 'SequentialMLP', or 'TwoLayerNet'"
        )

    criterion = nn.MSELoss()

    # Optimizer selection with model-aware defaults
    optimizer_name = config["training"]["optimizer"]

    # Auto-select optimizer if not specified or if 'auto'
    if optimizer_name == "auto" or (optimizer_name not in ["adam", "hybrid", "per_neuron"]):
        if model_type == "SequentialMLP":
            optimizer_name = "per_neuron"
            print(f"Auto-selected optimizer: {optimizer_name} (recommended for SequentialMLP)")
        else:
            optimizer_name = "adam"
            print(f"Auto-selected optimizer: {optimizer_name}")
    else:
        print(f"Using optimizer: {optimizer_name}")

    if optimizer_name == "adam":
        opt = optim.Adam(
            rnn_2d.parameters(),
            lr=config["training"]["learning_rate"],
            betas=tuple(config["training"]["betas"]),
            weight_decay=config["training"]["weight_decay"],
        )
    elif optimizer_name == "hybrid":
        if model_type != "QuadraticRNN":
            raise ValueError(
                f"'hybrid' optimizer is only supported for QuadraticRNN, got {model_type}"
            )
        opt = optimizer.HybridRNNOptimizer(
            rnn_2d,
            lr=1,
            scaling_factor=config["training"]["scaling_factor"],
            adam_lr=config["training"]["learning_rate"],
            adam_betas=tuple(config["training"]["betas"]),
            adam_eps=1e-8,
        )
    elif optimizer_name == "per_neuron":
        # Per-neuron scaled SGD (recommended for SequentialMLP)
        degree = config["training"]["degree"]
        lr = config["training"]["learning_rate"]

        # For SequentialMLP, use lr=1.0 by default if not specified
        if model_type == "SequentialMLP" and lr == 1.0e-3:
            print("  Note: Using lr=1.0 for per_neuron optimizer with SequentialMLP")
            lr = 1.0

        opt = optimizer.PerNeuronScaledSGD(
            rnn_2d,
            lr=lr,
            degree=degree,  # Will auto-infer as k+1 for SequentialMLP (k = sequence length)
        )
        print(f"  Degree of homogeneity: {opt.param_groups[0]['degree']}")
    else:
        raise ValueError(
            f"Invalid optimizer: {optimizer_name}. Must be 'adam', 'hybrid', or 'per_neuron'"
        )

    ### ----- CREATE DATA LOADERS ----- ###
    training_mode = config["training"]["mode"]

    if training_mode == "online":
        print("Using ONLINE data generation...")

        if group_name == "cn":
            # Training dataset
            train_dataset = dataset.OnlineModularAdditionDataset1D(
                p=config["data"]["p"],
                template=template_1d,
                k=config["data"]["k"],
                batch_size=config["data"]["batch_size"],
                device=device,
                return_all_outputs=config["model"]["return_all_outputs"],
            )

            # Validation dataset
            val_dataset = dataset.OnlineModularAdditionDataset1D(
                p=config["data"]["p"],
                template=template_1d,
                k=config["data"]["k"],
                batch_size=config["data"]["batch_size"],
                device=device,
                return_all_outputs=config["model"]["return_all_outputs"],
            )
        elif group_name == "cnxcn":
            # Training dataset
            train_dataset = dataset.OnlineModularAdditionDataset2D(
                p1=config["data"]["p1"],
                p2=config["data"]["p2"],
                template=template_2d,
                k=config["data"]["k"],
                batch_size=config["data"]["batch_size"],
                device=device,
                return_all_outputs=config["model"]["return_all_outputs"],
            )

            # Validation dataset
            val_dataset = dataset.OnlineModularAdditionDataset2D(
                p1=config["data"]["p1"],
                p2=config["data"]["p2"],
                template=template_2d,
                k=config["data"]["k"],
                batch_size=config["data"]["batch_size"],
                device=device,
                return_all_outputs=config["model"]["return_all_outputs"],
            )
        elif group_name in ["dihedral", "octahedral", "A5"]:
            # Online training for these groups is not yet implemented
            raise NotImplementedError(
                f"Online training mode is not yet implemented for {group_name}. "
                "Please use training.mode='offline' in the config."
            )
        else:
            raise ValueError(
                f"group_name must be 'cn', 'cnxcn', 'dihedral', 'octahedral', or 'A5', got {group_name}"
            )

        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

        num_steps = config["training"]["num_steps"]
        print(f"  Training for {num_steps} steps")

    elif training_mode == "offline":
        print("Using OFFLINE pre-generated dataset...")
        from torch.utils.data import TensorDataset

        if model_type == "TwoLayerNet":
            # TwoLayerNet: X=(N, k, group_size) -> flattened to (N, k*group_size)
            if group_name == "cn":
                pair_ds, _ = dataset.OfflineModularCompositionDataset.from_cn(
                    config["data"]["p"],
                    tpl,
                    k=config["data"]["k"],
                    mode="exhaustive",
                )
            elif group_name == "cnxcn":
                pair_ds, _ = dataset.OfflineModularCompositionDataset.from_cnxcn(
                    config["data"]["p1"],
                    config["data"]["p2"],
                    tpl,
                    k=config["data"]["k"],
                    mode="exhaustive",
                )
            elif group_name in ("dihedral", "octahedral", "A5"):
                pair_ds, _ = dataset.OfflineModularCompositionDataset.from_group(
                    tpl,
                    k=config["data"]["k"],
                    group=group,
                    mode="exhaustive",
                )
            else:
                raise ValueError(f"Unsupported group_name for TwoLayerNet: {group_name}")

            N_all = len(pair_ds)
            X_all = pair_ds.X.reshape(N_all, -1).to(device)
            Y_all = pair_ds.Y.to(device)

            # Apply dataset_fraction if configured
            dataset_fraction = config["data"].get("dataset_fraction", 1.0)
            if dataset_fraction < 1.0:
                N = X_all.shape[0]
                n_sample = int(np.ceil(N * dataset_fraction))
                indices = np.random.choice(N, size=n_sample, replace=False)
                X_all = X_all[indices]
                Y_all = Y_all[indices]

            # Split into train/val (90/10)
            N = X_all.shape[0]
            n_val = max(1, N // 10)
            n_train = N - n_val
            X_train_t, X_val_t = X_all[:n_train], X_all[n_train:]
            Y_train_t, Y_val_t = Y_all[:n_train], Y_all[n_train:]

        else:
            # Sequence models (QuadraticRNN, SequentialMLP) use sequence datasets
            if group_name == "cn":
                train_ds, _ = dataset.OfflineModularCompositionDataset.from_cn(
                    config["data"]["p"],
                    template_1d,
                    config["data"]["k"],
                    mode=config["data"]["mode"],
                    num_samples=config["data"]["num_samples"],
                    return_all_outputs=config["model"]["return_all_outputs"],
                )

                val_samples = max(1000, config["data"]["num_samples"] // 10)
                val_ds, _ = dataset.OfflineModularCompositionDataset.from_cn(
                    config["data"]["p"],
                    template_1d,
                    config["data"]["k"],
                    mode="sampled",
                    num_samples=val_samples,
                    return_all_outputs=config["model"]["return_all_outputs"],
                )
            elif group_name == "cnxcn":
                train_ds, _ = dataset.OfflineModularCompositionDataset.from_cnxcn(
                    config["data"]["p1"],
                    config["data"]["p2"],
                    template_2d,
                    config["data"]["k"],
                    mode=config["data"]["mode"],
                    num_samples=config["data"]["num_samples"],
                    return_all_outputs=config["model"]["return_all_outputs"],
                )

                val_samples = max(1000, config["data"]["num_samples"] // 10)
                val_ds, _ = dataset.OfflineModularCompositionDataset.from_cnxcn(
                    config["data"]["p1"],
                    config["data"]["p2"],
                    template_2d,
                    config["data"]["k"],
                    mode="sampled",
                    num_samples=val_samples,
                    return_all_outputs=config["model"]["return_all_outputs"],
                )
            elif group_name in ("dihedral", "octahedral", "A5"):
                train_ds, _ = dataset.OfflineModularCompositionDataset.from_group(
                    tpl,
                    config["data"]["k"],
                    group=group,
                    mode=config["data"]["mode"],
                    num_samples=config["data"]["num_samples"],
                    return_all_outputs=config["model"]["return_all_outputs"],
                )

                val_samples = max(1000, config["data"]["num_samples"] // 10)
                val_ds, _ = dataset.OfflineModularCompositionDataset.from_group(
                    tpl,
                    config["data"]["k"],
                    group=group,
                    mode="sampled",
                    num_samples=val_samples,
                    return_all_outputs=config["model"]["return_all_outputs"],
                )
            else:
                raise ValueError(
                    f"group_name must be 'cn', 'cnxcn', 'dihedral', 'octahedral', or 'A5', got {group_name}"
                )

            X_train_t = train_ds.X.to(device)
            Y_train_t = train_ds.Y.to(device)
            X_val_t = val_ds.X.to(device)
            Y_val_t = val_ds.Y.to(device)

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

    if training_mode == "online":
        from src import train as train_mod

        train_loss_hist, val_loss_hist, param_hist, param_save_indices, final_step = (
            train_mod.train_online(
                rnn_2d,
                train_loader,
                criterion,
                opt,
                num_steps=num_steps,
                verbose_interval=config["training"]["verbose_interval"],
                grad_clip=config["training"]["grad_clip"],
                eval_dataloader=val_loader,
                save_param_interval=config["training"]["save_param_interval"],
                reduction_threshold=reduction_threshold,
            )
        )
    else:  # offline
        from src import train as train_mod

        train_loss_hist, val_loss_hist, param_hist, param_save_indices, final_step = (
            train_mod.train(
                rnn_2d,
                train_loader,
                criterion,
                opt,
                epochs=epochs,
                verbose_interval=config["training"]["verbose_interval"],
                grad_clip=config["training"]["grad_clip"],
                eval_dataloader=val_loader,
                save_param_interval=config["training"]["save_param_interval"],
                reduction_threshold=reduction_threshold,
                dense_save_until=config["training"].get("dense_save_until", 0),
            )
        )

    training_time = time.time() - start_time

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
        rnn_2d,
        train_loss_hist,
        val_loss_hist,
        param_hist,
        param_save_indices,
        tpl,
        training_time,
        device,
    )

    ### ----- PRODUCE ALL PLOTS ----- ###
    if group_name == "cnxcn":
        # Produce detailed plots for 2D
        produce_plots_cnxcn(
            run_dir=run_dir,
            config=config,
            model=rnn_2d,
            param_hist=param_hist,
            param_save_indices=param_save_indices,
            train_loss_hist=train_loss_hist,
            template_2d=template_2d,
            training_mode=training_mode,
            device=device,
        )
    elif group_name == "cn":
        # Produce detailed plots for 1D
        produce_plots_cn(
            run_dir=run_dir,
            config=config,
            model=rnn_2d,
            param_hist=param_hist,
            param_save_indices=param_save_indices,
            train_loss_hist=train_loss_hist,
            template_1d=template_1d,
            training_mode=training_mode,
            device=device,
        )
    elif group_name in ("dihedral", "octahedral", "A5"):
        produce_plots_group(
            run_dir=run_dir,
            config=config,
            model=rnn_2d,
            param_hist=param_hist,
            param_save_indices=param_save_indices,
            train_loss_hist=train_loss_hist,
            template=tpl,
            device=device,
            group=group,
        )
    else:
        raise ValueError(
            f"group_name must be 'cn', 'cnxcn', 'dihedral', 'octahedral', or 'A5', got {group_name}"
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
        if (cached / "power_data.npz").exists() and (cached / "train_loss_history.npy").exists():
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
        description="Train QuadraticRNN or SequentialMLP on group modular addition"
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
        help="Produce the combined 2x5 loss-and-power figure for C11, C5xC5, D5, Oh, A5",
    )

    args = parser.parse_args()

    if args.regenerate:
        regenerate_plots(args.regenerate, device=_auto_device())
    elif args.combined_plot:
        make_combined_plot()
    else:
        config = load_config(args.config)
        main(config)
