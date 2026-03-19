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
import src.fourier as fourier
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


def produce_plots_2d(
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
    X_seq_2d, Y_seq_2d, _ = dataset.OnlineModularAdditionDataset2D.generate_dataset(
        config["data"]["p1"],
        config["data"]["p2"],
        template_2d,
        config["data"]["k"],
        mode="sampled",
        num_samples=min(config["data"]["num_samples"], 1000),
        return_all_outputs=config["model"]["return_all_outputs"],
    )
    X_seq_2d_t = torch.tensor(X_seq_2d, dtype=torch.float32, device=device)
    Y_seq_2d_t = torch.tensor(Y_seq_2d, dtype=torch.float32, device=device)
    print(f"  Generated {X_seq_2d_t.shape[0]} samples for visualization")

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

    ### ----- PLOT MODEL PREDICTIONS ----- ###
    if plot_predictions:
        print("Plotting model predictions over time...")
        viz.plot_predictions_2d(
            model,
            param_hist,
            X_seq_2d_t,
            Y_seq_2d_t,
            config["data"]["p1"],
            config["data"]["p2"],
            steps=checkpoint_indices,
            save_path=os.path.join(run_dir, "predictions_over_time.pdf"),
            show=False,
        )

    ### ----- PLOT W_MIX FREQUENCY STRUCTURE (QuadraticRNN only) ----- ###
    model_type = config["model"]["model_type"]
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


def produce_plots_1d(
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

    ### ----- GENERATE EVALUATION DATA ----- ###
    print("Generating evaluation data for visualization...")
    X_seq_1d, Y_seq_1d, _ = dataset.OnlineModularAdditionDataset1D.generate_dataset(
        config["data"]["p"],
        template_1d,
        config["data"]["k"],
        mode="sampled",
        num_samples=min(config["data"]["num_samples"], 1000),
        return_all_outputs=config["model"]["return_all_outputs"],
    )
    X_seq_1d_t = torch.tensor(X_seq_1d, dtype=torch.float32, device=device)
    Y_seq_1d_t = torch.tensor(Y_seq_1d, dtype=torch.float32, device=device)
    print(f"  Generated {X_seq_1d_t.shape[0]} samples for visualization")

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

        # Create a 2x2 subplot for different scale combinations
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "training_loss.pdf"), bbox_inches="tight", dpi=150)
        plt.close()
        print("  ✓ Saved training loss plot (all scales)")

    ### ----- PLOT MODEL PREDICTIONS ----- ###
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
    if plot_power_spectrum:
        print("Analyzing power spectrum of predictions over training...")
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

    print("\n✓ All 1D plots generated successfully!")


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
        X_raw, Y_raw = dataset.group_dataset(group, template)
        X_eval_t, Y_eval_t, device = dataset.move_dataset_to_device_and_flatten(
            X_raw, Y_raw, device=device
        )
        # Optionally subsample for visualization
        n_eval = min(len(X_eval_t), 1000)
        if n_eval < len(X_eval_t):
            indices = np.random.choice(len(X_eval_t), size=n_eval, replace=False)
            X_eval_t = X_eval_t[indices]
            Y_eval_t = Y_eval_t[indices]
    else:
        # Sequence models use the generic sequence dataset
        X_eval, Y_eval, _ = dataset.build_modular_addition_sequence_dataset_generic(
            template,
            k,
            group=group,
            mode="sampled",
            num_samples=min(config["data"]["num_samples"], 1000),
            return_all_outputs=config["model"]["return_all_outputs"],
        )
        X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
        Y_eval_t = torch.tensor(Y_eval, dtype=torch.float32, device=device)

    print(f"  Generated {X_eval_t.shape[0]} samples for visualization")

    ### ----- COMPUTE CHECKPOINT INDICES ----- ###
    total_checkpoints = len(param_hist)
    checkpoint_fractions = config["analysis"]["checkpoints"]
    checkpoint_indices = [int(f * (total_checkpoints - 1)) for f in checkpoint_fractions]
    print(f"Analysis checkpoints: {checkpoint_indices} (out of {total_checkpoints})")

    ### ----- PLOT TRAINING LOSS ----- ###
    if plot_training_loss:
        print("\nPlotting training loss...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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

        plt.suptitle(f"{group_label} Composition (k={k})", fontsize=14)
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
    if plot_power_spectrum:
        print("\nPlotting power spectrum over time...")
        optimizer_name = config["training"]["optimizer"]
        init_scale = config["model"]["init_scale"]
        viz.plot_power_group(
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
        )
        print(f"  ✓ Saved {os.path.join(run_dir, 'power_spectrum_analysis.pdf')}")

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
        elif template_type == "fourier":
            n_freqs = config["data"]["n_freqs"]
            template_1d = template.fourier_1d(p, n_freqs=n_freqs, seed=config["data"]["seed"])
        elif template_type == "gaussian":
            template_1d = template.gaussian_1d(p, n_gaussians=3, seed=config["data"]["seed"])
        elif template_type == "onehot":
            template_1d = template.onehot_1d(p)
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
        elif template_type == "fourier":
            n_freqs = config["data"]["n_freqs"]
            template_2d = template.unique_freqs_2d(
                p1, p2, n_freqs=n_freqs, seed=config["data"]["seed"]
            )
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
            irreps = group.irreps()
            irrep_dims = [ir.size for ir in irreps]

            assert len(powers) == len(irreps), (
                f"powers must have {len(irreps)} values (one per irrep), got {len(powers)}"
            )

            fourier_coef_diag_values = [
                np.sqrt(group_order * p / dim**2) if p > 0 else 0.0
                for p, dim in zip(powers, irrep_dims)
            ]

            print("Template type: custom_fourier")
            print(f"Desired powers (per irrep): {powers}")
            print(f"Fourier coef diagonal values: {fourier_coef_diag_values}")

            spectrum = []
            for i, irrep in enumerate(irreps):
                diag_val = fourier_coef_diag_values[i]
                diag_values = np.full(irrep.size, diag_val, dtype=float)
                mat = np.zeros((irrep.size, irrep.size), dtype=float)
                np.fill_diagonal(mat, diag_values)
                print(
                    f"  Irrep {i} (dim={irrep.size}): diag_value = {diag_val:.4f} -> power = {powers[i]}"
                )
                spectrum.append(mat)

            tpl = fourier.group_fourier_inverse(group, spectrum)
            tpl = tpl - np.mean(tpl)
            tpl = tpl.astype(np.float32)
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
            # TwoLayerNet uses binary pair datasets from src/datamodule.py
            # Data shape: X=(N, 2, group_size) -> flattened to (N, 2*group_size), Y=(N, group_size)
            if group_name == "cn":
                X_raw, Y_raw = dataset.cn_dataset(tpl)
            elif group_name == "cnxcn":
                X_raw, Y_raw = dataset.cnxcn_dataset(tpl)
            elif group_name in ("dihedral", "octahedral", "A5"):
                X_raw, Y_raw = dataset.group_dataset(group, tpl)
            else:
                raise ValueError(f"Unsupported group_name for TwoLayerNet: {group_name}")

            # Flatten X from (N, 2, group_size) to (N, 2*group_size) and convert to tensors
            X_all, Y_all, device = dataset.move_dataset_to_device_and_flatten(
                X_raw, Y_raw, device=device
            )

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
                # Generate training dataset
                X_train, Y_train, _ = dataset.OnlineModularAdditionDataset1D.generate_dataset(
                    config["data"]["p"],
                    template_1d,
                    config["data"]["k"],
                    mode=config["data"]["mode"],
                    num_samples=config["data"]["num_samples"],
                    return_all_outputs=config["model"]["return_all_outputs"],
                )

                # Generate validation dataset
                val_samples = max(1000, config["data"]["num_samples"] // 10)
                X_val, Y_val, _ = dataset.OnlineModularAdditionDataset1D.generate_dataset(
                    config["data"]["p"],
                    template_1d,
                    config["data"]["k"],
                    mode="sampled",
                    num_samples=val_samples,
                    return_all_outputs=config["model"]["return_all_outputs"],
                )
            elif group_name == "cnxcn":
                # Generate training dataset
                X_train, Y_train, _ = dataset.OnlineModularAdditionDataset2D.generate_dataset(
                    config["data"]["p1"],
                    config["data"]["p2"],
                    template_2d,
                    config["data"]["k"],
                    mode=config["data"]["mode"],
                    num_samples=config["data"]["num_samples"],
                    return_all_outputs=config["model"]["return_all_outputs"],
                )

                # Generate validation dataset
                val_samples = max(1000, config["data"]["num_samples"] // 10)
                X_val, Y_val, _ = dataset.OnlineModularAdditionDataset2D.generate_dataset(
                    config["data"]["p1"],
                    config["data"]["p2"],
                    template_2d,
                    config["data"]["k"],
                    mode="sampled",
                    num_samples=val_samples,
                    return_all_outputs=config["model"]["return_all_outputs"],
                )
            elif group_name in ("dihedral", "octahedral", "A5"):
                X_train, Y_train, _ = dataset.build_modular_addition_sequence_dataset_generic(
                    tpl,
                    config["data"]["k"],
                    group=group,
                    mode=config["data"]["mode"],
                    num_samples=config["data"]["num_samples"],
                    return_all_outputs=config["model"]["return_all_outputs"],
                )

                val_samples = max(1000, config["data"]["num_samples"] // 10)
                X_val, Y_val, _ = dataset.build_modular_addition_sequence_dataset_generic(
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

            X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
            Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
            Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device)

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
        tpl,
        training_time,
        device,
    )

    ### ----- PRODUCE ALL PLOTS ----- ###
    if group_name == "cnxcn":
        # Produce detailed plots for 2D
        produce_plots_2d(
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
        produce_plots_1d(
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

    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
