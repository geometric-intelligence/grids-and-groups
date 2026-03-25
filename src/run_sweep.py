#!/usr/bin/env python3
"""
Parameter sweep experiment runner for QuadraticRNN and SequentialMLP training.

Takes an experiment configuration file and runs all parameter combinations
with multiple seeds for uncertainty quantification.
"""

import argparse
import contextlib
import copy
import datetime
import multiprocessing
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def deep_merge_dict(base: dict, override: dict) -> dict:
    """Deep merge override dictionary into base dictionary."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_sweep_config(sweep_file: str) -> dict:
    """Load sweep configuration with base config and experiments."""
    if not os.path.exists(sweep_file):
        raise FileNotFoundError(f"Sweep file not found: {sweep_file}")

    with open(sweep_file) as f:
        sweep_config = yaml.safe_load(f)

    # Load base configuration
    base_config_path = sweep_config["base_config"]
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")

    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    sweep_config["_base_config"] = base_config
    return sweep_config


def expand_parameter_grid(parameter_grid: dict) -> list[dict]:
    """
    Expand a parameter grid specification into a list of parameter combinations.

    Args:
        parameter_grid: Nested dict with lists as leaf values.
                       Example: {'data': {'p': [5, 10], 'k': [2, 3]},
                                'model': {'hidden_dim': [64, 128]}}

    Returns:
        List of override dicts, one per combination.
        Example: [{'data': {'p': 5, 'k': 2}, 'model': {'hidden_dim': 64}}, ...]
    """

    def flatten_dict(d, parent_key=""):
        """Flatten nested dict to dot-notation keys."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def unflatten_dict(d):
        """Convert dot-notation keys back to nested dict."""
        result = {}
        for key, value in d.items():
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result

    # Flatten to get all parameter paths
    flat = flatten_dict(parameter_grid)

    # Get parameter names and their value lists
    param_names = list(flat.keys())
    param_values = [
        flat[name] if isinstance(flat[name], list) else [flat[name]] for name in param_names
    ]

    # Generate all combinations
    combinations = []
    for values in product(*param_values):
        combo = dict(zip(param_names, values))
        combinations.append(unflatten_dict(combo))

    return combinations


def generate_experiment_name(overrides: dict) -> str:
    """
    Generate a concise experiment name from parameter overrides.

    Example: {'data': {'p': 10, 'k': 2}, 'model': {'hidden_dim': 64}}
             -> "p10_k2_h64"
    """
    name_parts = []

    # Common parameter abbreviations
    abbrev = {
        "p": "p",
        "p1": "p1",
        "p2": "p2",
        "k": "k",
        "hidden_dim": "h",
        "n_freqs": "f",
        "learning_rate": "lr",
        "batch_size": "bs",
        "init_scale": "is",
    }

    def extract_params(d, prefix=""):
        """Recursively extract parameters."""
        for key, value in sorted(d.items()):
            if isinstance(value, dict):
                extract_params(value, prefix)
            else:
                param_name = abbrev.get(key, key)
                name_parts.append(f"{param_name}{value}")

    extract_params(overrides)
    return "_".join(name_parts)


def generate_experiment_configs(sweep_config: dict) -> list[tuple[str, dict]]:
    """
    Generate all individual experiment configurations from sweep.

    Supports two modes:
    1. Explicit experiments list (original behavior)
    2. Parameter grid (cartesian product of parameter lists)
    """
    base_config = sweep_config["_base_config"]
    global_overrides = sweep_config.get("global_overrides", {})

    # Check if using parameter grid or explicit experiments
    if "parameter_grid" in sweep_config:
        # Generate experiments from parameter grid
        print("Generating experiments from parameter grid...")
        param_combinations = expand_parameter_grid(sweep_config["parameter_grid"])

        experiment_configs = []
        for overrides in param_combinations:
            exp_name = generate_experiment_name(overrides)

            # Merge: base -> global_overrides -> specific overrides
            config = deep_merge_dict(base_config, global_overrides)
            merged_config = deep_merge_dict(config, overrides)
            experiment_configs.append((exp_name, merged_config))

        print(f"Generated {len(experiment_configs)} experiments from parameter grid")

    elif "experiments" in sweep_config:
        # Use explicit experiments list (original behavior)
        experiments = sweep_config["experiments"]

        experiment_configs = []
        for exp in experiments:
            exp_name = exp["name"]
            overrides = exp.get("overrides", {})

            # Merge: base -> global_overrides -> specific overrides
            config = deep_merge_dict(base_config, global_overrides)
            merged_config = deep_merge_dict(config, overrides)
            experiment_configs.append((exp_name, merged_config))

    else:
        raise ValueError("Sweep config must contain either 'experiments' or 'parameter_grid'")

    # Optionally rescale init_scale based on k: init_scale_k = base^(3/(k+1))
    if sweep_config.get("scale_init_with_k", False):
        for _, cfg in experiment_configs:
            k = cfg["data"]["k"]
            base_scale = cfg["model"]["init_scale"]
            adjusted = base_scale ** (3.0 / (k + 1))
            cfg["model"]["init_scale"] = adjusted
        print("Adjusted init_scale per k: init_scale_k = base^(3/(k+1))")

    # Validate: Check for duplicate experiment names
    exp_names = [name for name, _ in experiment_configs]
    if len(exp_names) != len(set(exp_names)):
        duplicates = [name for name in set(exp_names) if exp_names.count(name) > 1]
        raise ValueError(
            f"Duplicate experiment names found: {duplicates}. "
            "Each experiment must have a unique name. "
            "If using parameter_grid, ensure parameter combinations produce unique names."
        )

    return experiment_configs


def save_sweep_metadata(
    sweep_dir: Path, sweep_config: dict, experiment_configs: list[tuple[str, dict]]
) -> None:
    """Save sweep metadata and configurations."""

    # Save sweep metadata
    sweep_metadata = {
        "sweep_name": sweep_dir.name,
        "base_config_file": sweep_config["base_config"],
        "n_seeds": sweep_config["n_seeds"],
        "n_experiments": len(experiment_configs),
        "total_runs": len(experiment_configs) * sweep_config["n_seeds"],
        "created_at": datetime.datetime.now().isoformat(),
        "experiments": [name for name, _ in experiment_configs],
    }

    metadata_path = sweep_dir / "sweep_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(sweep_metadata, f, default_flow_style=False, indent=2)

    # Save individual experiment configs
    configs_dir = sweep_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    for exp_name, config in experiment_configs:
        config_path = configs_dir / f"{exp_name}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Sweep metadata saved to: {metadata_path}")


def run_single_seed(
    exp_name: str, config: dict, seed: int, sweep_dir: Path, gpu_id: int | None = None
) -> dict[str, Any]:
    """Run a single experiment seed (for multiprocessing).

    Args:
        exp_name: Name of the experiment
        config: Configuration dictionary
        seed: Random seed to use
        sweep_dir: Directory to save sweep results
        gpu_id: Optional GPU ID to use (overrides config device)

    Returns:
        Dictionary with run results
    """
    # Add project root to Python path for subprocess imports
    # This is needed because ProcessPoolExecutor spawns new processes that don't inherit PYTHONPATH
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Create seed config
    seed_config = copy.deepcopy(config)
    seed_config["data"]["seed"] = seed

    # Override device if GPU ID specified
    if gpu_id is not None:
        seed_config["device"] = f"cuda:{gpu_id}"

    # Create seed-specific run directory
    exp_dir = sweep_dir / exp_name
    seed_dir = exp_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Import here to avoid circular dependency
        import src.main as main

        # Check if quiet mode is enabled (suppress stdout but keep stderr for errors)
        quiet = seed_config.get("training", {}).get("quiet", False)

        # Run training (with optional stdout suppression)
        if quiet:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    result = main.train_single_run(seed_config, run_dir=seed_dir)
        else:
            result = main.train_single_run(seed_config, run_dir=seed_dir)

        # Save run summary
        run_summary = {
            "experiment_name": exp_name,
            "seed": seed,
            "status": "completed",
            "seed_dir": str(seed_dir),
            "final_train_loss": result.get("final_train_loss", None),
            "final_val_loss": result.get("final_val_loss", None),
            "training_time": result.get("training_time", None),
            "completed_at": datetime.datetime.now().isoformat(),
        }

        summary_path = seed_dir / "run_summary.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(run_summary, f, default_flow_style=False, indent=2)

        if not quiet:
            print(f"✓ {exp_name} seed {seed} completed successfully (GPU: {gpu_id})")
            if run_summary["final_train_loss"] is not None:
                print(f"  Train loss: {run_summary['final_train_loss']:.6f}")
            if run_summary["final_val_loss"] is not None:
                print(f"  Val loss: {run_summary['final_val_loss']:.6f}")

        return run_summary

    except Exception as e:
        print(f"✗ {exp_name} seed {seed} failed with error: {str(e)} (GPU: {gpu_id})")
        traceback.print_exc()

        error_summary = {
            "experiment_name": exp_name,
            "seed": seed,
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.datetime.now().isoformat(),
        }

        error_path = seed_dir / "error_summary.yaml"
        with open(error_path, "w") as f:
            yaml.dump(error_summary, f, default_flow_style=False, indent=2)

        return error_summary


def run_experiment(
    exp_name: str, config: dict, seeds: list[int], sweep_dir: Path, gpu_id: int = None
) -> list[dict[str, Any]]:
    """Run a single experiment configuration with multiple seeds.

    This function is kept for backward compatibility. For multi-GPU sweeps,
    use run_parameter_sweep with gpu_ids parameter instead.

    Args:
        exp_name: Name of the experiment
        config: Configuration dictionary
        seeds: List of random seeds to run
        sweep_dir: Directory to save sweep results
        gpu_id: Optional GPU ID to use (overrides config device)
    """

    print(f"\n{'=' * 80}")
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"Seeds: {seeds}")
    if gpu_id is not None:
        print(f"GPU: cuda:{gpu_id}")
    print(f"{'=' * 80}")

    # Create experiment directory
    exp_dir = sweep_dir / exp_name
    exp_dir.mkdir(exist_ok=True)

    # Run each seed sequentially
    run_results = []
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'-' * 60}")
        print(f"EXPERIMENT {exp_name} - SEED {seed_idx + 1}/{len(seeds)}: seed={seed}")
        print(f"{'-' * 60}")

        result = run_single_seed(exp_name, config, seed, sweep_dir, gpu_id)
        run_results.append(result)

    # Generate experiment summary
    successful_runs = [r for r in run_results if r["status"] == "completed"]
    train_losses = [
        r["final_train_loss"] for r in successful_runs if r.get("final_train_loss") is not None
    ]
    val_losses = [
        r["final_val_loss"] for r in successful_runs if r.get("final_val_loss") is not None
    ]

    exp_summary = {
        "experiment_name": exp_name,
        "experiment_completed_at": datetime.datetime.now().isoformat(),
        "total_seeds": len(run_results),
        "successful_runs": len(successful_runs),
        "failed_runs": len(run_results) - len(successful_runs),
        "success_rate": len(successful_runs) / len(run_results) if run_results else 0,
    }

    if train_losses:
        exp_summary["train_loss_stats"] = {
            "mean": float(np.mean(train_losses)),
            "std": float(np.std(train_losses)),
            "min": float(np.min(train_losses)),
            "max": float(np.max(train_losses)),
            "median": float(np.median(train_losses)),
        }

    if val_losses:
        exp_summary["val_loss_stats"] = {
            "mean": float(np.mean(val_losses)),
            "std": float(np.std(val_losses)),
            "min": float(np.min(val_losses)),
            "max": float(np.max(val_losses)),
            "median": float(np.median(val_losses)),
        }

    exp_summary["run_details"] = run_results

    # Save experiment summary
    summary_path = exp_dir / "experiment_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(exp_summary, f, default_flow_style=False, indent=2)

    print(f"\nExperiment {exp_name} complete!")
    print(f"Successful runs: {len(successful_runs)}/{len(run_results)}")
    if train_losses:
        print(
            f"Train loss: {exp_summary['train_loss_stats']['mean']:.6f} ± {exp_summary['train_loss_stats']['std']:.6f}"
        )
    if val_losses:
        print(
            f"Val loss: {exp_summary['val_loss_stats']['mean']:.6f} ± {exp_summary['val_loss_stats']['std']:.6f}"
        )

    return run_results


def generate_sweep_summary(sweep_dir: Path, all_results: dict[str, list[dict[str, Any]]]) -> None:
    """Generate overall sweep summary."""

    # Aggregate statistics across all experiments
    total_runs = sum(len(results) for results in all_results.values())
    total_successful = sum(
        len([r for r in results if r["status"] == "completed"]) for results in all_results.values()
    )

    # Per-experiment statistics
    experiment_stats = {}
    for exp_name, results in all_results.items():
        successful = [r for r in results if r["status"] == "completed"]
        train_losses = [
            r["final_train_loss"] for r in successful if r.get("final_train_loss") is not None
        ]
        val_losses = [
            r["final_val_loss"] for r in successful if r.get("final_val_loss") is not None
        ]

        stats = {
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(results) - len(successful),
            "success_rate": len(successful) / len(results) if results else 0,
        }

        if train_losses:
            stats["train_loss_stats"] = {
                "mean": float(np.mean(train_losses)),
                "std": float(np.std(train_losses)),
                "min": float(np.min(train_losses)),
                "max": float(np.max(train_losses)),
                "median": float(np.median(train_losses)),
            }

        if val_losses:
            stats["val_loss_stats"] = {
                "mean": float(np.mean(val_losses)),
                "std": float(np.std(val_losses)),
                "min": float(np.min(val_losses)),
                "max": float(np.max(val_losses)),
                "median": float(np.median(val_losses)),
            }

        experiment_stats[exp_name] = stats

    # Overall sweep summary
    sweep_summary = {
        "sweep_completed_at": datetime.datetime.now().isoformat(),
        "total_experiments": len(all_results),
        "total_runs": total_runs,
        "total_successful_runs": total_successful,
        "total_failed_runs": total_runs - total_successful,
        "overall_success_rate": total_successful / total_runs if total_runs else 0,
        "experiment_statistics": experiment_stats,
    }

    # Save sweep summary
    summary_path = sweep_dir / "sweep_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(sweep_summary, f, default_flow_style=False, indent=2)

    print(f"\n{'=' * 80}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {total_successful}/{total_runs}")
    print(f"Overall success rate: {sweep_summary['overall_success_rate']:.2%}")
    print(f"Results saved to: {sweep_dir}")
    print(f"Summary: {summary_path}")

    # Print per-experiment summary
    print("\nPer-experiment results:")
    for exp_name, stats in experiment_stats.items():
        print(
            f"  {exp_name}: {stats['successful_runs']}/{stats['total_runs']} successful",
            end="",
        )
        if "val_loss_stats" in stats:
            print(
                f" (val_loss: {stats['val_loss_stats']['mean']:.6f} ± {stats['val_loss_stats']['std']:.6f})"
            )
        else:
            print()


def run_parameter_sweep(
    sweep_file: str, gpu_ids: list[int] | None = None, gpu_id: int | None = None
):
    """Run full parameter sweep experiment.

    Args:
        sweep_file: Path to sweep configuration file
        gpu_ids: List of GPU IDs to use for parallel execution (e.g., [0,1,2,3,4,5,6,7]).
                 If None, falls back to gpu_id or config default.
        gpu_id: Single GPU ID for backward compatibility (deprecated, use gpu_ids instead)
    """
    print(f"Loading parameter sweep configuration: {sweep_file}")

    # Load sweep configuration
    sweep_config = load_sweep_config(sweep_file)
    n_seeds = sweep_config["n_seeds"]
    experiment_configs = generate_experiment_configs(sweep_config)

    # Determine GPU configuration
    if gpu_ids is None:
        if gpu_id is not None:
            # Backward compatibility: single GPU
            gpu_ids = [gpu_id]
            use_parallel = False
        else:
            # Auto-detect all available GPUs
            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
                use_parallel = len(gpu_ids) > 1
            else:
                gpu_ids = [None]  # CPU fallback
                use_parallel = False
    else:
        use_parallel = len(gpu_ids) > 1

    print("Parameter sweep configuration:")
    print(f"  Base config: {sweep_config['base_config']}")
    print(f"  Number of experiments: {len(experiment_configs)}")
    print(f"  Seeds per experiment: {n_seeds}")
    print(f"  Total runs: {len(experiment_configs) * n_seeds}")
    print(f"  Available GPUs: {gpu_ids}")
    print(f"  Parallel execution: {use_parallel}")
    print(f"  Experiments: {[name for name, _ in experiment_configs]}")

    # Create sweep directory
    sweep_name = os.path.splitext(os.path.basename(sweep_file))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path("sweep_results").resolve() / f"{sweep_name}_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSweep directory: {sweep_dir}")

    # Save sweep metadata
    save_sweep_metadata(sweep_dir, sweep_config, experiment_configs)

    # Generate seeds
    seeds = list(range(n_seeds))

    if use_parallel:
        # Parallel execution: distribute tasks across GPUs
        print(f"\n{'=' * 80}")
        print(
            f"PARALLEL EXECUTION MODE: Distributing {len(experiment_configs) * n_seeds} tasks across {len(gpu_ids)} GPUs"
        )
        print(f"{'=' * 80}")

        # Flatten experiments with seeds into individual tasks
        tasks = []
        for exp_name, config in experiment_configs:
            for seed in seeds:
                tasks.append((exp_name, config, seed, sweep_dir))

        # Distribute tasks across GPUs using round-robin
        all_results = {}
        with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
            # Submit all tasks
            future_to_task = {}
            for task_idx, task in enumerate(tasks):
                exp_name, config, seed, _ = task
                gpu_id = gpu_ids[task_idx % len(gpu_ids)]  # Round-robin assignment
                future = executor.submit(run_single_seed, exp_name, config, seed, sweep_dir, gpu_id)
                future_to_task[future] = (exp_name, seed, gpu_id)

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                exp_name, seed, assigned_gpu = future_to_task[future]
                completed += 1
                try:
                    result = future.result()
                    if exp_name not in all_results:
                        all_results[exp_name] = []
                    all_results[exp_name].append(result)
                    print(
                        f"[{completed}/{len(tasks)}] Completed: {exp_name} seed {seed} (GPU {assigned_gpu})"
                    )
                except Exception as e:
                    # This catch handles exceptions during result retrieval/serialization
                    # (run_single_seed already catches training exceptions and returns error dict)
                    print(
                        f"[{completed}/{len(tasks)}] Failed: {exp_name} seed {seed} (GPU {assigned_gpu}): {e}"
                    )
                    if exp_name not in all_results:
                        all_results[exp_name] = []
                    all_results[exp_name].append(
                        {
                            "experiment_name": exp_name,
                            "seed": seed,
                            "status": "failed",
                            "error": str(e),
                        }
                    )

        # Verify all tasks completed
        total_expected = len(experiment_configs) * len(seeds)
        total_completed = sum(len(results) for results in all_results.values())
        if total_completed != total_expected:
            print(
                f"\n⚠️  WARNING: Expected {total_expected} runs, but collected {total_completed} results"
            )
            print("This may indicate some tasks failed silently or were not properly tracked.")

        # Generate experiment summaries for each experiment
        for exp_name in all_results.keys():
            exp_dir = sweep_dir / exp_name
            run_results = all_results[exp_name]

            # Sort results by seed for consistent ordering
            run_results.sort(key=lambda x: x.get("seed", 0))

            successful_runs = [r for r in run_results if r["status"] == "completed"]
            train_losses = [
                r["final_train_loss"]
                for r in successful_runs
                if r.get("final_train_loss") is not None
            ]
            val_losses = [
                r["final_val_loss"] for r in successful_runs if r.get("final_val_loss") is not None
            ]

            exp_summary = {
                "experiment_name": exp_name,
                "experiment_completed_at": datetime.datetime.now().isoformat(),
                "total_seeds": len(run_results),
                "successful_runs": len(successful_runs),
                "failed_runs": len(run_results) - len(successful_runs),
                "success_rate": len(successful_runs) / len(run_results) if run_results else 0,
            }

            if train_losses:
                exp_summary["train_loss_stats"] = {
                    "mean": float(np.mean(train_losses)),
                    "std": float(np.std(train_losses)),
                    "min": float(np.min(train_losses)),
                    "max": float(np.max(train_losses)),
                    "median": float(np.median(train_losses)),
                }

            if val_losses:
                exp_summary["val_loss_stats"] = {
                    "mean": float(np.mean(val_losses)),
                    "std": float(np.std(val_losses)),
                    "min": float(np.min(val_losses)),
                    "max": float(np.max(val_losses)),
                    "median": float(np.median(val_losses)),
                }

            exp_summary["run_details"] = run_results

            # Save experiment summary
            summary_path = exp_dir / "experiment_summary.yaml"
            with open(summary_path, "w") as f:
                yaml.dump(exp_summary, f, default_flow_style=False, indent=2)
    else:
        # Sequential execution: run experiments one at a time (original behavior)
        print(f"\n{'=' * 80}")
        print(
            f"SEQUENTIAL EXECUTION MODE: Running on GPU {gpu_ids[0] if gpu_ids[0] is not None else 'CPU'}"
        )
        print(f"{'=' * 80}")

        all_results = {}
        for exp_name, config in experiment_configs:
            results = run_experiment(exp_name, config, seeds, sweep_dir, gpu_id=gpu_ids[0])
            all_results[exp_name] = results

    # Generate sweep summary
    generate_sweep_summary(sweep_dir, all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep experiment for QuadraticRNN or SequentialMLP"
    )
    parser.add_argument("--sweep", type=str, required=True, help="Path to sweep configuration file")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Single GPU ID to use (e.g., 0 or 1). Overrides device in config. Deprecated: use --gpus instead.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1,2,3,4,5,6,7') or 'auto' to use all available GPUs. "
        "If not specified, uses --gpu or config default. Parallel execution enabled when multiple GPUs specified.",
    )
    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = None
    if args.gpus:
        if args.gpus.lower() == "auto":
            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
                print(f"Auto-detected {len(gpu_ids)} GPUs: {gpu_ids}")
            else:
                print("No CUDA devices available, falling back to CPU")
                gpu_ids = [None]
        else:
            gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
            print(f"Using specified GPUs: {gpu_ids}")
    elif args.gpu is not None:
        # Backward compatibility: single GPU
        gpu_ids = [args.gpu]
        print(f"Using single GPU: {gpu_ids[0]}")

    run_parameter_sweep(args.sweep, gpu_ids=gpu_ids, gpu_id=args.gpu)


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This is required because CUDA cannot be re-initialized in forked subprocesses
    # Must be set before any ProcessPoolExecutor is created
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set, which is fine
        pass

    main()
