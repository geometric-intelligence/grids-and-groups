"""
Sweep result loading and visualization utilities.

Used by notebooks to analyze parameter sweep experiments.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_EXP_DIR_RE = re.compile(r"^k(\d+)_p(\d+)_h(\d+)$")


def load_sweep_config(config_path: str) -> tuple[list, list, list]:
    """Read p_values, hidden_dims, and k_values from a sweep YAML config.

    Returns:
        (p_values, hidden_dims, k_values)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    grid = cfg["parameter_grid"]
    return grid["data"]["p"], grid["model"]["hidden_dim"], grid["data"]["k"]


def discover_sweep_params(sweep_dir: str) -> tuple[list[int], list[int], list[int]]:
    """Scan a sweep results directory and return the (p, hidden_dim, k) values present.

    Parses experiment directory names of the form ``k{k}_p{p}_h{h}``.

    Returns:
        (p_values, hidden_dims, k_values) — each sorted ascending.
    """
    ks, ps, hs = set(), set(), set()
    for d in Path(sweep_dir).iterdir():
        if not d.is_dir():
            continue
        m = _EXP_DIR_RE.match(d.name)
        if m:
            ks.add(int(m.group(1)))
            ps.add(int(m.group(2)))
            hs.add(int(m.group(3)))
    return sorted(ps), sorted(hs), sorted(ks)


def _iter_experiment_seeds(sweep_dir: Path, k: int, p: int, h: int):
    """Yield (seed_dir, loss_history) for each completed seed of an experiment."""
    exp_dir = sweep_dir / f"k{k}_p{p}_h{h}"
    if not exp_dir.exists():
        return

    seed_0 = exp_dir / "seed_0"
    if not seed_0.exists() or not (seed_0 / "run_summary.yaml").exists():
        return

    for seed_dir in exp_dir.glob("seed_*"):
        loss_file = seed_dir / "train_loss_history.npy"
        if loss_file.exists():
            loss_history = np.load(loss_file)
            if len(loss_history) > 0:
                yield seed_dir, loss_history


def load_sweep_results_grid(
    sweep_dir: str,
    k: int,
    p_values: list,
    hidden_dims: list,
    metric: str = "final_loss",
    max_p: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load sweep results into a (hidden_dims × p_values) grid.

    Args:
        sweep_dir: Path to the sweep directory.
        k: Sequence length parameter.
        p_values: List of group size values.
        hidden_dims: List of hidden dimension values.
        metric: One of ``"final_loss"`` or ``"initial_loss"``.
        max_p: If set, skip experiments with p > max_p.

    Returns:
        (grid, std_grid) — 2D arrays of shape (len(hidden_dims), len(p_values)).
        Cells without data are NaN.
    """
    sweep_path = Path(sweep_dir)
    grid = np.full((len(hidden_dims), len(p_values)), np.nan)
    std_grid = np.full((len(hidden_dims), len(p_values)), np.nan)

    for i, h in enumerate(hidden_dims):
        for j, p in enumerate(p_values):
            if max_p is not None and p > max_p:
                continue

            values = []
            for _, loss_history in _iter_experiment_seeds(sweep_path, k, p, h):
                if metric == "final_loss":
                    values.append(loss_history[-1])
                elif metric == "initial_loss":
                    if loss_history[0] > 0:
                        values.append(loss_history[0])
                else:
                    raise ValueError(f"Unknown metric: {metric!r}")

            if values:
                grid[i, j] = np.mean(values)
                std_grid[i, j] = np.std(values) if len(values) > 1 else 0.0

    return grid, std_grid


def load_training_loss_curves(
    sweep_dir: str,
    k: int,
    hidden_dim: int,
    p_values: list,
) -> dict[int, list[np.ndarray]]:
    """Load training loss histories for different group sizes.

    Args:
        sweep_dir: Path to the sweep directory.
        k: Sequence length parameter (fixed).
        hidden_dim: Hidden dimension (fixed).
        p_values: Group sizes to load.

    Returns:
        Dictionary mapping p -> list of loss history arrays (one per seed).
    """
    sweep_path = Path(sweep_dir)
    curves: dict[int, list[np.ndarray]] = {}

    for p in p_values:
        histories = [h for _, h in _iter_experiment_seeds(sweep_path, k, p, hidden_dim)]
        if histories:
            curves[p] = histories

    return curves


def export_lightweight_data(
    sweep_dir: str,
    output_path: str,
    p_values: list[int],
    hidden_dims: list[int],
    k_values: list[int],
    *,
    downsample: int = 1000,
) -> None:
    """Pre-compute grids and downsampled curves, saving to a compressed ``.npz``.

    Args:
        sweep_dir: Path to the full sweep results directory.
        output_path: Where to write the ``.npz`` file.
        p_values, hidden_dims, k_values: Sweep axes.
        downsample: Number of evenly-spaced points to keep per loss curve.
    """
    out: dict[str, np.ndarray] = {
        "p_values": np.array(p_values),
        "hidden_dims": np.array(hidden_dims),
        "k_values": np.array(k_values),
    }

    for k in k_values:
        for metric in ("final_loss", "initial_loss"):
            grid, _ = load_sweep_results_grid(
                sweep_dir, k, p_values, hidden_dims, metric=metric
            )
            out[f"{metric}_k{k}"] = grid.astype(np.float32)

        for h in hidden_dims:
            curves = load_training_loss_curves(sweep_dir, k, h, p_values)
            mat = np.full((len(p_values), downsample), np.nan, dtype=np.float32)
            for j, p in enumerate(p_values):
                if p not in curves or not curves[p]:
                    continue
                raw = curves[p][0]
                indices = np.linspace(0, len(raw) - 1, downsample).astype(int)
                mat[j] = raw[indices].astype(np.float32)
            out[f"curves_k{k}_h{h}"] = mat

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Saved lightweight data to {output_path} ({size_mb:.1f} MB)")


def load_lightweight_data(path: str) -> dict[str, np.ndarray]:
    """Load pre-computed lightweight sweep data from a ``.npz`` file.

    Returns a dict-like object (``NpzFile``) with the same keys produced by
    ``export_lightweight_data``.
    """
    return dict(np.load(path))


def plot_theory_boundaries(
    ax_or_plt,
    k: int,
    p_values: list,
    hidden_dims: list,
    *,
    use_ax: bool = False,
    show_minor: bool = True,
):
    """Plot theory boundaries H = m * 2^{k-1} * |G| for m = 1, …, k+1.

    Args:
        ax_or_plt: A matplotlib Axes (if use_ax=True) or the plt module.
        k: Sequence length parameter.
        p_values: Group size values (x-axis).
        hidden_dims: Hidden dimension values (y-axis).
        use_ax: If True, call ax.step(); otherwise plt.step().
        show_minor: If False, only the top boundary (m=k+1) is drawn.
    """
    x_step = np.arange(len(p_values) + 1) - 0.5
    step_func = ax_or_plt.step if use_ax else plt.step
    h_arr = np.array(hidden_dims)

    for m in range(1, k + 2):
        if m < k + 1 and not show_minor:
            continue

        coeff = m * (2 ** (k - 1))
        y_step = [
            int(np.argmax(h_arr >= coeff * p)) if coeff * p <= h_arr[-1] else len(hidden_dims)
            for p in p_values
        ]
        y_step.append(y_step[-1])
        y_step = [y - 0.5 for y in y_step]

        if m == k + 1:
            style = dict(color="black", linewidth=3, linestyle="-")
            label = f"$m={m}$: $H$ ≥ ${m} \\cdot 2^{{k-1}} |G|$"
        else:
            style = dict(color="black", linewidth=2, linestyle="--")
            label = (
                f"$m=1$: $H$ ≥ $2^{{k-1}} |G|$"
                if m == 1
                else f"$m={m}$: $H$ ≥ ${m} \\cdot 2^{{k-1}} |G|$"
            )

        step_func(x_step, y_step, where="post", label=label, **style)
