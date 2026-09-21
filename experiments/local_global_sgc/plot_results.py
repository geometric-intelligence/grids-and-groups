"""Plot the four held-out loss curves from completed or running pilot jobs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_metrics(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no metrics in {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for metrics_path in sorted(args.run_root.glob("*_seed*/metrics.csv")):
        run_name = metrics_path.parent.name
        distribution, seed_text = run_name.rsplit("_seed", 1)
        records[(distribution, int(seed_text))] = read_metrics(metrics_path)
    if not records:
        raise ValueError(f"no run metrics found under {args.run_root}")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = {"local": "#E45756", "global": "#356AE6"}
    for axis, eval_distribution in zip(axes, ("local", "global")):
        for train_distribution in ("local", "global"):
            matching = [
                metrics
                for (distribution, _), metrics in records.items()
                if distribution == train_distribution
            ]
            for index, metrics in enumerate(matching):
                axis.plot(
                    metrics["step"],
                    metrics[f"{eval_distribution}_mse"],
                    color=colors[train_distribution],
                    alpha=0.35,
                    label=(f"train {train_distribution}" if index == 0 else None),
                )
        axis.set_yscale("log")
        axis.set_xlabel("optimizer step")
        axis.set_title(f"Evaluate on {eval_distribution} trajectories")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("held-out MSE")
    axes[0].legend(frameon=False)
    figure.tight_layout()
    output = args.output or args.run_root / "loss_matrix.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
