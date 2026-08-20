"""Plotting helpers for square-torus geometry."""

import matplotlib.pyplot as plt
import numpy as np

TRACK_COLOR = "#E45756"


def plot_grid_scalar(
    values: np.ndarray,
    *,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
):
    """Plot a scalar field as a square array of lattice cells."""
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"values must be two-dimensional, got {values.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    artist = ax.imshow(
        values.T,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.grid(which="minor", color=(1, 1, 1, 0.32), linewidth=0.3)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_grid_trajectory(
    group,
    exact_points: np.ndarray,
    predicted_points: np.ndarray,
    *,
    title: str = "Bump center trajectory",
    save_path: str | None = None,
):
    """Overlay exact and decoded paths on a neutral square lattice."""
    figure, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    background = np.full((group.p1, group.p2), 0.92)
    plot_grid_scalar(
        background,
        ax=ax,
        cmap="Greys_r",
        vmin=0,
        vmax=1,
        colorbar=False,
    )
    exact_points = np.asarray(exact_points)
    predicted_points = np.asarray(predicted_points)
    ax.scatter(
        exact_points[:, 0],
        exact_points[:, 1],
        s=36,
        color=TRACK_COLOR,
        alpha=0.4,
        linewidths=0,
        label="true bump path",
        zorder=2,
    )
    ax.plot(
        exact_points[:, 0],
        exact_points[:, 1],
        color=TRACK_COLOR,
        linewidth=2.4,
        label="true bump center",
        zorder=3,
    )
    ax.plot(
        predicted_points[:, 0],
        predicted_points[:, 1],
        "k--",
        linewidth=2,
        label="predicted theory peak",
        zorder=4,
    )
    ax.scatter(
        *exact_points[0],
        s=120,
        color=TRACK_COLOR,
        edgecolors="black",
        linewidths=0.8,
        label="start",
        zorder=5,
    )
    ax.scatter(
        *exact_points[-1],
        s=160,
        marker="*",
        color=TRACK_COLOR,
        edgecolors="black",
        linewidths=0.8,
        label="end",
        zorder=6,
    )
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight", dpi=300)
    return ax
