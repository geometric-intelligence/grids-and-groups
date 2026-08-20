"""Plotting helpers for discrete SE(2) geometry."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon

from .core import (
    align_rotation_slices,
    lattice_coordinates,
    lattice_path_coordinates,
    signal_to_tensor,
)


def plot_lattice_scalar(
    values: np.ndarray,
    *,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    coordinate_mode: str = "offset",
):
    """Plot a scalar field as a tightly packed triangular lattice of hexagons."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"values must be a square two-dimensional array, got {values.shape}")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    x, y = lattice_coordinates(values.shape[0], mode=coordinate_mode)
    norm = mcolors.Normalize(
        vmin=float(values.min()) if vmin is None else vmin,
        vmax=float(values.max()) if vmax is None else vmax,
    )
    patches = [
        RegularPolygon(
            (center_x, center_y),
            numVertices=6,
            radius=1 / np.sqrt(3),
            orientation=np.pi / 6,
        )
        for center_x, center_y in zip(x.ravel(), y.ravel())
    ]
    artist = PatchCollection(
        patches,
        array=values.ravel(),
        cmap=cmap,
        norm=norm,
        edgecolor=(0.15, 0.15, 0.15, 0.35),
        linewidth=0.25,
    )
    ax.add_collection(artist)
    radius = 1 / np.sqrt(3)
    ax.set_xlim(float(x.min()) - radius, float(x.max()) + radius)
    ax.set_ylim(float(y.min()) - radius, float(y.max()) + radius)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_group_signal(
    group,
    signal: np.ndarray,
    *,
    title: str | None = None,
    align_rotations: bool = False,
    reduction: str | None = None,
    cmap: str = "viridis",
    coordinate_mode: str = "offset",
):
    """Plot rotation slices or one rotation-reduced spatial field."""
    tensor = signal_to_tensor(group, signal)
    if align_rotations:
        tensor = align_rotation_slices(group, tensor)
    if reduction is not None:
        if reduction == "sum":
            values = tensor.sum(axis=0)
        elif reduction == "mean":
            values = tensor.mean(axis=0)
        else:
            raise ValueError("reduction must be None, 'sum', or 'mean'")
        return plot_lattice_scalar(
            values,
            title=title,
            cmap=cmap,
            coordinate_mode=coordinate_mode,
        )

    figure, axes = plt.subplots(
        1, group.m, figsize=(4.5 * group.m, 4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    vmin, vmax = float(tensor.min()), float(tensor.max())
    for rotation, ax in enumerate(axes):
        plot_lattice_scalar(
            tensor[rotation],
            ax=ax,
            title=f"rotation {rotation}",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=rotation == group.m - 1,
            coordinate_mode=coordinate_mode,
        )
    if title:
        figure.suptitle(title)
    return axes


def plot_lattice_trajectory(
    group,
    exact_points: np.ndarray,
    predicted_points: np.ndarray,
    *,
    exact_rotations: np.ndarray | None = None,
    predicted_rotations: np.ndarray | None = None,
    orientation_stride: int | None = None,
    orientation_length: float = 0.25,
    orientation_head_scale: float = 3.0,
    title: str = "Bump center trajectory",
    coordinate_mode: str = "offset",
    save_path: str | None = None,
):
    """Plot true and predicted pose trajectories in time-colored panels."""
    exact_points = np.asarray(exact_points)
    predicted_points = np.asarray(predicted_points)
    if exact_points.shape != predicted_points.shape:
        raise ValueError("predicted_points must match exact_points")
    if exact_points.ndim != 2 or exact_points.shape[1] != 2:
        raise ValueError("trajectory points must have shape (steps, 2)")
    if len(exact_points) == 0:
        raise ValueError("trajectory must contain at least one point")

    figure, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.2), constrained_layout=True, sharex=True, sharey=True
    )
    x, y = lattice_coordinates(group.n, mode=coordinate_mode)
    radius = 0.5
    exact_xy = lattice_path_coordinates(
        exact_points, group.n, mode=coordinate_mode
    )
    predicted_xy = lattice_path_coordinates(
        predicted_points, group.n, mode=coordinate_mode
    )
    time = np.arange(len(exact_points))
    norm = mcolors.Normalize(vmin=0, vmax=max(len(exact_points) - 1, 1))
    cmap = plt.colormaps["viridis"]

    for ax, points_xy, panel_title in zip(
        axes,
        (exact_xy, predicted_xy),
        ("True pose", "Predicted pose"),
    ):
        ax.scatter(
            x,
            y,
            s=7,
            color=(0.84, 0.84, 0.84),
            linewidths=0,
            zorder=0,
        )
        ax.scatter(
            points_xy[:, 0],
            points_xy[:, 1],
            c=time,
            cmap=cmap,
            norm=norm,
            s=18,
            alpha=0.85,
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            *points_xy[0],
            s=65,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            label="start",
            zorder=5,
        )
        ax.scatter(
            *points_xy[-1],
            s=85,
            marker="*",
            color="black",
            linewidths=0,
            label="end",
            zorder=6,
        )
        ax.set_xlim(float(x.min()) - radius, float(x.max()) + radius)
        ax.set_ylim(float(y.min()) - radius, float(y.max()) + radius)
        ax.set(aspect="equal", xticks=[], yticks=[], title=panel_title)
        ax.set_frame_on(False)
        ax.legend(frameon=False, loc="upper right")

    if (exact_rotations is None) != (predicted_rotations is None):
        raise ValueError(
            "exact_rotations and predicted_rotations must be supplied together"
        )
    if exact_rotations is not None:
        exact_rotations = np.asarray(exact_rotations, dtype=int)
        predicted_rotations = np.asarray(predicted_rotations, dtype=int)
        if exact_rotations.shape != (len(exact_points),):
            raise ValueError("exact_rotations must contain one value per trajectory step")
        if predicted_rotations.shape != exact_rotations.shape:
            raise ValueError("predicted_rotations must match exact_rotations")
        if orientation_stride is None:
            orientation_stride = max(1, len(exact_points) // 14)
        if orientation_stride < 1:
            raise ValueError("orientation_stride must be positive")
        if orientation_length <= 0:
            raise ValueError("orientation_length must be positive")
        if orientation_head_scale <= 0:
            raise ValueError("orientation_head_scale must be positive")
        arrow_indices = np.arange(0, len(exact_points), orientation_stride)

        def display_directions(rotations):
            axial = np.asarray(
                [
                    group.rotation_matrix(int(rotation)) @ np.asarray((1, 0))
                    for rotation in rotations
                ]
            )
            axial = np.where(axial > group.n // 2, axial - group.n, axial)
            return np.column_stack(
                (axial[:, 0] + 0.5 * axial[:, 1], np.sqrt(3) * axial[:, 1] / 2)
            )

        exact_directions = display_directions(exact_rotations[arrow_indices])
        predicted_directions = display_directions(predicted_rotations[arrow_indices])
        arrow_colors = cmap(norm(time[arrow_indices]))
        for ax, positions, directions in (
            (axes[0], exact_xy[arrow_indices], exact_directions),
            (axes[1], predicted_xy[arrow_indices], predicted_directions),
        ):
            ax.quiver(
                positions[:, 0],
                positions[:, 1],
                orientation_length * directions[:, 0],
                orientation_length * directions[:, 1],
                color=arrow_colors,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.004,
                headwidth=orientation_head_scale,
                headlength=orientation_head_scale + 1,
                headaxislength=orientation_head_scale,
                zorder=7,
            )

    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        fraction=0.03,
        pad=0.02,
    )
    colorbar.set_label("time step")
    figure.suptitle(title)
    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight", dpi=300)
    return axes
