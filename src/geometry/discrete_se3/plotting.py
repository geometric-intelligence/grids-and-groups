"""Plotting helpers for discrete SE(3) signals and trajectories."""

import matplotlib.pyplot as plt
import numpy as np

from .decoding import orientation_marginal


def plot_orthogonal_slices(
    volume: np.ndarray,
    *,
    center: tuple[int, int, int] | None = None,
    title: str | None = None,
    cmap: str = "viridis",
):
    """Plot orthogonal slices through a cubic scalar volume."""
    volume = np.asarray(volume)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"volume must be cubic, got {volume.shape}")
    if center is None:
        center = tuple(int(value) for value in np.unravel_index(np.argmax(volume), volume.shape))
    x, y, z = center
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    slices = (
        (volume[:, :, z], f"xy at z={z}"),
        (volume[:, y, :], f"xz at y={y}"),
        (volume[x, :, :], f"yz at x={x}"),
    )
    for ax, (values, label) in zip(axes, slices):
        image = ax.imshow(values.T, origin="lower", cmap=cmap)
        ax.set_title(label)
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if title:
        figure.suptitle(title)
    return axes


def plot_orientation_marginal(
    group,
    signal: np.ndarray,
    *,
    ax=None,
    title: str = "Orientation marginal",
):
    """Plot activity across the 24 cubic rotations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    values = orientation_marginal(group, signal)
    ax.bar(np.arange(group.num_rotations), values)
    ax.set_xlabel("Rotation index")
    ax.set_ylabel("Summed activity")
    ax.set_title(title)
    return ax


def plot_trajectory(
    positions: np.ndarray,
    *,
    ax=None,
    title: str = "Decoded spatial trajectory",
):
    """Plot a three-dimensional sequence of decoded positions."""
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (steps, 3), got {positions.shape}")
    if ax is None:
        figure = plt.figure(figsize=(6, 5), constrained_layout=True)
        ax = figure.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    return ax


def plot_pose_trajectory(
    group,
    exact_positions: np.ndarray,
    predicted_positions: np.ndarray,
    *,
    exact_rotations: np.ndarray | None = None,
    predicted_rotations: np.ndarray | None = None,
    orientation_stride: int | None = None,
    title: str = "Spatial pose trajectory",
):
    """Overlay wrapped trajectories and optional allocentric orientation frames."""
    exact_positions = np.asarray(exact_positions)
    predicted_positions = np.asarray(predicted_positions)
    expected_shape = (exact_positions.shape[0], 3)
    if exact_positions.shape != expected_shape:
        raise ValueError(
            f"exact_positions must have shape (steps, 3), got {exact_positions.shape}"
        )
    if predicted_positions.shape != exact_positions.shape:
        raise ValueError(
            "predicted_positions must have the same shape as exact_positions"
        )

    figure = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    grid = np.indices((group.n, group.n, group.n)).reshape(3, -1).T
    ax.scatter(
        grid[:, 0],
        grid[:, 1],
        grid[:, 2],
        s=12,
        color="0.85",
        alpha=0.45,
        depthshade=False,
    )

    def plot_segments(positions, *, color, linestyle, label, linewidth):
        jumps = np.any(np.abs(np.diff(positions, axis=0)) > 1, axis=1)
        start = 0
        first = True
        for stop in np.append(np.flatnonzero(jumps), len(positions) - 1):
            segment = positions[start : stop + 1]
            if len(segment):
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    segment[:, 2],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=0.9,
                    label=label if first else None,
                )
                first = False
            start = stop + 1

    plot_segments(
        exact_positions,
        color="#E45756",
        linestyle="-",
        label="exact pose",
        linewidth=2.4,
    )
    plot_segments(
        predicted_positions,
        color="black",
        linestyle="--",
        label="decoded pose",
        linewidth=1.8,
    )
    ax.scatter(
        *exact_positions[0],
        s=90,
        color="#E45756",
        edgecolors="black",
        linewidths=0.7,
        label="start",
    )
    ax.scatter(
        *exact_positions[-1],
        s=130,
        marker="*",
        color="#E45756",
        edgecolors="black",
        linewidths=0.7,
        label="end",
    )
    if (exact_rotations is None) != (predicted_rotations is None):
        raise ValueError(
            "exact_rotations and predicted_rotations must be supplied together"
        )
    if exact_rotations is not None:
        exact_rotations = np.asarray(exact_rotations, dtype=int)
        predicted_rotations = np.asarray(predicted_rotations, dtype=int)
        if exact_rotations.shape != (len(exact_positions),):
            raise ValueError("exact_rotations must contain one value per trajectory step")
        if predicted_rotations.shape != exact_rotations.shape:
            raise ValueError("predicted_rotations must match exact_rotations")
        if orientation_stride is None:
            orientation_stride = max(1, len(exact_positions) // 10)
        if orientation_stride < 1:
            raise ValueError("orientation_stride must be positive")
        frame_indices = np.arange(0, len(exact_positions), orientation_stride)
        axis_colors = ("#D62728", "#2CA02C", "#1F77B4")
        for positions, rotations, linestyle, alpha in (
            (exact_positions, exact_rotations, "-", 0.9),
            (predicted_positions, predicted_rotations, "--", 0.65),
        ):
            for sample_index in frame_indices:
                origin = positions[sample_index]
                rotation = group.rotation_matrix(rotations[sample_index])
                for axis_index, axis_color in enumerate(axis_colors):
                    direction = rotation[:, axis_index]
                    ax.plot(
                        [origin[0], origin[0] + 0.32 * direction[0]],
                        [origin[1], origin[1] + 0.32 * direction[1]],
                        [origin[2], origin[2] + 0.32 * direction[2]],
                        color=axis_color,
                        linestyle=linestyle,
                        linewidth=1.2,
                        alpha=alpha,
                    )
    ax.set(
        xlim=(-0.25, group.n - 0.75),
        ylim=(-0.25, group.n - 0.75),
        zlim=(-0.25, group.n - 0.75),
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title=title,
    )
    ax.set_box_aspect((1, 1, 1))
    ax.legend(frameon=False, loc="upper left")
    return ax


def plot_volume_scatter(
    volume: np.ndarray,
    *,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot a small cubic scalar volume as colored, activity-scaled points."""
    volume = np.asarray(volume)
    if volume.ndim != 3 or len(set(volume.shape)) != 1:
        raise ValueError(f"volume must be cubic, got {volume.shape}")
    if ax is None:
        figure = plt.figure(figsize=(5, 4.5), constrained_layout=True)
        ax = figure.add_subplot(111, projection="3d")
    coordinates = np.indices(volume.shape).reshape(3, -1).T
    values = volume.reshape(-1)
    lower = float(values.min()) if vmin is None else vmin
    upper = float(values.max()) if vmax is None else vmax
    span = upper - lower
    normalized = np.zeros_like(values) if span == 0 else (values - lower) / span
    artist = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=values,
        s=20 + 160 * np.clip(normalized, 0, 1),
        cmap=cmap,
        vmin=lower,
        vmax=upper,
        alpha=0.85,
        depthshade=False,
    )
    ax.set(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title=title,
    )
    ax.set_box_aspect((1, 1, 1))
    return artist
