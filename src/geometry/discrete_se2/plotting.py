"""Plotting helpers for discrete SE(2) geometry."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from plotly.subplots import make_subplots

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


def plotly_heading_stacks(
    group,
    signals,
    *,
    titles,
    highlighted_elements=None,
    arrangement: str = "horizontal",
    coordinate_mode: str = "axial",
    layer_spacing: float = 4.0,
    width: int = 750,
    height: int = 650,
):
    """Build interactive WebGL heading stacks in a row or column."""
    signals = [np.asarray(signal) for signal in signals]
    titles = list(titles)
    if len(signals) != len(titles):
        raise ValueError("signals and titles must have the same length")
    if arrangement not in {"horizontal", "vertical"}:
        raise ValueError("arrangement must be 'horizontal' or 'vertical'")
    if highlighted_elements is None:
        highlighted_elements = [set() for _ in signals]
    if len(highlighted_elements) != len(signals):
        raise ValueError("highlighted_elements must contain one collection per signal")

    all_values = np.concatenate(signals)
    color_min = float(all_values.min())
    color_max = float(all_values.max())
    if np.isclose(color_min, color_max):
        color_max = color_min + 1.0

    is_vertical = arrangement == "vertical"
    rows = len(signals) if is_vertical else 1
    columns = 1 if is_vertical else len(signals)
    figure = make_subplots(
        rows=rows,
        cols=columns,
        specs=[[{"type": "scene"} for _ in range(columns)] for _ in range(rows)],
        subplot_titles=titles,
        horizontal_spacing=0.015,
        vertical_spacing=0.04 if is_vertical and rows > 1 else 0.0,
    )
    for index, (signal, highlights) in enumerate(
        zip(signals, highlighted_elements)
    ):
        row = index + 1 if is_vertical else 1
        column = 1 if is_vertical else index + 1
        _add_plotly_heading_stack(
            figure,
            row,
            column,
            group,
            signal,
            highlighted_elements=set(highlights),
            coordinate_mode=coordinate_mode,
            layer_spacing=layer_spacing,
        )

    lattice_x, lattice_y = lattice_coordinates(group.n, mode=coordinate_mode)
    z_levels = layer_spacing * np.arange(group.m)
    padding = 1 / np.sqrt(3)
    axis_style = {
        "showbackground": False,
        "showgrid": False,
        "zeroline": False,
        "showline": True,
        "linecolor": "rgba(45, 45, 45, 0.8)",
        "linewidth": 2,
        "ticks": "outside",
        "tickfont": {"size": 10},
    }
    figure.update_scenes(
        xaxis={
            **axis_style,
            "title": {"text": "lattice x", "font": {"size": 12}},
            "range": [
                float(lattice_x.min() - padding),
                float(lattice_x.max() + padding),
            ],
        },
        yaxis={
            **axis_style,
            "title": {"text": "lattice y", "font": {"size": 12}},
            "range": [
                float(lattice_y.min() - padding),
                float(lattice_y.max() + padding),
            ],
        },
        zaxis={
            **axis_style,
            "title": {"text": "heading", "font": {"size": 12}},
            "range": [
                -0.5 * layer_spacing,
                float(z_levels[-1] + 0.5 * layer_spacing),
            ],
            "tickvals": z_levels.tolist(),
            "ticktext": [
                f"{degrees:.0f}°"
                for degrees in 360 * np.arange(group.m) / group.m
            ],
        },
        aspectmode="manual",
        aspectratio={"x": 1.0, "y": 0.9, "z": 0.75},
        camera={
            "projection": {"type": "orthographic"},
            "eye": {"x": 1.45, "y": -1.45, "z": 1.05},
        },
        bgcolor="rgba(0,0,0,0)",
    )
    figure.update_layout(
        width=width,
        height=height,
        margin={"l": 5, "r": 65, "t": 45, "b": 5},
        paper_bgcolor="white",
        coloraxis={
            "colorscale": "Viridis",
            "cmin": color_min,
            "cmax": color_max,
            "colorbar": {
                "title": "encoding value",
                "thickness": 14,
                "len": 0.7,
            },
        },
        showlegend=False,
    )
    return figure


def linked_plotly_html(figure, scene_pairs=()) -> str:
    """Render a Plotly figure as HTML with bidirectionally linked cameras."""
    pairs_json = repr([list(pair) for pair in scene_pairs]).replace("'", '"')
    camera_sync_script = f"""
const plot = document.getElementById('{{plot_id}}');
const scenePairs = {pairs_json};
let synchronizing = false;

plot.on('plotly_relayout', eventData => {{
    if (synchronizing) return;
    for (const [leftScene, rightScene] of scenePairs) {{
        for (const [sourceScene, targetScene] of [
            [leftScene, rightScene],
            [rightScene, leftScene]
        ]) {{
            const updates = {{}};
            const cameraKey = `${{sourceScene}}.camera`;
            if (eventData[cameraKey]) {{
                updates[`${{targetScene}}.camera`] = eventData[cameraKey];
            }}
            const cameraPrefix = `${{cameraKey}}.`;
            for (const [key, value] of Object.entries(eventData)) {{
                if (key.startsWith(cameraPrefix)) {{
                    const suffix = key.slice(cameraPrefix.length);
                    updates[`${{targetScene}}.camera.${{suffix}}`] = value;
                }}
            }}
            if (Object.keys(updates).length > 0) {{
                synchronizing = true;
                Plotly.relayout(plot, updates).finally(() => {{
                    window.requestAnimationFrame(() => {{
                        synchronizing = false;
                    }});
                }});
                return;
            }}
        }}
    }}
}});
"""
    return pio.to_html(
        figure,
        include_plotlyjs="cdn",
        full_html=False,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["zoom3d"],
            "scrollZoom": False,
            "responsive": True,
        },
        post_script=camera_sync_script,
    )


def _add_plotly_heading_stack(
    figure,
    row,
    column,
    group,
    signal,
    *,
    highlighted_elements,
    coordinate_mode,
    layer_spacing,
):
    tensor = signal_to_tensor(group, signal)
    lattice_x, lattice_y = lattice_coordinates(group.n, mode=coordinate_mode)
    highlighted_poses = {group.decode(element) for element in highlighted_elements}
    hex_radius = 1 / np.sqrt(3)
    hex_angles = np.pi / 6 + np.arange(6) * np.pi / 3

    for heading in range(group.m):
        z_level = layer_spacing * heading
        mesh_x, mesh_y, mesh_z = [], [], []
        triangle_i, triangle_j, triangle_k = [], [], []
        intensities = []
        border_x, border_y, border_z = [], [], []
        highlight_x, highlight_y, highlight_z = [], [], []

        for x_index in range(group.n):
            for y_index in range(group.n):
                center_x = lattice_x[x_index, y_index]
                center_y = lattice_y[x_index, y_index]
                corner_x = center_x + hex_radius * np.cos(hex_angles)
                corner_y = center_y + hex_radius * np.sin(hex_angles)
                value = float(tensor[heading, x_index, y_index])

                base = len(mesh_x)
                mesh_x.extend([center_x, *corner_x])
                mesh_y.extend([center_y, *corner_y])
                mesh_z.extend([z_level] * 7)
                intensities.extend([value] * 7)
                for corner in range(6):
                    triangle_i.append(base)
                    triangle_j.append(base + 1 + corner)
                    triangle_k.append(base + 1 + (corner + 1) % 6)

                target = (
                    (highlight_x, highlight_y, highlight_z)
                    if (x_index, y_index, heading) in highlighted_poses
                    else (border_x, border_y, border_z)
                )
                target[0].extend([*corner_x, corner_x[0], None])
                target[1].extend([*corner_y, corner_y[0], None])
                target[2].extend([z_level + 0.03] * 7 + [None])

        figure.add_trace(
            go.Mesh3d(
                x=mesh_x,
                y=mesh_y,
                z=mesh_z,
                i=triangle_i,
                j=triangle_j,
                k=triangle_k,
                intensity=intensities,
                intensitymode="vertex",
                coloraxis="coloraxis",
                flatshading=True,
                hoverinfo="skip",
                lighting={
                    "ambient": 1.0,
                    "diffuse": 0.0,
                    "specular": 0.0,
                    "roughness": 1.0,
                    "fresnel": 0.0,
                },
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter3d(
                x=border_x,
                y=border_y,
                z=border_z,
                mode="lines",
                line={"color": "rgba(255, 255, 255, 0.9)", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        if highlight_x:
            figure.add_trace(
                go.Scatter3d(
                    x=highlight_x,
                    y=highlight_y,
                    z=highlight_z,
                    mode="lines",
                    line={"color": "rgb(255, 59, 48)", "width": 7},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=column,
            )
