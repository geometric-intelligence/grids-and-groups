"""Render a step-by-step animation of a constructed discrete-SE(2) RNN rollout.

Run from the repository root:

    conda run -n group-agf python \
        notebooks/constructed_networks/animate_discrete_se2_rollout.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon

PROJECT_ROOT = next(
    parent for parent in (Path.cwd(), *Path.cwd().parents) if (parent / "src").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.finite_group_rnn import (  # noqa: E402
    build_finite_group_rnn,
    random_invertible_encoding,
)
from src.geometry.discrete_se2 import (  # noqa: E402
    decode_pose,
    gaussian_bump,
    make_momentum_motion_sequence,
    signal_to_tensor,
)
from src.geometry.discrete_se2.core import lattice_coordinates  # noqa: E402
from src.groups import DiscreteSE2Group  # noqa: E402

LOCAL_TRANSLATIONS = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
)
ORIENTATION_COLORS = ("#0072B2", "#E69F00", "#009E73")


@dataclass
class SignalPanel:
    axes: list
    collections: list[PatchCollection]
    heading_texts: list


@dataclass
class MarkerPanel:
    axes: list
    collections: list[PatchCollection]
    markers: list
    heading_texts: list


def centered_residue(value: int, modulus: int) -> int:
    """Choose the periodic representative nearest zero."""
    value %= modulus
    return value - modulus if value > modulus // 2 else value


def element_label(group, element: int) -> str:
    """Format a group element with centered translation coordinates."""
    x, y, rotation = group.decode(int(element))
    return (
        f"({centered_residue(x, group.n)}, "
        f"{centered_residue(y, group.n)}, "
        f"{360 * rotation // group.m}°)"
    )


def pose_label(group, pose: tuple[int, int, int]) -> str:
    """Format an absolute pose."""
    x, y, rotation = (int(value) for value in pose)
    return f"({x}, {y}, {360 * rotation // group.m}°)"


def make_hexagons(x_coordinates, y_coordinates):
    """Create one display hexagon per lattice point."""
    return [
        RegularPolygon(
            (float(x), float(y)),
            numVertices=6,
            radius=1 / np.sqrt(3),
            orientation=np.pi / 6,
        )
        for x, y in zip(x_coordinates.ravel(), y_coordinates.ravel())
    ]


def style_lattice_axis(ax, x_coordinates, y_coordinates, title: str):
    """Apply a shared compact lattice style."""
    padding = 1 / np.sqrt(3)
    ax.set_xlim(
        float(x_coordinates.min()) - padding,
        float(x_coordinates.max()) + padding,
    )
    ax.set_ylim(
        float(y_coordinates.min()) - padding,
        float(y_coordinates.max()) + padding,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=8, pad=2)


def add_signal_panel(
    figure,
    subplot_spec,
    group,
    *,
    title: str,
    norm,
    cmap: str = "viridis",
) -> SignalPanel:
    """Create one three-heading signal panel."""
    grid = subplot_spec.subgridspec(1, group.m, wspace=0.03)
    x_coordinates, y_coordinates = lattice_coordinates(group.n, mode="centered_axial")
    axes, collections, heading_texts = [], [], []
    for heading in range(group.m):
        ax = figure.add_subplot(grid[0, heading])
        collection = PatchCollection(
            make_hexagons(x_coordinates, y_coordinates),
            cmap=cmap,
            norm=norm,
            edgecolor=(1, 1, 1, 0.45),
            linewidth=0.2,
        )
        collection.set_array(np.zeros(group.n * group.n))
        ax.add_collection(collection)
        style_lattice_axis(
            ax,
            x_coordinates,
            y_coordinates,
            f"{360 * heading // group.m}°",
        )
        axes.append(ax)
        collections.append(collection)
        heading_texts.append(ax.title)
    axes[0].text(
        0.0,
        1.23,
        title,
        transform=axes[0].transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
    )
    return SignalPanel(axes, collections, heading_texts)


def add_marker_panel(
    figure,
    subplot_spec,
    group,
    *,
    title: str,
    support: np.ndarray | None = None,
    marker_color: str,
) -> MarkerPanel:
    """Create a three-heading group panel with one movable marker."""
    grid = subplot_spec.subgridspec(1, group.m, wspace=0.03)
    x_coordinates, y_coordinates = lattice_coordinates(group.n, mode="centered_axial")
    axes, collections, markers, heading_texts = [], [], [], []
    support_tensor = (
        np.zeros((group.m, group.n, group.n))
        if support is None
        else signal_to_tensor(group, support)
    )
    for heading in range(group.m):
        ax = figure.add_subplot(grid[0, heading])
        collection = PatchCollection(
            make_hexagons(x_coordinates, y_coordinates),
            cmap="Blues",
            norm=mcolors.Normalize(vmin=0, vmax=1),
            edgecolor=(0.35, 0.35, 0.35, 0.28),
            linewidth=0.25,
        )
        collection.set_array(support_tensor[heading].ravel())
        ax.add_collection(collection)
        (marker,) = ax.plot(
            [],
            [],
            marker="h",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor=marker_color,
            markeredgewidth=2.5,
            linestyle="none",
            zorder=5,
        )
        style_lattice_axis(
            ax,
            x_coordinates,
            y_coordinates,
            f"{360 * heading // group.m}°",
        )
        axes.append(ax)
        collections.append(collection)
        markers.append(marker)
        heading_texts.append(ax.title)
    axes[0].text(
        0.0,
        1.23,
        title,
        transform=axes[0].transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
    )
    return MarkerPanel(axes, collections, markers, heading_texts)


def set_signal(panel: SignalPanel, group, signal: np.ndarray, *, alpha: float):
    """Update a signal panel."""
    tensor = signal_to_tensor(group, signal)
    for heading, collection in enumerate(panel.collections):
        collection.set_array(tensor[heading].ravel())
        collection.set_alpha(alpha)


def set_marker(
    panel: MarkerPanel,
    group,
    element_or_pose,
    *,
    alpha: float,
    is_pose: bool,
):
    """Move a panel marker to one group element or absolute pose."""
    if is_pose:
        x_index, y_index, heading = (int(value) for value in element_or_pose)
    else:
        x_index, y_index, heading = group.decode(int(element_or_pose))
    x_coordinates, y_coordinates = lattice_coordinates(group.n, mode="centered_axial")
    for layer, marker in enumerate(panel.markers):
        if layer == heading and alpha > 0:
            marker.set_data([x_coordinates[x_index, y_index]], [y_coordinates[x_index, y_index]])
            marker.set_alpha(alpha)
        else:
            marker.set_data([], [])
    for collection in panel.collections:
        collection.set_alpha(max(0.08, alpha))


def build_local_sequence(group, initial_pose, *, steps: int, seed: int):
    """Generate only local inputs, with no preliminary repositioning token."""
    generated = make_momentum_motion_sequence(
        group,
        steps=steps + 1,
        seed=seed,
        include_rotations=True,
        momentum=True,
        turn_probability=0.18,
        stay_probability=0.04,
        start_xy=initial_pose[:2],
        initial_pose=initial_pose,
    )
    sequence = generated[1:]
    support = {
        group.encode(dx, dy, rotation)
        for dx, dy in LOCAL_TRANSLATIONS
        for rotation in range(group.m)
    }
    unexpected = set(int(element) for element in sequence) - support
    if unexpected:
        raise RuntimeError(f"generated inputs outside local support: {unexpected}")
    return sequence, support


def build_exact_states(group, initial_pose, sequence, action_side: str):
    """Compose absolute physical states under the selected action convention."""
    state = group.encode(*initial_pose)
    states = []
    for element in sequence:
        if action_side == "right":
            state = group.compose(state, int(element))
        else:
            state = group.compose(int(element), state)
        states.append(state)
    return np.asarray(states, dtype=int)


def render_animation(args) -> Path:
    """Construct the model, run it, and write the animation."""
    group = DiscreteSE2Group(n=args.n, m=3)
    initial_pose = (2, 2, 0)

    orientation_weights = np.zeros(group.m)
    orientation_weights[initial_pose[2]] = 1.0
    x_allo = gaussian_bump(
        group,
        center=initial_pose[:2],
        sigma=1.0,
        orientation_weights=orientation_weights,
    )
    x_ego = random_invertible_encoding(group, group.irreps(), seed=10)
    model = build_finite_group_rnn(
        group,
        x_ego,
        x_allo=x_allo,
        irrep_selection="power",
        num_irreps=None,
        q_rho=3,
        materialize_mix=False,
        action_side=args.action_side,
    )

    sequence, support_elements = build_local_sequence(
        group,
        initial_pose,
        steps=args.steps,
        seed=args.seed,
    )
    result = {
        key: value.detach().cpu().numpy() for key, value in model.rollout(x_allo, sequence).items()
    }
    exact_signals = result["true_outputs"]
    predicted_signals = result["predicted_outputs"]
    predicted_poses = np.asarray(
        [decode_pose(group, output) for output in predicted_signals],
        dtype=int,
    )
    exact_states = build_exact_states(
        group,
        initial_pose,
        sequence,
        args.action_side,
    )
    exact_poses = np.asarray([group.decode(state) for state in exact_states], dtype=int)

    support_signal = np.zeros(group.order)
    support_signal[list(support_elements)] = 1.0
    signal_min = float(min(x_allo.min(), exact_signals.min(), predicted_signals.min()))
    signal_max = float(max(x_allo.max(), exact_signals.max(), predicted_signals.max()))
    signal_norm = mcolors.Normalize(vmin=signal_min, vmax=signal_max)

    figure = plt.figure(figsize=(14, 11), facecolor="white")
    outer = figure.add_gridspec(
        3,
        2,
        left=0.035,
        right=0.985,
        top=0.90,
        bottom=0.10,
        hspace=0.43,
        wspace=0.10,
    )
    previous_panel = add_signal_panel(
        figure,
        outer[0, 0],
        group,
        title="1. Previous allocentric state",
        norm=signal_norm,
    )
    input_panel = add_marker_panel(
        figure,
        outer[0, 1],
        group,
        title="2. Incoming local group element",
        support=support_signal,
        marker_color="#D62728",
    )
    exact_signal_panel = add_signal_panel(
        figure,
        outer[1, 0],
        group,
        title=f"3. Exact state after {args.action_side} action",
        norm=signal_norm,
    )
    predicted_signal_panel = add_signal_panel(
        figure,
        outer[1, 1],
        group,
        title="4. Network output",
        norm=signal_norm,
    )
    predicted_pose_panel = add_marker_panel(
        figure,
        outer[2, 0],
        group,
        title="5. Pose decoded from network output",
        marker_color="#CC79A7",
    )
    exact_pose_panel = add_marker_panel(
        figure,
        outer[2, 1],
        group,
        title="6. Ground-truth composed pose",
        marker_color="#009E73",
    )

    title = figure.suptitle(
        "How an egocentric input drives the allocentric recurrent state",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    step_text = figure.text(0.5, 0.935, "", ha="center", va="center", fontsize=12)
    explanation_text = figure.text(
        0.5,
        0.045,
        "",
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.4,
    )
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=signal_norm, cmap="viridis"),
        ax=[
            *previous_panel.axes,
            *exact_signal_panel.axes,
            *predicted_signal_panel.axes,
        ],
        location="bottom",
        fraction=0.018,
        pad=0.08,
        aspect=45,
    )
    colorbar.set_label("signal value")

    initial_hold = max(1, round(args.fps * args.initial_seconds))
    transition_frames = max(2, round(args.fps * args.step_seconds))
    frame_plan = [(-1, 0.0)] * initial_hold
    for step in range(args.steps):
        frame_plan.extend((step, progress) for progress in np.linspace(0.0, 1.0, transition_frames))

    def update(frame_index):
        step, progress = frame_plan[frame_index]
        if step < 0:
            set_signal(previous_panel, group, x_allo, alpha=1.0)
            set_signal(exact_signal_panel, group, x_allo, alpha=1.0)
            set_signal(predicted_signal_panel, group, x_allo, alpha=1.0)
            set_marker(
                input_panel,
                group,
                group.identity(),
                alpha=0.0,
                is_pose=False,
            )
            set_marker(
                predicted_pose_panel,
                group,
                initial_pose,
                alpha=1.0,
                is_pose=True,
            )
            set_marker(
                exact_pose_panel,
                group,
                initial_pose,
                alpha=1.0,
                is_pose=True,
            )
            step_text.set_text(f"t = 0   initial state s₀ = {pose_label(group, initial_pose)}")
            explanation_text.set_text(
                "The allocentric signal encodes the initial group state. "
                "No egocentric motion has been applied yet."
            )
            return []

        previous_signal = x_allo if step == 0 else exact_signals[step - 1]
        previous_prediction = x_allo if step == 0 else predicted_signals[step - 1]
        previous_pose = initial_pose if step == 0 else tuple(exact_poses[step - 1])
        previous_decoded_pose = initial_pose if step == 0 else tuple(predicted_poses[step - 1])
        incoming = int(sequence[step])
        exact_pose = tuple(exact_poses[step])
        predicted_pose = tuple(predicted_poses[step])
        interpolated_exact = (1.0 - progress) * previous_signal + progress * exact_signals[step]
        interpolated_prediction = (
            1.0 - progress
        ) * previous_prediction + progress * predicted_signals[step]

        set_signal(previous_panel, group, previous_signal, alpha=1.0)
        set_marker(
            input_panel,
            group,
            incoming,
            alpha=1.0,
            is_pose=False,
        )
        set_signal(
            exact_signal_panel,
            group,
            interpolated_exact,
            alpha=1.0,
        )
        set_signal(
            predicted_signal_panel,
            group,
            interpolated_prediction,
            alpha=1.0,
        )
        set_marker(
            predicted_pose_panel,
            group,
            predicted_pose if progress >= 0.5 else previous_decoded_pose,
            alpha=1.0,
            is_pose=True,
        )
        set_marker(
            exact_pose_panel,
            group,
            exact_pose if progress >= 0.5 else previous_pose,
            alpha=1.0,
            is_pose=True,
        )

        step_text.set_text(
            f"t = {step + 1}   incoming gₜ = {element_label(group, incoming)}   "
            f"transition {progress:3.0%}"
        )
        operation = "sₜ₋₁ · gₜ" if args.action_side == "right" else "gₜ · sₜ₋₁"
        error = np.linalg.norm(predicted_signals[step] - exact_signals[step])
        explanation = (
            f"{operation}: {pose_label(group, previous_pose)} → "
            f"{pose_label(group, exact_pose)}    |    "
            f"decoded {pose_label(group, predicted_pose)}    |    "
            f"output error {error:.2e}"
        )
        explanation_text.set_text(explanation)
        return []

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frame_plan),
        interval=1000 / args.fps,
        blit=False,
        repeat=False,
    )
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=args.fps,
        codec="h264",
        bitrate=2200,
        metadata={"title": title.get_text()},
    )
    movie.save(output_path, writer=writer, dpi=args.dpi)
    plt.close(figure)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--action-side", choices=("left", "right"), default="right")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--initial-seconds", type=float, default=1.5)
    parser.add_argument("--step-seconds", type=float, default=1.2)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument(
        "--output",
        default="notebooks/constructed_networks/discrete_se2_rollout.mp4",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    rendered_path = render_animation(arguments)
    print(f"Wrote {rendered_path}")
