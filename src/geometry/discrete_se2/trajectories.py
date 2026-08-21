"""Trajectory generation for discrete SE(2) geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRACK_COLOR = "#E45756"


@dataclass(frozen=True)
class NaturalisticMotionConfig:
    """Probabilities and wall response for a forward-biased C6 motion policy.

    Translation directions are measured relative to the heading *after* the
    sampled turn.  The six nonzero directions are forward, forward-left,
    backward-left, backward, backward-right, and forward-right at 60-degree
    intervals.
    """

    stay_probability: float = 0.05
    forward_probability: float = 0.70
    forward_left_or_right_probability: float = 0.115
    backward_left_or_right_probability: float = 0.005
    backward_probability: float = 0.01
    turn_probability: float = 0.12
    turn_persistence: float = 0.20
    wall_lookahead: int = 3
    wall_avoidance_strength: float = 2.0
    minimum_wall_weight: float = 0.05

    def __post_init__(self) -> None:
        probabilities = {
            "stay_probability": self.stay_probability,
            "forward_probability": self.forward_probability,
            "forward_left_or_right_probability": (self.forward_left_or_right_probability),
            "backward_left_or_right_probability": (self.backward_left_or_right_probability),
            "backward_probability": self.backward_probability,
            "turn_probability": self.turn_probability,
            "turn_persistence": self.turn_persistence,
            "minimum_wall_weight": self.minimum_wall_weight,
        }
        for name, value in probabilities.items():
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1], got {value}")

        translation_total = (
            self.stay_probability
            + self.forward_probability
            + 2 * self.forward_left_or_right_probability
            + 2 * self.backward_left_or_right_probability
            + self.backward_probability
        )
        if not np.isclose(translation_total, 1.0):
            raise ValueError(f"translation probabilities must sum to 1, got {translation_total}")
        if 2 * self.turn_probability > 1:
            raise ValueError(
                "turn_probability is assigned to both left and right turns, "
                "so it must be at most 0.5"
            )
        if isinstance(self.wall_lookahead, bool) or self.wall_lookahead < 1:
            raise ValueError("wall_lookahead must be an integer greater than 0")
        if not isinstance(self.wall_lookahead, (int, np.integer)):
            raise ValueError("wall_lookahead must be an integer greater than 0")
        if not np.isfinite(self.wall_avoidance_strength) or self.wall_avoidance_strength < 0:
            raise ValueError("wall_avoidance_strength must be finite and nonnegative")


def make_naturalistic_motion_sequence(
    group,
    *,
    steps: int = 100,
    seed: int = 0,
    start_xy: tuple[int, int] | None = None,
    initial_pose: tuple[int, int, int] | None = None,
    margin: int = 0,
    action_side: str = "right",
    config: NaturalisticMotionConfig | None = None,
) -> np.ndarray:
    """Generate a forward-biased, heading-coupled trajectory on a C6 lattice.

    At every ordinary step, a persistent turn in ``{-60°, 0°, +60°}`` and a
    body-relative translation are sampled jointly.  Candidate actions that
    cross the display boundary are removed, and headings with little forward
    clearance are smoothly downweighted.  Reverse and oblique locomotion remain
    possible according to ``config``.

    The first returned element relocates ``initial_pose`` (or the identity) to
    ``start_xy``.  Later elements use the requested physical action convention:
    ``current * drive = next`` for ``"right"`` and
    ``drive * current = next`` for ``"left"``.  Right-action drives remain
    local body motions.  Left-action drives reproduce the same local pose path
    but can be nonlocal as group elements because they act in the world frame.
    """
    if group.m != 6:
        raise ValueError(
            f"naturalistic heading-coupled motion requires C6 orientations, got C{group.m}"
        )
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise ValueError("steps must be a positive integer")
    if steps < 1:
        raise ValueError("steps must be a positive integer")
    if action_side not in {"left", "right"}:
        raise ValueError("action_side must be 'left' or 'right'")

    config = NaturalisticMotionConfig() if config is None else config

    n = group.n
    if isinstance(margin, bool) or not isinstance(margin, (int, np.integer)):
        raise ValueError("margin must be a nonnegative integer")
    if margin < 0 or 2 * margin >= n:
        raise ValueError(f"margin must satisfy 0 <= 2 * margin < {n}")

    rng = np.random.default_rng(seed)
    if start_xy is None:
        start_xy = (n // 2, n // 2)
    x = int(np.clip(start_xy[0], margin, n - 1 - margin))
    y = int(np.clip(start_xy[1], margin, n - 1 - margin))
    heading = 0 if initial_pose is None else int(initial_pose[2]) % group.m
    base_pose = group.identity() if initial_pose is None else group.encode(*initial_pose)
    current_pose = group.encode(x, y, heading)

    def relative_element(current: int, next_pose: int) -> int:
        if action_side == "right":
            return group.compose(group.inverse(current), next_pose)
        return group.compose(next_pose, group.inverse(current))

    def inside_bounds(x_value: int, y_value: int) -> bool:
        return margin <= x_value <= n - 1 - margin and margin <= y_value <= n - 1 - margin

    def forward_clearance(
        x_value: int,
        y_value: int,
        heading_value: int,
    ) -> int:
        dx, dy = group.apply_rotation(heading_value, 1, 0)
        dx = int(dx if dx <= n // 2 else dx - n)
        dy = int(dy if dy <= n // 2 else dy - n)
        clearance = 0
        for distance in range(1, config.wall_lookahead + 1):
            if not inside_bounds(x_value + distance * dx, y_value + distance * dy):
                break
            clearance = distance
        return clearance

    sequence = [relative_element(base_pose, current_pose)]
    # Counterclockwise order in axial coordinates, beginning with forward.
    relative_directions = np.asarray(
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)],
        dtype=int,
    )
    translation_weights = np.asarray(
        [
            config.stay_probability,
            config.forward_probability,
            config.forward_left_or_right_probability,
            config.backward_left_or_right_probability,
            config.backward_probability,
            config.backward_left_or_right_probability,
            config.forward_left_or_right_probability,
        ],
        dtype=float,
    )
    turn_values = np.asarray([-1, 0, 1], dtype=int)
    base_turn_weights = np.asarray(
        [
            config.turn_probability,
            1 - 2 * config.turn_probability,
            config.turn_probability,
        ],
        dtype=float,
    )
    previous_turn = 0

    for _ in range(steps - 1):
        turn_weights = (1 - config.turn_persistence) * base_turn_weights
        turn_weights[previous_turn + 1] += config.turn_persistence

        candidates = []
        candidate_weights = []
        for turn_index, turn in enumerate(turn_values):
            next_heading = (heading + int(turn)) % group.m
            for translation_index, relative_direction in enumerate(relative_directions):
                dx_world, dy_world = group.apply_rotation(
                    next_heading,
                    int(relative_direction[0]),
                    int(relative_direction[1]),
                )
                dx_world = int(dx_world if dx_world <= n // 2 else dx_world - n)
                dy_world = int(dy_world if dy_world <= n // 2 else dy_world - n)
                x_next = x + dx_world
                y_next = y + dy_world
                if not inside_bounds(x_next, y_next):
                    continue

                clearance_fraction = (
                    forward_clearance(x_next, y_next, next_heading) / config.wall_lookahead
                )
                wall_weight = (
                    config.minimum_wall_weight
                    + (1 - config.minimum_wall_weight) * clearance_fraction
                ) ** config.wall_avoidance_strength
                weight = (
                    turn_weights[turn_index] * translation_weights[translation_index] * wall_weight
                )
                if weight <= 0:
                    continue
                next_pose = group.encode(x_next, y_next, next_heading)
                candidates.append(
                    (
                        next_pose,
                        x_next,
                        y_next,
                        next_heading,
                        int(turn),
                    )
                )
                candidate_weights.append(weight)

        if not candidates:
            raise RuntimeError(
                "naturalistic motion policy has no valid action; "
                "increase minimum_wall_weight or allow stationary motion"
            )
        candidate_weights = np.asarray(candidate_weights, dtype=float)
        candidate_weights /= candidate_weights.sum()
        chosen = int(rng.choice(len(candidates), p=candidate_weights))
        (
            next_pose,
            x,
            y,
            heading,
            previous_turn,
        ) = candidates[chosen]
        sequence.append(relative_element(current_pose, next_pose))
        current_pose = next_pose

    return np.asarray(sequence, dtype=int)


def make_momentum_motion_sequence(
    group,
    *,
    steps: int = 100,
    seed: int = 0,
    include_rotations: bool = False,
    momentum: bool = True,
    turn_probability: float = 0.18,
    stay_probability: float = 0.04,
    start_xy: tuple[int, int] | None = None,
    initial_pose: tuple[int, int, int] | None = None,
    margin: int = 2,
    max_resample: int = 20,
) -> np.ndarray:
    """Generate body-frame motions while keeping the displayed pose in bounds.

    The first returned element moves ``initial_pose`` (or the identity when it
    is omitted) to ``start_xy``.  Every later element is the local increment
    ``current_pose⁻¹ * next_pose``.
    """
    rng = np.random.default_rng(seed)
    n = group.n
    if start_xy is None:
        start_xy = (n // 2, n // 2)
    x = int(np.clip(start_xy[0], margin, n - 1 - margin))
    y = int(np.clip(start_xy[1], margin, n - 1 - margin))
    directions = np.asarray([(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)])

    def inside_bounds(x_new, y_new):
        return margin <= x_new <= n - 1 - margin and margin <= y_new <= n - 1 - margin

    heading = 0 if initial_pose is None else int(initial_pose[2]) % group.m
    base_pose = group.identity() if initial_pose is None else group.encode(*initial_pose)
    current_pose = group.encode(x, y, heading)
    sequence = [group.compose(group.inverse(base_pose), current_pose)]
    direction_index = int(rng.integers(0, len(directions)))
    for _ in range(steps - 1):
        if rng.random() < stay_probability:
            dx, dy = 0, 0
        else:
            accepted = False
            for _ in range(max_resample):
                proposed = direction_index
                if not momentum or rng.random() < turn_probability:
                    proposed = (direction_index + rng.choice([-1, 1])) % len(directions)
                dx, dy = directions[proposed]
                if inside_bounds(x + dx, y + dy):
                    direction_index = proposed
                    accepted = True
                    break
                direction_index = (direction_index + rng.choice([-1, 1])) % len(directions)
            if not accepted:
                dx, dy = 0, 0
        x += int(dx)
        y += int(dy)
        rotation_step = int(rng.choice([0, 0, 0, 1, 2])) if include_rotations else 0
        heading = (heading + rotation_step) % group.m
        next_pose = group.encode(x, y, heading)
        relative = group.compose(group.inverse(current_pose), next_pose)
        current_pose = next_pose
        sequence.append(relative)
    return np.asarray(sequence, dtype=int)
