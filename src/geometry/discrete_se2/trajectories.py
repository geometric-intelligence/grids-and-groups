"""Trajectory generation for discrete SE(2) geometry."""

import numpy as np

TRACK_COLOR = "#E45756"


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
    directions = np.asarray(
        [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    )

    def inside_bounds(x_new, y_new):
        return (
            margin <= x_new <= n - 1 - margin
            and margin <= y_new <= n - 1 - margin
        )

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
                    proposed = (direction_index + rng.choice([-1, 1])) % len(
                        directions
                    )
                dx, dy = directions[proposed]
                if inside_bounds(x + dx, y + dy):
                    direction_index = proposed
                    accepted = True
                    break
                direction_index = (
                    direction_index + rng.choice([-1, 1])
                ) % len(directions)
            if not accepted:
                dx, dy = 0, 0
        x += int(dx)
        y += int(dy)
        rotation_step = (
            int(rng.choice([0, 0, 0, 1, 2])) if include_rotations else 0
        )
        heading = (heading + rotation_step) % group.m
        next_pose = group.encode(x, y, heading)
        relative = group.compose(group.inverse(current_pose), next_pose)
        current_pose = next_pose
        sequence.append(relative)
    return np.asarray(sequence, dtype=int)
