"""Motion trajectories for square-torus geometry."""

import numpy as np


def make_momentum_motion_sequence(
    group,
    *,
    steps: int = 250,
    seed: int = 1,
    turn_probability: float = 0.18,
    stay_probability: float = 0.04,
    start_xy: tuple[int, int] | None = None,
    margin: int = 2,
    max_resample: int = 20,
) -> np.ndarray:
    """Generate a bounded persistent walk as relative translation elements."""
    rng = np.random.default_rng(seed)
    if start_xy is None:
        start_xy = (group.p1 // 2, group.p2 // 2)
    x = int(np.clip(start_xy[0], margin, group.p1 - 1 - margin))
    y = int(np.clip(start_xy[1], margin, group.p2 - 1 - margin))
    directions = np.asarray([(1, 0), (0, 1), (-1, 0), (0, -1)])

    def inside_bounds(x_new, y_new):
        return (
            margin <= x_new <= group.p1 - 1 - margin
            and margin <= y_new <= group.p2 - 1 - margin
        )

    sequence = [group.encode(x, y)]
    direction_index = int(rng.integers(0, len(directions)))
    for _ in range(steps - 1):
        if rng.random() < stay_probability:
            dx, dy = 0, 0
        else:
            accepted = False
            for _ in range(max_resample):
                proposed = direction_index
                if rng.random() < turn_probability:
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
        sequence.append(group.encode(int(dx), int(dy)))
    return np.asarray(sequence, dtype=int)
