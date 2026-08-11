"""Tests for the right-regular action implemented through the opposite group."""

import numpy as np

from src.groups import DihedralGroup, DiscreteSE2Group, OppositeGroup


def test_right_action_moves_state_by_body_frame_product():
    group = DiscreteSE2Group(n=5, m=4)
    opposite = OppositeGroup(group)
    state = group.encode(2, 2, 1)
    motion = group.encode(1, 0, 0)
    signal = np.zeros(group.order)
    signal[state] = 1.0

    transformed = opposite.left_action(motion, signal)

    assert group.decode(int(np.argmax(transformed))) == (2, 3, 1)
    assert int(np.argmax(transformed)) == group.compose(state, motion)


def test_opposite_irreps_are_homomorphisms_for_reversed_product():
    group = DiscreteSE2Group(n=3, m=3)
    opposite = OppositeGroup(group)
    left = group.encode(1, 0, 1)
    right = group.encode(0, 1, 0)

    for irrep in opposite.irreps():
        np.testing.assert_allclose(
            irrep(opposite.compose(left, right)),
            irrep(left) @ irrep(right),
            atol=1e-10,
        )


def test_opposite_group_can_infer_operations_from_regular_representation():
    group = DihedralGroup(N=3)
    opposite = OppositeGroup(group)
    identity = opposite.identity()

    for element in opposite.elements():
        inverse = opposite.inverse(element)
        assert opposite.compose(identity, element) == element
        assert opposite.compose(element, identity) == element
        assert opposite.compose(element, inverse) == identity
        assert opposite.compose(inverse, element) == identity

    regular = opposite.regular_rep()
    for left in opposite.elements():
        for right in opposite.elements():
            basis = np.zeros(group.order)
            basis[right] = 1.0
            assert int(np.argmax(regular[left] @ basis)) == opposite.compose(left, right)
