import numpy as np
import pytest

from src.groups.a5 import IcosahedralGroup
from src.groups.oh import OctahedralGroup


@pytest.fixture(
    params=[OctahedralGroup, IcosahedralGroup],
    ids=["octahedral", "icosahedral"],
)
def group(request):
    return request.param()


def test_order_and_irrep_dimensions(group):
    expected = {
        OctahedralGroup: [1, 3, 3, 2, 1],
        IcosahedralGroup: [1, 3, 5, 3, 4],
    }
    dimensions = [irrep.dim for irrep in group.irreps()]

    assert len(group.elements()) == group.order
    assert dimensions == expected[type(group)]
    assert sum(dimension**2 for dimension in dimensions) == group.order


def test_group_operations(group):
    elements = group.elements()
    assert all(group.compose(0, element) == element for element in elements)
    assert all(group.compose(element, 0) == element for element in elements)
    assert all(
        group.compose(element, group.inverse(element)) == 0 for element in elements
    )
    assert all(
        group.compose(group.inverse(element), element) == 0 for element in elements
    )

    rng = np.random.default_rng(42)
    for left, middle, right in rng.integers(0, group.order, size=(500, 3)):
        assert group.compose(group.compose(left, middle), right) == group.compose(
            left, group.compose(middle, right)
        )


def test_irreps_are_unitary_homomorphisms(group):
    identity = np.eye
    for irrep in group.irreps():
        np.testing.assert_allclose(irrep(0), identity(irrep.dim), atol=1e-10)
        for element in group.elements():
            matrix = irrep(element)
            np.testing.assert_allclose(
                matrix.conj().T @ matrix,
                identity(irrep.dim),
                atol=1e-10,
            )

        for left in group.elements():
            for right in group.elements():
                np.testing.assert_allclose(
                    irrep(group.compose(left, right)),
                    irrep(left) @ irrep(right),
                    atol=1e-10,
                )


def test_irrep_characters_are_orthonormal(group):
    characters = np.asarray(
        [
            [np.trace(irrep(element)) for element in group.elements()]
            for irrep in group.irreps()
        ]
    )
    inner_products = characters.conj() @ characters.T / group.order
    np.testing.assert_allclose(
        inner_products,
        np.eye(len(group.irreps())),
        atol=1e-10,
    )


def test_regular_representation_matches_group_product(group):
    regular = group.regular_rep()
    assert regular.shape == (group.order, group.order, group.order)
    for element in group.elements():
        expected = np.zeros((group.order, group.order))
        expected[
            [group.compose(element, right) for right in group.elements()],
            group.elements(),
        ] = 1.0
        np.testing.assert_array_equal(regular[element], expected)


def test_fourier_roundtrip(group):
    signal = np.random.default_rng(42).standard_normal(group.order)
    reconstructed = group.inverse_fourier(group.fourier(signal))
    np.testing.assert_allclose(reconstructed, signal, atol=1e-10)
