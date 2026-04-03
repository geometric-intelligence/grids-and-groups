import numpy as np

from src.groups import OctahedralGroup
import src.template as template


def test_group_fourier_inverse_is_identity():
    """Test that fourier followed by inverse_fourier reconstructs the original."""
    group = OctahedralGroup()

    tpl = template.fixed_group(group, powers=[100.0, 20.0, 0.0, 0.0, 0.0])

    fourier_coefs = group.fourier(tpl)
    reconstructed = group.inverse_fourier(fourier_coefs)

    assert np.allclose(tpl, reconstructed, atol=1e-10), (
        f"Inversion failed! max diff: {np.max(np.abs(tpl - reconstructed))}"
    )


def test_group_fourier_coefs_shape():
    """Test that fourier returns one coefficient matrix per irrep."""
    group = OctahedralGroup()
    tpl = template.fixed_group(group, powers=[100.0, 20.0, 0.0, 0.0, 0.0])

    fourier_coefs = group.fourier(tpl)

    assert len(fourier_coefs) == len(group.irreps())
    for coef, irrep in zip(fourier_coefs, group.irreps()):
        assert coef.shape == (irrep.size, irrep.size)


if __name__ == "__main__":
    test_group_fourier_inverse_is_identity()
    test_group_fourier_coefs_shape()
