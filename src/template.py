import numpy as np


def one_hot(group_size):
    """One-hot, with 0th frequency removed (!) encoding of an integer value in R^group_size."""
    vec = np.zeros(group_size)
    vec[1] = 10

    zeroth_freq = np.mean(vec)
    vec = vec - zeroth_freq
    return vec


def custom_fourier(group, powers):
    """Generate a template for a group from desired per-irrep power values.

    Each entry in *powers* specifies the desired spectral power for the
    corresponding irrep.  The function converts these to Fourier coefficient
    diagonal values, builds the spectrum, and returns the mean-centered
    template.

    Parameters
    ----------
    group : Group
        The group.
    powers : list of float
        Desired spectral power for each irrep (one entry per irrep).

    Returns
    -------
    template : np.ndarray, shape (group.order(),), dtype float32
        The mean-centered template.
    """
    group_order = group.order
    irreps = group.irreps()
    irrep_dims = [ir.size for ir in irreps]

    assert len(powers) == len(irreps), (
        f"powers must have {len(irreps)} values (one per irrep), got {len(powers)}"
    )

    fourier_coef_diag_values = [
        np.sqrt(group_order * p / dim**2) if p > 0 else 0.0 for p, dim in zip(powers, irrep_dims)
    ]

    spectrum = []
    for i, irrep in enumerate(irreps):
        diag_val = fourier_coef_diag_values[i]
        mat = np.zeros((irrep.size, irrep.size), dtype=float)
        np.fill_diagonal(mat, np.full(irrep.size, diag_val, dtype=float))
        spectrum.append(mat)

    template = group.inverse_fourier(spectrum)
    template = template - np.mean(template)
    template = template.astype(np.float32)

    return template


def make_template(config):
    """Create a template based on configuration.

    For ``custom_fourier`` templates, uses :func:`custom_fourier` with the
    ``Group`` object stored in ``config["group"]``.
    """
    if config["template_type"] == "custom_fourier":
        template = custom_fourier(config["group"], config["powers"])
    elif config["template_type"] == "one_hot":
        template = one_hot(config["group_size"])
    else:
        raise ValueError(f"Unknown template type: {config['template_type']}")
    return template


def mnist_1d(group_size: int, label: int, root: str = "data", axis: int = 0):
    """Return a (group_size,) 1D template from a random MNIST image by taking a slice or projection.

    Args:
        group_size: dimension of the cyclic group
        label: MNIST digit class (0-9)
        root: MNIST data directory
        axis: 0 for row average, 1 for column average, 2 for diagonal

    Returns:
        template: (group_size,) array
    """
    import torch
    import torchvision
    import torchvision.transforms as transforms

    if not (0 <= int(label) <= 9):
        raise ValueError("label must be an integer in [0, 9].")

    ds = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )
    cls_idxs = (ds.targets == int(label)).nonzero(as_tuple=True)[0]
    if cls_idxs.numel() == 0:
        raise ValueError(f"No samples for label {label}.")

    idx = cls_idxs[torch.randint(len(cls_idxs), (1,)).item()].item()
    img, _ = ds[idx]
    img = img[0].numpy()

    if axis == 0:
        signal = img.mean(axis=1)
    elif axis == 1:
        signal = img.mean(axis=0)
    elif axis == 2:
        signal = np.diag(img)
    else:
        raise ValueError("axis must be 0, 1, or 2")

    from scipy.interpolate import interp1d

    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, group_size)
    f = interp1d(x_old, signal, kind="cubic")
    template = f(x_new)

    return template.astype(np.float32)


def mnist_2d(p1: int, p2: int, label: int, root: str = "data"):
    """Return a (p1, p2) template from a random MNIST image.

    Args:
        p1, p2: dimensions
        label: MNIST digit class (0-9)
        root: MNIST data directory

    Returns:
        template: (p1, p2) array
    """
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms

    if not (0 <= int(label) <= 9):
        raise ValueError("label must be an integer in [0, 9].")

    ds = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )
    cls_idxs = (ds.targets == int(label)).nonzero(as_tuple=True)[0]
    if cls_idxs.numel() == 0:
        raise ValueError(f"No samples for label {label}.")

    idx = cls_idxs[torch.randint(len(cls_idxs), (1,)).item()].item()
    img, _ = ds[idx]
    img = nn.functional.interpolate(
        img.unsqueeze(0), size=(p1, p2), mode="bilinear", align_corners=False
    )[0, 0]
    return img.numpy().astype(np.float32)
