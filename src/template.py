import numpy as np

import src.fourier as fourier

def one_hot(p):
    """One-hot, with 0th frequency removed (!) encoding of an integer value in R^p."""
    vec = np.zeros(p)
    vec[1] = 10

    zeroth_freq = np.mean(vec)
    vec = vec - zeroth_freq
    return vec


def fixed_cn(p, powers):
    """Generate a template for cyclic group C_n from desired per-mode powers.

    Parameters
    ----------
    p : int
        p in C_p (group order).
    powers : list of float
        Desired spectral power for each frequency mode.
        Length must equal the number of modes: ``(p + 1) // 2``
        for odd ``group_size``, or ``group_size // 2 + 1`` for even.

    Returns
    -------
    template : np.ndarray, shape (p,), dtype float32
        Mean-centered template.
    """
    n_modes = (p + 1) // 2 if p % 2 == 1 else p // 2 + 1
    assert len(powers) == n_modes, (
        f"powers length {len(powers)} must equal number of frequency modes {n_modes} "
        f"for p={p}"
    )

    fourier_coef_mags = [0.0]
    for k_mode in range(1, len(powers)):
        mag = np.sqrt(powers[k_mode] * p / 2.0) if powers[k_mode] > 0 else 0.0
        fourier_coef_mags.append(mag)

    spectrum = np.zeros(p, dtype=complex)
    spectrum[0] = fourier_coef_mags[0]

    for i_mag in range(1, len(fourier_coef_mags)):
        spectrum[i_mag] = fourier_coef_mags[i_mag]
        spectrum[-i_mag] = np.conj(fourier_coef_mags[i_mag])

    template = np.fft.ifft(spectrum).real
    template = template - np.mean(template)
    template = template.astype(np.float32)

    return template


def fixed_cnxcn(p1, p2, powers):
    """Generate a template for product group C_n x C_n from desired per-mode powers.

    Modes are laid out as ``(1,0), (0,1), (1,1), (2,0), (0,2), (2,2), ...``
    (i.e. cycling through row-only, col-only, diagonal for each frequency
    band).

    Parameters
    ----------
    p1, p2 : int
        p1, p2 in C_{p1} x C_{p2}.
    powers : list of float
        Desired spectral power for each 2D mode (excluding the DC component,
        which is always set to zero).

    Returns
    -------
    template : np.ndarray, shape (p1 * p2,), dtype float32
        Mean-centered, flattened template.
    """
    p_flat = p1 * p2

    fourier_coef_mags = []
    for pw in powers:
        mag = np.sqrt(pw * p_flat / 2.0) if pw > 0 else 0.0
        fourier_coef_mags.append(mag)

    spectrum = np.zeros((p1, p2), dtype=complex)
    spectrum[0, 0] = 0.0

    def mode_selector(i_mag):
        i_mode = 1 + i_mag // 3
        mode_type = i_mag % 3
        if mode_type == 0:
            return (i_mode, 0)
        elif mode_type == 1:
            return (0, i_mode)
        else:
            return (i_mode, i_mode)

    for i_mag, mag in enumerate(fourier_coef_mags):
        mode = mode_selector(i_mag)
        spectrum[mode[0], mode[1]] = mag
        spectrum[-mode[0], -mode[1]] = np.conj(mag)

    template = np.fft.ifft2(spectrum).real
    template = template.flatten()
    template = template - np.mean(template)
    template = template.astype(np.float32)

    return template


def fixed_group(group, powers):
    """Generate a template for a group from desired per-irrep power values.

    Each entry in *powers* specifies the desired spectral power for the
    corresponding irrep.  The function converts these to Fourier coefficient
    diagonal values, builds the spectrum, and returns the mean-centered
    template.

    Parameters
    ----------
    group : Group (escnn object)
        The group.
    powers : list of float
        Desired spectral power for each irrep (one entry per irrep).

    Returns
    -------
    template : np.ndarray, shape (group.order(),), dtype float32
        The mean-centered template.
    """
    group_order = group.order()
    irreps = group.irreps()
    irrep_dims = [ir.size for ir in irreps]

    assert len(powers) == len(irreps), (
        f"powers must have {len(irreps)} values (one per irrep), got {len(powers)}"
    )

    fourier_coef_diag_values = [
        np.sqrt(group_order * p / dim**2) if p > 0 else 0.0
        for p, dim in zip(powers, irrep_dims)
    ]

    spectrum = []
    for i, irrep in enumerate(irreps):
        diag_val = fourier_coef_diag_values[i]
        mat = np.zeros((irrep.size, irrep.size), dtype=float)
        np.fill_diagonal(mat, np.full(irrep.size, diag_val, dtype=float))
        spectrum.append(mat)

    template = fourier.group_fourier_inverse(group, spectrum)
    template = template - np.mean(template)
    template = template.astype(np.float32)

    return template




def template_selector(config):
    """Select template based on configuration."""
    if config["template_type"] == "custom_fourier":
        if config["group_name"] == "cnxcn":
            template = fixed_cnxcn(config["p1"], config["p2"], config["powers"])
        elif config["group_name"] == "cn":
            template = fixed_cn(config["group_n"], config["powers"])
        else:
            template = fixed_group(config["group"], config["powers"])
    elif config["template_type"] == "one_hot":
        template = one_hot(config["group_size"])
    else:
        raise ValueError(f"Unknown template type: {config['template_type']}")
    return template


def mnist_1d(p: int, label: int, root: str = "data", axis: int = 0):
    """Return a (p,) 1D template from a random MNIST image by taking a slice or projection.

    Args:
        p: dimension of the cyclic group
        label: MNIST digit class (0-9)
        root: MNIST data directory
        axis: 0 for row average, 1 for column average, 2 for diagonal

    Returns:
        template: (p,) array
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
    x_new = np.linspace(0, 1, p)
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




def gaussian_1d(p: int, n_gaussians: int = 3, sigma_range: tuple = (0.5, 2.0), seed=None):
    """Generate 1D template as sum of Gaussians.

    Args:
        p: dimension of cyclic group
        n_gaussians: number of Gaussian bumps
        sigma_range: (min_sigma, max_sigma) for Gaussian widths
        seed: random seed

    Returns:
        template: (p,) real-valued array
    """
    rng = np.random.default_rng(seed)
    x = np.arange(p)
    template = np.zeros(p, dtype=np.float32)

    for _ in range(n_gaussians):
        center = rng.uniform(0, p)
        sigma = rng.uniform(*sigma_range)
        amplitude = rng.uniform(0.5, 1.0)

        dist = np.minimum(np.abs(x - center), p - np.abs(x - center))
        template += amplitude * np.exp(-(dist**2) / (2 * sigma**2))

    template -= template.mean()
    s = template.std()
    if s > 1e-12:
        template /= s

    return template.astype(np.float32)


def onehot_1d(p: int):
    """Generate 1D one-hot template for cyclic group C_p.

    Args:
        p: dimension of cyclic group

    Returns:
        template: (p,) array with template[0] = 1, all others = 0
    """
    template = np.zeros(p, dtype=np.float32)
    template[0] = 1.0
    return template


# --- 2D Synthetic Templates ---


def gaussian_mixture_2d(
    p1=20,
    p2=20,
    n_blobs=8,
    frac_broad=0.7,
    sigma_broad=(3.5, 6.0),
    sigma_narrow=(1.0, 2.0),
    amp_broad=1.0,
    amp_narrow=0.5,
    seed=None,
    normalize=True,
):
    """Build a (p1 x p2) template as a periodic mixture of Gaussians."""
    rng = np.random.default_rng(seed)
    H, W = p1, p2
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    k_broad = int(round(n_blobs * frac_broad))
    k_narrow = n_blobs - k_broad

    def add_blobs(k, sigma_range, amp):
        out = np.zeros((H, W), dtype=float)
        for _ in range(k):
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
            sigma = rng.uniform(*sigma_range)
            dy = np.minimum(np.abs(Y - cy), H - np.abs(Y - cy))
            dx = np.minimum(np.abs(X - cx), W - np.abs(X - cx))
            out += amp * np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
        return out

    template = add_blobs(k_broad, sigma_broad, amp_broad) + add_blobs(
        k_narrow, sigma_narrow, amp_narrow
    )

    if normalize:
        template -= template.mean()
        s = template.std()
        if s > 1e-12:
            template /= s
    return template.astype(np.float32)



def _fft_indices(n):
    """Return integer-like frequency indices aligned with numpy's FFT layout."""
    k = np.fft.fftfreq(n) * n
    return k.astype(int)


def hexagon_tie_2d(p1: int, p2: int, k0: float = 6.0, amp: float = 1.0):
    """Real template with hexagonal Fourier spectrum.

    Args:
        p1, p2: spatial dims
        k0: desired radius (index units)
        amp: amplitude per spike

    Returns:
        template: (p1, p2) real-valued array
    """
    assert p1 > 5 and p2 > 5, "p1 and p2 must be > 5"
    spec = np.zeros((p1, p2), dtype=np.complex128)

    thetas = np.arange(6) * (np.pi / 3.0)

    Kx = _fft_indices(p1)
    Ky = _fft_indices(p2)

    def put(kx, ky, val):
        spec[int(kx) % p1, int(ky) % p2] += val

    used = set()
    for th in thetas:
        kx_f = k0 * np.cos(th)
        ky_f = k0 * np.sin(th)
        kx = int(np.round(kx_f))
        ky = int(np.round(ky_f))
        if (kx, ky) == (0, 0):
            if abs(np.cos(th)) > abs(np.sin(th)):
                kx = 1 if kx >= 0 else -1
            else:
                ky = 1 if ky >= 0 else -1
        if (kx, ky) in used:
            continue
        used.add((kx, ky))
        used.add((-kx, -ky))

        put(kx, ky, amp)
        put(-kx, -ky, np.conjugate(amp))

    spec[0, 0] = 0.0

    x = np.fft.ifft2(spec).real
    return x


def ring_isotropic_2d(
    p1: int, p2: int, r0: float = 6.0, sigma: float = 0.5, total_power: float = 1.0
):
    """Real template with an isotropic ring in the 2D spectrum.

    Args:
        p1, p2: spatial dims
        r0: target radius (index units)
        sigma: radial width of the ring
        total_power: scales overall energy

    Returns:
        template: (p1, p2) real-valued array
    """
    assert p1 > 5 and p2 > 5, "p1 and p2 must be > 5"

    kx = _fft_indices(p1)[:, None]
    ky = _fft_indices(p2)[None, :]
    R = np.sqrt(kx**2 + ky**2)

    mag = np.exp(-0.5 * ((R - r0) / max(sigma, 1e-6)) ** 2)

    mag[0, 0] = 0.0

    power = np.sum(mag**2)
    if power > 0:
        mag *= np.sqrt(total_power / power)

    spec = mag.astype(np.complex128)

    x = np.fft.ifft2(spec).real
    return x


def gaussian_2d(
    p1: int,
    p2: int,
    center: tuple = None,
    sigma: float = 2.0,
    k_freqs: int = None,
) -> np.ndarray:
    """Generate 2D template with a single Gaussian, optionally band-limited.

    Args:
        p1: height dimension
        p2: width dimension
        center: (cx, cy) center position
        sigma: standard deviation of Gaussian
        k_freqs: if not None, keep only the top k frequencies by power

    Returns:
        template: (p1, p2) real-valued array
    """
    if center is None:
        center = (p1 / 2, p2 / 2)
    cx, cy = center
    x = np.arange(p1)
    y = np.arange(p2)
    X, Y = np.meshgrid(x, y, indexing="ij")
    template = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
    if k_freqs is not None:
        spectrum = np.fft.fft2(template)
        power = np.abs(spectrum) ** 2
        power_flat = power.flatten()
        kx_indices = np.arange(p1)
        ky_indices = np.arange(p2)
        KX, KY = np.meshgrid(kx_indices, ky_indices, indexing="ij")
        all_indices = list(zip(KX.flatten(), KY.flatten()))
        sorted_idx = np.argsort(-power_flat)
        top_k_idx = sorted_idx[:k_freqs]
        top_k_freqs = set([all_indices[i] for i in top_k_idx])
        mask = np.zeros((p1, p2), dtype=complex)
        for kx, ky in top_k_freqs:
            mask[kx, ky] = 1.0
        spectrum_masked = spectrum * mask
        template = np.fft.ifft2(spectrum_masked).real
    return template
