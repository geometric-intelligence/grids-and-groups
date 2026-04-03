#!/usr/bin/env python
"""One-time script: extract irrep and regular-rep matrices from escnn.

Saves .npy files into src/groups/data/ that are loaded at import time
by oh.py and a5.py.

Usage:
    conda activate group-agf
    python scripts/extract_escnn_data.py
"""

from pathlib import Path

import numpy as np
from escnn.group import Icosahedral, Octahedral


DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "groups" / "data"


def extract_group(escnn_group, prefix: str):
    """Extract and save all group data as .npy files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    elements = list(escnn_group.elements)
    irreps = escnn_group.irreps()

    print(f"  Order: {escnn_group.order()}")
    print(f"  Irreps: {len(irreps)}  dims={[ir.size for ir in irreps]}")
    print(f"  Irrep names: {[str(ir) for ir in irreps]}")

    names = [str(ir) for ir in irreps]
    np.save(DATA_DIR / f"{prefix}_irrep_names.npy", np.array(names))

    for i, irrep in enumerate(irreps):
        mats = np.array([irrep(g) for g in elements])
        path = DATA_DIR / f"{prefix}_irrep_{i}.npy"
        np.save(path, mats)
        print(f"  Saved {path.name}  shape={mats.shape}")

    regular_rep = escnn_group.representations["regular"]
    reg_mats = np.array([regular_rep(g) for g in elements])
    path = DATA_DIR / f"{prefix}_regular_rep.npy"
    np.save(path, reg_mats)
    print(f"  Saved {path.name}  shape={reg_mats.shape}")


def main():
    print("Extracting Octahedral ...")
    extract_group(Octahedral(), "oh")

    print("\nExtracting Icosahedral (A5) ...")
    extract_group(Icosahedral(), "a5")

    print(f"\nAll files saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
