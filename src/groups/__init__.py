"""Self-contained finite group implementations for Group-AGF."""

from src.groups.a5 import IcosahedralGroup
from src.groups.cn import CyclicGroup
from src.groups.cnxcn import ProductCyclicGroup
from src.groups.dn import DihedralGroup
from src.groups.group import Group
from src.groups.irrep import IrreducibleRepresentation
from src.groups.oh import OctahedralGroup

__all__ = [
    "Group",
    "IrreducibleRepresentation",
    "CyclicGroup",
    "ProductCyclicGroup",
    "DihedralGroup",
    "OctahedralGroup",
    "IcosahedralGroup",
    "make_group",
]


def make_group(group_name: str, config: dict) -> Group:
    """Instantiate the appropriate ``Group`` subclass from a run configuration.

    Parameters
    ----------
    group_name : str
        One of ``'cn'``, ``'cnxcn'``, ``'dihedral'``, ``'octahedral'``, ``'A5'``.
    config : dict
        Experiment config; the ``config["data"]`` sub-dict supplies group
        parameters (``p``, ``p1``/``p2``, ``group_n``, etc.).

    Returns
    -------
    Group
        A concrete ``Group`` instance ready for Fourier analysis and dataset
        construction.
    """
    data = config["data"]
    if group_name == "cn":
        return CyclicGroup(N=data["p"])
    if group_name == "cnxcn":
        return ProductCyclicGroup(p1=data["p1"], p2=data["p2"])
    if group_name == "dihedral":
        return DihedralGroup(N=data.get("group_n", 3))
    if group_name == "octahedral":
        return OctahedralGroup()
    if group_name == "A5":
        return IcosahedralGroup()
    raise ValueError(
        f"Unknown group_name '{group_name}'. "
        "Must be one of: 'cn', 'cnxcn', 'dihedral', 'octahedral', 'A5'."
    )
