"""Self-contained finite group implementations for Group-AGF."""

from src.groups.a5 import IcosahedralGroup
from src.groups.cn import CyclicGroup
from src.groups.cnxcn import ProductCyclicGroup
from src.groups.dn import DihedralGroup
from src.groups.group import Group
from src.groups.irrep import IrreducibleRepresentation, LazyIrreducibleRepresentation
from src.groups.oh import OctahedralGroup
from src.groups.znxzn_cm import DiscreteSE2Group
from src.groups.znxznxzn_oh import DiscreteSE3Group

__all__ = [
    "Group",
    "IrreducibleRepresentation",
    "LazyIrreducibleRepresentation",
    "CyclicGroup",
    "ProductCyclicGroup",
    "DihedralGroup",
    "OctahedralGroup",
    "IcosahedralGroup",
    "DiscreteSE2Group",
    "DiscreteSE3Group",
    "make_group",
]


def make_group(group_name: str, config: dict) -> Group:
    """Instantiate the appropriate ``Group`` subclass from a run configuration."""
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
    if group_name == "znxzn_cm":
        return DiscreteSE2Group(n=data["p"], m=data["m"])
    if group_name == "znxznxzn_oh":
        return DiscreteSE3Group(n=data["p"])
    raise ValueError(
        f"Unknown group_name '{group_name}'. "
        "Must be one of: 'cn', 'cnxcn', 'dihedral', 'octahedral', 'A5', "
        "'znxzn_cm', 'znxznxzn_oh'."
    )
