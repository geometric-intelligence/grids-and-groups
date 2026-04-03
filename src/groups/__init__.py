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
]
