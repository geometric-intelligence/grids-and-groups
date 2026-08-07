"""Geometry adapters for interpreting finite-group signals spatially.

The submodules are intentionally domain-specific:

- :mod:`src.geometry.cnxcn` for square periodic translations;
- :mod:`src.geometry.discrete_se2` for triangular translations and headings;
- :mod:`src.geometry.discrete_se3` for cubic translations and orientations.

They provide signal encodings, decoders, geometric error metrics, motion
sequences, and plots used to evaluate the geometry-agnostic RNN construction.
"""
