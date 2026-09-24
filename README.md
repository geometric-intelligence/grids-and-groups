<h1 align="center">The Algebra of Spatial Navigation</h1>

<h3 align="center">Exact and learned recurrent networks for path integration over finite groups</h3>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3120/"><img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> &bull;
  <a href="#installation">Install</a> &bull;
  <a href="#notebooks">Notebooks</a> &bull;
  <a href="#usage">Usage</a> &bull;
  <a href="#testing">Testing</a>
</p>

---

> **How can a recurrent neural circuit integrate a sequence of local, egocentric movements into a global, allocentric representation of position and orientation?**

This repository studies path integration as **sequential group composition**. A recurrent network receives an allocentric population code together with egocentric transformations and must maintain the allocentric code of their cumulative product:

$$
\left(x_{\mathrm{allo}},\; g_1 \cdot x_{\mathrm{ego}},\ldots,g_T \cdot x_{\mathrm{ego}}\right)
\longmapsto
(g_T\cdots g_1) \cdot x_{\mathrm{allo}}.
$$

The group $G$ specifies the geometry of the navigated space. Circular groups model head direction, product groups model periodic translations, and semidirect products model coupled rotations and translations in two and three dimensions.

The repository supports two complementary approaches:

1. **Constructed networks:** use finite-group Fourier analysis to derive QuadraticRNN weights that solve the task exactly when all irreducible representations are included.
2. **Trained networks:** learn group composition by gradient descent and analyze the resulting loss plateaus, Fourier content, recurrent structure, and neural tuning.

## Overview

### Algebraic formulation

For a finite group $G$, an encoding $x\in\mathbb R^{|G|}$ is equivalently a scalar function $x:G\to\mathbb R$. Group elements act by permuting its coordinates through the regular action.

The recurrent model uses a squared-ReLU activation,

$$
\sigma(z)=\mathrm{ReLU}(z)^2,
$$

and updates

$$
\begin{aligned}
h_1 &= \sigma \left(W_{\mathrm{in}}x_{\mathrm{allo}}
      +W_{\mathrm{drive}}(g_1 \cdot x_{\mathrm{ego}})\right),\\
h_t &= \sigma \left(W_{\mathrm{mix}}h_{t-1}
      +W_{\mathrm{drive}}(g_t \cdot x_{\mathrm{ego}})\right),\\
y_t &= W_{\mathrm{out}}h_t.
\end{aligned}
$$

The closed-form construction decomposes the computation into modules indexed by irreducible representations of $G$. The same representation-theoretic quantities are used to analyze networks learned by gradient descent.

### Navigation groups

| Group | Interpretation | Status |
| --- | --- | --- |
| $C_n$ | Circular variable or head direction | Training infrastructure |
| $C_n\times C_m$ | Periodic planar translations | Trained sequential notebook |
| $\mathbb Z_n^2\rtimes C_m$ | Discrete planar rigid motion (Discrete SE(2)) | Trained and constructed notebooks |
| $\mathbb Z_n^3\rtimes O$ | Discrete volumetric motion with 24 proper cubic rotations (Discrete SE(3)) | Constructed notebook |

The general training stack also includes cyclic, product-cyclic, dihedral, octahedral, and icosahedral benchmark groups.


## Installation

### Prerequisite

- [Conda](https://docs.conda.io/) or Miniconda

### Setup

```bash
git clone git@github.com:geometric-intelligence/grids-and-groups.git
cd grids-and-groups

conda env create -f conda.yaml
conda activate group-agf
poetry install
```


## Notebooks

Notebooks give analytically constructed networks. See [`notebooks/README.md`](notebooks/README.md) for a short navigation note.

| Notebook | Purpose |
| --- | --- |
| [`rnn_constructed_cnxcn.ipynb`](notebooks/constructed_networks/rnn_constructed_cnxcn.ipynb) | Exact and Fourier-truncated translation RNNs on $C_n\times C_n$ |
| [`rnn_constructed_discrete_se2_c6.ipynb`](notebooks/constructed_networks/rnn_constructed_discrete_se2_c6.ipynb) | C6 construction, regular actions, and naturalistic rollout |
| [`rnn_constructed_discrete_se2_c6_tuning.ipynb`](notebooks/constructed_networks/rnn_constructed_discrete_se2_c6_tuning.ipynb) | Empirical and theoretical tuning comparisons |
| [`rnn_constructed_discrete_se2_c6_manifolds.ipynb`](notebooks/constructed_networks/rnn_constructed_discrete_se2_c6_manifolds.ipynb) | Fixed-point module manifolds and persistent homology |
| [`rnn_constructed_discrete_SE3.ipynb`](notebooks/constructed_networks/rnn_constructed_discrete_SE3.ipynb) | Exact and cost-aware truncated QuadraticRNNs on $\mathbb Z_n^3\rtimes O$ |



## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
