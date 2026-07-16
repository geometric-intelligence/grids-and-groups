<h1 align="center">The Algebra of Spatial Navigation</h1>

<h3 align="center">Exact and learned recurrent networks for path integration over finite groups</h3>

<p align="center">
  <a href="https://github.com/geometric-intelligence/grids-and-groups/actions/workflows/ci.yml"><img src="https://github.com/geometric-intelligence/grids-and-groups/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
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

<p align="center">
  <b>Daniel Kunin &middot; Christopher J. Kymn &middot; Francisco Acosta &middot; Giovanni Luca Marchetti &middot; Nina Miolane</b>
</p>

---

> **How can a recurrent neural circuit integrate a sequence of local, egocentric movements into a global, allocentric representation of position and orientation?**

This repository studies path integration as **sequential group composition**. A recurrent network receives an allocentric population code together with egocentric transformations and must maintain the allocentric code of their cumulative product:

\[
\left(x_{\mathrm{allo}},\; g_1\!\cdot x_{\mathrm{ego}},\ldots,g_T\!\cdot x_{\mathrm{ego}}\right)
\longmapsto
(g_T\cdots g_1)\!\cdot x_{\mathrm{allo}}.
\]

The group \(G\) specifies the geometry of the navigated space. Circular groups model head direction, product groups model periodic translations, and semidirect products model coupled rotations and translations in two and three dimensions.

The repository supports two complementary approaches:

1. **Constructed networks:** use finite-group Fourier analysis to derive QuadraticRNN weights that solve the task exactly when all irreducible representations are included.
2. **Trained networks:** learn group composition by gradient descent and analyze the resulting loss plateaus, Fourier content, recurrent structure, and neural tuning.

## Overview

### Algebraic formulation

For a finite group \(G\), an encoding \(x\in\mathbb R^{|G|}\) is equivalently a scalar function \(x:G\to\mathbb R\). Group elements act by permuting its coordinates through the regular action.

The recurrent model uses a squared-ReLU activation,

\[
\sigma(z)=\operatorname{ReLU}(z)^2,
\]

and updates

\[
\begin{aligned}
h_1 &= \sigma\!\left(W_{\mathrm{in}}x_{\mathrm{allo}}
      +W_{\mathrm{drive}}(g_1\!\cdot x_{\mathrm{ego}})\right),\\
h_t &= \sigma\!\left(W_{\mathrm{mix}}h_{t-1}
      +W_{\mathrm{drive}}(g_t\!\cdot x_{\mathrm{ego}})\right),\\
y_t &= W_{\mathrm{out}}h_t.
\end{aligned}
\]

The closed-form construction decomposes the computation into modules indexed by irreducible representations of \(G\). The same representation-theoretic quantities are used to analyze networks learned by gradient descent.

### Navigation groups

| Group | Interpretation | Status |
| --- | --- | --- |
| \(C_n\) | Circular variable or head direction | Training infrastructure |
| \(C_n\times C_m\) | Periodic planar translations | Trained sequential notebook |
| \(\mathbb Z_n^2\rtimes C_m\) | Discrete planar rigid motion | Trained and constructed notebooks |
| \(\mathbb Z_n^3\rtimes O\) | Discrete volumetric motion with 24 proper cubic rotations | Constructed notebook |

The general training stack also includes cyclic, product-cyclic, dihedral, octahedral, and icosahedral benchmark groups.

### Main capabilities

- finite-group Fourier transforms, inverse transforms, and power spectra;
- dense and lazy irreducible representations;
- exact closed-form QuadraticRNN construction;
- cost-aware Fourier truncation for larger navigation groups;
- factored recurrent mixing without a dense \(H\times H\) matrix;
- offline and online composition datasets;
- MLP and recurrent training;
- loss-plateau and representation-power analysis;
- triangular and cubic spatial encodings;
- position, orientation, trajectory, and tuning-curve visualization;
- parameter sweeps and saved-run analysis.

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

Register the environment as a Jupyter kernel if needed:

```bash
python -m ipykernel install --user \
  --name group-agf \
  --display-name "Python (group-agf)"
```

In Cursor or Jupyter, select **Python (group-agf)**.

## Notebooks

Notebooks are divided into trained and analytically constructed networks. See [`notebooks/README.md`](notebooks/README.md) for detailed descriptions and results.

### Trained networks

| Notebook | Purpose |
| --- | --- |
| [`sequential_cnxcn.ipynb`](notebooks/trained_networks/sequential_cnxcn.ipynb) | Train a QuadraticRNN on length-three composition in \(C_3\times C_3\) |
| [`discrete_se2.ipynb`](notebooks/trained_networks/discrete_se2.ipynb) | Compare an MLP and QuadraticRNN on discrete SE(2) |
| [`discrete_se2_rnn.ipynb`](notebooks/trained_networks/discrete_se2_rnn.ipynb) | Main end-to-end discrete-SE(2) training experiment |
| [`discrete_se2_analysis.ipynb`](notebooks/trained_networks/discrete_se2_analysis.ipynb) | Analyze checkpoints and parameter histories without retraining |
| [`discrete_se2_local_composition.ipynb`](notebooks/trained_networks/discrete_se2_local_composition.ipynb) | Test whether locally trained composition generalizes globally |

### Constructed networks

| Notebook | Purpose |
| --- | --- |
| [`rnn_constructed_discrete_SE2_m3.ipynb`](notebooks/constructed_networks/rnn_constructed_discrete_SE2_m3.ipynb) | Exact and Fourier-truncated QuadraticRNNs on \(\mathbb Z_n^2\rtimes C_3\) |
| [`rnn_constructed_discrete_SE3.ipynb`](notebooks/constructed_networks/rnn_constructed_discrete_SE3.ipynb) | Exact and cost-aware truncated QuadraticRNNs on \(\mathbb Z_n^3\rtimes O\) |

The constructed notebooks distinguish between:

- a small **all-irrep verification**, which demonstrates exact group composition to floating-point precision; and
- a larger **Fourier-truncated experiment**, which studies the trade-off between network width and reconstruction quality.

## Usage

### Run a configured training experiment

```bash
conda activate group-agf
python -m src.main --config src/configs/config_d5.yaml
```

Outputs include loss histories, checkpoints, parameter snapshots, and representation-power analyses.

### Run a parameter sweep

```bash
python -m src.run_sweep \
  --sweep src/sweep_configs/example_sweep.yaml
```

For multiple GPUs:

```bash
python -m src.run_sweep \
  --sweep src/sweep_configs/example_sweep.yaml \
  --gpus auto
```

### Construct an exact finite-group RNN

The group-agnostic construction lives in `src/finite_group_rnn.py`:

```python
import numpy as np

from src.finite_group_rnn import (
    build_finite_group_rnn,
    random_invertible_encoding,
    rollout,
)
from src.groups import DiscreteSE2Group

group = DiscreteSE2Group(n=2, m=3)
irreps = group.irreps()

x_allo = np.random.default_rng(0).normal(size=group.order)
x_ego = random_invertible_encoding(group, irreps, seed=1)

params = build_finite_group_rnn(
    group,
    x_ego,
    irrep_selection="all",
    materialize_mix=False,
)

sequence = [
    group.encode(1, 0, 0),
    group.encode(0, 0, 1),
]
result = rollout(params, x_allo, sequence)
```

With `materialize_mix=False`, recurrent mixing is applied as

\[
W_{\mathrm{mix}}h=W_{\mathrm{in}}(W_{\mathrm{out}}h),
\]

avoiding the storage cost of a dense hidden-by-hidden matrix.

## Repository structure

```text
grids-and-groups/
├── notebooks/
│   ├── trained_networks/          # Networks learned by gradient descent
│   ├── constructed_networks/      # Closed-form representation-theoretic RNNs
│   └── README.md                  # Notebook guide and experimental results
├── src/
│   ├── groups/                    # Finite groups and irreducible representations
│   ├── configs/                   # Training configurations
│   ├── sweep_configs/             # Parameter-sweep configurations
│   ├── finite_group_rnn.py        # Closed-form QuadraticRNN construction
│   ├── discrete_se2_geometry.py   # Triangular geometry and SE(2) decoding
│   ├── discrete_se3_geometry.py   # Cubic geometry and SE(3) pose decoding
│   ├── model.py                   # TwoLayerMLP and QuadraticRNN
│   ├── dataset.py                 # Group-composition datasets
│   ├── template.py                # Population-code construction
│   ├── train.py                   # Training loops
│   ├── optimizer.py               # Custom optimizers
│   ├── viz.py                     # Fourier and learned-network visualizations
│   ├── main.py                    # Configured experiment entry point
│   └── run_sweep.py               # Sweep entry point
├── test/                          # Unit, integration, and notebook tests
├── conda.yaml                     # Conda environment
├── pyproject.toml                 # Package and tool configuration
└── poetry.lock                    # Locked Python dependencies
```

### Key modules

- **`src/groups/`** — group laws, regular actions, character orbits, induced irreps, and Fourier analysis.
- **`src/finite_group_rnn.py`** — analytical weights, Fourier selection, factored recurrence, rollout, and hidden-state probes.
- **`src/discrete_se2_geometry.py`** — triangular periodic distance, spatial bumps, direction alignment, and center decoding.
- **`src/discrete_se3_geometry.py`** — cubic periodic geometry, anisotropic landmarks, position/orientation decoding, and trajectory plots.
- **`src/model.py`** — trainable feedforward and recurrent architectures.
- **`src/dataset.py`** — sampled and exhaustive sequential-composition datasets.

## Testing

Run the complete test suite:

```bash
conda activate group-agf
pytest -q
```

Run the analytical RNN tests:

```bash
pytest test/test_finite_group_rnn.py -q
```

Run notebook execution tests:

```bash
NOTEBOOK_TEST_MODE=1 pytest test/test_notebooks.py -q
```

Run lint checks:

```bash
ruff check .
```

## Current experimental landmarks

- The all-irrep discrete-SE(2) and discrete-SE(3) constructions reproduce mixed group actions to floating-point precision.
- The budgeted \(n=3\) discrete-SE(3) construction reduces hidden width from 71,040 to 3,360 while retaining approximately 48.6% of the encoding's Fourier power.
- In the local-composition SE(2) experiment, near-perfect local fitting does not generalize to the full group law.

These numbers are notebook-scale reference experiments, not benchmark claims.

## Manuscript

This repository accompanies the manuscript draft:

> **The Algebra of Spatial Navigation**
> Daniel Kunin, Christopher J. Kymn, Francisco Acosta, Giovanni Luca Marchetti, and Nina Miolane.

```bibtex
@unpublished{kunin2026algebra,
  title  = {The Algebra of Spatial Navigation},
  author = {Kunin, Daniel and Kymn, Christopher J. and Acosta, Francisco and Marchetti, Giovanni Luca and Miolane, Nina},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
