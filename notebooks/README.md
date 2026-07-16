# Notebooks

The notebooks are organized by how the recurrent network is obtained:

```text
notebooks/
├── trained_networks/
│   ├── sequential_cnxcn.ipynb
│   ├── discrete_se2.ipynb
│   ├── discrete_se2_rnn.ipynb
│   ├── discrete_se2_analysis.ipynb
│   └── discrete_se2_local_composition.ipynb
└── constructed_networks/
    ├── rnn_constructed_discrete_SE2_m3.ipynb
    └── rnn_constructed_discrete_SE3.ipynb
```

The notebooks in `trained_networks/` learn group composition from data using PyTorch. The notebooks in `constructed_networks/` derive the weights analytically from finite-group irreducible representations; no gradient training is used.

## Overview

| Notebook | Method | Purpose |
| --- | --- | --- |
| [`trained_networks/sequential_cnxcn.ipynb`](trained_networks/sequential_cnxcn.ipynb) | Trained QuadraticRNN | Sequential composition on \(C_3\times C_3\) with sequence length three |
| [`trained_networks/discrete_se2.ipynb`](trained_networks/discrete_se2.ipynb) | Trained MLP and QuadraticRNN | Initial discrete-SE(2) prototype and architecture comparison |
| [`trained_networks/discrete_se2_rnn.ipynb`](trained_networks/discrete_se2_rnn.ipynb) | Trained QuadraticRNN | Main end-to-end discrete-SE(2) training experiment |
| [`trained_networks/discrete_se2_analysis.ipynb`](trained_networks/discrete_se2_analysis.ipynb) | Post-hoc analysis | Load and analyze saved discrete-SE(2) runs without retraining |
| [`trained_networks/discrete_se2_local_composition.ipynb`](trained_networks/discrete_se2_local_composition.ipynb) | Trained QuadraticRNN | Test whether local generator compositions generalize globally |
| [`constructed_networks/rnn_constructed_discrete_SE2_m3.ipynb`](constructed_networks/rnn_constructed_discrete_SE2_m3.ipynb) | Closed-form QuadraticRNN | Exact and Fourier-truncated constructions on \(\mathbb Z_n^2\rtimes C_3\) |
| [`constructed_networks/rnn_constructed_discrete_SE3.ipynb`](constructed_networks/rnn_constructed_discrete_SE3.ipynb) | Closed-form QuadraticRNN | Exact and cost-aware truncated constructions on \(\mathbb Z_n^3\rtimes O\) |

## Trained networks

### `sequential_cnxcn.ipynb`

This notebook trains a `QuadraticRNN` to compose sequences of three elements from \(C_3\times C_3\). It builds a template with controlled Fourier structure, trains a 200-unit model, compares the loss curve with theoretical Fourier plateaus, and visualizes predictions.

The saved 10,000-epoch run reaches a loss of approximately \(0.0436\).

### `discrete_se2.ipynb`

The original integrated experiment for \(\mathbb Z_6^2\rtimes C_3\). It:

- constructs the exhaustive binary-composition dataset;
- visualizes the Cayley table and irreducible representations;
- trains a `TwoLayerMLP` and a `QuadraticRNN`;
- compares their learning curves; and
- examines learned representation power and output weights.

In the saved run, the MLP plateaus near \(0.176\), while the QuadraticRNN reaches approximately \(2.6\times10^{-4}\). This notebook remains useful for the direct MLP-versus-RNN comparison.

### `discrete_se2_rnn.ipynb`

The main training notebook for a QuadraticRNN on \(\mathbb Z_{10}^2\rtimes C_3\). It covers:

- translation-character orbits and irreps;
- the exhaustive 90,000-pair composition dataset;
- a 3,000-unit QuadraticRNN;
- parameter snapshots and Fourier power over training; and
- detailed output-weight analysis.

The saved run reaches a loss of approximately \(0.0238\). Some cells near the end contain unfinished exploratory analysis and should not be assumed to execute unchanged.

### `discrete_se2_analysis.ipynb`

A post-hoc companion to `discrete_se2_rnn.ipynb`. It loads configuration, template, losses, parameter history, and a final checkpoint from an existing run. It then reproduces loss, Fourier-power, output-weight, and hidden-tuning analyses without retraining.

Use this notebook for saved runs and parameter sweeps.

### `discrete_se2_local_composition.ipynb`

A local-to-global generalization experiment. The model is trained on short compositions formed from unit translations and rotations, then evaluated on the complete group law.

The saved experiment reports:

- local loss: approximately \(1.23\times10^{-5}\);
- global loss: approximately \(1.61\times10^{-1}\); and
- global/local ratio: approximately 13,126.

The model fits local compositions but does not infer global composition.

## Constructed networks

The constructed notebooks share the group-agnostic implementation in `src/finite_group_rnn.py`. Geometry and decoding live in:

- `src/discrete_se2_geometry.py`;
- `src/discrete_se3_geometry.py`.

Recurrent mixing is kept factored as

\[
W_{\mathrm{mix}}h=W_{\mathrm{in}}(W_{\mathrm{out}}h),
\]

which avoids allocating a dense hidden-by-hidden matrix.

### `rnn_constructed_discrete_SE2_m3.ipynb`

This notebook constructs a QuadraticRNN analytically for

\[
\mathbb Z_n^2\rtimes C_3.
\]

It separates:

1. a small all-irrep experiment that verifies exact translation, rotation, and mixed composition to floating-point precision; and
2. a moderate \(n=8\) experiment using four high-power irreps.

The truncated experiment includes triangular-lattice geometry, translation-only and rotation-containing rollouts, separate signal and center errors, and static hidden-unit tuning.

### `rnn_constructed_discrete_SE3.ipynb`

This notebook applies the same construction to

\[
\mathbb Z_n^3\rtimes O,
\]

where \(O\) is the 24-element group of proper cubic rotations.

It first uses \(n=2\), all 20 irreps, and 9,312 hidden units. Translation, cubic-rotation, and mixed-composition errors are below \(1.5\times10^{-13}\).

The \(n=3\) experiment selects six irreps under a hidden-width budget:

- selected width: 3,360;
- all-irrep width: 71,040;
- width reduction: approximately 95.3%;
- retained Fourier power: approximately 48.6%.

The three-dimensional encoding makes both position and orientation observable. The notebook reports full-signal, decoded-position, and decoded-rotation errors separately and includes orthogonal-slice, orientation, and trajectory plots.

## Which notebook to use

- For a simple sequential product-group example, use `trained_networks/sequential_cnxcn.ipynb`.
- For the original MLP/RNN comparison, use `trained_networks/discrete_se2.ipynb`.
- For end-to-end discrete-SE(2) training, use `trained_networks/discrete_se2_rnn.ipynb`.
- For analysis of saved runs, use `trained_networks/discrete_se2_analysis.ipynb`.
- For local-to-global generalization, use `trained_networks/discrete_se2_local_composition.ipynb`.
- For the analytical discrete-SE(2) construction, use `constructed_networks/rnn_constructed_discrete_SE2_m3.ipynb`.
- For the analytical discrete-SE(3) construction, use `constructed_networks/rnn_constructed_discrete_SE3.ipynb`.

## Environment

Use the repository environment and select its Jupyter kernel:

```bash
conda activate group-agf
```

Kernel: **Python (group-agf)**
