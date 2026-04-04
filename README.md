<h1 align="center">Sequential Group Composition</h1>

<h3 align="center">A Window into the Mechanics of Deep Learning</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2602.03655"><img src="https://img.shields.io/badge/arXiv-2602.03655-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/geometric-intelligence/group-agf/actions/workflows/ci.yml"><img src="https://github.com/geometric-intelligence/group-agf/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.03655">Paper</a> &bull;
  <a href="https://arxiv.org/pdf/2602.03655">PDF</a> &bull;
  <a href="#installation">Install</a> &bull;
  <a href="#usage">Usage</a> &bull;
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <b>Giovanni Luca Marchetti* &middot; Daniel Kunin* &middot; Adele Myers &middot; Francisco Acosta &middot; Nina Miolane</b>
</p>

---

> **How do neural networks trained over sequences acquire the ability to perform structured operations, such as arithmetic, geometric, and algorithmic computation?**
>
> We introduce the *sequential group composition* task: networks receive a sequence of elements from a finite group encoded in a real vector space and must predict their cumulative product. We prove that two-layer networks learn this task **one irreducible representation at a time**, in an order determined by the Fourier statistics of the encoding -- producing a characteristic *staircase* in the training loss.

<p align="center">
  <img src="assets/binary_composition.png" width="100%" alt="Staircase learning across five finite groups">
</p>
<p align="center">
  <em><b>Two-layer networks learn group composition one irreducible representation at a time.</b> Top: power spectrum of the learned function over training. Bottom: training loss showing a characteristic staircase. Each column is a different finite group.</em>
</p>

---

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda)

### Setup (Linux)

```bash
# Install gfortran (required by numerical dependencies)
sudo apt install -y gfortran

# Create and activate the conda environment
conda env create -f conda.yaml
conda activate group-agf

# Install all Python dependencies (pinned versions from poetry.lock)
poetry install
```

### Setup (macOS)

On macOS several dependencies require extra steps because (a) `gfortran` and `libomp` (OpenMP) are not bundled with Apple's toolchain, and (b) PyTorch ≤ 2.2 requires NumPy < 2.

```bash
# 1. Install build prerequisites via Homebrew
brew install gcc        # provides gfortran
brew install libomp     # OpenMP runtime (needed by py3nj / escnn)

# 2. Create and activate the conda environment
conda env create -f conda.yaml
conda activate group-agf

# 3. Install lie-learn from source (the PyPI release fails on Python 3.12+)
pip install cython
pip install --no-build-isolation git+https://github.com/AMLab-Amsterdam/lie_learn.git

# 4. Build py3nj with OpenMP made optional
#    (Apple clang + Rosetta can't always locate libomp for Meson builds)
pip download py3nj --no-binary :all: -d /tmp/py3nj_src --no-deps --no-build-isolation
cd /tmp && tar xzf /tmp/py3nj_src/py3nj-*.tar.gz
sed -i '' "s/dependency('openmp')/dependency('openmp', required: false)/" /tmp/py3nj-*/meson.build
pip install --no-build-isolation /tmp/py3nj-*/
cd -

# 5. Install all Python dependencies (without poetry.lock, which is Linux-specific)
pip install torch torchvision 'numpy<2' scipy matplotlib jupyter jupyterlab \
    pandas scikit-image wandb jupyterlab-code-formatter pytest ipykernel \
    pykernel scikit-learn tqdm seaborn jupyter-black ruff pre-commit escnn
```

> **Note (macOS):**
> - PyTorch on x86_64 macOS is capped at 2.2.x and requires `numpy<2`.
> - `py3nj` is built without OpenMP parallelization; this affects only the speed of Wigner symbol computation, not correctness.
> - The `lie-learn` data files (`J_dense_0-150.npy`, `J_block_0-150.npy`) are downloaded automatically on first use by `escnn`. If they are missing, run:
>   ```bash
>   python -c "
>   import os, requests
>   target = os.path.join(os.path.dirname(__import__('lie_learn').representations.SO3.pinchon_hoggan.__file__))
>   base = 'https://github.com/AMLab-Amsterdam/lie_learn/raw/master/lie_learn/representations/SO3/pinchon_hoggan'
>   for f in ['J_dense_0-150.npy', 'J_block_0-150.npy']:
>       p = os.path.join(target, f)
>       if not os.path.exists(p):
>           open(p, 'wb').write(requests.get(f'{base}/{f}').content)
>   "
>   ```

## Usage

### Single Run

Train a model on a specific group:

```bash
python src/main.py --config src/configs/config_d5.yaml
```

Results (loss curves, predictions, power spectra) are saved to a timestamped directory under `runs/`.

### Supported Groups

The repository includes preconfigured experiments for eight groups:

| Group | Config | Order | k | Architecture |
|:------|:-------|:-----:|:-:|:-------------|
| Cyclic $C_{10}$ | `config_c10_k3.yaml` | 10 | 3 | TwoLayerMLP |
| Cyclic $C_{11}$ | `config_c11.yaml` | 11 | 2 | TwoLayerMLP |
| Product $C_4 \times C_4$ | `config_c4x4_k3.yaml` | 16 | 3 | TwoLayerMLP |
| Product $C_5 \times C_5$ | `config_c5xc5.yaml` | 25 | 2 | TwoLayerMLP |
| Dihedral $D_3$ | `config_d3.yaml` | 6 | 2 | TwoLayerMLP |
| Dihedral $D_5$ | `config_d5.yaml` | 10 | 2 | TwoLayerMLP |
| Octahedral $O_h$ | `config_oh.yaml` | 24 | 2 | TwoLayerMLP |
| Icosahedral $A_5$ | `config_a5.yaml` | 60 | 2 | TwoLayerMLP |

### Reproduce Paper's Figure

Reproduce the paper's figure (training loss + power spectrum for C11, C5xC5, D5, Oh, A5):

```bash
python src/main.py --combined-plot
```

This uses precomputed data from `runs_data/` and produces `combined_loss_and_power.pdf` in seconds. No GPU or training is needed.

If you want to retrain from scratch instead, delete `runs_data/` first — the command will automatically detect CUDA, train each group, and generate the plot.

### Parameter Sweeps

Run experiments across multiple configurations and random seeds:

```bash
python src/run_sweep.py --sweep src/sweep_configs/example_sweep.yaml
```

Multi-GPU support:

```bash
# Auto-detect and use all available GPUs
python src/run_sweep.py --sweep src/sweep_configs/example_sweep.yaml --gpus auto

# Use specific GPUs
python src/run_sweep.py --sweep src/sweep_configs/example_sweep.yaml --gpus 0,1,2,3
```

Sweep results are saved to `sweeps/{sweep_name}_{timestamp}/` with per-seed results and aggregated summaries.

## Configuration

Key parameters in the YAML config files:

| Parameter | Options | Description |
|:----------|:--------|:------------|
| `data.group_name` | `cn`, `cnxcn`, `dihedral`, `octahedral`, `A5` | Group to learn |
| `data.k` | integer | Number of elements to compose |
| `data.template_type` | `custom_fourier`, `onehot`, `mnist` | Template generation method |
| `model.model_type` | `QuadraticRNN`, `TwoLayerMLP` | Architecture |
| `model.hidden_dim` | integer | Hidden layer size |
| `model.init_scale` | float | Weight initialization scale |
| `training.optimizer` | `auto`, `adam`, `per_neuron`, `hybrid` | Optimizer (`auto` recommended) |
| `training.learning_rate` | float | Base learning rate |
| `training.mode` | `online`, `offline` | Training mode |
| `training.epochs` | integer | Number of epochs (offline mode) |

<details>
<summary><b>Example config -- D5 with custom Fourier template</b></summary>

```yaml
data:
  group_name: dihedral
  group_n: 5
  k: 2
  template_type: custom_fourier
  powers: [0.0, 3000.0, 2000.0, 1000.0]

model:
  model_type: TwoLayerMLP
  hidden_dim: 300
  init_scale: 0.0001

training:
  optimizer: per_neuron
  learning_rate: 0.006
  mode: offline
  epochs: 5000
```

</details>

## Repository Structure

```
group-agf/
├── src/                          # Source code
│   ├── main.py                   # Training entry point (CLI)
│   ├── model.py                  # TwoLayerMLP, QuadraticRNN
│   ├── optimizer.py              # PerNeuronScaledSGD, HybridRNNOptimizer
│   ├── dataset.py                # Dataset generation and loading
│   ├── template.py               # Template construction functions
│   ├── viz.py                    # Plotting, visualization, and power spectrum helpers
│   ├── train.py                  # Training loops (offline and online)
│   ├── run_sweep.py              # Parameter sweep runner
│   ├── sweep_analysis.py         # Sweep result loading and analysis
│   ├── groups/                   # Finite group implementations
│   │   ├── group.py              # Abstract Group base class (Fourier, power spectrum)
│   │   ├── cn.py                 # CyclicGroup (C_n)
│   │   ├── cnxcn.py              # ProductCyclicGroup (C_n × C_m)
│   │   ├── dn.py                 # DihedralGroup (D_n)
│   │   ├── oh.py                 # OctahedralGroup (O_h)
│   │   ├── a5.py                 # AlternatingGroup (A_5 / icosahedral)
│   │   └── irrep.py              # IrreducibleRepresentation helper
│   ├── configs/                  # Group-specific configurations
│   │   └── config_*.yaml
├── runs_data/                    # Precomputed data for combined plot (Figure 1)
│   ├── {C11,C5xC5,D5,Oh,A5}/
│   │   ├── config.yaml
│   │   ├── train_loss_history.npy
│   │   └── power_data.npz
├── test/                         # Unit and integration tests
├── notebooks/                    # Jupyter notebooks for exploration
├── pyproject.toml                # Project metadata and dependencies
├── poetry.lock                   # Pinned dependency versions
└── conda.yaml                    # Conda environment specification
```

<details>
<summary><b>Module details</b></summary>

### `model.py` -- Neural Network Architectures

| Model | Description | Input |
|:------|:------------|:------|
| **TwoLayerMLP** | Two-layer feedforward network with configurable nonlinearity (square, relu, tanh, gelu) | Flattened binary pair `(N, 2 * group_size)` |
| **QuadraticRNN** | Recurrent network: `h_t = (W_mix h_{t-1} + W_drive x_t)^2` | Sequence `(N, k, p)` |

### `optimizer.py` -- Custom Optimizers

| Optimizer | Description | Recommended for |
|:----------|:------------|:----------------|
| **PerNeuronScaledSGD** | SGD with per-neuron learning rate scaling exploiting model homogeneity | TwoLayerMLP |
| **HybridRNNOptimizer** | Scaled SGD for MLP weights + Adam for recurrent weights | QuadraticRNN |
| Adam (PyTorch built-in) | Standard Adam | QuadraticRNN |

### `groups/` -- Finite Group Implementations

- **`Group`** (abstract base class) -- defines the interface (`order`, `elements`, `irreps`, `regular_rep`) and provides concrete Fourier analysis methods: `fourier`, `inverse_fourier`, `power_spectrum`
- **`CyclicGroup`** ($C_n$), **`ProductCyclicGroup`** ($C_n \times C_m$), **`DihedralGroup`** ($D_n$), **`OctahedralGroup`** ($O_h$), **`AlternatingGroup`** ($A_5$)

### `dataset.py` -- Data Generation

- **`GroupCompositionDataset`** -- PyTorch map-style dataset for offline group composition; supports arbitrary sequence length `k`, sampled/exhaustive mode, and `return_all_outputs`
- **`_OnlineModularAdditionDataset1D`**, **`_OnlineModularAdditionDataset2D`** -- generate samples on-the-fly (GPU-accelerated) via `__iter__`

### `template.py` -- Template Construction

- `one_hot` -- one-hot encoding with zeroth frequency removed
- `custom_fourier` -- template from desired per-irrep power values
- `make_template` -- config-driven template factory (dispatches to the above)
- `mnist_1d`, `mnist_2d` -- templates derived from MNIST images

### `viz.py` -- Visualization and Power Spectrum Helpers

Plotting and analysis functions for training: `plot_train_loss_with_theory`, `plot_predictions_group`, `plot_power_group`, `plot_wmix_structure`, `plot_irreps`, `plot_combined_loss_and_power`, `plot_loss_power_and_weight_power`, and more. Also includes power spectrum helpers moved from the former `power.py`: `loss_plateau_predictions`, `powers_per_neuron_rows`, `model_power_over_time`, `topk_template_freqs`.

### `train.py` -- Training Loops

- `train(model, loader, criterion, optimizer, ...)` -- epoch-based offline training
- `train_online(model, loader, criterion, optimizer, ...)` -- step-based online training

### `sweep_analysis.py` -- Sweep Result Analysis

Utilities for loading, aggregating, and plotting parameter sweep results: `load_sweep_results_grid`, `load_training_loss_curves`, `export_lightweight_data`, `plot_theory_boundaries`.

</details>

## Testing

```bash
# All tests (unit + integration)
pytest test/ -v

# Notebook tests only (requires jupyter/nbconvert)
NOTEBOOK_TEST_MODE=1 pytest test/test_notebooks.py -v
```

## Development

```bash
# Install pre-commit hooks
pre-commit install

# Run linting
ruff check .
ruff format --check .
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{marchetti2026sequential,
  title   = {Sequential Group Composition: A Window into the Mechanics of Deep Learning},
  author  = {Marchetti, Giovanni Luca and Kunin, Daniel and Myers, Adele and Acosta, Francisco and Miolane, Nina},
  journal = {arXiv preprint arXiv:2602.03655},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
