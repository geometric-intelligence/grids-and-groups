# Refactoring Notes: Unifying cn/cnxcn/group Function Triplicates

## Overview

This refactoring eliminated legacy code duplication where functions existed
in three variants (`*_cn`, `*_cnxcn`, `*_group`) due to the codebase
predating the unified `Group` class hierarchy. Every set of triplicates has
been collapsed into a single `*_group` function that accepts a `Group`
object.

## Removed Functions

### `template.py`

| Removed            | Replacement                  |
| ------------------ | ---------------------------- |
| `fixed_cn`         | `custom_fourier(group, powers)` |
| `fixed_cnxcn`      | `custom_fourier(group, powers)` |

**Behavioral difference**: `fixed_cn` accepted `(n+1)//2` powers (one per
real-FFT frequency bin with Hermitian folding); `fixed_cnxcn` accepted an
arbitrary number of 2D mode powers with a `mode_selector` layout.
`custom_fourier` accepts exactly `len(group.irreps())` powers (one per irrep).
For `CyclicGroup(N)` that is `N` values; for `ProductCyclicGroup(p1, p2)`
that is `p1 * p2` values. Config files that used the old `powers` lists
must be updated to provide one power per irrep.

### `power.py`

| Removed                           | Replacement                                       |
| --------------------------------- | ------------------------------------------------- |
| `loss_plateau_predictions_cyclic` | `loss_plateau_predictions(template, group)`        |
| `loss_plateau_predictions_group`  | `loss_plateau_predictions(template, group)`        |
| `powers_per_neuron_rows_cyclic`   | `powers_per_neuron_rows(W, group)`                 |

**Normalization fix**: `Group.power_spectrum` returns `|F[k]|^2` (total
sums to `|G| * ||x||^2` by Plancherel). The old cyclic code used
`|rfft(x)|^2 / N` with Hermitian doubling (total sums to `||x||^2`). The
unified `loss_plateau_predictions` divides the group power spectrum by `|G|`
to match the MSE normalization (`nn.MSELoss` averages over the output
dimension `|G|`). This correction was verified in
`test/test_refactor_equivalence.py::TestPowerNormalization`.

The old `loss_plateau_predictions_group` was off by a factor of `|G|` for
**all** groups (including D_n, Oh, A5). The new function produces correct
MSE plateau predictions.

`model_power_over_time` now takes `(group, model, param_history,
model_inputs)` instead of `(group_name, model, ...)`. It uses
`group.power_spectrum` uniformly, so the power values are in group
convention (not FFT-normalized). Downstream consumers that only compare
ratios (e.g., dominant-mode fraction) are unaffected by the scale
difference.

### `dataset.py`

| Removed       | Replacement                                           |
| ------------- | ----------------------------------------------------- |
| `_build_cn`   | `_build_group(template, k, group, ...)`               |
| `_build_cnxcn`| `_build_group(template, k, group, ...)`               |

**Equivalence**: `_build_cn` used `np.roll` and `_build_cnxcn` used 2D
`np.roll`. `_build_group` uses `group.regular_rep()` (matrix
multiplication). For `CyclicGroup` and `ProductCyclicGroup`, these produce
identical results as verified in
`test/test_refactor_equivalence.py::TestDatasetGroupPath`.

`GroupCompositionDataset` now takes a `Group` object as its first argument
instead of a string `group_name`. For online mode, it dispatches to
`_OnlineModularAdditionDataset1D` / `2D` via `isinstance` checks on
`CyclicGroup` / `ProductCyclicGroup`.

### `viz.py`

| Removed                | Replacement                |
| ---------------------- | -------------------------- |
| `plot_power_cn`        | `plot_power_group`         |
| `plot_power_cnxcn`     | `plot_power_group`         |
| `plot_predictions_1d`  | `plot_predictions_group`   |
| `plot_predictions_2d`  | `plot_predictions_group`   |

`plot_predictions_group` uses a bar-chart representation that works for any
group order. The 2D heatmap view for `ProductCyclicGroup` is no longer
generated (the bar-chart is used uniformly).

`compute_w_dominant_irrep_fraction_data`, `plot_w_dominant_irrep_fraction`,
`plot_loss_power_and_weight_power`, `maybe_save_w_dominant_irrep_fraction_npz`,
and `load_w_dominant_irrep_fraction_for_run_dir` all accept a `Group`
object instead of `group_name`/`group_size`/`p1`/`p2` parameters.

### `main.py`

| Removed               | Replacement       |
| --------------------- | ----------------- |
| `produce_plots_cn`    | `produce_plots`   |
| `produce_plots_cnxcn` | `produce_plots`   |
| `produce_plots_group` | `produce_plots`   |

`train_single_run` now calls `make_group(group_name, config)` at the top
and threads the `group` object through template generation, dataset
creation, and plotting.

`regenerate_plots` uses `make_group` + the unified `produce_plots`.

## New Functions

### `src/groups/__init__.py :: make_group(group_name, config)`

Factory function that maps `(group_name, config)` to the appropriate
`Group` subclass:

- `"cn"` → `CyclicGroup(N=config["data"]["p"])`
- `"cnxcn"` → `ProductCyclicGroup(p1=config["data"]["p1"], p2=config["data"]["p2"])`
- `"dihedral"` → `DihedralGroup(N=config["data"].get("group_n", 3))`
- `"octahedral"` → `OctahedralGroup()`
- `"A5"` → `IcosahedralGroup()`

### `src/power.py :: loss_plateau_predictions(template, group)`

Unified MSE loss plateau prediction. Divides `group.power_spectrum` by
`|G|` for correct MSE normalization.

## Online Datasets

`_OnlineModularAdditionDataset1D` and `_OnlineModularAdditionDataset2D`
are **not** generalized. They exploit modular addition structure for GPU
throughput and remain as performance specializations gated by `isinstance`
checks.

## Test Coverage

`test/test_refactor_equivalence.py` verifies:

1. `_build_group` produces identical datasets for `CyclicGroup` and
   `ProductCyclicGroup` (formerly verified against `_build_cn`/`_build_cnxcn`).
2. The `|G|` normalization relationship between `Group.power_spectrum` and
   the old FFT-based power.
3. `loss_plateau_predictions` with the `/|G|` correction matches the old
   cyclic plateaus.
4. `powers_per_neuron_rows` total power matches the cyclic version after
   normalization.
5. `custom_fourier` produces valid templates for all group types.

All 117 tests pass after the refactoring.
