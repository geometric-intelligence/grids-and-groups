# Trained networks

This directory contains the optional learned-network side of the project:

- trainable PyTorch models and optimizers;
- group-composition datasets and optimization loops;
- configured experiments and parameter sweeps;
- post-training visualization and checkpoint analysis;
- training notebooks, tests, and fixtures.

Composition datasets use body-frame/right-regular updates by default, so a
sequence ``(g_1, ..., g_T)`` is labeled by ``g_1 * ... * g_T``. Pass
``action_side="left"`` to `GroupCompositionDataset` only for spatial/world-frame
experiments.

The dependency direction is intentionally one-way: `trained_networks` may import the
group theory and signal utilities in `src`, while `src` must not import
`trained_networks`. This keeps the closed-form construction usable independently and
allows this directory to become a separate package or repository later.

Run a configured experiment from the repository root:

```bash
python -m trained_networks.main --config trained_networks/configs/config.yaml
```

Run a sweep:

```bash
python -m trained_networks.run_sweep --sweep trained_networks/sweep_configs/example_sweep.yaml
```

Run the training-specific tests:

```bash
pytest trained_networks/tests -q
```
