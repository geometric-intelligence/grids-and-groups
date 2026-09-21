# Local-vs-global SGC training

This sandbox compares two training distributions for a quadratic RNN on
`C_n^2 \rtimes C_6` while evaluating both models on the same fixed local and
global validation sets throughout training.

Nothing in `src/` or `trained_networks/` is modified. The experiment imports
the local translation and rotation support used by
`src.experiments.discrete_se2`, so the definition stays tied to the constructed
network analysis.

For both distributions, the first input is uniform on the whole group. Only
the subsequent egocentric increments differ:

- `global`: each increment is uniform on the whole group;
- `local`: each increment is uniform on
  `experiment.local_egocentric_elements` (the 7 local translations crossed
  with rotations `{-1, 0, 1}`).

Each run writes `config.json`, `metrics.csv`, and `final_model.pt` to its own
output directory. `metrics.csv` contains the held-out local and global MSE and
exact decoded-product accuracy at every evaluation checkpoint.

`--supervision-stride L` applies the training loss every `L` recurrent updates
and always at the final update. Evaluation still records dense all-prefix,
supervised-checkpoint, and final-step metrics separately.

Example:

```bash
python experiments/local_global_sgc/run_experiment.py \
  --train-distribution local \
  --seed 0 \
  --device cuda \
  --output-dir /data/facosta/grids-and-groups/local_global_sgc/local_seed0
```

The initial pilot uses `n=4`, `m=6`, three inputs per trajectory (one initial
state and two increments), all-prefix supervision, Adam, and paired seeds for
the local/global models.
