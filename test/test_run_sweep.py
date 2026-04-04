"""
Tests for src/run_sweep.py

Unit tests exercise config loading, experiment generation,
parameter grid expansion, and helper utilities.

Integration tests run actual sweeps with minimal test configs
to verify the end-to-end pipeline.

Expected runtime: < 1 minute total

Usage:
    pytest test/test_run_sweep.py -v
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.run_sweep import (
    deep_merge_dict,
    expand_parameter_grid,
    generate_experiment_configs,
    generate_experiment_name,
    load_sweep_config,
)

TEST_DIR = Path(__file__).parent
SWEEP_CONFIGS = {
    "example": TEST_DIR / "test_sweep_example.yaml",
    "learning_rate": TEST_DIR / "test_sweep_learning_rate.yaml",
    "model_size": TEST_DIR / "test_sweep_model_size.yaml",
    "onehot_grid": TEST_DIR / "test_sweep_onehot_grid.yaml",
}


# ---------------------------------------------------------------------------
# Unit tests (always run)
# ---------------------------------------------------------------------------


class TestDeepMergeDict:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge_dict(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = deep_merge_dict(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 2}}
        deep_merge_dict(base, override)
        assert base["a"]["x"] == 1


class TestExpandParameterGrid:
    def test_single_param(self):
        grid = {"data": {"p": [5, 10]}}
        combos = expand_parameter_grid(grid)
        assert len(combos) == 2
        assert combos[0] == {"data": {"p": 5}}
        assert combos[1] == {"data": {"p": 10}}

    def test_cartesian_product(self):
        grid = {"data": {"p": [5, 7], "k": [2, 3]}}
        combos = expand_parameter_grid(grid)
        assert len(combos) == 4

    def test_nested_grid(self):
        grid = {"data": {"p": [5]}, "model": {"hidden_dim": [8, 16]}}
        combos = expand_parameter_grid(grid)
        assert len(combos) == 2

    def test_scalar_treated_as_single_value(self):
        grid = {"data": {"p": 5}}
        combos = expand_parameter_grid(grid)
        assert len(combos) == 1
        assert combos[0] == {"data": {"p": 5}}


class TestGenerateExperimentName:
    def test_simple(self):
        overrides = {"data": {"p": 10}, "model": {"hidden_dim": 64}}
        name = generate_experiment_name(overrides)
        assert "p10" in name
        assert "h64" in name

    def test_empty_overrides(self):
        name = generate_experiment_name({})
        assert name == ""


class TestLoadSweepConfig:
    def test_loads_example(self):
        config = load_sweep_config(str(SWEEP_CONFIGS["example"]))
        assert "_base_config" in config
        assert "experiments" in config
        assert config["n_seeds"] == 1
        assert "data" in config["_base_config"]

    def test_loads_grid(self):
        config = load_sweep_config(str(SWEEP_CONFIGS["onehot_grid"]))
        assert "_base_config" in config
        assert "parameter_grid" in config

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_sweep_config("/nonexistent/sweep.yaml")


class TestGenerateExperimentConfigs:
    def test_explicit_experiments(self):
        config = load_sweep_config(str(SWEEP_CONFIGS["example"]))
        experiments = generate_experiment_configs(config)
        assert len(experiments) == 2
        names = [name for name, _ in experiments]
        assert "hidden_dim_4" in names
        assert "hidden_dim_8" in names

    def test_grid_experiments(self):
        config = load_sweep_config(str(SWEEP_CONFIGS["onehot_grid"]))
        experiments = generate_experiment_configs(config)
        # p: [5, 7], k: [2], hidden_dim: [8] -> 2 combos
        assert len(experiments) == 2

    def test_global_overrides_applied(self):
        config = load_sweep_config(str(SWEEP_CONFIGS["example"]))
        experiments = generate_experiment_configs(config)
        for _, exp_config in experiments:
            assert exp_config["device"] == "cpu"
            assert exp_config["training"]["epochs"] == 2

    def test_learning_rate_configs(self):
        config = load_sweep_config(str(SWEEP_CONFIGS["learning_rate"]))
        experiments = generate_experiment_configs(config)
        assert len(experiments) == 2
        names = [name for name, _ in experiments]
        assert "adam_lr_1e-2" in names
        assert "hybrid_scale_-3" in names

    def test_model_size_configs(self):
        config = load_sweep_config(str(SWEEP_CONFIGS["model_size"]))
        experiments = generate_experiment_configs(config)
        assert len(experiments) == 2


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_all_plots():
    """Mock produce_plots and plt.savefig/close to skip visualization."""
    import src.main as main  # noqa: F401

    with (
        patch("src.main.produce_plots") as mock_plots,
        patch("matplotlib.pyplot.savefig") as mock_savefig,
        patch("matplotlib.pyplot.close") as mock_close,
    ):
        yield {
            "produce_plots": mock_plots,
            "savefig": mock_savefig,
            "close": mock_close,
        }


def _get_repo_root():
    """Get the repository root directory."""
    return Path(__file__).parent.parent


def _run_sweep_and_check(sweep_config_path, mock_all_plots, expected_experiments):
    """Helper: run a sweep, assert all experiments completed successfully."""
    from src.run_sweep import run_parameter_sweep

    with tempfile.TemporaryDirectory() as tmpdir:
        # run_parameter_sweep creates sweep_results/ relative to cwd.
        # We chdir to tmpdir so output goes there, but the sweep config
        # uses relative paths like "src/configs/config.yaml" resolved from repo root.
        # Fix: rewrite base_config to absolute path before running.
        import yaml

        with open(sweep_config_path) as f:
            sweep_data = yaml.safe_load(f)

        repo_root = _get_repo_root()
        abs_base = str(repo_root / sweep_data["base_config"])
        sweep_data["base_config"] = abs_base

        patched_config = Path(tmpdir) / "sweep_config.yaml"
        with open(patched_config, "w") as f:
            yaml.dump(sweep_data, f)

        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            run_parameter_sweep(str(patched_config), gpu_ids=[None])
        finally:
            os.chdir(original_cwd)

        # Check that sweep_results directory was created
        sweep_results_dir = Path(tmpdir) / "sweep_results"
        assert sweep_results_dir.exists(), "sweep_results/ directory not created"

        # Find the sweep directory (timestamped)
        sweep_dirs = list(sweep_results_dir.iterdir())
        assert len(sweep_dirs) == 1, f"Expected 1 sweep dir, got {len(sweep_dirs)}"
        sweep_dir = sweep_dirs[0]

        # Check metadata
        metadata_path = sweep_dir / "sweep_metadata.yaml"
        assert metadata_path.exists(), "sweep_metadata.yaml not found"

        # Check summary
        summary_path = sweep_dir / "sweep_summary.yaml"
        assert summary_path.exists(), "sweep_summary.yaml not found"

        with open(summary_path) as f:
            summary = yaml.safe_load(f)

        assert summary["total_experiments"] == expected_experiments
        assert summary["total_successful_runs"] == expected_experiments
        assert summary["total_failed_runs"] == 0

        # Check each experiment has a directory with results
        for exp_name in summary["experiment_statistics"]:
            exp_dir = sweep_dir / exp_name
            assert exp_dir.exists(), f"Experiment dir {exp_name} not found"
            exp_summary = sweep_dir / exp_name / "experiment_summary.yaml"
            assert exp_summary.exists(), f"experiment_summary.yaml not found for {exp_name}"


def test_sweep_example(mock_all_plots):
    """Run full sweep end-to-end with example config (2 explicit experiments)."""
    _run_sweep_and_check(SWEEP_CONFIGS["example"], mock_all_plots, expected_experiments=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
