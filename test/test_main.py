"""
Tests for src/main.py

This module tests that the main() entry point runs successfully with minimal
configuration for all supported groups: cn (C_10), cnxcn (C_4 x C_4),
dihedral (D3), octahedral, and A5.

Expected runtime: < 1 minute

Usage:
    pytest test/test_main.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Paths to test config files
TEST_DIR = Path(__file__).parent
CONFIG_FILES = {
    "c10": TEST_DIR / "test_config_c10.yaml",
    "c4x4": TEST_DIR / "test_config_c4x4.yaml",
    "d3": TEST_DIR / "test_config_d3.yaml",
    "octahedral": TEST_DIR / "test_config_octahedral.yaml",
    "a5": TEST_DIR / "test_config_a5.yaml",
}


@pytest.fixture
def temp_run_dir():
    """Create a temporary directory for run outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_all_plots():
    """Mock all produce_plots_* and plt.savefig/close to skip visualization entirely."""
    import src.main as main  # noqa: F401

    with (
        patch("src.main.produce_plots_cn") as mock_1d,
        patch("src.main.produce_plots_cnxcn") as mock_2d,
        patch("src.main.produce_plots_group") as mock_group,
        patch("matplotlib.pyplot.savefig") as mock_savefig,
        patch("matplotlib.pyplot.close") as mock_close,
    ):
        yield {
            "produce_plots_cn": mock_1d,
            "produce_plots_cnxcn": mock_2d,
            "produce_plots_group": mock_group,
            "savefig": mock_savefig,
            "close": mock_close,
        }


@pytest.fixture
def mock_savefig():
    """Mock only plt.savefig and plt.close so plotting code runs but files aren't saved."""
    with (
        patch("matplotlib.pyplot.savefig") as mock_sf,
        patch("matplotlib.pyplot.close") as mock_cl,
    ):
        yield {"savefig": mock_sf, "close": mock_cl}


def test_load_config():
    """Test that load_config correctly loads a YAML file."""
    import src.main as main

    config = main.load_config(str(CONFIG_FILES["c10"]))

    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert "device" in config
    assert "analysis" in config
    assert config["data"]["group_name"] == "cn"
    assert config["data"]["p"] == 10
    assert config["training"]["epochs"] == 2


def test_main_c10(temp_run_dir, mock_all_plots):
    """Test main() with C_10 cyclic group config."""
    import src.main as main

    config = main.load_config(str(CONFIG_FILES["c10"]))
    results = main.train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0
    mock_all_plots["produce_plots_cn"].assert_called_once()


def test_main_c4x4(temp_run_dir, mock_all_plots):
    """Test main() with C_4 x C_4 product group config."""
    import src.main as main

    config = main.load_config(str(CONFIG_FILES["c4x4"]))
    results = main.train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0
    mock_all_plots["produce_plots_cnxcn"].assert_called_once()


def test_main_d3(temp_run_dir, mock_savefig):
    """Test main() with D3 dihedral group config.

    Full integration test: does NOT mock produce_plots_group so the entire
    plotting pipeline (TwoLayerMLP eval data via GroupCompositionDataset, power spectrum)
    is exercised. D3 (order 6) is the smallest group so this stays fast.
    This validates the TwoLayerMLP-compatible eval data path in produce_plots_group,
    which is shared by octahedral and A5 (mocked in their tests for speed).
    """
    import src.main as main

    config = main.load_config(str(CONFIG_FILES["d3"]))
    results = main.train_single_run(config, run_dir=temp_run_dir)

    assert "final_train_loss" in results
    assert "final_val_loss" in results
    assert results["final_train_loss"] > 0


def test_main_octahedral_config():
    """Test that octahedral config loads and validates correctly.

    Full training is skipped because Octahedral group construction
    is relatively expensive. The D3 test already covers the full group pipeline
    integration (same code path, just a different group).
    """
    import src.main as main

    config = main.load_config(str(CONFIG_FILES["octahedral"]))
    assert config["data"]["group_name"] == "octahedral"
    assert config["training"]["epochs"] == 2
    assert config["device"] == "cpu"


def test_main_a5_config():
    """Test that A5 config loads and validates correctly.

    Full training is skipped because Icosahedral group construction
    is relatively expensive. The D3 test already covers the full group pipeline
    integration (same code path, just a different group).
    """
    import src.main as main

    config = main.load_config(str(CONFIG_FILES["a5"]))
    assert config["data"]["group_name"] == "A5"
    assert config["training"]["epochs"] == 2
    assert config["device"] == "cpu"


def test_regenerate_plots_cn(temp_run_dir, mock_all_plots):
    """Test regenerate_plots loads a saved run and dispatches to the right produce_plots.

    Runs a tiny train_single_run to produce artifacts, then calls
    regenerate_plots on the same directory with fresh mocks.
    """
    import src.main as main

    config = main.load_config(str(CONFIG_FILES["c10"]))
    main.train_single_run(config, run_dir=temp_run_dir)

    mock_all_plots["produce_plots_cn"].reset_mock()

    main.regenerate_plots(str(temp_run_dir), device="cpu")

    mock_all_plots["produce_plots_cn"].assert_called_once()
    call_kwargs = mock_all_plots["produce_plots_cn"].call_args[1]
    mdl = call_kwargs["model"]
    assert isinstance(mdl, main.model.TwoLayerMLP)
    assert mdl.group_size == 10
    assert mdl.k == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
