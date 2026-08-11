"""
Tests for constructed-notebook execution in the group-agf repository.

This module runs the notebooks in ``notebooks/constructed_networks`` to verify
that they execute without errors. Training notebooks and their dependencies
live under ``trained_networks/`` and are intentionally outside this core test.

Usage:
    pytest test/test_notebooks.py -v
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def get_repo_root():
    """Get the repository root directory."""
    return Path(__file__).parent.parent


def get_notebooks_dir():
    """Get the notebooks directory."""
    return get_repo_root() / "notebooks" / "constructed_networks"


# Notebooks to skip (with reasons)
SKIP_NOTEBOOKS = {
    # Add notebooks here if they need to be skipped, e.g.:
    # "notebook_name": "Reason for skipping",
}


def get_notebook_files():
    """Get list of all notebook files in the notebooks directory."""
    notebooks_dir = get_notebooks_dir()
    if not notebooks_dir.exists():
        return []
    return sorted(notebooks_dir.glob("*.ipynb"))


# Get list of notebooks for parametrization
NOTEBOOKS = get_notebook_files()
NOTEBOOK_IDS = [nb.stem for nb in NOTEBOOKS]


@pytest.fixture(scope="module")
def notebook_test_env():
    """Set up environment for notebook testing."""
    env = os.environ.copy()
    env["NOTEBOOK_TEST_MODE"] = "1"
    repo_root = str(get_repo_root())
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def execute_notebook(notebook_path, env):
    """
    Execute a Jupyter notebook using nbconvert.

    Args:
        notebook_path: Path to the notebook file
        env: Environment dictionary for the subprocess

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.ipynb")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--ExecutePreprocessor.timeout=300",  # 5 minute timeout per notebook
                    "--ExecutePreprocessor.kernel_name=python3",
                    "--output",
                    output_path,
                    str(notebook_path),
                ],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(get_repo_root()),
                timeout=360,  # 6 minute overall timeout
            )

            if result.returncode != 0:
                error_msg = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                return False, error_msg
            return True, None

    except subprocess.TimeoutExpired:
        return False, "Notebook execution timed out (>6 minutes)"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_notebook_execution(notebook_path, notebook_test_env):
    """
    Test that a notebook executes without errors.
    """
    notebook_name = notebook_path.stem

    # Skip notebooks with known issues
    if notebook_name in SKIP_NOTEBOOKS:
        pytest.skip(f"Skipped: {SKIP_NOTEBOOKS[notebook_name]}")

    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"

    success, error_msg = execute_notebook(notebook_path, notebook_test_env)

    if not success:
        pytest.fail(f"Notebook {notebook_path.name} failed to execute:\n{error_msg}")


def test_notebooks_directory_exists():
    """Test that the notebooks directory exists."""
    notebooks_dir = get_notebooks_dir()
    assert notebooks_dir.exists(), f"Notebooks directory not found: {notebooks_dir}"


def test_at_least_one_notebook_exists():
    """Test that there is at least one notebook to test."""
    notebooks = get_notebook_files()
    assert len(notebooks) > 0, "No notebooks found in notebooks/ directory"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
