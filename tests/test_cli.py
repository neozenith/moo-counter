"""Test command-line interface functionality."""

import subprocess
import sys
from pathlib import Path


def test_cli_python_engine():
    """Test that CLI works with Python engine."""
    result = subprocess.run(
        [sys.executable, "-m", "moo_counter.moo_counter",
         "--puzzle", "micro",
         "--strategy", "greedy-high",
         "--iterations", "1",
         "--engine", "python"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "Moo Counter - PYTHON Engine" in result.stdout
    assert "RESULTS" in result.stdout


def test_cli_help():
    """Test that CLI help works."""
    result = subprocess.run(
        [sys.executable, "-m", "moo_counter.moo_counter", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    assert result.returncode == 0
    assert "--puzzle" in result.stdout
    assert "--engine" in result.stdout


def test_cli_invalid_engine():
    """Test that CLI rejects invalid engine."""
    result = subprocess.run(
        [sys.executable, "-m", "moo_counter.moo_counter",
         "--puzzle", "micro",
         "--engine", "nonexistent"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr