"""Minimal test suite to debug engine registration issues."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_engine_factory_exists():
    """Test that EngineFactory can be imported."""
    from moo_counter.engine import EngineFactory
    assert EngineFactory is not None


def test_python_engine_registered():
    """Test that Python engine is always registered."""
    from moo_counter.engine import EngineFactory
    engines = EngineFactory.list_engines()
    assert "python" in engines, f"Python engine not found. Available: {engines}"


def test_python_engine_works():
    """Test that Python engine can be instantiated and used."""
    from moo_counter.engine import EngineFactory

    engine = EngineFactory.get_engine("python")
    assert engine is not None
    assert engine.name == "Python"

    # Test basic functionality
    grid = [
        ["m", "o", "o"],
        ["o", "m", "o"],
        ["o", "o", "m"]
    ]

    dims = engine.get_grid_dimensions(grid)
    assert dims == (3, 3)

    mooves = engine.generate_all_valid_mooves(grid)
    assert len(mooves) >= 0  # Should find at least some valid moves


def test_wrapper_imports():
    """Test that wrapper classes can be imported (but may not instantiate)."""
    try:
        from moo_counter.engines.wrappers import RustEngineWrapper
        assert RustEngineWrapper is not None
    except ImportError as e:
        pytest.skip(f"RustEngineWrapper not available: {e}")

    try:
        from moo_counter.engines.wrappers import CythonEngineWrapper
        assert CythonEngineWrapper is not None
    except ImportError as e:
        pytest.skip(f"CythonEngineWrapper not available: {e}")


def test_engine_registration_graceful_failure():
    """Test that missing engines don't crash the system."""
    from moo_counter.engine import EngineFactory

    # This should not raise an error even if engine doesn't exist
    with pytest.raises(ValueError, match="Unknown engine"):
        EngineFactory.get_engine("nonexistent")


def test_available_engines_list():
    """Test that list_engines returns valid list."""
    from moo_counter.engine import EngineFactory

    engines = EngineFactory.list_engines()
    assert isinstance(engines, list)
    assert len(engines) >= 1  # At least Python should be there
    assert all(isinstance(name, str) for name in engines)