"""Engine implementations for moo-counter."""

# Try to import Cython engine if available
try:
    from .cython_engine import CythonEngine
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    CythonEngine = None

# Try to import Rust engine if available
try:
    from moo_counter_rust import RustEngine
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustEngine = None

__all__ = ["CythonEngine", "RustEngine", "CYTHON_AVAILABLE", "RUST_AVAILABLE"]