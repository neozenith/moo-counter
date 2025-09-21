"""Core game engine abstraction and implementations."""

# Standard Library
import time
from typing import Protocol, runtime_checkable
from .moo_types import (
    DIRECTIONS,
    BoardState,
    Direction,
    Grid,
    GridDimensions,
    Moove,
    MooveDirection,
    MooveOverlapGraph,
    MooveSequence,
    Position,
    SimulationResult,
)
from .engines.python_engine import PythonEngine

@runtime_checkable
class GameEngine(Protocol):
    """Protocol for game engine implementations.

    This allows for different implementations (Python, Rust, C) to be swapped
    at runtime while maintaining the same interface.
    """

    @property
    def name(self) -> str:
        """Return the name of this engine implementation."""
        ...

    def is_valid_moove(self, moove: Moove, grid: Grid) -> bool:
        """Check if three positions form a valid 'moo' move."""
        ...

    def generate_moove(self, start: Position, direction: Direction) -> Moove:
        """Generate a 'moo' move given a starting position and direction."""
        ...

    def generate_all_valid_mooves(self, grid: Grid) -> MooveSequence:
        """Generate all possible 'moo' moves on the grid."""
        ...

    def generate_overlaps_graph(self, mooves: MooveSequence) -> MooveOverlapGraph:
        """Generate a graph of overlapping mooves."""
        ...

    def do_mooves_overlap(self, m1: Moove, m2: Moove) -> bool:
        """Check if two mooves overlap."""
        ...

    def generate_empty_board(self, dims: GridDimensions) -> BoardState:
        """Generate an empty board."""
        ...

    def get_moove_coverage(self, board: BoardState, moove: Moove) -> int:
        """Get the number of positions already covered by a moove."""
        ...

    def update_board_with_moove(self, board: BoardState, moo_count: int, moove: Moove) -> tuple[BoardState, int, int]:
        """Update the board with a moove and return new state, count, and coverage gain."""
        ...

    def simulate_board(self, mooves: MooveSequence, dims: GridDimensions) -> SimulationResult:
        """Simulate a complete game with a sequence of mooves."""
        ...

    def get_grid_dimensions(self, grid: Grid) -> GridDimensions:
        """Get the dimensions of a grid."""
        ...

    def benchmark(self, grid: Grid, iterations: int = 1000) -> float:
        """Benchmark this engine's performance."""
        ...



class EngineFactory:
    """Factory for creating and managing game engines."""

    _engines: dict[str, GameEngine] = {"python": PythonEngine()}
    _default_engine: str = "python"

    @classmethod
    def register_engine(cls, name: str, engine: GameEngine):
        """Register a new engine implementation."""
        cls._engines[name.lower()] = engine

    @classmethod
    def get_engine(cls, name: str | None = None) -> GameEngine:
        """Get an engine by name, or the default engine."""
        if name is None:
            name = cls._default_engine

        name = name.lower()
        if name not in cls._engines:
            raise ValueError(f"Unknown engine: {name}. Available: {list(cls._engines.keys())}")

        return cls._engines[name]

    @classmethod
    def set_default_engine(cls, name: str):
        """Set the default engine."""
        if name.lower() not in cls._engines:
            raise ValueError(f"Unknown engine: {name}")
        cls._default_engine = name.lower()

    @classmethod
    def list_engines(cls) -> list[str]:
        """List all available engines."""
        return list(cls._engines.keys())

    @classmethod
    def benchmark_all_engines(cls, grid: Grid, iterations: int = 1000) -> dict[str, float]:
        """Benchmark all registered engines."""
        results = {}
        for name, engine in cls._engines.items():
            print(f"\nBenchmarking {name} engine...")
            results[name] = engine.benchmark(grid, iterations)
        return results




