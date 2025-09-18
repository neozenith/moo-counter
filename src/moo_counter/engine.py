"""Core game engine abstraction and implementations."""

from typing import Protocol, runtime_checkable
import time

from .moo_types import (
    Grid, GridDimensions, BoardState, Position, Direction,
    Moove, MooveSequence, MooveDirection, SimulationResult,
    MooveOverlapGraph, DIRECTIONS
)


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

    def update_board_with_moove(
        self, board: BoardState, moo_count: int, moove: Moove
    ) -> tuple[BoardState, int, int]:
        """Update the board with a moove and return new state, count, and coverage gain."""
        ...

    def simulate_board(
        self, mooves: MooveSequence, dims: GridDimensions
    ) -> SimulationResult:
        """Simulate a complete game with a sequence of mooves."""
        ...

    def get_grid_dimensions(self, grid: Grid) -> GridDimensions:
        """Get the dimensions of a grid."""
        ...

    def benchmark(self, grid: Grid, iterations: int = 1000) -> float:
        """Benchmark this engine's performance."""
        ...


class PythonEngine:
    """Pure Python implementation of the game engine."""

    @property
    def name(self) -> str:
        return "Python"

    def get_grid_dimensions(self, grid: Grid) -> GridDimensions:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        return (rows, cols)

    def is_valid_moove(self, moove: Moove, grid: Grid) -> bool:
        """Check if three positions form a valid 'moo' move.

        Rules:
        - t1 must be 'm' and t2, t3 must be 'o's
        - t2 must be adjacent to t1 (horizontally, vertically, or diagonally)
        - t3 must follow the same direction as t2 from t1
        """
        height, width = self.get_grid_dimensions(grid)
        t1, t2, t3 = moove
        r1, c1 = t1
        r2, c2 = t2
        r3, c3 = t3

        # Check bounds
        if not (0 <= r1 < height and 0 <= c1 < width):
            return False
        if not (0 <= r2 < height and 0 <= c2 < width):
            return False
        if not (0 <= r3 < height and 0 <= c3 < width):
            return False

        # Check letters spell 'moo'
        if grid[r1][c1] != "m" or grid[r2][c2] != "o" or grid[r3][c3] != "o":
            return False

        # Calculate directions
        d1: MooveDirection = (r2 - r1, c2 - c1)
        d2: MooveDirection = (r3 - r2, c3 - c2)

        # Check t2 is adjacent to t1
        if abs(d1[0]) > 1 or abs(d1[1]) > 1:
            return False

        # Check t3 follows same direction from t2
        return d1 == d2

    def generate_moove(self, start: Position, direction: Direction) -> Moove:
        """Generate a 'moo' move given a starting position and direction."""
        d = DIRECTIONS[direction]
        t1 = start
        t2 = (start[0] + d[0], start[1] + d[1])
        t3 = (start[0] + 2 * d[0], start[1] + 2 * d[1])
        return Moove(t1, t2, t3)

    def generate_all_valid_mooves(self, grid: Grid) -> MooveSequence:
        """Generate all possible 'moo' moves on the grid."""
        height, width = self.get_grid_dimensions(grid)
        mooves: MooveSequence = []

        for r in range(height):
            for c in range(width):
                for direction in range(8):
                    moove = self.generate_moove((r, c), direction)
                    if self.is_valid_moove(moove, grid):
                        mooves.append(moove)

        return mooves

    def generate_overlaps_graph(self, mooves: MooveSequence) -> MooveOverlapGraph:
        """Generate a graph of overlapping mooves."""
        overlaps: MooveOverlapGraph = {}

        for i, moove1 in enumerate(mooves):
            for moove2 in mooves[i + 1:]:
                if self.do_mooves_overlap(moove1, moove2):
                    overlaps.setdefault(moove1, set()).add(moove2)
                    overlaps.setdefault(moove2, set()).add(moove1)

        return overlaps

    def do_mooves_overlap(self, m1: Moove, m2: Moove) -> bool:
        """Check if two mooves share any positions."""
        positions1 = set(m1)
        positions2 = set(m2)
        return not positions1.isdisjoint(positions2)

    def generate_empty_board(self, dims: GridDimensions) -> BoardState:
        """Generate an empty board of False values."""
        height, width = dims
        return [[False for _ in range(width)] for _ in range(height)]

    def get_moove_coverage(self, board: BoardState, moove: Moove) -> int:
        """Get the number of positions already covered by a moove."""
        coverage = 0
        for t in moove:
            if board[t[0]][t[1]]:
                coverage += 1
        return coverage

    def update_board_with_moove(
        self, board: BoardState, moo_count: int, moove: Moove
    ) -> tuple[BoardState, int, int]:
        """Update the board with a 'moo' move."""
        t1, t2, t3 = moove
        r1, c1 = t1
        r2, c2 = t2
        r3, c3 = t3

        # Deep copy the board
        output_board = [row[:] for row in board]

        # Determine coverage
        moo_coverage = self.get_moove_coverage(board, moove)

        # Place moove on board
        if moo_coverage < 3:
            output_board[r1][c1] = True
            output_board[r2][c2] = True
            output_board[r3][c3] = True
            moo_count += 1

        coverage_gain = 3 - moo_coverage
        return output_board, moo_count, coverage_gain

    def simulate_board(
        self, mooves: MooveSequence, dims: GridDimensions
    ) -> SimulationResult:
        """Simulate the board with a sequence of 'moo' moves."""
        board = self.generate_empty_board(dims)
        moo_count = 0
        moo_count_sequence = []
        moo_coverage_gain_sequence = []

        for moove in mooves:
            board, moo_count, coverage_gain = self.update_board_with_moove(
                board, moo_count, moove
            )
            moo_count_sequence.append(moo_count)
            moo_coverage_gain_sequence.append(coverage_gain)

        return SimulationResult(
            board=board,
            moo_count=moo_count,
            moove_sequence=mooves,
            moo_count_sequence=moo_count_sequence,
            moo_coverage_gain_sequence=moo_coverage_gain_sequence
        )

    def benchmark(self, grid: Grid, iterations: int = 1000) -> float:
        """Benchmark this engine's performance."""
        dims = self.get_grid_dimensions(grid)
        all_mooves = self.generate_all_valid_mooves(grid)

        # Benchmark simulation speed
        start_time = time.time()
        for _ in range(iterations):
            _ = self.simulate_board(all_mooves[:50], dims)

        duration = time.time() - start_time
        simulations_per_second = iterations / duration

        print(f"[{self.name} Engine] {simulations_per_second:.0f} simulations/second")
        return simulations_per_second


class EngineFactory:
    """Factory for creating and managing game engines."""

    _engines: dict[str, GameEngine] = {}
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


# Register default Python engine
EngineFactory.register_engine("python", PythonEngine())