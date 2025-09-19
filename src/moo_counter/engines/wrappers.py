"""Python wrappers for Rust and Cython engines to match the GameEngine protocol."""

from ..moo_types import (
    Grid, GridDimensions, BoardState, Position, Direction,
    Moove, MooveSequence, MooveOverlapGraph, SimulationResult
)


class RustEngineWrapper:
    """Wrapper for Rust engine to match GameEngine protocol."""

    def __init__(self):
        from moo_counter_rust import RustEngine
        self._engine = RustEngine()

    @property
    def name(self) -> str:
        return self._engine.name

    def get_grid_dimensions(self, grid: Grid) -> GridDimensions:
        return self._engine.get_grid_dimensions(grid)

    def is_valid_moove(self, moove: Moove, grid: Grid) -> bool:
        # Convert Moove namedtuple to tuple for Rust
        moove_tuple = (moove.t1, moove.t2, moove.t3)
        return self._engine.is_valid_moove(moove_tuple, grid)

    def generate_moove(self, start: Position, direction: Direction) -> Moove:
        moove_tuple = self._engine.generate_moove(start, direction)
        return Moove(*moove_tuple)

    def generate_all_valid_mooves(self, grid: Grid) -> MooveSequence:
        mooves_tuples = self._engine.generate_all_valid_mooves_parallel(grid)
        return [Moove(*m) for m in mooves_tuples]

    def do_mooves_overlap(self, m1: Moove, m2: Moove) -> bool:
        m1_tuple = (m1.t1, m1.t2, m1.t3)
        m2_tuple = (m2.t1, m2.t2, m2.t3)
        return self._engine.do_mooves_overlap(m1_tuple, m2_tuple)

    def generate_overlaps_graph(self, mooves: MooveSequence) -> MooveOverlapGraph:
        # Convert Moove namedtuples to tuples
        mooves_tuples = [(m.t1, m.t2, m.t3) for m in mooves]
        graph_dict = self._engine.generate_overlaps_graph(mooves_tuples)

        # Convert back to Moove namedtuples
        overlaps = {}
        for moove_tuple, overlap_set in graph_dict.items():
            moove = Moove(*moove_tuple)
            overlaps[moove] = {Moove(*m) for m in overlap_set}

        return overlaps

    def generate_empty_board(self, dims: GridDimensions) -> BoardState:
        return self._engine.generate_empty_board(dims)

    def get_moove_coverage(self, board: BoardState, moove: Moove) -> int:
        moove_tuple = (moove.t1, moove.t2, moove.t3)
        return self._engine.get_moove_coverage(board, moove_tuple)

    def update_board_with_moove(
        self, board: BoardState, moo_count: int, moove: Moove
    ) -> tuple[BoardState, int, int]:
        moove_tuple = (moove.t1, moove.t2, moove.t3)
        return self._engine.update_board_with_moove(board, moo_count, moove_tuple)

    def simulate_board(
        self, mooves: MooveSequence, dims: GridDimensions
    ) -> SimulationResult:
        mooves_tuples = [(m.t1, m.t2, m.t3) for m in mooves]
        result_dict = self._engine.simulate_board(mooves_tuples, dims)

        # Convert dict back to SimulationResult
        return SimulationResult(
            board=result_dict["board"],
            moo_count=result_dict["moo_count"],
            moove_sequence=[Moove(*m) for m in result_dict["moove_sequence"]],
            moo_count_sequence=result_dict["moo_count_sequence"],
            moo_coverage_gain_sequence=result_dict["moo_coverage_gain_sequence"]
        )

    def benchmark(self, grid: Grid, iterations: int = 1000) -> float:
        return self._engine.benchmark(grid, iterations)


class CythonEngineWrapper:
    """Wrapper for Cython engine to match GameEngine protocol."""

    def __init__(self):
        from .cython_engine import CythonEngine
        self._engine = CythonEngine()

    @property
    def name(self) -> str:
        return self._engine.name

    def get_grid_dimensions(self, grid: Grid) -> GridDimensions:
        return self._engine.get_grid_dimensions(grid)

    def is_valid_moove(self, moove: Moove, grid: Grid) -> bool:
        # Convert Moove namedtuple to tuple for Cython
        moove_tuple = (moove.t1, moove.t2, moove.t3)
        return self._engine.is_valid_moove(moove_tuple, grid)

    def generate_moove(self, start: Position, direction: Direction) -> Moove:
        moove_tuple = self._engine.generate_moove(start, direction)
        return Moove(*moove_tuple)

    def generate_all_valid_mooves(self, grid: Grid) -> MooveSequence:
        mooves_tuples = self._engine.generate_all_valid_mooves(grid)
        return [Moove(*m) for m in mooves_tuples]

    def do_mooves_overlap(self, m1: Moove, m2: Moove) -> bool:
        m1_tuple = (m1.t1, m1.t2, m1.t3)
        m2_tuple = (m2.t1, m2.t2, m2.t3)
        return self._engine.do_mooves_overlap(m1_tuple, m2_tuple)

    def generate_overlaps_graph(self, mooves: MooveSequence) -> MooveOverlapGraph:
        # Convert Moove namedtuples to tuples
        mooves_tuples = [(m.t1, m.t2, m.t3) for m in mooves]
        graph_dict = self._engine.generate_overlaps_graph(mooves_tuples)

        # Convert back to Moove namedtuples
        overlaps = {}
        for moove_tuple, overlap_set in graph_dict.items():
            moove = Moove(*moove_tuple)
            overlaps[moove] = {Moove(*m) for m in overlap_set}

        return overlaps

    def generate_empty_board(self, dims: GridDimensions) -> BoardState:
        return self._engine.generate_empty_board(dims)

    def get_moove_coverage(self, board: BoardState, moove: Moove) -> int:
        moove_tuple = (moove.t1, moove.t2, moove.t3)
        return self._engine.get_moove_coverage(board, moove_tuple)

    def update_board_with_moove(
        self, board: BoardState, moo_count: int, moove: Moove
    ) -> tuple[BoardState, int, int]:
        moove_tuple = (moove.t1, moove.t2, moove.t3)
        return self._engine.update_board_with_moove(board, moo_count, moove_tuple)

    def simulate_board(
        self, mooves: MooveSequence, dims: GridDimensions
    ) -> SimulationResult:
        mooves_tuples = [(m.t1, m.t2, m.t3) for m in mooves]
        result_dict = self._engine.simulate_board(mooves_tuples, dims)

        # Convert dict back to SimulationResult
        return SimulationResult(
            board=result_dict["board"],
            moo_count=result_dict["moo_count"],
            moove_sequence=[Moove(*m) for m in result_dict["moove_sequence"]],
            moo_count_sequence=result_dict["moo_count_sequence"],
            moo_coverage_gain_sequence=result_dict["moo_coverage_gain_sequence"]
        )

    def benchmark(self, grid: Grid, iterations: int = 1000) -> float:
        return self._engine.benchmark(grid, iterations)