"""Python wrappers for Rust, Cython, and C engines to match the GameEngine protocol."""

from ..moo_types import (
    Grid, GridDimensions, BoardState, Position, Direction,
    Moove, MooveSequence, MooveOverlapGraph, SimulationResult
)


class CEngineWrapper:
    """Wrapper for C engine to match GameEngine protocol."""

    def __init__(self):
        from .c_engine import CEngine
        self._engine = CEngine()

    @property
    def name(self) -> str:
        return self._engine.get_name()

    def get_grid_dimensions(self, grid: Grid) -> GridDimensions:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        return (rows, cols)

    def is_valid_moove(self, moove: Moove, grid: Grid) -> bool:
        rows, cols = self.get_grid_dimensions(grid)
        # Convert moove to start position and direction index
        # This is a simplified check - the C engine uses a different interface
        t1 = moove.t1
        t2 = moove.t2

        # Determine direction from t1 to t2
        dr = t2[0] - t1[0]
        dc = t2[1] - t1[1]

        # Map to direction index
        directions = [
            (-1, 0),   # up
            (-1, 1),   # up-right
            (0, 1),    # right
            (1, 1),    # down-right
            (1, 0),    # down
            (1, -1),   # down-left
            (0, -1),   # left
            (-1, -1),  # up-left
        ]

        try:
            dir_idx = directions.index((dr, dc))
            return self._engine.is_valid_moove(t1[0], t1[1], dir_idx, rows, cols)
        except ValueError:
            return False

    def generate_moove(self, start: Position, direction: Direction) -> Moove:
        # Generate a moove from start position and direction
        t1 = start
        t2 = (start[0] + direction[0], start[1] + direction[1])
        t3 = (start[0] + 2 * direction[0], start[1] + 2 * direction[1])
        return Moove(t1, t2, t3)

    def generate_all_valid_mooves(self, grid: Grid) -> MooveSequence:
        rows, cols = self.get_grid_dimensions(grid)
        mooves_raw = self._engine.generate_all_valid_mooves(rows, cols)

        # Convert from (row, col, dir_idx) to Moove namedtuples
        directions = [
            (-1, 0),   # up
            (-1, 1),   # up-right
            (0, 1),    # right
            (1, 1),    # down-right
            (1, 0),    # down
            (1, -1),   # down-left
            (0, -1),   # left
            (-1, -1),  # up-left
        ]

        result = []
        for row, col, dir_idx in mooves_raw:
            dr, dc = directions[dir_idx]
            t1 = (row, col)
            t2 = (row + dr, col + dc)
            t3 = (row + 2*dr, col + 2*dc)
            result.append(Moove(t1, t2, t3))

        return result

    def do_mooves_overlap(self, m1: Moove, m2: Moove) -> bool:
        # Check if two mooves share any cells
        cells1 = {m1.t1, m1.t2, m1.t3}
        cells2 = {m2.t1, m2.t2, m2.t3}
        return bool(cells1 & cells2)

    def generate_overlaps_graph(self, mooves: MooveSequence) -> MooveOverlapGraph:
        # Build overlap graph
        overlaps = {}
        for i, m1 in enumerate(mooves):
            overlaps[m1] = set()
            for j, m2 in enumerate(mooves):
                if i != j and self.do_mooves_overlap(m1, m2):
                    overlaps[m1].add(m2)
        return overlaps

    def generate_empty_board(self, dims: GridDimensions) -> BoardState:
        rows, cols = dims
        return [[0 for _ in range(cols)] for _ in range(rows)]

    def get_moove_coverage(self, board: BoardState, moove: Moove) -> int:
        count = 0
        for pos in [moove.t1, moove.t2, moove.t3]:
            if board[pos[0]][pos[1]] != 0:  # Count already covered cells
                count += 1
        return count

    def update_board_with_moove(
        self, board: BoardState, moo_count: int, moove: Moove
    ) -> tuple[BoardState, int, int]:
        import copy
        new_board = copy.deepcopy(board)
        coverage_gain = 0

        for pos in [moove.t1, moove.t2, moove.t3]:
            if new_board[pos[0]][pos[1]] == 0:
                coverage_gain += 1
            new_board[pos[0]][pos[1]] = moo_count

        return new_board, moo_count + 1, coverage_gain

    def simulate_board(
        self, mooves: MooveSequence, dims: GridDimensions
    ) -> SimulationResult:
        board = self.generate_empty_board(dims)
        moo_count = 1
        moove_sequence = []
        moo_count_sequence = []
        moo_coverage_gain_sequence = []

        for moove in mooves:
            board, moo_count, coverage_gain = self.update_board_with_moove(
                board, moo_count, moove
            )
            moove_sequence.append(moove)
            moo_count_sequence.append(moo_count - 1)
            moo_coverage_gain_sequence.append(coverage_gain)

        return SimulationResult(
            board=board,
            moo_count=moo_count - 1,
            moove_sequence=moove_sequence,
            moo_count_sequence=moo_count_sequence,
            moo_coverage_gain_sequence=moo_coverage_gain_sequence
        )

    def benchmark(self, grid: Grid, iterations: int = 1000) -> float:
        import time
        start = time.time()
        for _ in range(iterations):
            self.generate_all_valid_mooves(grid)
        return time.time() - start


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