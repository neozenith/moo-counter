# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

# Third Party
import cython
from cython.parallel import prange

# Third Party
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy


cdef class CythonEngine:
    cdef public str name
    cdef list directions

    def __init__(self):
        self.name = "Cython"
        self.directions = [
            (-1, 0),   # up
            (-1, 1),   # up-right
            (0, 1),    # right
            (1, 1),    # down-right
            (1, 0),    # down
            (1, -1),   # down-left
            (0, -1),   # left
            (-1, -1),  # up-left
        ]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_grid_dimensions(self, list grid):
        cdef int rows = len(grid)
        cdef int cols = len(grid[0]) if rows > 0 else 0
        return (rows, cols)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint is_valid_moove(self, tuple moove, list grid):
        cdef int height = len(grid)
        cdef int width = len(grid[0]) if height > 0 else 0

        cdef tuple t1 = moove[0]
        cdef tuple t2 = moove[1]
        cdef tuple t3 = moove[2]

        cdef int r1 = t1[0]
        cdef int c1 = t1[1]
        cdef int r2 = t2[0]
        cdef int c2 = t2[1]
        cdef int r3 = t3[0]
        cdef int c3 = t3[1]

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
        cdef int d1_r = r2 - r1
        cdef int d1_c = c2 - c1
        cdef int d2_r = r3 - r2
        cdef int d2_c = c3 - c2

        # Check t2 is adjacent to t1
        if abs(d1_r) > 1 or abs(d1_c) > 1:
            return False

        # Check t3 follows same direction from t2
        return d1_r == d2_r and d1_c == d2_c

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def generate_moove(self, tuple start, int direction):
        cdef tuple d = self.directions[direction]
        cdef int dr = d[0]
        cdef int dc = d[1]
        cdef int r = start[0]
        cdef int c = start[1]

        cdef tuple t1 = start
        cdef tuple t2 = (r + dr, c + dc)
        cdef tuple t3 = (r + 2 * dr, c + 2 * dc)

        return (t1, t2, t3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def generate_all_valid_mooves(self, list grid):
        cdef int height = len(grid)
        cdef int width = len(grid[0]) if height > 0 else 0
        cdef list mooves = []

        cdef int r, c, direction
        cdef tuple moove

        for r in range(height):
            for c in range(width):
                for direction in range(8):
                    moove = self.generate_moove((r, c), direction)
                    if self.is_valid_moove(moove, grid):
                        mooves.append(moove)

        return mooves

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint do_mooves_overlap(self, tuple m1, tuple m2):
        cdef set positions1 = {m1[0], m1[1], m1[2]}
        cdef set positions2 = {m2[0], m2[1], m2[2]}
        return len(positions1.intersection(positions2)) > 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def generate_overlaps_graph(self, list mooves):
        cdef dict overlaps = {}
        cdef int i, j
        cdef tuple moove1, moove2

        for i in range(len(mooves)):
            moove1 = mooves[i]
            for j in range(i + 1, len(mooves)):
                moove2 = mooves[j]
                if self.do_mooves_overlap(moove1, moove2):
                    if moove1 not in overlaps:
                        overlaps[moove1] = set()
                    if moove2 not in overlaps:
                        overlaps[moove2] = set()
                    overlaps[moove1].add(moove2)
                    overlaps[moove2].add(moove1)

        return overlaps

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def generate_empty_board(self, tuple dims):
        cdef int height = dims[0]
        cdef int width = dims[1]
        return [[False for _ in range(width)] for _ in range(height)]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int get_moove_coverage(self, list board, tuple moove):
        cdef int coverage = 0
        cdef tuple pos

        for pos in moove:
            if board[pos[0]][pos[1]]:
                coverage += 1

        return coverage

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_board_with_moove(self, list board, int moo_count, tuple moove):
        cdef tuple t1 = moove[0]
        cdef tuple t2 = moove[1]
        cdef tuple t3 = moove[2]
        cdef int r1 = t1[0]
        cdef int c1 = t1[1]
        cdef int r2 = t2[0]
        cdef int c2 = t2[1]
        cdef int r3 = t3[0]
        cdef int c3 = t3[1]

        # Deep copy the board
        cdef list output_board = [row[:] for row in board]

        # Determine coverage
        cdef int moo_coverage = self.get_moove_coverage(board, moove)

        # Place moove on board
        if moo_coverage < 3:
            output_board[r1][c1] = True
            output_board[r2][c2] = True
            output_board[r3][c3] = True
            moo_count += 1

        cdef int coverage_gain = 3 - moo_coverage
        return output_board, moo_count, coverage_gain

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def simulate_board(self, list mooves, tuple dims):
        cdef list board = self.generate_empty_board(dims)
        cdef int moo_count = 0
        cdef list moo_count_sequence = []
        cdef list moo_coverage_gain_sequence = []
        cdef tuple moove
        cdef int coverage_gain

        for moove in mooves:
            board, moo_count, coverage_gain = self.update_board_with_moove(
                board, moo_count, moove
            )
            moo_count_sequence.append(moo_count)
            moo_coverage_gain_sequence.append(coverage_gain)

        return {
            "board": board,
            "moo_count": moo_count,
            "moove_sequence": mooves,
            "moo_count_sequence": moo_count_sequence,
            "moo_coverage_gain_sequence": moo_coverage_gain_sequence
        }

    def benchmark(self, list grid, int iterations=1000):
        # Standard Library
        import time

        cdef tuple dims = self.get_grid_dimensions(grid)
        cdef list all_mooves = self.generate_all_valid_mooves(grid)

        # Take first 50 mooves for benchmark
        cdef list test_mooves = all_mooves[:50]

        cdef double start_time = time.time()
        cdef int i

        for i in range(iterations):
            board = self.generate_empty_board(dims)
            moo_count = 0

            for moove in test_mooves:
                board, moo_count, _ = self.update_board_with_moove(
                    board, moo_count, moove
                )

        cdef double duration = time.time() - start_time
        cdef double simulations_per_second = iterations / duration

        print(f"[Cython Engine] {simulations_per_second:.0f} simulations/second")
        return simulations_per_second