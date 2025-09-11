#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx",
#   "beautifulsoup4",
# ]
# ///
# Standard Library
import argparse
import itertools
import json
import pathlib
import random
import sys
import os
import time
from multiprocessing import Pool

import httpx
from bs4 import BeautifulSoup

OUTPUT_DIR = pathlib.Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

OUTPUT_HISTOGRAM = OUTPUT_DIR / "moo_count_histogram.json"
OUTPUT_COVERAGE_BOARD = OUTPUT_DIR / "moo_coverage_board.txt"
OUTPUT_MAX_MOO_SEQUENCE = OUTPUT_DIR / "max_moo_sequence.yml"
OUTPUT_MIN_MOO_SEQUENCE = OUTPUT_DIR / "min_moo_sequence.yml"

# Custom Types
Grid = list[list[str]]
GridDimensions = tuple[int, int]  # (rows, columns)
BoardState = list[list[bool]] # This is the current board coverage state, True if occupied, False if not.

Position = tuple[int, int]  # (row, column)
Moove = tuple[Position, Position, Position]  # ((r1,c1), (r2,c2), (r3,c3))
MooveCandidate = tuple[Moove, int]  # (moove, coverage_gain)
MooveSequence = list[Moove]
Direction = int  # 0-7 representing 8 possible directions
MooveDirection = tuple[int, int]  # (dr, dc)

MooCount = int # Total scored number of 'moo' moves placed

MooveCountSequence = list[MooCount]  # Sequence of moo counts after each move
MooveCoverageGainSequence = list[int]  # Sequence of coverage gains after each move

SimulationResult = tuple[BoardState, MooCount, MooveSequence, MooveCountSequence, MooveCoverageGainSequence]
MooCountHistogram = dict[str, int]  # Histogram of moo counts from multiple simulations

MooveOverlapGraph = dict[Moove, set[Moove]]  # Graph of mooves that have a relationship to another moove because they overlap

def grid_from_live() -> Grid:
    """Generate the grid from the live puzzle input."""
    url = "https://find-a-moo.kleeut.com/plain-text"
    
    response = httpx.get(url)
    response.raise_for_status()
    text = response.text
    # TODO: let it finish rendering the javascript before scraping.

    soup = BeautifulSoup(text, "html.parser")
    content = soup.select("body > pre")[0].get_text()

    grid = [list(line.replace(" ", "")) for line in content.splitlines()]
    return grid


def grid_from_file(path: pathlib.Path) -> Grid:
    with open(path, "r") as f:
        lines = f.readlines()

    grid = []
    for line in lines:
        row = list(line.strip())
        grid.append(row)

    return grid

def grid_dimensions(grid: Grid) -> GridDimensions:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    return (rows, cols)

def is_valid_moove(m: Moove, grid: Grid) -> bool:
    """Check if three positions form a valid 'moo' move.

    m is a tuple of three tuples, each representing (row, column) positions on the grid.
    The rules for a valid 'moo' move are:
    - t1 must be the 'm' and t2 and t3 must be the 'o's.
    - t2 must be adjacent to t1 (horizontally or vertically or diagonally)
    - t3 must follow the same direction as t2 from t1.
    """
    height, width = grid_dimensions(grid)
    # Unpack rows and columns
    t1, t2, t3 = m
    r1, c1 = t1
    r2, c2 = t2
    r3, c3 = t3

    # Check if positions are within bounds
    if r1 < 0 or r1 >= height or c1 < 0 or c1 >= width:
        return False
    if r2 < 0 or r2 >= height or c2 < 0 or c2 >= width:
        return False
    if r3 < 0 or r3 >= height or c3 < 0 or c3 >= width:
        return False

    # Calculate direction from t1 to t2
    d1: MooveDirection = (r2 - r1, c2 - c1)

    # Calculate direction from t2 to t3
    d2: MooveDirection = (r3 - r2, c3 - c2)

    # Assume true and then test for invalidations

    # Check if it spells 'moo'
    # Check if t1 is 'm' and t2, t3 are 'o's
    if grid[r1][c1] != "m" or grid[r2][c2] != "o" or grid[r3][c3] != "o":
        return False

    # Check if t2 is adjacent to t1
    if abs(d1[0]) > 1 or abs(d1[1]) > 1:
        return False

    # Final check
    # Check if t3 follows the same direction from t2
    # They should have the same vector d1(dr1, dc1) == d2(dr2, dc2)
    return d1 == d2


def generate_moove(start: Position, direction: Direction) -> Moove:
    """Generate a 'moo' move given a starting position and a direction.

    - start is the position of 'm' (t1).
    - direction is an integer from 0 to 7 representing the 8 possible directions clockwise
    """
    directions: list[MooveDirection] = [
        (-1, 0),  # up
        (-1, 1),  # up-right
        (0, 1),  # right
        (1, 1),  # down-right
        (1, 0),  # down
        (1, -1),  # down-left
        (0, -1),  # left
        (-1, -1),  # up-left
    ]
    d = directions[direction]

    t1 = start
    t2 = (start[0] + d[0], start[1] + d[1])
    t3 = (start[0] + 2 * d[0], start[1] + 2 * d[1])

    return (t1, t2, t3)


def generate_all_valid_mooves(grid: Grid) -> MooveSequence:
    """Generate all possible 'moo' moves on the grid."""
    height, width = grid_dimensions(grid)
    mooves: MooveSequence = []
    for r in range(height):
        for c in range(width):
            for direction in range(8):
                moove: Moove = generate_moove((r, c), direction)
                if is_valid_moove(moove, grid):
                    mooves.append(moove)
    return mooves

def generate_overlaps_graph(mooves: MooveSequence) -> dict[Moove, set[Moove]]:
    overlaps: dict[Moove, set[Moove]] = {}
    for i, moove1 in enumerate(mooves):
        for moove2 in mooves[i + 1:]:
            if do_mooves_overlap(moove1, moove2):
                overlaps.setdefault(moove1, set()).add(moove2)
                overlaps.setdefault(moove2, set()).add(moove1)
    return overlaps

def do_mooves_overlap(m1: Moove, m2: Moove) -> bool:
    positions1 = set(m1)
    positions2 = set(m2)
    return not positions1.isdisjoint(positions2)


def generate_empty_board(dims: GridDimensions) -> BoardState:
    """Generate an empty board of False values."""
    height, width = dims
    return [[False for _ in range(width)] for _ in range(height)]


def update_board_with_moove(
    board: BoardState, moo_count: int, moove: Moove
) -> tuple[BoardState, int, int]:
    """Update the board with a 'moo' move."""
    t1, t2, t3 = moove
    r1, c1 = t1
    r2, c2 = t2
    r3, c3 = t3

    output_board_state = [row[:] for row in board]  # Deep copy of the board

    # Determine how much of the existing board state is covered already by this move.
    moo_coverage = 0
    for t in moove:
        if board[t[0]][t[1]] is True:
            moo_coverage += 1

    # Place 'm' and 'o's on the board
    if moo_coverage < 3:
        # Mark positions as occupied
        output_board_state[r1][c1] = True
        output_board_state[r2][c2] = True
        output_board_state[r3][c3] = True
        # Update the moo count ONLY if at least one position was newly occupied
        moo_count += 1
    else:
        # All positions were already occupied, do not increment moo_count
        pass

    return output_board_state, moo_count, 3 - moo_coverage  # Return also how many new positions were covered


def render_board(board: BoardState, grid: Grid) -> str:
    output = ""
    for r, row in enumerate(board):
        row_str = ""
        for c, cell in enumerate(row):
            if cell is True:
                row_str += grid[r][c].upper() + " "
            elif cell is False:
                row_str += grid[r][c].lower() + " "
            else:
                row_str += f"{cell} "
        output += row_str.strip() + "\n"
    return output

def determine_direction_from_moove(moove: Moove) -> Direction:
    t1, t2, t3 = moove
    r1, c1 = t1
    r2, c2 = t2

    d = (r2 - r1, c2 - c1)

    direction_mapping = {
        (-1, 0): 0,  # up
        (-1, 1): 1,  # up-right
        (0, 1): 2,   # right
        (1, 1): 3,   # down-right
        (1, 0): 4,   # down
        (1, -1): 5,  # down-left
        (0, -1): 6,  # left
        (-1, -1): 7  # up-left
    }

    return direction_mapping.get(d, None)

def render_direction_arrow(direction: Direction) -> str:
    arrows = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖']
    return arrows[direction]

def simulate_board(
    mooves: MooveSequence,
    dims: GridDimensions = (15, 15)
) -> SimulationResult:
    """Simulate the board with a sequence of 'moo' moves."""
    board = generate_empty_board(dims)
    moo_count = 0
    moo_count_sequence = []
    moo_coverage_gain_sequence = []
    for moove in mooves:
        board, moo_count, moo_coverage_gain = update_board_with_moove(board, moo_count, moove)
        moo_count_sequence.append(moo_count)
        moo_coverage_gain_sequence.append(moo_coverage_gain)

    return board, moo_count, mooves, moo_count_sequence, moo_coverage_gain_sequence

def generate_sequence_from_strategy(
    strategy: str, all_valid_mooves: MooveSequence, dims: GridDimensions, graph: MooveOverlapGraph, random_seed: int = 42
) -> MooveSequence:
    """Generate a sequence of mooves based on the given strategy."""
    random.seed(random_seed)
    if strategy == 'all':
        strategy = random.choice(['random', 'greedy-high', 'greedy-low'])

    if strategy == 'random':
        return random.sample(all_valid_mooves, len(all_valid_mooves))
    elif strategy == 'greedy-high':
        return generate_sequence_greedily(all_valid_mooves=all_valid_mooves, dims=dims, graph=graph, seed=random_seed, highest_score=True)
    elif strategy == 'greedy-low':
        return generate_sequence_greedily(all_valid_mooves=all_valid_mooves, dims=dims, graph=graph, seed=random_seed, highest_score=False)
    elif strategy == 'greedy':
        return generate_sequence_greedily(all_valid_mooves=all_valid_mooves, dims=dims, graph=graph, seed=random_seed, highest_score=random.choice([True, False]))
    elif strategy == 'beam-search':
        return generate_sequence_beam_search(all_valid_mooves=all_valid_mooves, dims=dims, graph=graph, seed=random_seed, highest_score=random.choice([True, False]))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def worker_simulate(args: tuple[int, MooveSequence, GridDimensions, MooveOverlapGraph, str]) -> SimulationResult:
    """Worker function that generates and simulates a random permutation."""
    seed, all_valid_mooves, dims, graph, strategy = args
    sequence = generate_sequence_from_strategy(strategy, all_valid_mooves=all_valid_mooves, dims=dims, graph=graph, random_seed=seed)
    return simulate_board(sequence, dims)
 # Offload the shuffle to the worker.

def render_moo_count_histogram(all_moo_counts: MooveCountSequence) -> MooCountHistogram:
    moo_count_histogram = {}
    for count in all_moo_counts:
        if count in moo_count_histogram.keys():
            moo_count_histogram[count] += 1
        else:
            moo_count_histogram[count] = 1

    histogram_max_frequency = float(max(moo_count_histogram.values()))
    screen_width = 40.0
    max_stars = max(screen_width, histogram_max_frequency)

    sorted_moo_count_histogram_keys = sorted(moo_count_histogram.keys())
    for key in sorted_moo_count_histogram_keys:
        print(
            f"Moo count {key}: {int((moo_count_histogram[key] / max_stars) * screen_width) * '🐮'} {moo_count_histogram[key]}"
        )
    
    # sort the dictionary by keys for easier reading
    moo_count_histogram = {k: moo_count_histogram[k] for k in sorted(moo_count_histogram.keys())}
    return moo_count_histogram

def render_moove(moove: Moove) -> str:
    """Render a single moove in the format 'A, 1 →' or 'H,12 ↖'."""
    t1, t2, t3 = moove
    return f"'{chr(t1[0] + 65)},{t1[1]+1:>2} {render_direction_arrow(determine_direction_from_moove(moove))}'"

def render_moove_sequence(
        moove_sequence: MooveSequence, 
        moo_count_sequence: MooveCountSequence, 
        moo_coverage_sequence: MooveCoverageGainSequence
    ) -> str:
    accumulative_coverage = 0
    output = "mooves:      # Moove Number, Moo Count, Coverage Gain, Accumulative Coverage Gain\n"
    for i, (moove, moo_count, moo_coverage_gain) in enumerate(zip(moove_sequence, moo_count_sequence, moo_coverage_sequence)):
        t1, t2, t3 = moove
        accumulative_coverage += moo_coverage_gain
        d = determine_direction_from_moove(moove)
        direction_arrow = render_direction_arrow(d) if d is not None else "?"
        # TODO: map the row number to a letter
        row_letter = chr(t1[0] + 65)  # Map 0->A, 1->B, 2->C, ...
        row_str = f"{row_letter}"
        col_str = f"{t1[1]+1:>2}"
        if moo_coverage_gain > 0:
            # Maximum number of digits in steps is 3 but render 5 zero padded so it looks like the word M00
            comment_annotation = f"# M{i:05d} {moo_count} {moo_coverage_gain} {accumulative_coverage}"
            moove_record = f"'{row_str},{col_str} {direction_arrow}'"
            output += f"  - {moove_record} {comment_annotation}\n"
        
    output += ""
    return output


def parallel_process_simulations(grid: Grid, iterations: int, workers: int, strategy: str) -> None:
    all_valid_mooves = generate_all_valid_mooves(grid)
    dims = grid_dimensions(grid)
    height, width = dims
    all_cells = height * width
    graph = generate_overlaps_graph(all_valid_mooves)

    print(f"Total valid 'moo' moves found: {len(all_valid_mooves)}")
    print(f"Graph of overlapping Mooves has {len(graph)} nodes. And highest degree node has {max(len(v) for v in graph.values())} overlaps.")
    # The total permuations is NumberOfValidMooves Factorial for a 15x15 grid which has like >300 zeroes in the count of permutations.
    # We therefore cannot brute force it.
    # I have added some strategies on how a candidate sequence is generated and those can be selected randomly.
    # The strategies are:
    # - Random shuffle of all valid mooves (Monte Carlo trials)
    # - Greedily select the next moove that gives the highest immediate score (coverage gain) when mooves are equally weighted then randomly select among those.
    # - Greedily select the next moove that gives the lowest immediate score (coverage gain) when mooves are equally weighted then randomly select among those.
    # - Greedily select either of the above two at random for each iteration.
    # - Beam search (not implemented yet)
    # - All of the above strategies randomly selected for each iteration.

    all_moo_counts = []
    max_mooves = 0
    max_board = None
    max_moove_sequence = None
    N_iters = iterations

    # Time tracking from here since creating worker args can take some time.
    time_start = time.time()
    # Create args for workers
    worker_args = [(i, all_valid_mooves, dims, graph, strategy) for i in range(N_iters)]

    # Multiprocessing with proper chunksize
    P = workers if workers > 0 else (os.cpu_count() or 4)
    optimal_chunksize = max(1, N_iters // (P * 4)) # Aim for at least 4 tasks per worker
    print(f"Using {P} processes with chunksize {optimal_chunksize}")

    time_sims_start = time.time()
    with Pool(P) as pool:
        results_iter = pool.imap_unordered(worker_simulate, worker_args, chunksize=optimal_chunksize)

        all_simulations: list[SimulationResult] = list(results_iter)

    time_parallel_end = time.time()
    time_parallel_duration = time_parallel_end - time_sims_start
    print(f"Simulations complete took {time_parallel_duration:.2f}s, ({N_iters / time_parallel_duration:.0f} simulations per second)")
    
    all_moo_counts = [final_moo_count for _, final_moo_count, _, _, _ in all_simulations]
    max_result = max(all_simulations, key=lambda x: x[1])  # x[1] is moo_count
    min_result = min(all_simulations, key=lambda x: x[1])  # x[1] is moo_count
    max_board, max_mooves, max_moove_sequence, max_moo_count_sequence, max_moo_count_coverage_sequence = max_result
    min_board, min_mooves, min_moove_sequence, min_moo_count_sequence, min_moo_count_coverage_sequence = min_result

    time_reduce_end = time.time()
    time_reduce_duration = time_reduce_end - time_parallel_end
    print(
        f"Result processing took {time_reduce_duration:.2f}s after {time_parallel_duration:.2f}s of parallel simulation."
    )

    time_now = time.time()
    total_time = time_now - time_start
    total_sims_time = time_now - time_sims_start

    print(f"Time taken for parallel simulation: {total_sims_time:.2f}s")
    print(f"Total time taken: {total_time:.2f}s")
    print(f"Simulations per second: {N_iters / total_sims_time:.0f}")
    print()

    OUTPUT_HISTOGRAM.write_text(json.dumps(render_moo_count_histogram(all_moo_counts), indent=2))
    OUTPUT_COVERAGE_BOARD.write_text(render_board(max_board, grid))
    OUTPUT_MAX_MOO_SEQUENCE.write_text(
        render_moove_sequence(
            max_moove_sequence, 
            max_moo_count_sequence, 
            max_moo_count_coverage_sequence
        )
    )
    OUTPUT_MIN_MOO_SEQUENCE.write_text(
        render_moove_sequence(
            min_moove_sequence, 
            min_moo_count_sequence, 
            min_moo_count_coverage_sequence
        )
    )

    
    
    print(f"Graph of overlapping Mooves has {len(graph)} nodes. And highest degree node has {max(len(v) for v in graph.values())} overlaps.")

    print(f"Total valid 'moo' moves found: {len(all_valid_mooves)}")
    print(f"Max Board coverage: {sum(max_moo_count_coverage_sequence)} --> {all_cells - sum(max_moo_count_coverage_sequence)} dead squares")
    print(f"MAX moo count: {max_mooves}")
    print(f"MIN moo count: {min_mooves}")

def generate_sequence_greedily(
    all_valid_mooves: MooveSequence, dims: GridDimensions, graph: MooveOverlapGraph, seed: int = 42, highest_score: bool = True
) -> MooveSequence:

    board = generate_empty_board(dims)
    moove_sequence = []
    moo_count = 0
    moo_count_sequence = []
    moo_coverage_gain_sequence = []

    remaining_mooves = set(all_valid_mooves)

    iteration = 0

    while remaining_mooves:
        
        best_moove_candidates: list[MooveCandidate] = []
        best_coverage_gain = 3

        # Find next best moove
        for moove in remaining_mooves:
            _, _, coverage_gain = update_board_with_moove(
                board, moo_count, moove
            )
            if coverage_gain <= 0:
                continue  # Invalid move, skip

            moove_candidate = (moove, coverage_gain)
            best_moove_candidates.append(moove_candidate)
        
        # Subset to only those with the maximum / minimum coverage gain for this round.
        if highest_score:
            min_coverage_gain = min((cg for _, cg in best_moove_candidates), default=3)
            best_moove_candidates = [mc for mc in best_moove_candidates if mc[1] == min_coverage_gain]
        else:
            max_coverage_gain = max((cg for _, cg in best_moove_candidates), default=0)
            best_moove_candidates = [mc for mc in best_moove_candidates if mc[1] == max_coverage_gain]
        
        # Random selection among the best candidates
        best_candidate = random.choice(best_moove_candidates) if best_moove_candidates else None

        best_moove = best_candidate[0] if best_candidate else None
        best_coverage_gain = best_candidate[1] if best_candidate else 0

        if best_moove is None or best_coverage_gain == 0:
            break  # No more beneficial moves

        moove_sequence.append(best_moove)
        board, moo_count, coverage_gain = update_board_with_moove(board, moo_count, best_moove)

        moo_count_sequence.append(moo_count)
        moo_coverage_gain_sequence.append(coverage_gain)
        remaining_mooves.remove(best_moove)
        iteration += 1
    return moove_sequence

def generate_sequence_beam_search(
        all_valid_mooves: MooveSequence, 
        dims: GridDimensions, 
        graph: MooveOverlapGraph, 
        seed: int = 42, 
        highest_score: bool = True
    ) -> MooveSequence:
    """Generate a sequence of mooves using beam search."""
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moo Counter")
    parser.add_argument("--puzzle", type=str, help="Path to the puzzle file")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for random sampling")
    parser.add_argument(
        "--workers", type=int, default=-1, 
        help="Number of worker processes to use. Default is number of CPU cores."
    )
    parser.add_argument(
        "--strategy", type=str, default="all", choices=["random", "greedy-high", "greedy-low", "greedy", "beam-search", "all"],
        help="Strategy for generating moove sequences."
    )
    args = parser.parse_args(sys.argv[1:])

    grid = grid_from_live() if args.puzzle == 'live' else grid_from_file(pathlib.Path(args.puzzle))
    
    parallel_process_simulations(grid, args.iterations, args.workers, args.strategy)
    
    
