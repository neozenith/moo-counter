#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx",
#   "playwright",
# ]
# ///
# Standard Library
from dataclasses import dataclass
import argparse
import itertools
import json
import pathlib
import random
import sys
import os
import time
from multiprocessing import Pool
import math

from playwright.sync_api import sync_playwright

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

OUTPUT_PUZZLES_DIR = PROJECT_ROOT / "puzzles"
OUTPUT_PUZZLES_DIR.mkdir(exist_ok=True, parents=True)

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

valid_sizes = ["micro", "mini", "maxi"]

def today_date_str() -> str:
    return time.strftime("%Y%m%d", time.localtime())

def fetch_live_puzzle_input(size: str) -> str:
    """Fetch the live puzzle input from the website."""
    url = f"https://find-a-moo.kleeut.com/plain-text?size={size}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")
        content = page.locator("body > pre").inner_text()
        browser.close()
    return content.replace(" ", "")

def grid_from_live(size: str) -> tuple[Grid, str]:
    """Generate the grid from the live puzzle input."""
    content = fetch_live_puzzle_input(size)
    grid = [list(line.replace(" ", "")) for line in content.splitlines()]
    return grid, content


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


def get_moove_coverage(board: BoardState, moove: Moove) -> int:
    moo_coverage = 0
    for t in moove:
        if board[t[0]][t[1]] is True:
            moo_coverage += 1
    return moo_coverage


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

    moo_coverage = get_moove_coverage(board, moove)    

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
        strategy = random.choice(['random', 'greedy', 'mcts'])

    if strategy == 'random':
        return random.sample(all_valid_mooves, len(all_valid_mooves))
    elif strategy == 'greedy-high':
        return generate_sequence_greedily(all_valid_mooves=all_valid_mooves, dims=dims, graph=graph, seed=random_seed, highest_score=True)
    elif strategy == 'greedy-low':
        return generate_sequence_greedily(all_valid_mooves=all_valid_mooves, dims=dims, graph=graph, seed=random_seed, highest_score=False)
    elif strategy == 'greedy':
        return generate_sequence_greedily(all_valid_mooves=all_valid_mooves, dims=dims, graph=graph, seed=random_seed, highest_score=random.choice([True, False]))
    elif strategy == 'mcts':
        return generate_sequence_mcts(all_valid_mooves=all_valid_mooves, dims=dims, graph=graph, seed=random_seed, highest_score=True, iterations=200)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def worker_simulate(args: tuple[int, MooveSequence, GridDimensions, MooveOverlapGraph, str]) -> SimulationResult:
    """Worker function that generates and simulates a random permutation."""
    seed, all_valid_mooves, dims, graph, strategy = args
    sequence = generate_sequence_from_strategy(strategy, all_valid_mooves=all_valid_mooves, dims=dims, graph=graph, random_seed=seed)
    return simulate_board(sequence, dims)


def build_moo_count_histogram(all_moo_counts: MooveCountSequence) -> MooCountHistogram:
    moo_count_histogram = {}
    for count in all_moo_counts:
        if count in moo_count_histogram.keys():
            moo_count_histogram[count] += 1
        else:
            moo_count_histogram[count] = 1
    moo_count_histogram = {k: moo_count_histogram[k] for k in sorted(moo_count_histogram.keys())}
    return moo_count_histogram

def render_moo_count_histogram(histogram: MooCountHistogram, screen_width: int = 40) -> str:
    histogram_max_frequency = float(max(histogram.values()))
    max_stars = max(screen_width, histogram_max_frequency)
    output = ""
    for key, value in histogram.items():
        scaled_bar_length = int((value / max_stars) * screen_width)
        bar = '🐮' * scaled_bar_length
        output += f"Moo count {key}: {bar} {value}\n"
    return output

def render_moove(moove: Moove) -> str:
    """Render a single moove in the format 'A, 1 →' or 'H,12 ↖'."""
    t1, _, _ = moove
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


def parallel_process_simulations(grid: Grid, iterations: int, workers: int, strategy: str):
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

    histogram = build_moo_count_histogram(all_moo_counts)
    max_coverage = sum(max_moo_count_coverage_sequence)
    dead_cells = all_cells - max_coverage

    return (all_valid_mooves, max_coverage, dead_cells, 
            max_mooves, min_mooves, histogram, graph, 
            max_moove_sequence, max_moo_count_sequence, max_moo_count_coverage_sequence,
            min_moove_sequence, min_moo_count_sequence, min_moo_count_coverage_sequence
        )

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
        dead_moves = set()
        # Find next best moove
        for moove in remaining_mooves:
            _, _, coverage_gain = update_board_with_moove(
                board, moo_count, moove
            )
            if coverage_gain <= 0:
                # This moove is already dead at this point.
                dead_moves.add(moove)
            else:
                moove_candidate = (moove, coverage_gain)
                best_moove_candidates.append(moove_candidate)
        
        remaining_mooves -= dead_moves
        # Select mooves based on strategy:
        # For MAX score: Want MIN coverage gain (= MAX overlap) to conserve coverage
        # For MIN score: Want MAX coverage gain (= MIN overlap) to cover efficiently
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

def generate_cytoscape_graph(
    all_valid_mooves: MooveSequence,
    graph: MooveOverlapGraph,
    max_moove_sequence: MooveSequence,
    max_moo_count_sequence: MooveCountSequence,
    max_moo_coverage_sequence: MooveCoverageGainSequence,
    dims: GridDimensions
) -> dict:
    """Generate Cytoscape.js compatible graph data for visualization.

    Creates a graph with:
    - START and END terminal nodes
    - All valid mooves as nodes
    - Path edges showing the max_moove_sequence
    - Possible edges showing other connections
    """
    nodes = []
    edges = []

    # Create node IDs for all mooves
    def moove_to_id(moove: Moove) -> str:
        """Convert a moove to a unique ID."""
        t1, _, _ = moove
        direction = determine_direction_from_moove(moove)
        arrow = render_direction_arrow(direction)
        return f"{chr(t1[0] + 65)}_{t1[1]+1}_{arrow}"

    # Add START node
    nodes.append({
        "data": {
            "id": "start",
            "label": "START",
            "type": "terminal"
        },
        "classes": "start-node"
    })

    # Add all moove nodes
    moove_to_node_map = {}
    for moove in all_valid_mooves:
        node_id = moove_to_id(moove)
        moove_to_node_map[moove] = node_id

        # Check if this moove is in the max sequence
        in_path = moove in max_moove_sequence

        nodes.append({
            "data": {
                "id": node_id,
                "label": render_moove(moove).strip("'"),
                "degree": len(graph.get(moove, [])),
                "position": {"row": moove[0][0], "col": moove[0][1]},
                "in_path": in_path
            },
            "classes": "path-node" if in_path else "regular-node"
        })

    # Add END node
    total_coverage = sum(max_moo_coverage_sequence) if max_moo_coverage_sequence else 0
    nodes.append({
        "data": {
            "id": "end",
            "label": "END",
            "type": "terminal",
            "coverage": total_coverage,
            "score": len(max_moove_sequence)
        },
        "classes": "end-node"
    })

    # Add path edges for max_moove_sequence
    if max_moove_sequence:
        # Edge from START to first moove
        first_moove_id = moove_to_node_map[max_moove_sequence[0]]
        edges.append({
            "data": {
                "id": f"start_to_{first_moove_id}",
                "source": "start",
                "target": first_moove_id,
                "sequence": 0,
                "coverage_gain": max_moo_coverage_sequence[0] if max_moo_coverage_sequence else 0,
                "total_coverage": max_moo_coverage_sequence[0] if max_moo_coverage_sequence else 0,
                "moo_count": 1
            },
            "classes": "path-edge"
        })

        # Edges between consecutive mooves in the sequence
        total_coverage = max_moo_coverage_sequence[0] if max_moo_coverage_sequence else 0
        for i in range(len(max_moove_sequence) - 1):
            source_id = moove_to_node_map[max_moove_sequence[i]]
            target_id = moove_to_node_map[max_moove_sequence[i + 1]]
            coverage_gain = max_moo_coverage_sequence[i + 1] if i + 1 < len(max_moo_coverage_sequence) else 0
            total_coverage += coverage_gain

            edges.append({
                "data": {
                    "id": f"{source_id}_to_{target_id}",
                    "source": source_id,
                    "target": target_id,
                    "sequence": i + 1,
                    "coverage_gain": coverage_gain,
                    "total_coverage": total_coverage,
                    "moo_count": max_moo_count_sequence[i + 1] if i + 1 < len(max_moo_count_sequence) else 0
                },
                "classes": "path-edge"
            })

        # Edge from last moove to END
        last_moove_id = moove_to_node_map[max_moove_sequence[-1]]
        edges.append({
            "data": {
                "id": f"{last_moove_id}_to_end",
                "source": last_moove_id,
                "target": "end",
                "sequence": len(max_moove_sequence),
                "coverage_gain": 0,
                "total_coverage": total_coverage,
                "moo_count": max_moo_count_sequence[-1] if max_moo_count_sequence else 0
            },
            "classes": "path-edge"
        })

    # Add ALL possible edges based on the overlap graph (but not part of the path)
    max_moove_set = set(max_moove_sequence)

    for moove1, overlapping_mooves in graph.items():
        source_id = moove_to_node_map[moove1]

        for moove2 in overlapping_mooves:
            # Skip if this edge is part of the path
            if moove1 in max_moove_set and moove2 in max_moove_set:
                idx1 = max_moove_sequence.index(moove1) if moove1 in max_moove_sequence else -1
                idx2 = max_moove_sequence.index(moove2) if moove2 in max_moove_sequence else -1
                if abs(idx1 - idx2) == 1:
                    continue

            target_id = moove_to_node_map[moove2]
            edge_id = f"{source_id}_overlap_{target_id}"

            # Only add if not already added (avoid duplicates)
            if not any(e["data"]["id"] == edge_id for e in edges):
                edges.append({
                    "data": {
                        "id": edge_id,
                        "source": source_id,
                        "target": target_id
                    },
                    "classes": "possible-edge"
                })

    # Create the complete Cytoscape data structure
    cytoscape_data = {
        "elements": {
            "nodes": nodes,
            "edges": edges
        },
        "style": [
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "background-color": "#666",
                    "text-valign": "center",
                    "text-halign": "center",
                    "width": 30,
                    "height": 30,
                    "font-size": 10
                }
            },
            {
                "selector": ".start-node",
                "style": {
                    "background-color": "#27ae60",
                    "shape": "diamond",
                    "width": 50,
                    "height": 50,
                    "font-weight": "bold"
                }
            },
            {
                "selector": ".end-node",
                "style": {
                    "background-color": "#e74c3c",
                    "shape": "diamond",
                    "width": 50,
                    "height": 50,
                    "font-weight": "bold"
                }
            },
            {
                "selector": ".path-node",
                "style": {
                    "background-color": "#3498db",
                    "border-width": 3,
                    "border-color": "#2980b9"
                }
            },
            {
                "selector": ".regular-node",
                "style": {
                    "background-color": "#95a5a6",
                    "opacity": 0.7
                }
            },
            {
                "selector": ".path-edge",
                "style": {
                    "line-color": "#3498db",
                    "target-arrow-color": "#3498db",
                    "width": 4,
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "label": "data(sequence)",
                    "text-rotation": "autorotate",
                    "text-margin-y": -10,
                    "font-size": 12,
                    "font-weight": "bold"
                }
            },
            {
                "selector": ".possible-edge",
                "style": {
                    "line-color": "#333",
                    "width": 1,
                    "line-style": "dashed",
                    "opacity": 0.3
                }
            }
        ],
        "layout": {
            "name": "cose",
            "animate": False,
            "nodeDimensionsIncludeLabels": True,
            "nodeRepulsion": 400000,
            "idealEdgeLength": 100,
            "edgeElasticity": 100,
            "nestingFactor": 5,
            "gravity": 80,
            "numIter": 1000,
            "componentSpacing": 100
        }
    }

    return cytoscape_data


def generate_sequence_mcts(
        all_valid_mooves: MooveSequence,
        dims: GridDimensions,
        graph: MooveOverlapGraph,
        seed: int = 42,
        highest_score: bool = True,
        iterations: int = 1000,
        exploration_constant: float = 1.414
    ) -> MooveSequence:
    """Generate a sequence using Monte Carlo Tree Search (MCTS).
    
    MCTS builds a search tree by:
    1. Selection: Use UCB1 to select promising nodes
    2. Expansion: Add a new child node
    3. Simulation: Random rollout from the new node
    4. Backpropagation: Update statistics back up the tree
    """
    import math
    from dataclasses import dataclass, field
    from typing import Optional
    
    random.seed(seed)
    
    @dataclass
    class MCTSNode:
        board: BoardState
        moo_count: int
        moove: Optional[Moove]  # The moove that led to this state
        remaining_mooves: set[Moove]
        parent: Optional['MCTSNode'] = None
        children: list['MCTSNode'] = field(default_factory=list)
        visits: int = 0
        total_score: float = 0.0
        unexplored_mooves: list[Moove] = field(default_factory=list)
        
        def __post_init__(self):
            # Initialize unexplored mooves with all viable mooves
            if not self.unexplored_mooves and self.remaining_mooves:
                self.unexplored_mooves = [
                    m for m in self.remaining_mooves 
                    if get_moove_coverage(self.board, m) < 3
                ]
                random.shuffle(self.unexplored_mooves)
        
        @property
        def average_score(self) -> float:
            return self.total_score / self.visits if self.visits > 0 else 0.0
        
        @property
        def is_fully_expanded(self) -> bool:
            return len(self.unexplored_mooves) == 0
        
        @property
        def is_terminal(self) -> bool:
            # Terminal if no viable mooves remain
            return all(get_moove_coverage(self.board, m) >= 3 
                      for m in self.remaining_mooves)
        
        def ucb1_score(self, exploration: float) -> float:
            """Calculate UCB1 score for node selection."""
            if self.visits == 0:
                return float('inf')
            
            exploitation = self.average_score
            exploration_term = exploration * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
            
            # For MAX, we want high scores; for MIN, we want low scores
            if highest_score:
                return exploitation + exploration_term
            else:
                return -exploitation + exploration_term
        
        def select_child(self, exploration: float) -> 'MCTSNode':
            """Select best child using UCB1."""
            return max(self.children, key=lambda c: c.ucb1_score(exploration))
        
        def expand(self) -> 'MCTSNode':
            """Expand by adding a new child node."""
            if not self.unexplored_mooves:
                return self
            
            # Take next unexplored moove
            moove = self.unexplored_mooves.pop()
            
            # Create new board state
            new_board, new_count, _ = update_board_with_moove(
                self.board, self.moo_count, moove
            )
            
            # Create child node
            child = MCTSNode(
                board=new_board,
                moo_count=new_count,
                moove=moove,
                remaining_mooves=self.remaining_mooves - {moove},
                parent=self
            )
            
            self.children.append(child)
            return child
        
        def rollout(self) -> int:
            """Perform random rollout from this node to estimate value."""
            # Use greedy strategy for rollout (faster and often better)
            current_board = [row[:] for row in self.board]
            current_count = self.moo_count
            current_remaining = set(self.remaining_mooves)
            
            while current_remaining:
                # Find best moove greedily
                best_mooves = []
                best_coverage = -1 if highest_score else 4
                
                for moove in current_remaining:
                    coverage = get_moove_coverage(current_board, moove)
                    if coverage >= 3:
                        continue
                    
                    new_coverage = 3 - coverage
                    
                    if highest_score:
                        # MAX: prefer minimum new coverage (maximum overlap)
                        if best_coverage == -1 or new_coverage < best_coverage:
                            best_coverage = new_coverage
                            best_mooves = [moove]
                        elif new_coverage == best_coverage:
                            best_mooves.append(moove)
                    else:
                        # MIN: prefer maximum new coverage (minimum overlap)
                        if best_coverage == 4 or new_coverage > best_coverage:
                            best_coverage = new_coverage
                            best_mooves = [moove]
                        elif new_coverage == best_coverage:
                            best_mooves.append(moove)
                
                if not best_mooves:
                    break
                
                # Random selection among best
                selected = random.choice(best_mooves)
                current_board, current_count, _ = update_board_with_moove(
                    current_board, current_count, selected
                )
                current_remaining.remove(selected)
            
            return current_count
        
        def backpropagate(self, score: int):
            """Backpropagate score up the tree."""
            self.visits += 1
            self.total_score += score
            if self.parent:
                self.parent.backpropagate(score)
    
    # Initialize root node
    root = MCTSNode(
        board=generate_empty_board(dims),
        moo_count=0,
        moove=None,
        remaining_mooves=set(all_valid_mooves)
    )
    
    # Run MCTS iterations
    for _ in range(iterations):
        node = root
        
        # Selection: traverse tree using UCB1
        while not node.is_terminal and node.is_fully_expanded:
            node = node.select_child(exploration_constant)
        
        # Expansion: add new child if not terminal
        if not node.is_terminal and not node.is_fully_expanded:
            node = node.expand()
        
        # Simulation: rollout from the node
        if not node.is_terminal:
            score = node.rollout()
        else:
            score = node.moo_count
        
        # Backpropagation: update statistics
        node.backpropagate(score)
    
    # Extract best path from root to leaf
    sequence = []
    current = root
    
    # Follow the most visited path in the tree
    while current.children:
        # Select child with highest visit count (most explored)
        best_child = max(current.children, key=lambda c: c.visits)
        if best_child.moove:
            sequence.append(best_child.moove)
        current = best_child
    
    # If the tree doesn't reach a terminal state, complete the game greedily
    # This is necessary because MCTS may not fully expand the tree in limited iterations
    if not current.is_terminal and current.remaining_mooves:
        # Complete the sequence using greedy rollout from final tree node
        current_board = [row[:] for row in current.board]
        current_count = current.moo_count
        current_remaining = set(current.remaining_mooves)
        
        while current_remaining:
            # Find best moove greedily
            best_mooves = []
            best_coverage = -1 if highest_score else 4
            
            for moove in current_remaining:
                coverage = get_moove_coverage(current_board, moove)
                if coverage >= 3:
                    continue
                
                new_coverage = 3 - coverage
                
                if highest_score:
                    # MAX: prefer minimum new coverage (maximum overlap)
                    if best_coverage == -1 or new_coverage < best_coverage:
                        best_coverage = new_coverage
                        best_mooves = [moove]
                    elif new_coverage == best_coverage:
                        best_mooves.append(moove)
                else:
                    # MIN: prefer maximum new coverage (minimum overlap)
                    if best_coverage == 4 or new_coverage > best_coverage:
                        best_coverage = new_coverage
                        best_mooves = [moove]
                    elif new_coverage == best_coverage:
                        best_mooves.append(moove)
            
            if not best_mooves:
                break
            
            # Random selection among best
            selected = random.choice(best_mooves)
            sequence.append(selected)
            current_board, current_count, _ = update_board_with_moove(
                current_board, current_count, selected
            )
            current_remaining.remove(selected)
    
    return sequence


def nth_permutation(elements, n):
    """
    Get the nth permutation of elements directly without iteration.
    n is 0-indexed.
    """

    elements = list(elements)
    k = len(elements)

    # Check bounds
    if n >= math.factorial(k):
        raise ValueError(f"Index {n} out of range for {k} elements")

    result = []
    available = elements.copy()

    # Convert n to factorial number system
    for i in range(k, 0, -1):
        factorial = math.factorial(i - 1)
        index = n // factorial
        n = n % factorial

        result.append(available.pop(index))

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moo Counter")
    parser.add_argument("--puzzle", type=str, help="Path to the puzzle file")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for random sampling")
    parser.add_argument(
        "--workers", type=int, default=-1, 
        help="Number of worker processes to use. Default is number of CPU cores."
    )
    parser.add_argument(
        "--strategy", type=str, default="all", 
        choices=["random", "greedy-high", "greedy-low", "greedy", "beam-search", "mcts", "all"],
        help="Strategy for generating moove sequences."
    )
    args = parser.parse_args(sys.argv[1:])

    if args.puzzle in valid_sizes:
        grid, content = grid_from_live(args.puzzle)
        dims = grid_dimensions(grid)
        output_puzzle_file = OUTPUT_PUZZLES_DIR / f"{today_date_str()}-{dims[0]}-{args.puzzle}.moo"
        output_puzzle_file.write_text(content)
        output_filepath = (OUTPUT_DIR / output_puzzle_file.name).with_suffix(".json")
    else:
        grid = grid_from_file(pathlib.Path(args.puzzle))
        dims = grid_dimensions(grid)
        output_filepath = (OUTPUT_DIR / f"{pathlib.Path(args.puzzle).name}").with_suffix(".json")
    
    print(render_board(generate_empty_board(dims), grid))
    all_valid_mooves = generate_all_valid_mooves(grid)
    graph = generate_overlaps_graph(all_valid_mooves)
    graph_degrees = {render_moove(k):len(graph[k]) for k in graph}
    graph_degrees = dict(sorted(graph_degrees.items(), key=lambda item: item[1], reverse=True))
    top_3 = list(graph_degrees.items())[:3]

    simulation_output = parallel_process_simulations(grid, args.iterations, args.workers, args.strategy)
    # Unpack simulation output
    (
        all_valid_mooves, max_coverage, dead_cells, 
        max_mooves, min_mooves, histogram, graph,
        max_moove_sequence, max_moo_count_sequence, max_moo_count_coverage_sequence,
        min_moove_sequence, min_moo_count_sequence, min_moo_count_coverage_sequence
    ) = simulation_output
    
    output = {
        "puzzle": str(output_filepath.stem),
        "dimensions": dims,
        "total_cells": dims[0] * dims[1],
        "total_valid_mooves": len(all_valid_mooves),
        "max_coverage": max_coverage,
        "dead_cells": dead_cells,
        "max_mooves": max_mooves,
        "min_mooves": min_mooves,
        "moo_count_histogram": histogram,
        "graph_degrees": graph_degrees,
        "max_moove_sequence": max_moove_sequence,
        "max_moo_count_sequence": max_moo_count_sequence,
        "max_moo_coverage_sequence": max_moo_count_coverage_sequence,
        "min_moove_sequence": min_moove_sequence,
        "min_moo_count_sequence": min_moo_count_sequence,
        "min_moo_coverage_sequence": min_moo_count_coverage_sequence,
        "rendered_max_moove_sequence": render_moove_sequence(
            max_moove_sequence, 
            max_moo_count_sequence, 
            max_moo_count_coverage_sequence
        ),
        "rendered_min_moove_sequence": render_moove_sequence(
            min_moove_sequence, 
            min_moo_count_sequence, 
            min_moo_count_coverage_sequence
        )
    }
    output_filepath.write_text(json.dumps(output, indent=2))

    # Generate Cytoscape graph data
    cytoscape_data = generate_cytoscape_graph(
        all_valid_mooves,
        graph,
        max_moove_sequence,
        max_moo_count_sequence,
        max_moo_count_coverage_sequence,
        dims
    )

    # Write Cytoscape graph to separate file
    cytoscape_filepath = output_filepath.with_name(f"{output_filepath.stem}_graph.json")
    cytoscape_filepath.write_text(json.dumps(cytoscape_data, indent=2))

    print(f"PUZZLE: {output_filepath.stem}")
    print(render_moove_sequence(
        max_moove_sequence,
        max_moo_count_sequence,
        max_moo_count_coverage_sequence
    ))
    print(render_moo_count_histogram(histogram, screen_width=40))
    print(f"Max mooves: {max_mooves}, Min mooves: {min_mooves}, Max coverage: {max_coverage}, Dead cells: {dead_cells}")
    print(f"Output written to {output_filepath}")
    print(f"Cytoscape graph written to {cytoscape_filepath}")
    