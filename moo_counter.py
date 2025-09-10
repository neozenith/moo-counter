import itertools
import pathlib
import random
import time
import sys
import argparse


directions = [
    (-1, 0),   # up
    (-1, 1),   # up-right
    (0, 1),    # right
    (1, 1),    # down-right
    (1, 0),    # down
    (1, -1),   # down-left
    (0, -1),   # left
    (-1, -1)   # up-left
]

def grid_from_file(path: pathlib.Path) -> list[list[str]]:
    with open(path, 'r') as f:
        lines = f.readlines()
    
    grid = []
    for line in lines:
        row = list(line.strip())
        grid.append(row)
    
    return grid



def is_valid_moove(m: tuple[tuple[int, int], tuple[int, int], tuple[int, int]], grid: list[list[str]]) -> bool:
    """Check if three positions form a valid 'moo' move.

    m is a tuple of three tuples, each representing (row, column) positions on the grid.
    The rules for a valid 'moo' move are:
    - t1 must be the 'm' and t2 and t3 must be the 'o's.
    - t2 must be adjacent to t1 (horizontally or vertically or diagonally)
    - t3 must follow the same direction as t2 from t1.
    """

    # Unpack rows and columns
    t1, t2, t3 = m
    r1, c1 = t1
    r2, c2 = t2
    r3, c3 = t3

    # Check if positions are within bounds
    if r1 < 0 or r1 >= 15 or c1 < 0 or c1 >= 15:
        return False
    if r2 < 0 or r2 >= 15 or c2 < 0 or c2 >= 15:
        return False
    if r3 < 0 or r3 >= 15 or c3 < 0 or c3 >= 15:
        return False

    # Calculate direction from t1 to t2
    dr1 = r2 - r1
    dc1 = c2 - c1

    # Calculate direction from t2 to t3
    dr2 = r3 - r2
    dc2 = c3 - c2

    # Assume true and then test for invalidations

    # Check if it spells 'moo'
    # Check if t1 is 'm' and t2, t3 are 'o's
    if grid[r1][c1] != 'm' or grid[r2][c2] != 'o' or grid[r3][c3] != 'o':
        return False
    
    # Check if t2 is adjacent to t1
    if abs(dr1) > 1 or abs(dc1) > 1:
        return False
    
    # Check if t3 follows the same direction from t2
    # They should have the same vector (dr1, dc1) == (dr2, dc2)
    if dr1 != dr2 or dc1 != dc2:
        return False
    
    return True


def generate_moove(start: tuple[int, int], direction: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Generate a 'moo' move given a starting position and a direction.

    - start is the position of 'm' (t1).
    - direction is an integer from 0 to 7 representing the 8 possible directions clockwise
    """
    r1, c1 = start
    dr, dc = directions[direction]

    t1 = (r1, c1)
    t2 = (r1 + dr, c1 + dc)
    t3 = (r1 + 2 * dr, c1 + 2 * dc)

    return (t1, t2, t3)

def generate_all_valid_mooves(grid: list[list[str]]) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """Generate all possible 'moo' moves on the grid."""
    mooves = []
    for r in range(15):
        for c in range(15):
            for direction in range(8):
                moove = generate_moove((r, c), direction)
                if is_valid_moove(moove, grid):
                    mooves.append(moove)
    return mooves

def generate_all_permutations_of_moove_sequences(mooves: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]], length: int):
    """Generate all possible permutations of moove sequences of a given length."""
    return itertools.permutations(mooves, length)

def generate_empty_board() -> list[list[bool]]:
    """Generate an empty 15x15 board of False values."""
    return [[False for _ in range(15)] for _ in range(15)]

def update_board_with_moove(board: list[list[bool]], moo_count: int, moove: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> tuple[list[list[bool]], int]:
    """Update the board with a 'moo' move."""
    t1, t2, t3 = moove
    r1, c1 = t1
    r2, c2 = t2
    r3, c3 = t3

    # Place 'm' and 'o's on the board
    if board[r1][c1] and board[r2][c2] and board[r3][c3]:
        # Position already occupied. Moo count does not increase.
        ...
    else:
        # Mark positions as occupied
        board[r1][c1] = True
        board[r2][c2] = True
        board[r3][c3] = True
        # Update the moo count ONLY if at least one position was newly occupied
        moo_count += 1

    return board, moo_count

def render_board(board: list[list[bool]], grid: list[list[str]]) -> None:
    for r, row in enumerate(board):
        row_str = ''
        for c, cell in enumerate(row):
            if cell is True:
                row_str += grid[r][c].upper() + ' '
            elif cell is False:
                row_str += grid[r][c].lower() + ' '
            else:
                row_str += f'{cell} '
        print(row_str)

def simulate_board(mooves: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]) -> tuple[list[list[bool]], int]:
    """Simulate the board with a sequence of 'moo' moves."""
    board = generate_empty_board()
    moo_count = 0

    for moove in mooves:
        board, moo_count = update_board_with_moove(board, moo_count, moove)

    return board, moo_count

def render_moo_count_histogram(all_moo_counts: dict) -> None:
    moo_count_histogram = {}
    for count in all_moo_counts.values():
        if count in moo_count_histogram.keys():
            moo_count_histogram[count] += 1
        else:
            moo_count_histogram[count] = 1

    histogram_max_frequency = float(max(moo_count_histogram.values()))
    screen_width = 40.0
    max_stars = max(screen_width, histogram_max_frequency)

    sorted_moo_count_histogram_keys = sorted(moo_count_histogram.keys())
    for key in sorted_moo_count_histogram_keys:
        print(f"Moo count {key}: {int((moo_count_histogram[key] / max_stars) * screen_width) * '🐮'} {moo_count_histogram[key]}")

def main(puzzle_path: pathlib.Path, iterations: int) -> None:
    grid = grid_from_file(puzzle_path)
    all_valid_mooves = generate_all_valid_mooves(grid)
    print(f"Total valid 'moo' moves found: {len(all_valid_mooves)}")
    # The total permuations is 172! for today's grid which has like >300 zeroes in the count of permutations.
    # So we will try monte carlo style random sampling of the permutations.
    # The longer we run for the more permuations we will try.
    # We will track the maximum moo count found and how long it took to find each new maximum to find a point of diminishing returns.
    # We will also keep a track of past trials to prevent re-trying the same permutation.

    all_moo_counts = {} # key: hash of moove sequence, value: moo count

    # Time tracking to see diminishing returns for simulation.
    time_start = time.time()
    new_max_timestamps = [(time_start, 0.0, 0)]  # (absolute time, cumulative time, max mooves so far)

    max_mooves = 0
    max_board = None
    N_iters = iterations
    for i in range(N_iters):

        if (i+1) % 1000 == 0 or i == 0:
            print(f"Iteration {i+1}/{N_iters}")

        mooves = random.sample(all_valid_mooves, len(all_valid_mooves))
        hash_of_mooves = hash(tuple(mooves))
        if hash_of_mooves in all_moo_counts.keys():
            print("This set of mooves has already been tried. Skipping.")
            continue

        current_board, current_moo_count = simulate_board(mooves)

        all_moo_counts[hash_of_mooves] = current_moo_count

        if current_moo_count > max_mooves:
            max_mooves = current_moo_count
            max_board = current_board
            time_now = time.time()
            last_record = new_max_timestamps[-1] 
            cumulative_time = time_now - time_start
            time_since_last = time_now - last_record[0]
            new_max_time_record = (time_now, cumulative_time, max_mooves)
            new_max_timestamps.append(new_max_time_record)
            print(f"New maximum moo count found: {max_mooves} (previously {last_record[2]}) after {cumulative_time:.2f}s ( +{time_since_last:.2f}s )")
    
    time_now = time.time()
    total_time = time_now - time_start
    render_moo_count_histogram(all_moo_counts)
    print(f"Total valid 'moo' moves found: {len(all_valid_mooves)}")
    print(f"Theoretical maximum moo count: {max_mooves} after {N_iters} iterations")
    print(f"Total time taken: {total_time:.2f}s")
    render_board(max_board, grid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moo Counter")
    parser.add_argument("--puzzle", type=pathlib.Path, help="Path to the puzzle file")
    parser.add_argument("--iterations", type=int, default=3000000, help="Number of iterations for random sampling")
    args = parser.parse_args(sys.argv[1:])

    main(args.puzzle, args.iterations)
