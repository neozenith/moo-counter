"""Utility functions for Moo Counter."""

import pathlib
import time
from playwright.sync_api import sync_playwright

from .moo_types import Grid, VALID_SIZES


def today_date_str() -> str:
    """Get today's date as a string in YYYYMMDD format."""
    return time.strftime("%Y%m%d", time.localtime())


def fetch_live_puzzle_input(size: str) -> str:
    """Fetch the live puzzle input from the website.

    Args:
        size: One of 'micro', 'mini', 'maxi'

    Returns:
        The puzzle content as a string
    """
    if size not in VALID_SIZES:
        raise ValueError(f"Invalid size: {size}. Must be one of {VALID_SIZES}")

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
    """Generate the grid from the live puzzle input.

    Args:
        size: One of 'micro', 'mini', 'maxi'

    Returns:
        Tuple of (grid, raw_content)
    """
    content = fetch_live_puzzle_input(size)
    grid = [list(line.replace(" ", "")) for line in content.splitlines()]
    return grid, content


def grid_from_file(path: pathlib.Path) -> Grid:
    """Load a grid from a file.

    Args:
        path: Path to the puzzle file

    Returns:
        The loaded grid
    """
    if not path.exists():
        raise FileNotFoundError(f"Puzzle file not found: {path}")

    with open(path, "r") as f:
        lines = f.readlines()

    grid = []
    for line in lines:
        row = list(line.strip())
        grid.append(row)

    return grid


def save_puzzle(content: str, size: str, output_dir: pathlib.Path) -> pathlib.Path:
    """Save a puzzle to a file with today's date.

    Args:
        content: The puzzle content
        size: The puzzle size
        output_dir: Directory to save the puzzle

    Returns:
        Path to the saved file
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get dimensions from content
    lines = content.splitlines()
    height = len(lines)

    filename = f"{today_date_str()}-{height}-{size}.moo"
    output_path = output_dir / filename
    output_path.write_text(content)

    return output_path


def get_output_filename(puzzle_path: str | pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    """Generate output filename for results.

    Args:
        puzzle_path: Path to the puzzle file or size name
        output_dir: Directory for output files

    Returns:
        Path for the output JSON file
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    if isinstance(puzzle_path, str) and puzzle_path in VALID_SIZES:
        # Live puzzle
        from .engine import PythonEngine
        engine = PythonEngine()
        grid, _ = grid_from_live(puzzle_path)
        dims = engine.get_grid_dimensions(grid)
        filename = f"{today_date_str()}-{dims[0]}-{puzzle_path}.json"
    else:
        # File-based puzzle
        path = pathlib.Path(puzzle_path)
        filename = f"{path.stem}.json"

    return output_dir / filename


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "1m 23s" or "45.2s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"


def calculate_permutation_space(n: int) -> str:
    """Calculate and format the size of the permutation space.

    Args:
        n: Number of elements

    Returns:
        Human-readable string describing the permutation space
    """
    import math

    factorial = math.factorial(n)

    if factorial < 1000:
        return str(factorial)
    elif factorial < 1_000_000:
        return f"{factorial / 1000:.1f}K"
    elif factorial < 1_000_000_000:
        return f"{factorial / 1_000_000:.1f}M"
    else:
        # Use scientific notation for very large numbers
        exponent = math.log10(factorial)
        return f"10^{exponent:.0f}"