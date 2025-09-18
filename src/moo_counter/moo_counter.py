#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx",
#   "playwright",
# ]
# ///
"""Moo Counter - Refactored main entry point."""

import argparse
import json
import pathlib

# Import from refactored modules
from .moo_types import VALID_SIZES
from .engine import EngineFactory, PythonEngine
from .display import (
    render_board, render_moove_sequence, render_moo_count_histogram,
    generate_cytoscape_graph
)
from .utils import (
    grid_from_live, grid_from_file, save_puzzle, get_output_filename
)
from .parallel import ParallelSimulator, benchmark_engines
from .analysis import analyze_graph_degrees, get_top_overlapping_mooves


# Project paths
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

OUTPUT_PUZZLES_DIR = PROJECT_ROOT / "puzzles"
OUTPUT_PUZZLES_DIR.mkdir(exist_ok=True, parents=True)


def main():
    """Main entry point for Moo Counter."""
    parser = argparse.ArgumentParser(description="Moo Counter - Find optimal 'moo' sequences")

    parser.add_argument(
        "--puzzle",
        type=str,
        required=True,
        help="Path to puzzle file or size (micro/mini/maxi) for live puzzle"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations for simulation (default: 1000)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of worker processes (-1 for auto, default: -1)"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["random", "greedy-high", "greedy-low", "greedy", "mcts", "all"],
        help="Strategy for generating moove sequences (default: all)"
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="python",
        choices=["python"],  # Add more as they become available
        help="Game engine to use (default: python)"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparison of all available engines"
    )

    parser.add_argument(
        "--compare-engines",
        action="store_true",
        help="Compare different engine implementations"
    )

    args = parser.parse_args()

    # Get engine
    engine = EngineFactory.get_engine(args.engine)

    # Load or fetch puzzle
    if args.puzzle in VALID_SIZES:
        grid, content = grid_from_live(args.puzzle)
        dims = engine.get_grid_dimensions(grid)

        # Save the live puzzle
        puzzle_path = save_puzzle(content, args.puzzle, OUTPUT_PUZZLES_DIR)
        output_filepath = get_output_filename(args.puzzle, OUTPUT_DIR)
    else:
        puzzle_path = pathlib.Path(args.puzzle)
        grid = grid_from_file(puzzle_path)
        dims = engine.get_grid_dimensions(grid)
        output_filepath = get_output_filename(puzzle_path, OUTPUT_DIR)

    print(f"\n{'='*50}")
    print(f"Moo Counter - {args.engine.upper()} Engine")
    print(f"{'='*50}")
    print(f"Puzzle: {puzzle_path.stem}")
    print(f"Dimensions: {dims[0]}x{dims[1]}")
    print(f"Strategy: {args.strategy}")
    print(f"Iterations: {args.iterations}")
    print()

    # Display empty board
    print("Initial board:")
    print(render_board(engine.generate_empty_board(dims), grid))

    # Run benchmark if requested
    if args.benchmark:
        print(f"\n{'='*50}")
        print("Running engine benchmark...")
        print(f"{'='*50}")

        engines = {"python": PythonEngine()}
        # Add more engines as they become available
        # engines["rust"] = RustEngine()
        # engines["c"] = CEngine()

        benchmark_results = benchmark_engines(
            engines, grid, args.iterations, args.workers, args.strategy
        )

        # Save benchmark results
        benchmark_file = output_filepath.with_name(f"{output_filepath.stem}_benchmark.json")
        benchmark_file.write_text(json.dumps(benchmark_results, indent=2))
        print(f"\nBenchmark results saved to: {benchmark_file}")
        return

    # Run simulations
    simulator = ParallelSimulator(engine)
    results = simulator.run_simulations(grid, args.iterations, args.workers, args.strategy)

    # Extract results
    all_valid_mooves = results["all_valid_mooves"]
    max_result = results["max_result"]
    min_result = results["min_result"]
    histogram = results["histogram"]
    graph = results["graph"]
    max_coverage = results["max_coverage"]
    dead_cells = results["dead_cells"]

    # Analyze graph degrees
    graph_degrees = analyze_graph_degrees(graph)
    top_3 = get_top_overlapping_mooves(graph, 3)

    # Prepare output data
    output = {
        "puzzle": str(output_filepath.stem),
        "engine": engine.name,
        "dimensions": dims,
        "total_cells": dims[0] * dims[1],
        "total_valid_mooves": len(all_valid_mooves),
        "max_coverage": max_coverage,
        "dead_cells": dead_cells,
        "max_mooves": max_result.moo_count,
        "min_mooves": min_result.moo_count,
        "moo_count_histogram": histogram,
        "graph_degrees": graph_degrees,
        "top_overlapping_mooves": [{"moove": m, "overlaps": c} for m, c in top_3],
        "max_moove_sequence": [
            (m.t1, m.t2, m.t3) for m in max_result.moove_sequence
        ],  # Convert Moove to tuple for JSON
        "max_moo_count_sequence": max_result.moo_count_sequence,
        "max_moo_coverage_sequence": max_result.moo_coverage_gain_sequence,
        "min_moove_sequence": [
            (m.t1, m.t2, m.t3) for m in min_result.moove_sequence
        ],
        "min_moo_count_sequence": min_result.moo_count_sequence,
        "min_moo_coverage_sequence": min_result.moo_coverage_gain_sequence,
        "rendered_max_moove_sequence": render_moove_sequence(
            max_result.moove_sequence,
            max_result.moo_count_sequence,
            max_result.moo_coverage_gain_sequence
        ),
        "rendered_min_moove_sequence": render_moove_sequence(
            min_result.moove_sequence,
            min_result.moo_count_sequence,
            min_result.moo_coverage_gain_sequence
        )
    }

    # Save main output
    output_filepath.write_text(json.dumps(output, indent=2))

    # Generate and save Cytoscape graph
    cytoscape_data = generate_cytoscape_graph(
        all_valid_mooves,
        graph,
        max_result.moove_sequence,
        max_result.moo_count_sequence,
        max_result.moo_coverage_gain_sequence,
        dims
    )

    cytoscape_filepath = output_filepath.with_name(f"{output_filepath.stem}_graph.json")
    cytoscape_filepath.write_text(json.dumps(cytoscape_data, indent=2))

    # Display results
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"PUZZLE: {output_filepath.stem}")
    print(f"Engine: {engine.name}")
    print()
    print("Best sequence found:")
    print(render_moove_sequence(
        max_result.moove_sequence,
        max_result.moo_count_sequence,
        max_result.moo_coverage_gain_sequence
    ))

    print("\nScore distribution:")
    print(render_moo_count_histogram(histogram, screen_width=40))

    print("\nStatistics:")
    print(f"  Max mooves: {max_result.moo_count}")
    print(f"  Min mooves: {min_result.moo_count}")
    print(f"  Max coverage: {max_coverage}")
    print(f"  Dead cells: {dead_cells}")
    print(f"  Total valid mooves: {len(all_valid_mooves)}")

    if top_3:
        print("\nMost overlapping mooves:")
        for moove_str, overlaps in top_3:
            print(f"  {moove_str}: {overlaps} overlaps")

    print(f"\nOutput written to: {output_filepath}")
    print(f"Graph data written to: {cytoscape_filepath}")

    # Compare engines if requested
    if args.compare_engines:
        print(f"\n{'='*50}")
        print("Comparing engine implementations...")
        print(f"{'='*50}")

        # For now, just show the Python engine
        # In the future, this will compare Python vs Rust vs C
        engines = {"python": PythonEngine()}

        for name, eng in engines.items():
            print(f"\n{name} engine:")
            eng.benchmark(grid, min(args.iterations, 100))


if __name__ == "__main__":
    main()