"""Parallel simulation management for Moo Counter."""

# Standard Library
import os
import time
from collections.abc import Mapping
from multiprocessing import Pool

from .analysis import build_moo_count_histogram
from .engine import EngineFactory, GameEngine
from .moo_types import (
    Grid,
    GridDimensions,
    MooveOverlapGraph,
    MooveSequence,
    SimulationResult,
)
from .strategies import create_strategy
from .engines.wrappers import PythonEngine, CythonEngineWrapper, RustEngineWrapper, CEngineWrapper


EngineFactory.register_engine("python", PythonEngine())
EngineFactory.register_engine("cython", CythonEngineWrapper())
EngineFactory.register_engine("rust", RustEngineWrapper())
EngineFactory.register_engine("c", CEngineWrapper())

def worker_simulate(args: tuple[int, MooveSequence, GridDimensions, MooveOverlapGraph, str, str]) -> SimulationResult:
    """Worker function that generates and simulates a random permutation."""
    seed, all_valid_mooves, dims, graph, strategy_name, engine_name = args

    # Create engine and strategy for this worker
    # Use EngineFactory to get the correct engine implementation
    engine = EngineFactory.get_engine(engine_name)

    strategy = create_strategy(strategy_name, engine, seed)
    sequence = strategy.generate_sequence(all_valid_mooves, dims, graph)

    return engine.simulate_board(sequence, dims)


class ParallelSimulator:
    """Manages parallel execution of simulations."""

    def __init__(self, engine: GameEngine):
        self.engine = engine

    def run_simulations(self, grid: Grid, iterations: int, workers: int, strategy: str) -> dict:
        """Run multiple simulations in parallel and return results."""

        # Prepare data
        all_valid_mooves = self.engine.generate_all_valid_mooves(grid)
        dims = self.engine.get_grid_dimensions(grid)
        height, width = dims
        all_cells = height * width
        graph = self.engine.generate_overlaps_graph(all_valid_mooves)

        print(f"Total valid 'moo' moves found: {len(all_valid_mooves)}")
        print(f"Graph of overlapping Mooves has {len(graph)} nodes.")
        if graph:
            max_degree = max(len(v) for v in graph.values())
            print(f"Highest degree node has {max_degree} overlaps.")

        # Setup parallel processing
        time_start = time.time()

        # Create worker arguments (pass engine name instead of engine object)
        worker_args = [
            (i, all_valid_mooves, dims, graph, strategy, self.engine.name.lower()) for i in range(iterations)
        ]

        # Determine number of processes
        num_processes = workers if workers > 0 else (os.cpu_count() or 4)
        optimal_chunksize = max(1, iterations // (num_processes * 4))

        print(f"Using {num_processes} processes with chunksize {optimal_chunksize}")

        # Run simulations in parallel
        time_sims_start = time.time()

        with Pool(num_processes) as pool:
            results_iter = pool.imap_unordered(worker_simulate, worker_args, chunksize=optimal_chunksize)
            all_simulations: list[SimulationResult] = list(results_iter)

        time_parallel_end = time.time()
        time_parallel_duration = time_parallel_end - time_sims_start

        print(
            f"Simulations complete took {time_parallel_duration:.2f}s, "
            f"({iterations / time_parallel_duration:.0f} simulations per second)"
        )

        # Process results
        all_moo_counts = [result.moo_count for result in all_simulations]

        # Find max and min results
        max_result = max(all_simulations, key=lambda x: x.moo_count)
        min_result = min(all_simulations, key=lambda x: x.moo_count)

        # Build histogram
        histogram = build_moo_count_histogram(all_moo_counts)

        # Calculate coverage
        max_coverage = sum(max_result.moo_coverage_gain_sequence)
        dead_cells = all_cells - max_coverage

        time_reduce_end = time.time()
        time_reduce_duration = time_reduce_end - time_parallel_end

        print(
            f"Result processing took {time_reduce_duration:.2f}s after "
            f"{time_parallel_duration:.2f}s of parallel simulation."
        )

        total_time = time.time() - time_start
        total_sims_time = time.time() - time_sims_start

        print(f"Time taken for parallel simulation: {total_sims_time:.2f}s")
        print(f"Total time taken: {total_time:.2f}s")
        print(f"Simulations per second: {iterations / total_sims_time:.0f}")
        print()

        return {
            "all_valid_mooves": all_valid_mooves,
            "max_coverage": max_coverage,
            "dead_cells": dead_cells,
            "max_result": max_result,
            "min_result": min_result,
            "histogram": histogram,
            "graph": graph,
            "dims": dims,
            "all_simulations": all_simulations,
        }


def benchmark_engines(
    engines: Mapping[str, GameEngine],
    grid: Grid,
    iterations: int = 100,
    workers: int = -1,
    strategy: str = "greedy-high",
) -> dict[str, dict]:
    """Benchmark multiple engines with parallel simulations."""

    results = {}

    for name, engine in engines.items():
        print(f"\n{'='*50}")
        print(f"Benchmarking {name} engine")
        print(f"{'='*50}")

        simulator = ParallelSimulator(engine)
        start_time = time.time()

        sim_results = simulator.run_simulations(grid, iterations, workers, strategy)

        duration = time.time() - start_time

        results[name] = {
            "duration": duration,
            "max_score": sim_results["max_result"].moo_count,
            "min_score": sim_results["min_result"].moo_count,
            "histogram": sim_results["histogram"],
            "simulations_per_second": iterations / duration,
        }

        print(f"{name} engine completed in {duration:.2f}s")
        print(f"Max score: {results[name]['max_score']}, Min score: {results[name]['min_score']}")

    # Print comparison
    print(f"\n{'='*50}")
    print("Engine Comparison")
    print(f"{'='*50}")

    for name, result in results.items():
        print(
            f"{name:15} - {result['simulations_per_second']:.0f} sims/sec, "
            f"Max: {result['max_score']}, Min: {result['min_score']}"
        )

    # Find fastest
    fastest = min(results.items(), key=lambda x: x[1]["duration"])
    print(f"\nFastest engine: {fastest[0]} ({fastest[1]['simulations_per_second']:.0f} sims/sec)")

    return results
