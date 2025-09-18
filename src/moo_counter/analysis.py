"""Analysis functions for simulation results."""

import math
from typing import Any

from .moo_types import (
    MooveCountSequence, MooCountHistogram, Moove, MooveSequence
)
from .display import render_moove


def build_moo_count_histogram(all_moo_counts: MooveCountSequence) -> MooCountHistogram:
    """Build a histogram of moo counts from multiple simulations."""
    moo_count_histogram = {}
    for count in all_moo_counts:
        if count in moo_count_histogram:
            moo_count_histogram[count] += 1
        else:
            moo_count_histogram[count] = 1

    # Sort by keys
    moo_count_histogram = {
        k: moo_count_histogram[k]
        for k in sorted(moo_count_histogram.keys())
    }

    return moo_count_histogram


def analyze_graph_degrees(graph: dict[Moove, set[Moove]]) -> dict[str, int]:
    """Analyze the degree distribution of the moove overlap graph."""
    graph_degrees = {
        render_moove(k): len(graph[k])
        for k in graph
    }

    # Sort by degree (descending)
    graph_degrees = dict(
        sorted(graph_degrees.items(), key=lambda item: item[1], reverse=True)
    )

    return graph_degrees


def get_top_overlapping_mooves(
    graph: dict[Moove, set[Moove]],
    n: int = 3
) -> list[tuple[str, int]]:
    """Get the top n mooves with the most overlaps."""
    graph_degrees = analyze_graph_degrees(graph)
    return list(graph_degrees.items())[:n]


def calculate_statistics(moo_counts: MooveCountSequence) -> dict[str, float]:
    """Calculate statistical measures for moo counts."""
    if not moo_counts:
        return {
            "mean": 0,
            "median": 0,
            "std_dev": 0,
            "min": 0,
            "max": 0,
            "range": 0
        }

    n = len(moo_counts)
    mean = sum(moo_counts) / n

    # Median
    sorted_counts = sorted(moo_counts)
    if n % 2 == 0:
        median = (sorted_counts[n//2 - 1] + sorted_counts[n//2]) / 2
    else:
        median = sorted_counts[n//2]

    # Standard deviation
    variance = sum((x - mean) ** 2 for x in moo_counts) / n
    std_dev = math.sqrt(variance)

    return {
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "min": min(moo_counts),
        "max": max(moo_counts),
        "range": max(moo_counts) - min(moo_counts)
    }


def analyze_coverage_efficiency(
    moove_sequence: MooveSequence,
    coverage_gains: list[int]
) -> dict[str, Any]:
    """Analyze how efficiently the sequence covers the board."""
    if not moove_sequence or not coverage_gains:
        return {
            "total_mooves": 0,
            "total_coverage": 0,
            "average_coverage_per_moove": 0,
            "efficiency_ratio": 0,
            "wasted_mooves": 0
        }

    total_mooves = len(moove_sequence)
    total_coverage = sum(coverage_gains)
    max_possible_coverage = total_mooves * 3  # Each moove can cover 3 cells max

    # Count mooves that didn't add any new coverage
    wasted_mooves = sum(1 for gain in coverage_gains if gain == 0)

    return {
        "total_mooves": total_mooves,
        "total_coverage": total_coverage,
        "average_coverage_per_moove": total_coverage / total_mooves if total_mooves > 0 else 0,
        "efficiency_ratio": total_coverage / max_possible_coverage if max_possible_coverage > 0 else 0,
        "wasted_mooves": wasted_mooves,
        "wasted_percentage": (wasted_mooves / total_mooves * 100) if total_mooves > 0 else 0
    }


def compare_strategies(results_by_strategy: dict[str, dict]) -> dict[str, Any]:
    """Compare results from different strategies."""
    comparison = {}

    for strategy_name, results in results_by_strategy.items():
        histogram = results.get("histogram", {})
        all_counts = []

        # Reconstruct counts from histogram
        for count, freq in histogram.items():
            all_counts.extend([int(count)] * freq)

        stats = calculate_statistics(all_counts)

        comparison[strategy_name] = {
            "statistics": stats,
            "best_score": results.get("max_score", 0),
            "worst_score": results.get("min_score", 0),
            "consistency": 1 - (stats["std_dev"] / stats["mean"]) if stats["mean"] > 0 else 0
        }

    # Find best strategy by different metrics
    if comparison:
        best_by_max = max(comparison.items(), key=lambda x: x[1]["best_score"])
        best_by_mean = max(comparison.items(), key=lambda x: x[1]["statistics"]["mean"])
        most_consistent = max(comparison.items(), key=lambda x: x[1]["consistency"])

        comparison["summary"] = {
            "best_by_max_score": best_by_max[0],
            "best_by_average": best_by_mean[0],
            "most_consistent": most_consistent[0]
        }

    return comparison


def nth_permutation(elements: list, n: int) -> list:
    """Get the nth permutation of elements directly without iteration.

    This uses the factorial number system (factoradic) to directly
    compute the nth permutation.

    Args:
        elements: List of elements to permute
        n: 0-indexed permutation number

    Returns:
        The nth permutation of the elements
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