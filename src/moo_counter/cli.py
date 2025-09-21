# Standard Library
import argparse


def parse_arguments(available_engines: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moo Counter - Find optimal 'moo' sequences")

    parser.add_argument(
        "--puzzle", type=str, required=True, help="Path to puzzle file or size (micro/mini/maxi) for live puzzle"
    )

    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of iterations for simulation (default: 1000)"
    )

    parser.add_argument("--workers", type=int, default=-1, help="Number of worker processes (-1 for auto, default: -1)")

    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["random", "greedy-high", "greedy-low", "greedy", "mcts", "all"],
        help="Strategy for generating moove sequences (default: all)",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="python",
        choices=available_engines,
        help=f"Game engine to use. Available: {', '.join(available_engines)} (default: python)",
    )

    parser.add_argument("--benchmark", action="store_true", help="Run benchmark comparison of all available engines")

    parser.add_argument("--compare-engines", action="store_true", help="Compare different engine implementations")

    args = parser.parse_args()
    return args
