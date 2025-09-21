"""Strategy implementations for generating moove sequences."""

# Standard Library
import random

from ..engine import GameEngine
from ..moo_types import (
    GridDimensions,
    MooveOverlapGraph,
    MooveSequence,
)
from .greedy import GreedyStrategy, MixedGreedyStrategy
from .mcts import MCTSStrategy
from .strategy import RandomStrategy, Strategy


class AllStrategiesStrategy(Strategy):
    """Randomly selects from all available strategies."""

    def __init__(self, engine: GameEngine, seed: int = 42):
        super().__init__(engine, seed)
        self.strategies = [
            RandomStrategy(engine, seed),
            GreedyStrategy(engine, seed, highest_score=True),
            GreedyStrategy(engine, seed, highest_score=False),
            MCTSStrategy(engine, seed),
        ]

    @property
    def name(self) -> str:
        return "all"

    def generate_sequence(
        self, all_valid_mooves: MooveSequence, dims: GridDimensions, graph: MooveOverlapGraph
    ) -> MooveSequence:
        """Randomly select a strategy and use it."""
        strategy = random.choice(self.strategies)
        return strategy.generate_sequence(all_valid_mooves, dims, graph)


def create_strategy(name: str, engine: GameEngine, seed: int = 42) -> Strategy:
    """Factory function to create strategies by name."""
    strategies = {
        "random": lambda: RandomStrategy(engine, seed),
        "greedy-high": lambda: GreedyStrategy(engine, seed, highest_score=True),
        "greedy-low": lambda: GreedyStrategy(engine, seed, highest_score=False),
        "greedy": lambda: MixedGreedyStrategy(engine, seed),
        "mcts": lambda: MCTSStrategy(engine, seed),
        "all": lambda: AllStrategiesStrategy(engine, seed),
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name]()


__all__ = [
    "Strategy",
    "RandomStrategy",
    "GreedyStrategy",
    "AllStrategiesStrategy",
    "MixedGreedyStrategy",
    "MCTSStrategy",
    "create_strategy",
]
