"""Strategy implementations for generating moove sequences."""

# Standard Library
import random
from abc import ABC, abstractmethod

from ..engine import GameEngine
from ..moo_types import (
    GridDimensions,
    MooveOverlapGraph,
    MooveSequence,
)


class Strategy(ABC):
    """Abstract base class for sequence generation strategies."""

    def __init__(self, engine: GameEngine, seed: int = 42):
        self.engine = engine
        self.seed = seed
        random.seed(seed)

    @abstractmethod
    def generate_sequence(
        self, all_valid_mooves: MooveSequence, dims: GridDimensions, graph: MooveOverlapGraph
    ) -> MooveSequence:
        """Generate a sequence of mooves using this strategy."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy."""
        pass


class RandomStrategy(Strategy):
    """Random shuffle of all valid mooves."""

    @property
    def name(self) -> str:
        return "random"

    def generate_sequence(
        self, all_valid_mooves: MooveSequence, dims: GridDimensions, graph: MooveOverlapGraph
    ) -> MooveSequence:
        """Generate a random permutation of all valid mooves."""
        return random.sample(all_valid_mooves, len(all_valid_mooves))
