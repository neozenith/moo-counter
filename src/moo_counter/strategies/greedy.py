"""Strategy implementations for generating moove sequences."""

# Standard Library
import random

from ..engine import GameEngine
from ..moo_types import (
    GridDimensions,
    MooveCandidate,
    MooveOverlapGraph,
    MooveSequence,
)
from .strategy import Strategy


class GreedyStrategy(Strategy):
    """Greedy strategy that selects mooves based on immediate coverage gain."""

    def __init__(self, engine: GameEngine, seed: int = 42, highest_score: bool = True):
        super().__init__(engine, seed)
        self.highest_score = highest_score

    @property
    def name(self) -> str:
        return f"greedy-{'high' if self.highest_score else 'low'}"

    def generate_sequence(
        self, all_valid_mooves: MooveSequence, dims: GridDimensions, graph: MooveOverlapGraph
    ) -> MooveSequence:
        """Generate sequence by greedily selecting best coverage gain."""
        board = self.engine.generate_empty_board(dims)
        moove_sequence = []
        moo_count = 0
        remaining_mooves = set(all_valid_mooves)

        while remaining_mooves:
            best_moove_candidates: list[MooveCandidate] = []
            dead_moves = set()

            # Find next best moove
            for moove in remaining_mooves:
                _, _, coverage_gain = self.engine.update_board_with_moove(board, moo_count, moove)
                if coverage_gain <= 0:
                    dead_moves.add(moove)
                else:
                    best_moove_candidates.append(MooveCandidate(moove, coverage_gain))

            remaining_mooves -= dead_moves

            if not best_moove_candidates:
                break

            # Select based on strategy
            if self.highest_score:
                # MAX score: Want MIN coverage gain (= MAX overlap)
                min_coverage = min(mc.coverage_gain for mc in best_moove_candidates)
                candidates = [mc for mc in best_moove_candidates if mc.coverage_gain == min_coverage]
            else:
                # MIN score: Want MAX coverage gain (= MIN overlap)
                max_coverage = max(mc.coverage_gain for mc in best_moove_candidates)
                candidates = [mc for mc in best_moove_candidates if mc.coverage_gain == max_coverage]

            # Random selection among best candidates
            best_candidate = random.choice(candidates)
            best_moove = best_candidate.moove

            moove_sequence.append(best_moove)
            board, moo_count, _ = self.engine.update_board_with_moove(board, moo_count, best_moove)
            remaining_mooves.remove(best_moove)

        return moove_sequence


class MixedGreedyStrategy(GreedyStrategy):
    """Randomly alternates between greedy-high and greedy-low."""

    @property
    def name(self) -> str:
        return "greedy-mixed"

    def generate_sequence(
        self, all_valid_mooves: MooveSequence, dims: GridDimensions, graph: MooveOverlapGraph
    ) -> MooveSequence:
        """Randomly choose between high and low greedy."""
        self.highest_score = random.choice([True, False])
        return super().generate_sequence(all_valid_mooves, dims, graph)
