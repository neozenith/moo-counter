"""Strategy implementations for generating moove sequences."""

import random
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .moo_types import (
    MooveSequence, GridDimensions, MooveOverlapGraph, Moove,
    BoardState, MooveCandidate
)
from .engine import GameEngine


class Strategy(ABC):
    """Abstract base class for sequence generation strategies."""

    def __init__(self, engine: GameEngine, seed: int = 42):
        self.engine = engine
        self.seed = seed
        random.seed(seed)

    @abstractmethod
    def generate_sequence(
        self,
        all_valid_mooves: MooveSequence,
        dims: GridDimensions,
        graph: MooveOverlapGraph
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
        self,
        all_valid_mooves: MooveSequence,
        dims: GridDimensions,
        graph: MooveOverlapGraph
    ) -> MooveSequence:
        """Generate a random permutation of all valid mooves."""
        return random.sample(all_valid_mooves, len(all_valid_mooves))


class GreedyStrategy(Strategy):
    """Greedy strategy that selects mooves based on immediate coverage gain."""

    def __init__(self, engine: GameEngine, seed: int = 42, highest_score: bool = True):
        super().__init__(engine, seed)
        self.highest_score = highest_score

    @property
    def name(self) -> str:
        return f"greedy-{'high' if self.highest_score else 'low'}"

    def generate_sequence(
        self,
        all_valid_mooves: MooveSequence,
        dims: GridDimensions,
        graph: MooveOverlapGraph
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
                _, _, coverage_gain = self.engine.update_board_with_moove(
                    board, moo_count, moove
                )
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
        self,
        all_valid_mooves: MooveSequence,
        dims: GridDimensions,
        graph: MooveOverlapGraph
    ) -> MooveSequence:
        """Randomly choose between high and low greedy."""
        self.highest_score = random.choice([True, False])
        return super().generate_sequence(all_valid_mooves, dims, graph)


class MCTSStrategy(Strategy):
    """Monte Carlo Tree Search strategy for sequence generation."""

    def __init__(
        self,
        engine: GameEngine,
        seed: int = 42,
        highest_score: bool = True,
        iterations: int = 200,
        exploration_constant: float = 1.414
    ):
        super().__init__(engine, seed)
        self.highest_score = highest_score
        self.iterations = iterations
        self.exploration_constant = exploration_constant

    @property
    def name(self) -> str:
        return "mcts"

    def generate_sequence(
        self,
        all_valid_mooves: MooveSequence,
        dims: GridDimensions,
        graph: MooveOverlapGraph
    ) -> MooveSequence:
        """Generate sequence using Monte Carlo Tree Search."""

        # Store engine and highest_score in local variables for closure
        engine = self.engine
        highest_score = self.highest_score

        @dataclass
        class MCTSNode:
            board: BoardState
            moo_count: int
            moove: Optional[Moove]
            remaining_mooves: set[Moove]
            parent: Optional['MCTSNode'] = None
            children: list['MCTSNode'] = field(default_factory=list)
            visits: int = 0
            total_score: float = 0.0
            unexplored_mooves: list[Moove] = field(default_factory=list)

            def __post_init__(self):
                if not self.unexplored_mooves and self.remaining_mooves:
                    self.unexplored_mooves = [
                        m for m in self.remaining_mooves
                        if engine.get_moove_coverage(self.board, m) < 3
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
                return all(
                    engine.get_moove_coverage(self.board, m) >= 3
                    for m in self.remaining_mooves
                )

            def ucb1_score(self, exploration: float) -> float:
                if self.visits == 0:
                    return float('inf')

                exploitation = self.average_score
                exploration_term = exploration * math.sqrt(
                    math.log(self.parent.visits) / self.visits
                )

                if highest_score:
                    return exploitation + exploration_term
                else:
                    return -exploitation + exploration_term

            def select_child(self, exploration: float) -> 'MCTSNode':
                return max(self.children, key=lambda c: c.ucb1_score(exploration))

            def expand(self) -> 'MCTSNode':
                if not self.unexplored_mooves:
                    return self

                moove = self.unexplored_mooves.pop()
                new_board, new_count, _ = engine.update_board_with_moove(
                    self.board, self.moo_count, moove
                )

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
                """Perform greedy rollout to estimate value."""
                current_board = [row[:] for row in self.board]
                current_count = self.moo_count
                current_remaining = set(self.remaining_mooves)

                while current_remaining:
                    best_mooves = []
                    best_coverage = -1 if highest_score else 4

                    for moove in current_remaining:
                        coverage = engine.get_moove_coverage(current_board, moove)
                        if coverage >= 3:
                            continue

                        new_coverage = 3 - coverage

                        if highest_score:
                            if best_coverage == -1 or new_coverage < best_coverage:
                                best_coverage = new_coverage
                                best_mooves = [moove]
                            elif new_coverage == best_coverage:
                                best_mooves.append(moove)
                        else:
                            if best_coverage == 4 or new_coverage > best_coverage:
                                best_coverage = new_coverage
                                best_mooves = [moove]
                            elif new_coverage == best_coverage:
                                best_mooves.append(moove)

                    if not best_mooves:
                        break

                    selected = random.choice(best_mooves)
                    current_board, current_count, _ = engine.update_board_with_moove(
                        current_board, current_count, selected
                    )
                    current_remaining.remove(selected)

                return current_count

            def backpropagate(self, score: int):
                self.visits += 1
                self.total_score += score
                if self.parent:
                    self.parent.backpropagate(score)

        # Initialize root node
        root = MCTSNode(
            board=engine.generate_empty_board(dims),
            moo_count=0,
            moove=None,
            remaining_mooves=set(all_valid_mooves)
        )

        # Run MCTS iterations
        for _ in range(self.iterations):
            node = root

            # Selection
            while not node.is_terminal and node.is_fully_expanded:
                node = node.select_child(self.exploration_constant)

            # Expansion
            if not node.is_terminal and not node.is_fully_expanded:
                node = node.expand()

            # Simulation
            score = node.rollout() if not node.is_terminal else node.moo_count

            # Backpropagation
            node.backpropagate(score)

        # Extract best path
        sequence = []
        current = root

        while current.children:
            best_child = max(current.children, key=lambda c: c.visits)
            if best_child.moove:
                sequence.append(best_child.moove)
            current = best_child

        # Complete sequence if needed
        if not current.is_terminal and current.remaining_mooves:
            current_board = [row[:] for row in current.board]
            current_count = current.moo_count
            current_remaining = set(current.remaining_mooves)

            while current_remaining:
                best_mooves = []
                best_coverage = -1 if highest_score else 4

                for moove in current_remaining:
                    coverage = engine.get_moove_coverage(current_board, moove)
                    if coverage >= 3:
                        continue

                    new_coverage = 3 - coverage

                    if highest_score:
                        if best_coverage == -1 or new_coverage < best_coverage:
                            best_coverage = new_coverage
                            best_mooves = [moove]
                        elif new_coverage == best_coverage:
                            best_mooves.append(moove)
                    else:
                        if best_coverage == 4 or new_coverage > best_coverage:
                            best_coverage = new_coverage
                            best_mooves = [moove]
                        elif new_coverage == best_coverage:
                            best_mooves.append(moove)

                if not best_mooves:
                    break

                selected = random.choice(best_mooves)
                sequence.append(selected)
                current_board, current_count, _ = engine.update_board_with_moove(
                    current_board, current_count, selected
                )
                current_remaining.remove(selected)

        return sequence


class AllStrategiesStrategy(Strategy):
    """Randomly selects from all available strategies."""

    def __init__(self, engine: GameEngine, seed: int = 42):
        super().__init__(engine, seed)
        self.strategies = [
            RandomStrategy(engine, seed),
            GreedyStrategy(engine, seed, highest_score=True),
            GreedyStrategy(engine, seed, highest_score=False),
            MCTSStrategy(engine, seed)
        ]

    @property
    def name(self) -> str:
        return "all"

    def generate_sequence(
        self,
        all_valid_mooves: MooveSequence,
        dims: GridDimensions,
        graph: MooveOverlapGraph
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
        "all": lambda: AllStrategiesStrategy(engine, seed)
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name]()