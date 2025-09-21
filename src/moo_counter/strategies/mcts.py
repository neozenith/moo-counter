# noqa: C901
"""Strategy implementations for generating moove sequences."""

# Standard Library
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ..engine import GameEngine
from ..moo_types import (
    BoardState,
    GridDimensions,
    Moove,
    MooveOverlapGraph,
    MooveSequence,
)
from .strategy import Strategy


class MCTSStrategy(Strategy):
    """Monte Carlo Tree Search strategy for sequence generation."""

    def __init__(
        self,
        engine: GameEngine,
        seed: int = 42,
        highest_score: bool = True,
        iterations: int = 200,
        exploration_constant: float = 1.414,
    ):
        super().__init__(engine, seed)
        self.highest_score = highest_score
        self.iterations = iterations
        self.exploration_constant = exploration_constant

    @property
    def name(self) -> str:
        return "mcts"

    def generate_sequence(
        self, all_valid_mooves: MooveSequence, dims: GridDimensions, graph: MooveOverlapGraph
    ) -> MooveSequence:
        """Generate sequence using Monte Carlo Tree Search."""

        # Store engine and highest_score in local variables for closure
        engine = self.engine
        highest_score = self.highest_score

        @dataclass
        class MCTSNode:
            board: BoardState
            moo_count: int
            moove: Moove | None
            remaining_mooves: set[Moove]
            parent: Optional["MCTSNode"] = None
            children: list["MCTSNode"] = field(default_factory=list)
            visits: int = 0
            total_score: float = 0.0
            unexplored_mooves: list[Moove] = field(default_factory=list)

            def __post_init__(self):
                if not self.unexplored_mooves and self.remaining_mooves:
                    self.unexplored_mooves = [
                        m for m in self.remaining_mooves if engine.get_moove_coverage(self.board, m) < 3
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
                return all(engine.get_moove_coverage(self.board, m) >= 3 for m in self.remaining_mooves)

            def ucb1_score(self, exploration: float) -> float:
                if self.visits == 0:
                    return float("inf")

                exploitation = self.average_score
                exploration_term = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

                if highest_score:
                    return exploitation + exploration_term
                else:
                    return -exploitation + exploration_term

            def select_child(self, exploration: float) -> "MCTSNode":
                return max(self.children, key=lambda c: c.ucb1_score(exploration))

            def expand(self) -> "MCTSNode":
                if not self.unexplored_mooves:
                    return self

                moove = self.unexplored_mooves.pop()
                new_board, new_count, _ = engine.update_board_with_moove(self.board, self.moo_count, moove)

                child = MCTSNode(
                    board=new_board,
                    moo_count=new_count,
                    moove=moove,
                    remaining_mooves=self.remaining_mooves - {moove},
                    parent=self,
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
            board=engine.generate_empty_board(dims), moo_count=0, moove=None, remaining_mooves=set(all_valid_mooves)
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
                current_board, current_count, _ = engine.update_board_with_moove(current_board, current_count, selected)
                current_remaining.remove(selected)

        return sequence
