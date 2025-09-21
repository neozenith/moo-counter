"""Type definitions for the Moo Counter game."""

# Standard Library
from dataclasses import dataclass
from typing import NamedTuple

# Basic types
type Grid = list[list[str]]
type GridDimensions = tuple[int, int]  # (rows, columns)
type BoardState = list[list[bool]]  # Current board coverage state
type Position = tuple[int, int]  # (row, column)
type Direction = int  # 0-7 representing 8 possible directions
type MooveDirection = tuple[int, int]  # (dr, dc)
type MooCount = int  # Total scored number of 'moo' moves placed


class Moove(NamedTuple):
    """Represents a 'moo' move with three positions."""

    t1: Position  # 'm' position
    t2: Position  # first 'o' position
    t3: Position  # second 'o' position


# Sequences and collections
type MooveSequence = list[Moove]
type MooveCountSequence = list[MooCount]
type MooveCoverageGainSequence = list[int]
type MooveOverlapGraph = dict[Moove, set[Moove]]


@dataclass
class MooveCandidate:
    """A candidate move with its coverage gain score."""

    moove: Moove
    coverage_gain: int


@dataclass
class SimulationResult:
    """Result of a single simulation run."""

    board: BoardState
    moo_count: MooCount
    moove_sequence: MooveSequence
    moo_count_sequence: MooveCountSequence
    moo_coverage_gain_sequence: MooveCoverageGainSequence


# Analysis types
type MooCountHistogram = dict[str, int]

# Constants
VALID_SIZES = ["micro", "mini", "maxi"]

# Direction mappings
DIRECTIONS: list[MooveDirection] = [
    (-1, 0),  # up
    (-1, 1),  # up-right
    (0, 1),  # right
    (1, 1),  # down-right
    (1, 0),  # down
    (1, -1),  # down-left
    (0, -1),  # left
    (-1, -1),  # up-left
]

DIRECTION_ARROWS = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]

DIRECTION_MAPPING = {
    (-1, 0): 0,  # up
    (-1, 1): 1,  # up-right
    (0, 1): 2,  # right
    (1, 1): 3,  # down-right
    (1, 0): 4,  # down
    (1, -1): 5,  # down-left
    (0, -1): 6,  # left
    (-1, -1): 7,  # up-left
}
