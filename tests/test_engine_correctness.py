"""Test suite to validate engine correctness against Python reference implementation."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moo_counter.engine import EngineFactory, PythonEngine
from moo_counter.moo_types import Moove


class TestEngineCorrectness:
    """Test all engines produce identical results to the Python reference implementation."""

    @pytest.fixture
    def test_grids(self):
        """Provide test grids of varying complexity."""
        return {
            "simple_3x3": [
                ["m", "o", "o"],
                ["o", "m", "o"],
                ["o", "o", "m"]
            ],
            "micro_5x5": [
                ['o', 'o', 'o', 'm', 'm'],
                ['m', 'm', 'm', 'o', 'm'],
                ['o', 'o', 'o', 'o', 'o'],
                ['o', 'o', 'o', 'o', 'm'],
                ['o', 'm', 'm', 'm', 'm']
            ],
            "no_moves": [
                ["o", "o", "o"],
                ["o", "o", "o"],
                ["o", "o", "o"]
            ],
            "single_moo": [
                ["m", "o", "o"],
                ["x", "x", "x"],
                ["x", "x", "x"]
            ],
            "complex_pattern": [
                ["m", "o", "o", "m"],
                ["o", "m", "o", "o"],
                ["o", "o", "m", "o"],
                ["m", "o", "o", "m"]
            ]
        }

    @pytest.fixture
    def python_engine(self):
        """Get the Python reference engine."""
        return EngineFactory.get_engine("python")

    @pytest.fixture
    def available_engines(self):
        """Get all available engines."""
        return EngineFactory.list_engines()

    def test_python_engine_as_reference(self, python_engine, test_grids):
        """Test that Python engine works correctly as our reference."""
        # Test simple grid
        grid = test_grids["simple_3x3"]
        dims = python_engine.get_grid_dimensions(grid)
        assert dims == (3, 3), "Grid dimensions should be (3, 3)"

        # Test finding moves
        mooves = python_engine.generate_all_valid_mooves(grid)
        assert isinstance(mooves, list), "Should return a list of mooves"

        # Test with no moves grid
        no_moves_grid = test_grids["no_moves"]
        mooves = python_engine.generate_all_valid_mooves(no_moves_grid)
        assert len(mooves) == 0, "Should find no moves in all-o grid"

        # Test single moo
        single_grid = test_grids["single_moo"]
        mooves = python_engine.generate_all_valid_mooves(single_grid)
        assert len(mooves) == 1, "Should find exactly one move"
        assert mooves[0] == Moove((0, 0), (0, 1), (0, 2)), "Should find horizontal moo at top"

    @pytest.mark.parametrize("engine_name", ["python", "c", "cython", "rust"])
    def test_engine_dimensions(self, engine_name, test_grids, python_engine):
        """Test that all engines report same dimensions as Python engine."""
        if engine_name not in EngineFactory.list_engines():
            pytest.skip(f"{engine_name} engine not available")

        engine = EngineFactory.get_engine(engine_name)

        for grid_name, grid in test_grids.items():
            expected_dims = python_engine.get_grid_dimensions(grid)
            actual_dims = engine.get_grid_dimensions(grid)

            assert actual_dims == expected_dims, \
                f"{engine_name} engine dimensions {actual_dims} != Python {expected_dims} for {grid_name}"

    @pytest.mark.parametrize("engine_name", ["python", "c", "cython", "rust"])
    def test_valid_mooves_count(self, engine_name, test_grids, python_engine):
        """Test that all engines find the same number of valid moves."""
        if engine_name not in EngineFactory.list_engines():
            pytest.skip(f"{engine_name} engine not available")

        engine = EngineFactory.get_engine(engine_name)

        for grid_name, grid in test_grids.items():
            expected_mooves = python_engine.generate_all_valid_mooves(grid)
            actual_mooves = engine.generate_all_valid_mooves(grid)

            assert len(actual_mooves) == len(expected_mooves), \
                f"{engine_name} found {len(actual_mooves)} moves != Python {len(expected_mooves)} for {grid_name}"

    @pytest.mark.parametrize("engine_name", ["python", "c", "cython", "rust"])
    def test_valid_mooves_content(self, engine_name, test_grids, python_engine):
        """Test that all engines find exactly the same moves."""
        if engine_name not in EngineFactory.list_engines():
            pytest.skip(f"{engine_name} engine not available")

        engine = EngineFactory.get_engine(engine_name)

        for grid_name, grid in test_grids.items():
            expected_mooves = set(python_engine.generate_all_valid_mooves(grid))
            actual_mooves = set(engine.generate_all_valid_mooves(grid))

            # Check for missing moves
            missing = expected_mooves - actual_mooves
            extra = actual_mooves - expected_mooves

            assert len(missing) == 0, \
                f"{engine_name} missing moves for {grid_name}: {missing}"
            assert len(extra) == 0, \
                f"{engine_name} has extra moves for {grid_name}: {extra}"

    @pytest.mark.parametrize("engine_name", ["python", "c", "cython", "rust"])
    def test_overlap_detection(self, engine_name, python_engine):
        """Test that all engines detect overlaps correctly."""

        engine = EngineFactory.get_engine(engine_name)

        # Test overlapping mooves
        m1 = Moove((0, 0), (0, 1), (0, 2))
        m2 = Moove((0, 1), (0, 2), (0, 3))  # Overlaps with m1
        m3 = Moove((1, 0), (1, 1), (1, 2))  # No overlap

        assert engine.do_mooves_overlap(m1, m2) == python_engine.do_mooves_overlap(m1, m2), \
            f"{engine_name} overlap detection differs from Python for overlapping mooves"

        assert engine.do_mooves_overlap(m1, m3) == python_engine.do_mooves_overlap(m1, m3), \
            f"{engine_name} overlap detection differs from Python for non-overlapping mooves"

    @pytest.mark.parametrize("engine_name", ["python", "c", "cython", "rust"])
    def test_simulation_results(self, engine_name, python_engine):
        """Test that all engines produce identical simulation results."""        

        engine = EngineFactory.get_engine(engine_name)

        # Create a simple test sequence
        test_mooves = [
            Moove((0, 0), (0, 1), (0, 2)),
            Moove((1, 0), (1, 1), (1, 2)),
            Moove((0, 1), (1, 1), (2, 1))  # This overlaps with previous moves
        ]
        dims = (3, 3)

        expected_result = python_engine.simulate_board(test_mooves, dims)
        actual_result = engine.simulate_board(test_mooves, dims)

        assert actual_result.moo_count == expected_result.moo_count, \
            f"{engine_name} moo_count {actual_result.moo_count} != Python {expected_result.moo_count}"

        assert actual_result.moo_count_sequence == expected_result.moo_count_sequence, \
            f"{engine_name} moo_count_sequence differs from Python"

        assert actual_result.moo_coverage_gain_sequence == expected_result.moo_coverage_gain_sequence, \
            f"{engine_name} coverage_gain_sequence differs from Python"

    @pytest.mark.parametrize("engine_name", ["python", "c", "cython", "rust"])
    def test_empty_board_generation(self, engine_name, python_engine):
        """Test that all engines generate empty boards correctly."""

        engine = EngineFactory.get_engine(engine_name)

        test_dimensions = [(3, 3), (5, 5), (10, 10)]

        for dims in test_dimensions:
            expected_board = python_engine.generate_empty_board(dims)
            actual_board = engine.generate_empty_board(dims)

            # Check dimensions
            assert len(actual_board) == len(expected_board), \
                f"{engine_name} board height {len(actual_board)} != Python {len(expected_board)}"

            if len(actual_board) > 0:
                assert len(actual_board[0]) == len(expected_board[0]), \
                    f"{engine_name} board width {len(actual_board[0])} != Python {len(expected_board[0])}"

            # Check all cells are False/0
            for row in actual_board:
                for cell in row:
                    assert cell in [False, 0], \
                        f"{engine_name} empty board contains non-False value: {cell}"

    def test_all_engines_available(self, available_engines):
        """Test that we have multiple engines available for testing."""
        assert len(available_engines) >= 1, "At least Python engine should be available"
        print(f"Available engines for testing: {available_engines}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])