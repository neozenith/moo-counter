use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

type Position = (usize, usize);
type Moove = (Position, Position, Position);
type Grid = Vec<Vec<char>>;
type BoardState = Vec<Vec<bool>>;
type MooveDirection = (i32, i32);

const DIRECTIONS: [(i32, i32); 8] = [
    (-1, 0),  // up
    (-1, 1),  // up-right
    (0, 1),   // right
    (1, 1),   // down-right
    (1, 0),   // down
    (1, -1),  // down-left
    (0, -1),  // left
    (-1, -1), // up-left
];

#[pyclass]
pub struct RustEngine {
    name: String,
}

#[pymethods]
impl RustEngine {
    #[new]
    fn new() -> Self {
        RustEngine {
            name: "Rust".to_string(),
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn get_grid_dimensions(&self, grid: Vec<Vec<String>>) -> (usize, usize) {
        let rows = grid.len();
        let cols = if rows > 0 { grid[0].len() } else { 0 };
        (rows, cols)
    }

    fn is_valid_moove(&self, moove: ((usize, usize), (usize, usize), (usize, usize)), grid: Vec<Vec<String>>) -> bool {
        let height = grid.len();
        let width = if height > 0 { grid[0].len() } else { 0 };

        let (t1, t2, t3) = moove;
        let (r1, c1) = t1;
        let (r2, c2) = t2;
        let (r3, c3) = t3;

        // Check bounds
        if r1 >= height || c1 >= width || r2 >= height || c2 >= width || r3 >= height || c3 >= width {
            return false;
        }

        // Check letters spell 'moo'
        if grid[r1][c1] != "m" || grid[r2][c2] != "o" || grid[r3][c3] != "o" {
            return false;
        }

        // Calculate directions
        let d1 = (r2 as i32 - r1 as i32, c2 as i32 - c1 as i32);
        let d2 = (r3 as i32 - r2 as i32, c3 as i32 - c2 as i32);

        // Check t2 is adjacent to t1
        if d1.0.abs() > 1 || d1.1.abs() > 1 {
            return false;
        }

        // Check t3 follows same direction from t2
        d1 == d2
    }

    fn generate_moove(&self, start: (usize, usize), direction: usize) -> ((usize, usize), (usize, usize), (usize, usize)) {
        let d = DIRECTIONS[direction];
        let t1 = start;
        let t2 = (
            (start.0 as i32 + d.0) as usize,
            (start.1 as i32 + d.1) as usize,
        );
        let t3 = (
            (start.0 as i32 + 2 * d.0) as usize,
            (start.1 as i32 + 2 * d.1) as usize,
        );
        (t1, t2, t3)
    }

    fn generate_all_valid_mooves(&self, grid: Vec<Vec<String>>) -> Vec<((usize, usize), (usize, usize), (usize, usize))> {
        let height = grid.len();
        let width = if height > 0 { grid[0].len() } else { 0 };
        let mut mooves = Vec::new();

        for r in 0..height {
            for c in 0..width {
                for direction in 0..8 {
                    let moove = self.generate_moove((r, c), direction);
                    if self.is_valid_moove(moove, grid.clone()) {
                        mooves.push(moove);
                    }
                }
            }
        }

        mooves
    }

    fn generate_all_valid_mooves_parallel(&self, grid: Vec<Vec<String>>) -> Vec<((usize, usize), (usize, usize), (usize, usize))> {
        let height = grid.len();
        let width = if height > 0 { grid[0].len() } else { 0 };

        // Generate all possible positions and directions
        let mut positions_directions = Vec::new();
        for r in 0..height {
            for c in 0..width {
                for direction in 0..8 {
                    positions_directions.push(((r, c), direction));
                }
            }
        }

        // Process in parallel
        positions_directions
            .par_iter()
            .filter_map(|&(start, direction)| {
                let moove = self.generate_moove(start, direction);
                if self.is_valid_moove(moove, grid.clone()) {
                    Some(moove)
                } else {
                    None
                }
            })
            .collect()
    }

    fn do_mooves_overlap(&self, m1: ((usize, usize), (usize, usize), (usize, usize)),
                         m2: ((usize, usize), (usize, usize), (usize, usize))) -> bool {
        let positions1 = vec![m1.0, m1.1, m1.2];
        let positions2 = vec![m2.0, m2.1, m2.2];

        for p1 in &positions1 {
            for p2 in &positions2 {
                if p1 == p2 {
                    return true;
                }
            }
        }
        false
    }

    fn generate_overlaps_graph(&self, py: Python, mooves: Vec<((usize, usize), (usize, usize), (usize, usize))>) -> PyResult<PyObject> {
        let mut overlaps: HashMap<usize, HashSet<usize>> = HashMap::new();

        for i in 0..mooves.len() {
            for j in (i + 1)..mooves.len() {
                if self.do_mooves_overlap(mooves[i], mooves[j]) {
                    overlaps.entry(i).or_insert_with(HashSet::new).insert(j);
                    overlaps.entry(j).or_insert_with(HashSet::new).insert(i);
                }
            }
        }

        // Convert to Python dict
        let py_dict = pyo3::types::PyDict::new_bound(py);
        for (i, overlap_set) in overlaps {
            let moove = mooves[i];
            let py_moove = (moove.0, moove.1, moove.2).to_object(py);

            let py_set = pyo3::types::PySet::new_bound(py, &Vec::<PyObject>::new())?;
            for &j in &overlap_set {
                let other_moove = mooves[j];
                let py_other = (other_moove.0, other_moove.1, other_moove.2).to_object(py);
                py_set.add(py_other)?;
            }

            py_dict.set_item(py_moove, py_set)?;
        }

        Ok(py_dict.into())
    }

    fn generate_empty_board(&self, dims: (usize, usize)) -> Vec<Vec<bool>> {
        let (height, width) = dims;
        vec![vec![false; width]; height]
    }

    fn get_moove_coverage(&self, board: Vec<Vec<bool>>, moove: ((usize, usize), (usize, usize), (usize, usize))) -> usize {
        let mut coverage = 0;
        let positions = vec![moove.0, moove.1, moove.2];

        for pos in positions {
            if board[pos.0][pos.1] {
                coverage += 1;
            }
        }

        coverage
    }

    fn update_board_with_moove(&self, board: Vec<Vec<bool>>, moo_count: usize,
                               moove: ((usize, usize), (usize, usize), (usize, usize)))
                               -> (Vec<Vec<bool>>, usize, usize) {
        let mut output_board = board.clone();
        let moo_coverage = self.get_moove_coverage(board, moove);

        let mut new_moo_count = moo_count;
        if moo_coverage < 3 {
            output_board[moove.0.0][moove.0.1] = true;
            output_board[moove.1.0][moove.1.1] = true;
            output_board[moove.2.0][moove.2.1] = true;
            new_moo_count += 1;
        }

        let coverage_gain = 3 - moo_coverage;
        (output_board, new_moo_count, coverage_gain)
    }

    fn simulate_board(&self, py: Python, mooves: Vec<((usize, usize), (usize, usize), (usize, usize))>,
                      dims: (usize, usize)) -> PyResult<PyObject> {
        let mut board = self.generate_empty_board(dims);
        let mut moo_count = 0;
        let mut moo_count_sequence = Vec::new();
        let mut moo_coverage_gain_sequence = Vec::new();

        for moove in &mooves {
            let result = self.update_board_with_moove(board.clone(), moo_count, *moove);
            board = result.0;
            moo_count = result.1;
            let coverage_gain = result.2;

            moo_count_sequence.push(moo_count);
            moo_coverage_gain_sequence.push(coverage_gain);
        }

        // Create Python dict for SimulationResult
        let py_dict = pyo3::types::PyDict::new_bound(py);
        py_dict.set_item("board", board)?;
        py_dict.set_item("moo_count", moo_count)?;
        py_dict.set_item("moove_sequence", mooves)?;
        py_dict.set_item("moo_count_sequence", moo_count_sequence)?;
        py_dict.set_item("moo_coverage_gain_sequence", moo_coverage_gain_sequence)?;

        Ok(py_dict.into())
    }

    fn benchmark(&self, grid: Vec<Vec<String>>, iterations: usize) -> f64 {
        use std::time::Instant;

        let dims = self.get_grid_dimensions(grid.clone());
        let all_mooves = self.generate_all_valid_mooves(grid);

        // Take first 50 mooves for benchmark
        let test_mooves: Vec<_> = all_mooves.into_iter().take(50).collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let mut board = self.generate_empty_board(dims);
            let mut moo_count = 0;

            for moove in &test_mooves {
                let result = self.update_board_with_moove(board.clone(), moo_count, *moove);
                board = result.0;
                moo_count = result.1;
            }
        }

        let duration = start.elapsed();
        let simulations_per_second = iterations as f64 / duration.as_secs_f64();

        println!("[Rust Engine] {:.0} simulations/second", simulations_per_second);
        simulations_per_second
    }
}

#[pymodule]
fn rust_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustEngine>()?;
    Ok(())
}