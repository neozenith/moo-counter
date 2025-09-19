# Moo Counter Engine Implementations

This project includes three engine implementations for performance comparison:

1. **Python Engine** - Pure Python implementation (default)
2. **Rust Engine** - High-performance Rust implementation using PyO3
3. **Cython Engine** - Optimized Cython implementation

## Building the Engines

### Prerequisites

```bash
# Install build dependencies
uv sync --group dev

# The dependencies are installed automatically with uv sync
```

### Build Rust Engine

The Rust engine uses `maturin` for building Python extensions:

```bash
# Build and install in development mode
maturin develop --release

# Or use the Makefile
make build-rust
```

### Build Cython Engine

```bash
# Build Cython extension
cd src/moo_counter/engines
python setup.py build_ext --inplace
cd ../../..

# Or use the Makefile
make build-cython
```

### Build All Engines

```bash
make build-engines
```

## Using the Engines

### Command Line

Select an engine using the `--engine` flag:

```bash
# Use Python engine (default)
uv run src/moo_counter/moo_counter.py --puzzle micro --strategy greedy-high --iterations 100

# Use Rust engine
uv run src/moo_counter/moo_counter.py --puzzle micro --strategy greedy-high --iterations 100 --engine rust

# Use Cython engine
uv run src/moo_counter/moo_counter.py --puzzle micro --strategy greedy-high --iterations 100 --engine cython
```

### Benchmarking

Compare all available engines:

```bash
# Run benchmark of all engines
uv run src/moo_counter/moo_counter.py --puzzle micro --benchmark --iterations 1000

# Compare engines side by side
uv run src/moo_counter/moo_counter.py --puzzle micro --compare-engines --iterations 100
```

## Performance Expectations

Based on typical performance characteristics:

- **Python Engine**: Baseline performance, ~1x speed
- **Cython Engine**: 2-5x faster than Python
- **Rust Engine**: 5-20x faster than Python (especially with parallel processing)

Actual performance will vary based on:
- Puzzle size
- Number of valid moves
- CPU cores available (Rust engine uses parallel processing)
- System architecture

## Development

### Clean Build Artifacts

```bash
make clean-engines
```

### Testing All Engines

```bash
make test-engines
```

## Implementation Details

### Python Engine
- Pure Python implementation
- Easy to modify and debug
- Good for algorithm development

### Rust Engine
- Uses PyO3 for Python bindings
- Parallel processing with Rayon
- Memory-efficient data structures
- Best for production workloads

### Cython Engine
- C-level performance with Python-like syntax
- Compile-time optimizations
- Good balance of performance and maintainability

## Troubleshooting

### Rust Engine Not Available

If the Rust engine doesn't appear in available engines:

1. Ensure Rust is installed: `rustc --version`
2. Install maturin: `uv pip install maturin`
3. Build the engine: `maturin develop --release`
4. Check for errors in the build output

### Cython Engine Not Available

If the Cython engine doesn't appear:

1. Install Cython: `uv pip install cython`
2. Build the extension: `cd src/moo_counter/engines && python setup.py build_ext --inplace`
3. Check that `.so` (Linux/Mac) or `.pyd` (Windows) files were created

### Import Errors

If you get import errors when using the engines:

1. Ensure you're running from the project root
2. Use `uv run` to ensure proper Python environment
3. Rebuild the engines with `make -f Makefile.engines build-all`