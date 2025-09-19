daily-micro: .venv/deps
	uv run -m moo_counter --puzzle micro --strategy greedy-high  --iterations 100
daily-mini: .venv/deps
	uv run -m moo_counter --puzzle mini --strategy greedy-high  --iterations 1000
daily-maxi: .venv/deps
	uv run -m moo_counter --puzzle maxi --strategy greedy-high  --iterations 10000

daily:	daily-micro daily-mini daily-maxi test test-engines

page: daily
	mkdir -p site
	
	cp docs/index.html site/index.html
	cp docs/styles.css site/styles.css
	cp docs/script.js site/script.js
	cp docs/graph.html site/graph.html

	cp output/*-micro.json docs/micro.json
	cp output/*-mini.json docs/mini.json
	cp output/*-maxi.json docs/maxi.json

	cp output/*-micro_graph.json docs/micro_graph.json
	cp output/*-mini_graph.json docs/mini_graph.json
	cp output/*-maxi_graph.json docs/maxi_graph.json

	cp output/*-micro.json site/micro.json
	cp output/*-mini.json site/mini.json
	cp output/*-maxi.json site/maxi.json

	cp output/*-micro_graph.json site/micro_graph.json
	cp output/*-mini_graph.json site/mini_graph.json
	cp output/*-maxi_graph.json site/maxi_graph.json

ghpages:
	mkdir -p site
	
	cp docs/index.html site/index.html
	cp docs/styles.css site/styles.css
	cp docs/script.js site/script.js
	cp docs/graph.html site/graph.html

	cp output/*-micro.json site/micro.json
	cp output/*-mini.json site/mini.json
	cp output/*-maxi.json site/maxi.json

	cp output/*-micro_graph.json site/micro_graph.json
	cp output/*-mini_graph.json site/mini_graph.json
	cp output/*-maxi_graph.json site/maxi_graph.json

docs-local: 
	uv run -m http.server --directory docs 8000

site-local: page
	uv run -m http.server --directory site 8000

######################################################################
# SETUP
######################################################################
.venv: pyproject.toml
	uv sync --all-groups
	uvx playwright install chromium-headless-shell --only-shell

# Create a "touch file" as a single target
# when the output of a prior target was many files.
.venv/deps: .venv
	touch $@

######################################################################
# ENGINE BUILDING
######################################################################

# Build Rust engine with maturin
build-rust: .venv/deps
	uvx maturin develop --release
	cp target/release/libmoo_counter_rust.dylib src/moo_counter_rust.so

test-engine-rust: .venv/deps build-rust
	uv run -m moo_counter --puzzle micro --strategy greedy-high --iterations 10 --engine rust

# Build Cython engine
build-cython: .venv/deps
	cd src/moo_counter/engines && uv run python setup.py build_ext --inplace

test-engine-cython: .venv/deps build-cython
	uv run -m moo_counter --puzzle micro --strategy greedy-high --iterations 10 --engine cython

# Build all engines
build-engines: build-rust build-cython

# Clean build artifacts
clean-engines:
	rm -rf target/
	rm -rf build/
	rm -rf src/moo_counter/engines/*.c
	rm -rf src/moo_counter/engines/*.so
	rm -rf src/moo_counter/engines/*.pyd
	rm -rf src/moo_counter/engines/build/
	find . -name "*.so" -delete
	find . -name "*.pyd" -delete

# Test all engines
test-engines: .venv/deps build-engines
	uv run -m moo_counter --puzzle micro --strategy greedy-high --iterations 10 --engine python
	uv run -m moo_counter --puzzle micro --strategy greedy-high --iterations 10 --engine rust
	uv run -m moo_counter --puzzle micro --strategy greedy-high --iterations 10 --engine cython
	
# Benchmark all engines
benchmark-engines: .venv/deps
	uv run python -m moo_counter.moo_counter --puzzle mini --benchmark --strategy greedy-high --iterations 10000

# Compare all engines side-by-side
compare-engines: .venv/deps
	uv run python -m moo_counter.moo_counter --puzzle maxi --strategy greedy-high --compare-engines --iterations 10000


######################################################################
# DOCUMENTATION
######################################################################
docs:
	uvx --from md-toc md_toc --in-place github --header-levels 2 *.md
	uvx rumdl check . --fix --respect-gitignore -d MD013,MD033 --exclude docs/api/*.md

######################################################################
# QUALITY ASSURANCE
######################################################################

format: .venv/deps docs
	uvx ruff format src/ --respect-gitignore --line-length 120
	uvx isort src/ --profile 'black'

check: .venv/deps format
	uvx ruff check src/
	uvx isort src/ --check-only
	uv run mypy src/

# Run minimal test suite
test: .venv/deps test-engines
	uv run pytest tests/ -v --tb=short

# Run specific test for engine debugging
test-engine-registration: .venv/deps
	uv run pytest tests/test_engine_registration.py -v --tb=short


clean: clean-engines
	rm -rf dist
	rm -rf .venv
	rm -rf htmlcov/
	rm -rf coverage.json
	rm -rf .*_cache
	rm -rf .coverage
	rm -rf output/
	rm -rf site/

.PHONY: format check test test-engine-registration docs build clean clean-engines \
		daily-micro daily-mini daily-maxi daily page \
		build-rust build-cython build-engines test-engines \
		benchmark-engines compare-engines
