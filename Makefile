######################################################################
# SETUP
######################################################################
.venv: pyproject.toml
	uv sync --all-groups
	uvx playwright install

# Create a "touch file" as a single target
# when the output of a prior target was many files.
.venv/deps: .venv
	touch $@

######################################################################
# WEBSITE
######################################################################
daily-micro: .venv/deps .venv/build-engines
	uv run -m moo_counter --puzzle micro --strategy greedy-high  --iterations 100 --engine python
daily-mini: .venv/deps .venv/build-engines
	uv run -m moo_counter --puzzle mini --strategy greedy-high  --iterations 1000 --engine python
daily-maxi: .venv/deps .venv/build-engines
	uv run -m moo_counter --puzzle maxi --strategy greedy-high  --iterations 10000 --engine python

daily:	daily-micro daily-mini daily-maxi 

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
# ENGINE BUILDING
######################################################################

# Clean build artifacts
clean-engines:
	rm -rf target/
	rm -rf build/
	rm -rf src/moo_counter/engines/cython_*.c
	rm -rf src/moo_counter/engines/*.so
	rm -rf src/moo_counter/engines/*.pyd
	rm -rf src/moo_counter/engines/build/
	find . -name "*.so" -delete
	find . -name "*.pyd" -delete

# Build Rust engine with maturin
build-rust: .venv/deps
	uvx maturin develop --release
	cp target/release/libmoo_counter_rust.dylib src/moo_counter/engines/rust_engine.so

# Build Cython engine
build-cython: .venv/deps
	cd src/moo_counter/engines && uv run python setup.py build_ext --inplace

# Build C engine (builds together with Cython since they share setup.py)
build-c: .venv/deps
	cd src/moo_counter/engines && uv run python setup.py build_ext --inplace

# Build all engines
build-engines: .venv/build-engines
.venv/build-engines: build-rust build-cython build-c
	touch $@


test-engine-python: .venv/deps
	uv run -m moo_counter --puzzle micro --strategy random --iterations 10 --engine python

test-engine-rust: .venv/deps build-rust
	uv run -m moo_counter --puzzle micro --strategy random --iterations 10 --engine rust
	
test-engine-cython: .venv/deps build-cython
	uv run -m moo_counter --puzzle micro --strategy random --iterations 10 --engine cython
	
test-engine-c: .venv/deps build-c
	uv run -m moo_counter --puzzle micro --strategy random --iterations 10 --engine c

# Test all engines
test-engines: .venv/deps test-engine-python test-engine-rust test-engine-cython test-engine-c
	
# Benchmark all engines
benchmark-engines: .venv/deps
	uv run -m moo_counter.moo_counter --puzzle mini --benchmark --strategy random --iterations 10000

# Compare all engines side-by-side
compare-engines: .venv/deps
	uv run -m moo_counter.moo_counter --puzzle maxi --strategy random --compare-engines --iterations 10000


######################################################################
# DOCUMENTATION
######################################################################
docs:
	uvx --from md-toc md_toc --in-place github --header-levels 2 *.md
	uvx rumdl check . --fix --respect-gitignore -d MD013,MD033,MD036 --exclude docs/api/*.md

######################################################################
# QUALITY ASSURANCE
######################################################################

format: .venv/deps docs
	uvx ruff format src/ --respect-gitignore --line-length 120
	uvx isort src/
	uvx isort src/moo_counter/engines/cython_engine.pyx

check: .venv/deps format
	uvx ruff check src/
	uvx isort src/ --check-only
	uv run mypy src/

# Run minimal test suite
test: .venv/deps test-engines
	uv run pytest tests/ -v --tb=short

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
		build-rust build-cython build-c build-engines test-engines \
		test-engine-rust test-engine-cython test-engine-c \
		benchmark-engines compare-engines
