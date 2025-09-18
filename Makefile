daily-micro: .venv/deps
	uv run -m moo_counter --puzzle micro --strategy greedy-high  --iterations 100
daily-mini: .venv/deps
	uv run -m moo_counter --puzzle mini --strategy greedy-high  --iterations 1000
daily-maxi: .venv/deps
	uv run -m moo_counter --puzzle maxi --strategy greedy-high  --iterations 10000

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


clean:
	rm -rf dist
	rm -rf .venv
	rm -rf htmlcov/
	rm -rf coverage.json
	rm -rf .*_cache
	rm -rf .coverage

.PHONY: format check test docs build clean \
		daily-micro daily-mini daily-maxi daily page
