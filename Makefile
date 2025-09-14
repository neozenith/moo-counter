daily:
	uv run src/moo_counter/moo_counter.py --puzzle micro --strategy greedy-high  --iterations 100
	uv run src/moo_counter/moo_counter.py --puzzle mini --strategy greedy-high  --iterations 1000
	uv run src/moo_counter/moo_counter.py --puzzle maxi --strategy greedy-high  --iterations 10000


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

docs-local: 
	uv run -m http.server --directory docs 8000

site-local: page
	uv run -m http.server --directory site 8000

######################################################################
# SETUP
######################################################################
.venv/deps: .venv pyproject.toml
	uv sync --all-groups
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

.PHONY: format check test docs build diag clean publish publish-test docs-install docs-serve docs-build docs-deploy docs-clean daily page
