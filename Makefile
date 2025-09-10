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

.PHONY: format check test docs build diag clean publish publish-test docs-install docs-serve docs-build docs-deploy docs-clean
