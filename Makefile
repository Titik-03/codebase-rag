.PHONY: all install run debug clean lint

PYTHON = uv run python
ARGS ?= --help

all: install

install:
	uv sync

run:
	$(PYTHON) -m src $(ARGS)

debug: ARGS =
debug:
	$(PYTHON) -m pdb -m src $(ARGS)

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	find src -type f -name "*.py[co]" -delete
	find src -depth -type d -name "__pycache__" -empty -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf build dist
