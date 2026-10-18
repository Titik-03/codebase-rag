.PHONY: all install run debug clean lint

PYTHON = uv run python
ARGS ?= --help

all: install

install:
	uv sync

run:
	$(PYTHON) -m src $(ARGS)

debug:
	$(PYTHON) -m pdb -m src $(ARGS)

lint:
	uv run --with flake8 flake8 src
	uv run --with mypy mypy src

clean:
	find src -type d -name "__pycache__" -exec rm -rf {} +
	find src -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf build dist