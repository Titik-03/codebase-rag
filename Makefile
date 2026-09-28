PYTHON ?= /opt/pyenv/versions/3.13.1/bin/python
MOULINETTE ?= ./moulinette-ubuntu

K ?= 5
MAX_CONTEXT_LENGTH ?= 2000

# Datasets and search output paths
CODE_RESULTS ?= data/output/search_results/code_result.json
CODE_DATASET ?= data/datasets/AnsweredQuestions/dataset_code_public.json

DOCS_RESULTS ?= data/output/search_results/docs_result.json
DOCS_DATASET ?= data/datasets/AnsweredQuestions/dataset_docs_public.json

.PHONY: all pipeline run index retrieve test eval eval-code eval-docs help

# Default target: run the full pipeline
all: pipeline

# Run the entire pipeline (indexing, retrieval, and evaluation)
pipeline: run test

# Run indexing and retrieval
run: index retrieve

# 1. Indexing step
index:
	$(PYTHON) src/indexing.py

# 2. Retrieval step
retrieve:
	$(PYTHON) src/retrieval.py

# 3. Evaluation / Test step (both code and docs)
test: eval
eval: eval-code eval-docs

# Evaluate code search results
eval-code:
	$(MOULINETTE) evaluate_student_search_results \
		$(CODE_RESULTS) \
		$(CODE_DATASET) \
		--k $(K) \
		--max_context_length $(MAX_CONTEXT_LENGTH)

# Evaluate docs search results
eval-docs:
	$(MOULINETTE) evaluate_student_search_results \
		$(DOCS_RESULTS) \
		$(DOCS_DATASET) \
		--k $(K) \
		--max_context_length $(MAX_CONTEXT_LENGTH)

help:
	@echo "Available targets:"
	@echo "  make pipeline   - Run full pipeline: index, retrieve, and evaluate (default)"
	@echo "  make run        - Run indexing and retrieval"
	@echo "  make index      - Run src/indexing.py"
	@echo "  make retrieve   - Run src/retrieval.py"
	@echo "  make test       - Run evaluation on both code and docs results"
	@echo "  make eval-code  - Evaluate code search results"
	@echo "  make eval-docs  - Evaluate docs search results"
	@echo "  make help       - Show this help message"
