*This project has been created as part of the 42 curriculum by mgamal.*

# RAG against the machine

## Description

A local Retrieval-Augmented Generation (RAG) system that answers questions
about the provided vLLM 0.10.1 repository. BM25 retrieves source snippets;
**Qwen/Qwen3-0.6B** generates answers from that context. Python Fire provides
the CLI, Pydantic validates exchanged data, and tqdm shows batch progress.

## Instructions

The current project requires Python **3.14 or later** and `uv`. A GPU is not
required. Allow disk space for PyTorch, model weights, the corpus, and indexes.
Models download on first use and are then cached. The semantic model currently
loads when the CLI starts, including for ordinary BM25 commands.

```sh
uv sync
# Or: make install
```

Extract the supplied corpus and public datasets into these directories:

```text
data/raw/vllm-0.10.1/
data/datasets/UnansweredQuestions/dataset_docs_public.json
data/datasets/UnansweredQuestions/dataset_code_public.json
data/datasets/AnsweredQuestions/dataset_docs_public.json
data/datasets/AnsweredQuestions/dataset_code_public.json
```

Index the corpus, search, and generate an answer:

```sh
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How to configure OpenAI server?" --k 5
uv run python -m src answer "How to configure OpenAI server?" --k 5
```

Run the dataset pipeline:

```sh
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 \
  --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Keep answered and unanswered dataset outputs in separate directories because
their filenames can be identical. Dataset search chooses its index from the
`dataset_docs_` or `dataset_code_` filename prefix.

The Makefile exposes the required tasks:

```sh
make install
make run                         # CLI help
make run ARGS="search 'What is vLLM?' --k 5"
make debug                       # Start the CLI under pdb
make lint                        # flake8 and required mypy checks
make clean                       # Python/build caches; keeps data and models
```

## System architecture

1. `chunking.py` records Python/documentation chunks with their source paths
   and character spans.
2. `indexing.py` tokenizes chunks and saves term frequencies, an inverted
   index, and metadata in `data/processed/`.
3. `retrieval.py` searches separate code/docs indexes for datasets;
   `combined_search.py` searches both together for single queries.
4. `generation.py` reads retrieved spans, selects whole chunks within the
   model's input budget, and generates up to 200 new tokens.
5. `evaluation.py` computes recall@1, recall@3, and recall@5. The supplied
   moulinette remains the official evaluator.

`models.py` defines the Pydantic models. Batch retrieval uses
`StudentSearchResults`; batch answers use `StudentSearchResultsAndAnswer`.
Single-query search currently prints `MinimalSearchResults`, and single-query
answer returns plain text.

## Chunking strategy

Python chunking uses the AST to group adjacent top-level nodes when they fit
and recursively split oversized nodes. Nodes without splittable children
use overlapping character windows. Markdown, text, and reStructuredText use
sliding windows with a default overlap of 120 characters.

The default maximum is 2,000 characters, configurable with `--max_chunk_size`.
Smaller chunks give precise locations but can lose useful surrounding text.
Larger chunks preserve context but can dilute keyword relevance and consume
more generation tokens. The measurements below use the default size; no
chunk-size comparison is claimed.

## Retrieval method and design decisions

BM25 combines term frequency, inverse document frequency, and length
normalization. Defaults are `k1=1.5` and `b=0.75`. Token scores are summed,
then matching chunks are sorted from highest to lowest score.

Tokenization keeps lowercase identifiers and splits underscores. Code chunks
also include file-path tokens. An inverted index avoids scanning every chunk
for every query token. Separate indexes support dataset evaluation; the
combined index supports questions requiring either code or documentation.
Pickle keeps local persistence simple; only load indexes you trust.

Generation reserves output tokens before selecting context. The prompt asks
the model to use only retrieved evidence and say "Insufficient context" when
it cannot answer. Whole-chunk selection avoids cutting a selected snippet
midway, at the cost of sometimes leaving unused context space.

## Performance analysis

A 2026-11-02 audit rebuilt lexical indexes in a temporary workspace and used
the official moulinette on two provided private datasets of 100 questions
each. Both result files were valid.

| Measurement | Result | Requirement |
| --- | --- | --- |
| Docs recall@5 | 87% | At least 80% |
| Code recall@5 | 50% | At least 50% |
| Whole-corpus lexical indexing | 51.71 seconds | At most 300 seconds |
| Search over 200 questions | 61.99 seconds | At most 90 seconds |

The audit limited CPU thread settings to four. Results depend on the machine;
code recall has no margin above the threshold. Private datasets are corrector
material and are not distributed with the project. The supplied reliability
script passed all four cases.

Fresh Qwen generation did not finish its first answer within a 110-second
audit limit. Live answer quality and HTTP behavior still need verification.

## Bonus features and challenges

- **Semantic retrieval:** normalized `all-MiniLM-L6-v2` vectors score chunks
  against a query. Existing `.npy` files are reused when fingerprints and
  dimensions match. Keep matching `*_chunks.pkl` and `*_embeddings.sha256`
  files with them. Mismatches trigger expensive embedding regeneration.
- **Hybrid retrieval:** reciprocal rank fusion combines lexical and semantic
  candidates. A CPU query using saved vectors worked; no recall improvement
  is claimed.
- **Caching:** indexes are cached in memory; query results are stored on disk
  with index-version keys. Repeated-query consistency was checked.
- **HTTP API:** FastAPI provides a form plus `/search` and `/answer`. The
  audit stalled in HTTP/AnyIO execution, so a live demonstration is pending.

Incremental indexing is not implemented.

```sh
uv run python -m src hybrid "How to configure OpenAI server?" --k 5
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`, or send a search request:

```sh
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is vLLM?","k":5}'
```

Paraphrased questions and exact identifiers are difficult to match with one
method. Identifier splitting, path tokens, and semantic search address
different aspects of this problem. Fingerprints prevent accidental reuse of
vectors for changed chunks. CPU generation remains a practical limitation.

Known issues remain in Unicode AST offsets, Python windows of 50 characters
or less, custom combined-index directories, and some filesystem errors.
Use default paths and 2,000-character chunks for the documented workflow.
Single-query output differs from the subject's required wrapper schema.
These behaviors were preserved during the typing/documentation cleanup.

## Resources and AI usage

- [vLLM documentation](https://docs.vllm.ai/): indexed project concepts.
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): generator model card.
- [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2):
  embedding model card.
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/):
  lexical search and probabilistic ranking.
- [uv](https://docs.astral.sh/uv/),
  [Python Fire](https://github.com/google/python-fire),
  [Pydantic](https://docs.pydantic.dev/), and
  [mypy](https://mypy.readthedocs.io/): development tool references.

AI assistance was used in this development session to compare the subject
and correction sheet with the implementation, inspect code, run bounded
checks, remove unused imports, add annotations and docstrings, configure
lint tooling and the Makefile, and draft this README. Earlier AI usage is
not established here. The author remains responsible for understanding and
verifying the submitted implementation.
