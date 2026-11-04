#!/usr/bin/env bash
# Pre-submission checks for 42 "RAG against the machine".
# Run from repository root: bash rag_pre_submission_check.sh
# Logs are written outside the repository; no source files are modified.
set -u

ROOT="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${TMPDIR:-/tmp}/rag_preflight_${STAMP}"
mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/summary.txt"
: > "$SUMMARY"
PASS=0
FAIL=0
SKIP=0

pass() { printf 'PASS: %s\n' "$1" | tee -a "$SUMMARY"; PASS=$((PASS+1)); }
fail() { printf 'FAIL: %s\n' "$1" | tee -a "$SUMMARY"; FAIL=$((FAIL+1)); }
skip() { printf 'SKIP: %s\n' "$1" | tee -a "$SUMMARY"; SKIP=$((SKIP+1)); }
run() {
  local label="$1" slug="$2"; shift 2
  local log="$REPORT_DIR/${slug}.log"
  printf '\n=== %s ===\n' "$label"
  printf 'Command: %s\n' "$*"
  if "$@" >"$log" 2>&1; then
    pass "$label"
  else
    fail "$label (see $log)"
    tail -n 18 "$log"
  fi
}

printf 'RAG pre-submission check\nRepository: %s\nLogs: %s\n' "$ROOT" "$REPORT_DIR"

if [[ ! -f pyproject.toml || ! -f uv.lock || ! -f Makefile || ! -f README.md || ! -d src ]]; then
  fail 'Required project root files: pyproject.toml, uv.lock, Makefile, README.md, src/'
else
  pass 'Required project root files exist'
fi

if command -v uv >/dev/null 2>&1; then
  run 'Python 3.10+' python_version uv run python -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
  run 'Locked dependency environment (uv sync --frozen)' uv_sync uv sync --frozen
else
  fail 'uv executable is available'
fi

if [[ -f Makefile ]]; then
  run 'Makefile exposes install/run/debug/clean/lint' make_rules bash -c 'for name in install run debug clean lint; do grep -Eq "^${name}:" Makefile || exit 1; done'
  if command -v make >/dev/null 2>&1; then
    run 'Mandatory make lint (flake8 + mypy)' make_lint make lint
  else
    fail 'make command exists'
  fi
fi

run 'All 8 required Pydantic model classes and their fields' pydantic_models uv run python -c '
from src import models
required = {
    "MinimalSource": {"file_path", "first_character_index", "last_character_index"},
    "UnansweredQuestion": {"question_id", "question"},
    "AnsweredQuestion": {"question_id", "question", "sources", "answer"},
    "RagDataset": {"rag_questions"},
    "MinimalSearchResults": {"question_id", "question", "retrieved_sources"},
    "MinimalAnswer": {"question_id", "question", "retrieved_sources", "answer"},
    "StudentSearchResults": {"search_results", "k"},
    "StudentSearchResultsAndAnswer": {"search_results", "k"},
}
for name, fields in required.items():
    cls = getattr(models, name)
    assert fields <= set(cls.model_fields), (name, fields - set(cls.model_fields))
print("Pydantic models verified")'

run 'Chunking: Unicode AST offsets and safe small chunks' chunking_regression uv run python -c '
import ast
import tempfile
from pathlib import Path
from src.chunking import calculate_char_offsets, get_node_char, parse_nodes, chunk_markdown
text = "x = \"é\"; y = 42\n"
tree = ast.parse(text)
starts = calculate_char_offsets(text)
lines = text.splitlines(keepends=True)
a, b = get_node_char(tree.body[1], starts, lines)
assert text[a:b] == "y = 42", repr(text[a:b])
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    (root / "test.py").write_text("def f():\n    x = \"abc\" * 30\n    return x\n", encoding="utf-8")
    (root / "test.md").write_text("sample text " * 20, encoding="utf-8")
    for size in (2, 20, 50, 2000):
        py_chunks = parse_nodes(root, max_chunk_size=size)
        md_chunks = chunk_markdown(root, max_chunk_size=size)
        for c in py_chunks + md_chunks:
            length = c["last_character_index"] - c["first_character_index"]
            assert 0 < length <= size, (size, c)
        assert len(py_chunks) + len(md_chunks) < 1000, size
print("Chunking regression tests passed")'

if [[ -f exams/scripts/exam_edge_cases.sh ]]; then
  run 'Official reliability exam (4 degenerate cases)' exam_edge_cases bash exams/scripts/exam_edge_cases.sh --student-path "$ROOT"
else
  skip 'Official reliability exam (exams/scripts/exam_edge_cases.sh missing)'
fi

if [[ -f exams/scripts/exam_retrieval.sh && -x moulinette-ubuntu && -f data/datasets/private/AnsweredQuestions/dataset_docs_private.json ]]; then
  run 'Official PRIVATE retrieval exam (indexing, throughput, docs/code recall)' exam_retrieval bash exams/scripts/exam_retrieval.sh --student-path "$ROOT" --moulinette-path "$ROOT/moulinette-ubuntu"
else
  skip 'Official private retrieval exam (need exams script, moulinette, extracted private datasets)'
fi

SEARCH_JSON="$REPORT_DIR/single_search.json"
ANSWER_JSON="$REPORT_DIR/single_answer.json"
run 'Single-query search CLI and configurable JSON output' single_search uv run python -m src search 'How to configure OpenAI server?' --k 10 --output_path "$SEARCH_JSON"
if [[ -f "$SEARCH_JSON" ]]; then
  run 'Single-query search schema and max source size' search_schema uv run python -c "
from pathlib import Path
from src.models import StudentSearchResults
x = StudentSearchResults.model_validate_json(Path('$SEARCH_JSON').read_text())
assert x.k == 10 and len(x.search_results) == 1
for source in x.search_results[0].retrieved_sources:
    assert 0 < source.last_character_index - source.first_character_index <= 2000
    assert Path(source.file_path).is_file(), source.file_path
print('Search schema/offsets passed')"
fi

# GNU timeout is present on the campus Linux machines. This is only a smoke test;
# the correction sheet separately requires 2 satisfactory answers out of 3.
if command -v timeout >/dev/null 2>&1; then
  run 'Single-query Qwen answer generation (180-second cap)' single_answer timeout 180s uv run python -m src answer 'How to configure OpenAI server?' --k 10 --output_path "$ANSWER_JSON"
else
  skip 'Single-query Qwen answer test (timeout executable unavailable)'
fi
if [[ -f "$ANSWER_JSON" ]]; then
  run 'Single-query answer JSON schema (quality still needs human judgment)' answer_schema uv run python -c "
from pathlib import Path
from src.models import StudentSearchResultsAndAnswer
x = StudentSearchResultsAndAnswer.model_validate_json(Path('$ANSWER_JSON').read_text())
assert x.k == 10 and len(x.search_results) == 1
assert x.search_results[0].answer.strip()
print('ANSWER:', x.search_results[0].answer)
"
fi

if [[ -f README.md ]]; then
  run 'README required first-line declaration' readme_firstline bash -c 'head -n 1 README.md | grep -Eq "^\\*This project has been created as part of the 42 curriculum by "'
  run 'README topic headings exist (manual quality review still required)' readme_sections bash -c 'for term in Description Instructions Resources Architecture Chunking Retrieval Performance "Design" "Example"; do grep -qi "$term" README.md || { echo "Missing topic: $term"; exit 1; }; done'
fi
run 'No moulinette import in student Python files' no_moulinette_import bash -c '! grep -rEn "^[[:space:]]*(import|from)[[:space:]]+moulinette" src/ --include="*.py"'

run 'No generated outputs/private data tracked in Git' git_hygiene bash -c '
  ! git ls-files | grep -E "(^\\.venv/|^private/|^exams/|^evaluations/|^data/cache/|^data/processed/|^data/output/|__pycache__|\\.npy$|moulinette-ubuntu$|private\\.zip$|exams\\.zip$)"'

printf '\n=========================\n'
printf 'Pass: %s | Fail: %s | Skip: %s\n' "$PASS" "$FAIL" "$SKIP"
printf 'Full logs: %s\n' "$REPORT_DIR"
printf '\nMANUAL CHECKS STILL REQUIRED:\n'
printf '  1) bash exams/scripts/exam_answer.sh --student-path . --moulinette-path ./moulinette-ubuntu\n'
printf '     Rate at least 2 of 3 answers satisfactory (grounded, coherent, on-point).\n'
printf '  2) Oral defense: answer at least 4 of the 5 rubric questions.\n'
printf '  3) Demonstrate semantic, hybrid, caching, and HTTP bonuses; excluded incremental.\n'
printf '  4) Verify README detail, Makefile behavior, and final clean Git checkout.\n'
if (( FAIL > 0 )); then exit 1; fi
