import fire
import sys
from pathlib import Path
from pydantic import ValidationError
from .indexing import build_index
from functools import lru_cache
from .retrieval import load_data, search as retrieve
from .retrieval import search_dataset as retrieve_dataset, dump_json
from .hybrid import hybrid_search
from .evaluation import evaluate_dataset
from .combined_search import (
    combined_search,
    load_combined_index,
    build_and_save_combined
)
from .retrieval import get_sources
from .models import MinimalSearchResults, StudentSearchResults, RagDataset
from .cache import (
    make_cache_key,
    get_index_version,
    load_cached_result,
    save_cached_result
)


@lru_cache(maxsize=2)
def load_index(
    name: str,
    index_directory: str = "data/processed"
):
    path = Path(index_directory)

    chunks = load_data(path / f"{name}_chunks.pkl")
    index_chunks = load_data(path / f"{name}_index.pkl")
    inverted = load_data(path / f"{name}_inverted_index.pkl")
    metadata = load_data(path / f"{name}_metadata.pkl")

    return chunks, index_chunks, inverted, metadata


def index_command(
    max_chunk_size: int = 2000,
    raw_directory: str = "data/raw/vllm-0.10.1",
    index_directory: str = "data/processed"
):
    try:
        build_index(
            max_chunk_size=max_chunk_size,
            raw_directory=raw_directory,
            index_directory=index_directory
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def search_dataset_command(
    dataset_path: str,
    k: int = 5,
    save_directory: str = "data/output/search_results",
    index_directory: str = "data/processed"
):
    if k <= 0:
        raise ValueError("k must be greater than 0")
    dataset_name = Path(dataset_path).name

    if dataset_name.startswith("dataset_docs_"):
        index_type = "docs"
    elif dataset_name.startswith("dataset_code_"):
        index_type = "code"
    else:
        raise ValueError("Cannot determine dataset type from filename")

    chunks, index_chunks, inverted, metadata = load_index(index_type,
                                                          index_directory)

    results = retrieve_dataset(
        dataset_path,
        chunks,
        index_chunks,
        inverted,
        metadata["avgdl"],
        index_type=index_type,
        k=k,
        index_directory=index_directory
    )

    output_path = Path(save_directory) / dataset_name
    dump_json(results, output_path)

    print(f"Results saved to: {output_path}")


def hybrid_command(
    query: str,
    k: int = 5,
    candidate_k: int = 20
):
    validate_query(query, k)

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    results = hybrid_search(
        query,
        k=k,
        candidate_k=candidate_k
    )

    chunks, _, _, _ = load_combined_index()

    sources = get_sources(results, chunks)

    final_result = MinimalSearchResults(
        question_id="hybrid-query",
        question=query,
        retrieved_sources=sources
    )

    print(final_result.model_dump_json(indent=2))


def search_both(
    query: str,
    k: int = 5,
    index_directory: str = "data/processed"
):
    validate_query(query, k)

    index_version = get_index_version(
        "combined",
        index_directory
    )

    cache_key = make_cache_key(
        query,
        k,
        index_version
    )

    cached = load_cached_result(cache_key)

    if cached is not None:
        return cached

    chunks, _, _, _ = load_combined_index()

    results = combined_search(query, k)

    sources = get_sources(results, chunks)

    final_result = MinimalSearchResults(
        question_id="cli-query",
        question=query,
        retrieved_sources=sources
    )

    save_cached_result(cache_key, final_result)

    return final_result


def search_command(
    query: str,
    k: int = 5,
    index_directory: str = "data/processed"
):
    result = search_both(query, k, index_directory)
    print(result.model_dump_json(indent=2))


def query_answer(
    query: str,
    k: int = 5,
    index_directory: str = "data/processed"
):
    try:
        results = search_both(query, k, index_directory)

        from .generation import generate_answer

        answer = generate_answer(
            question=query,
            sources=results.retrieved_sources
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    return answer


def answer_dataset_command(
    student_search_results_path: str,
    save_directory: str
):

    with open(student_search_results_path, "r", encoding="utf-8") as f:
        dataset = StudentSearchResults.model_validate_json(f.read())

    if dataset.k <= 0:
        raise ValueError("k must be greater than 0")
    from .generation import generate_dataset
    result = generate_dataset(dataset)
    filename = Path(student_search_results_path).name
    output_path = Path(save_directory) / filename
    dump_json(result, output_path)

    print(f"Answers saved to: {output_path}")


def evaluate_command(
    student_search_results_path: str,
    dataset_path: str
):
    with open(student_search_results_path, "r", encoding="utf-8") as f:
        student = StudentSearchResults.model_validate_json(f.read())

    with open(dataset_path, "r", encoding="utf-8") as f:
        reference = RagDataset.model_validate_json(f.read())

    results = evaluate_dataset(reference, student)

    for metric, score in results.items():
        print(f"{metric}: {score:.3f} ({score * 100:.1f}%)")


def validate_query(query: str, k: int):
    if not isinstance(query, str):
        raise ValueError("Query must be a string")

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")


if __name__ == "__main__":
    try:
        fire.Fire({
            "index": index_command,
            "search": search_command,
            "search_dataset": search_dataset_command,
            "answer": query_answer,
            "answer_dataset": answer_dataset_command,
            "evaluate": evaluate_command,
            "hybrid": hybrid_command
        })

    except (ValueError, FileNotFoundError)as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
