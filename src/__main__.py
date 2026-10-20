import fire
import sys
from pathlib import Path
from pydantic import ValidationError
from .indexing import build_index
from functools import lru_cache
from .retrieval import load_data, search as retrieve
from .retrieval import search_dataset as retrieve_dataset, dump_json
from .evaluation import evaluate_dataset
from .models import MinimalSearchResults, StudentSearchResults, RagDataset
from .cache import (
    make_cache_key,
    get_index_version,
    load_cached_result,
    save_cached_result
)

@lru_cache(maxsize=2)
def load_index(name: str):
    chunks = load_data(f"data/processed/{name}_chunks.pkl")
    index_chunks = load_data(f"data/processed/{name}_index.pkl")
    inverted = load_data(f"data/processed/{name}_inverted_index.pkl")
    metadata = load_data(f"data/processed/{name}_metadata.pkl")

    return chunks, index_chunks, inverted, metadata


def search_dataset_command(
    dataset_path: str,
    k: int = 5,
    save_directory: str = "data/output/search_results"
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

    chunks, index_chunks, inverted, metadata = load_index(index_type)

    results = retrieve_dataset(
        dataset_path,
        chunks,
        index_chunks,
        inverted,
        metadata["avgdl"],
        index_type=index_type,
        k=k
    )

    output_path = Path(save_directory) / dataset_name
    dump_json(results, output_path)

    print(f"Results saved to: {output_path}")


def merge_results(result, k):
    merged = []
    docs = result["docs"]
    code = result["code"]
    for i in range(max(len(docs), len(code))):
        if i < len(docs):
            merged.append(docs[i])
        if i < len(code):
            merged.append(code[i])
        if len(merged) >= k:
            break
    return merged[:k]


def search_both(query: str, k: int = 5):
    validate_query(query, k)
    docs_version = get_index_version("docs")
    code_version = get_index_version("code")

    index_version = docs_version + ":" + code_version

    cache_key = make_cache_key(query, k, index_version) 
    cached = load_cached_result(cache_key)

    if cached is not None:
        return cached

    results = {}

    for index_type in ("docs", "code"):
        chunks, index_chunks, inverted, metadata = load_index(index_type)

        result = retrieve(
            question_id="cli-query",
            query=query,
            chunks=chunks,
            index_chunks=index_chunks,
            inverted=inverted,
            avgdl=metadata["avgdl"],
            k=k
        )

        results[index_type] = result.retrieved_sources
    merged_sources = merge_results(results, k)
    final_result = MinimalSearchResults(
        question_id="cli-query",
        question=query,
        retrieved_sources=merged_sources
    )
    save_cached_result(cache_key, final_result)

    return final_result


def search_command(query: str, k: int = 5):
    result = search_both(query, k)
    print(result.model_dump_json(indent=2))


def query_answer(query: str, k: int = 5):

    results = search_both(query, k)
    from .generation import generate_answer
    answer = generate_answer(
        question=query,
        sources=results.retrieved_sources
    )

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
    filename = Path(student_search_results_path).stem + "_answer.json"
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
    if not query.strip():
        raise ValueError("Query cannot be empty")

    if k <= 0:
        raise ValueError("k must be greater than 0")


if __name__ == "__main__":
    try:
        fire.Fire({
            "index": build_index,
            "search": search_command,
            "search_dataset": search_dataset_command,
            "answer": query_answer,
            "answer_dataset": answer_dataset_command,
            "evaluate": evaluate_command
        })

    except (ValueError, FileNotFoundError)as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
