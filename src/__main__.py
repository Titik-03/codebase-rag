import fire
from pathlib import Path
from .indexing import build_index
from .retrieval import load_data, search as retrieve
from .retrieval import search_dataset as retrieve_dataset, dump_json
from .models import MinimalSearchResults

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

    return final_result


if __name__ == "__main__":
    fire.Fire({
        "index": build_index,
        "search": search_command,
        "search_dataset": search_dataset_command
    })