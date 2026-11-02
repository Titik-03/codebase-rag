from pathlib import Path
import pickle
from .indexing import bm25_search
from .indexing import index, inverted_index, average_lenght
from functools import lru_cache


def load_data(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def build_combined():
    path_proc = Path("data/processed")
    docs_chunks = load_data(path_proc / "docs_chunks.pkl")
    code_chunks = load_data(path_proc / "code_chunks.pkl")
    combined_chunks = docs_chunks + code_chunks
    combined_index = index(combined_chunks)
    combined_inverted = inverted_index(combined_index)
    combined_average_lenght = average_lenght(combined_index)
    return (
        combined_chunks,
        combined_index,
        combined_inverted,
        combined_average_lenght
    )


def save_combined_index(chunks, idx, inverted, avgdl):
    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)

    data = {
        "combined_chunks.pkl": chunks,
        "combined_index.pkl": idx,
        "combined_inverted_index.pkl": inverted,
        "combined_metadata.pkl": {
            "N": len(idx),
            "avgdl": avgdl
        }
    }

    for filename, content in data.items():
        with (output / filename).open("wb") as file:
            pickle.dump(content, file)


@lru_cache(maxsize=1)
def load_combined_index():
    base = Path("data/processed")

    chunks = load_data(base / "combined_chunks.pkl")
    idx = load_data(base / "combined_index.pkl")
    inverted = load_data(base / "combined_inverted_index.pkl")
    metadata = load_data(base / "combined_metadata.pkl")

    return chunks, idx, inverted, metadata


def build_and_save_combined():
    chunks, idx, inverted, avgdl = build_combined()

    save_combined_index(chunks, idx, inverted, avgdl)

    load_combined_index.cache_clear()

    print("Combined index created successfully.")


def combined_search(query: str, k: int = 5):
    chunks, idx, inverted, metadata = load_combined_index()

    results = bm25_search(
        query,
        idx,
        inverted,
        metadata["avgdl"],
        top_k=k
    )

    return results


if __name__ == "__main__":
    queries = [
        "How does vLLM allocate GPU memory?",
        "How to install vLLM?",
        "How to configure tensor parallelism?",
        "What is continuous batching?",
    ]

    chunks, _, _, _ = load_combined_index()

    for query in queries:
        print(f"\nQUERY: {query}")

        results = combined_search(query, k=5)

        for chunk_id, score in results:
            chunk = chunks[chunk_id]
            file_path = chunk["file_path"]

            chunk_type = (
                "CODE" if file_path.endswith(".py") else "DOCS"
            )

            print(
                f"[{chunk_type}] "
                f"{score:.4f} - {file_path}"
            )
