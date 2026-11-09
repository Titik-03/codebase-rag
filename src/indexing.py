import re
from .chunking import parse_nodes, chunk_markdown, add_source_to_chunks
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import pickle
import math
from typing import Any


def tokenize_text(text: str) -> list[str]:
    """Extract lowercase identifiers and their underscore-separated parts."""
    raw_tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*",
        text
    )

    tokens = []

    for token in raw_tokens:
        token = token.lower()

        tokens.append(token)

        if "_" in token:
            tokens.extend(
                part for part in token.split("_") if part
            )

    return tokens


def tokenize(chunks: list[dict[str, Any]], name: str) -> None:
    """Attach tokens to chunks, including file paths for code chunks."""
    for c in tqdm(chunks, desc=f"Tokenizing {name}", unit="chunk"):
        tokens = tokenize_text(c["text"])

        if name == "code":
            tokens += tokenize_text(c["file_path"])

        c["tokens"] = tokens


def index(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build term frequencies and lengths for tokenized chunks."""
    index = []

    for chunk_id, chunk in enumerate(chunks):
        c: dict[str, Any] = {}

        c["chunk_id"] = chunk_id
        c["tokens"] = chunk["tokens"]
        c["tf"] = Counter(c["tokens"])
        c["length"] = len(c["tokens"])

        index.append(c)

    return index


def inverted_index(
    index: list[dict[str, Any]]
) -> dict[str, dict[int, int]]:
    """Map each token to its chunk frequencies."""
    inverted: dict[str, dict[int, int]] = {}
    for chunk in index:
        chunk_id = chunk["chunk_id"]
        for token, freq in chunk["tf"].items():
            if token not in inverted:
                inverted[token] = {}
            inverted[token][chunk_id] = freq

    return inverted


def df(inverted: dict[str, dict[int, int]]) -> dict[str, int]:
    """Count the chunks containing each token."""
    df = {}
    for token, chunks in inverted.items():
        df[token] = len(chunks)
    return df


def average_lenght(index: list[dict[str, Any]]) -> float:
    """Calculate the mean token count across indexed chunks."""
    i: int = sum(chunk["length"] for chunk in index)
    j = len(index)
    return i/j


def bm25_idf(
    token: str, inverted: dict[str, dict[int, int]], N: int
) -> float:
    """Calculate the BM25 inverse document frequency of a token."""
    if token not in inverted:
        return 0

    df = len(inverted[token])

    idf = math.log(
        1 + (N - df + 0.5) / (df + 0.5)
    )

    return idf


def bm25_tf(
    tf: int, chunk_length: int, avgdl: float, k1: float = 1.5, b: float = 0.75
) -> float:

    """Calculate the length-normalized BM25 term frequency."""
    numerator = tf * (k1 + 1)

    length_normalization = (
        1 - b + b * (chunk_length / avgdl)
    )

    denominator = tf + k1 * length_normalization

    return numerator / denominator


def bm25_term_score(
    token: str,
    chunk: dict[str, Any],
    inverted: dict[str, dict[int, int]],
    N: int,
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75
) -> float:
    """Calculate one query token contribution to a chunk score."""
    if token not in chunk["tf"]:
        return 0

    tf = chunk["tf"][token]

    idf = bm25_idf(
        token,
        inverted,
        N
    )

    adjusted_tf = bm25_tf(
        tf,
        chunk["length"],
        avgdl,
        k1,
        b
    )

    return idf * adjusted_tf


def bm25_search(
    query: str,
    index: list[dict[str, Any]],
    inverted: dict[str, dict[int, int]],
    avgdl: float,
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75
) -> list[tuple[int, float]]:
    """Rank matching chunks by their summed BM25 scores."""
    query_tokens = tokenize_text(query)

    scores: dict[int, float] = {}
    N = len(index)

    for token in query_tokens:

        if token not in inverted:
            continue

        for chunk_id in inverted[token]:

            chunk = index[chunk_id]

            term_score = bm25_term_score(
                token,
                chunk,
                inverted,
                N,
                avgdl,
                k1,
                b
            )

            if chunk_id not in scores:
                scores[chunk_id] = 0

            scores[chunk_id] += term_score

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]


def save_data(path: str | Path, data: Any) -> None:
    """Persist an object as a pickle file, creating parent directories."""
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f)


def process_chunks(
    chunks: list[dict[str, Any]],
    name: str,
    index_directory: str = "data/processed"
) -> None:
    """Tokenize chunks and save their BM25 index and metadata."""
    add_source_to_chunks(chunks)
    tokenize(chunks, name)

    index_chunks = index(chunks)
    inv = inverted_index(index_chunks)

    metadata = {
        "N": len(index_chunks),
        "avgdl": average_lenght(index_chunks)
    }

    path = Path(index_directory)

    save_data(path / f"{name}_chunks.pkl", chunks)
    save_data(path / f"{name}_index.pkl", index_chunks)
    save_data(path / f"{name}_inverted_index.pkl", inv)
    save_data(path / f"{name}_metadata.pkl", metadata)


def build_index(max_chunk_size: int = 2000,
                raw_directory: str = "data/raw/vllm-0.10.1",
                index_directory: str = "data/processed"
                ) -> None:
    """Chunk the corpus and persist separate and combined lexical indexes."""
    if not isinstance(max_chunk_size, int) or not 1 < max_chunk_size <= 2000:
        raise ValueError("chunk size must be an integer between 2 and 2000")
    chunks_py = parse_nodes(
        raw_directory,
        max_chunk_size=max_chunk_size
    )

    chunks_doc = chunk_markdown(
        raw_directory,
        max_chunk_size=max_chunk_size
    )

    if not chunks_py:
        raise ValueError(
            f"No Python chunks found in '{raw_directory}'"
        )

    if not chunks_doc:
        raise ValueError(
            f"No documentation chunks found in '{raw_directory}'"
        )

    process_chunks(chunks_py, "code", index_directory)
    process_chunks(chunks_doc, "docs", index_directory)

    from .combined_search import build_and_save_combined
    build_and_save_combined(index_directory)

    print("Indexing finished.")


if __name__ == "__main__":
    build_index()
