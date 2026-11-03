from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import pickle
from .retrieval import get_sources
from .models import MinimalSearchResults
import hashlib
import json
from typing import Any


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def embed_query(query: str) -> Any:
    """Encode a query as a normalized semantic vector."""
    tokens: Any = model.encode(
        query,
        normalize_embeddings=True
    )
    return tokens


def chunks_fingerprint(chunks: list[dict[str, Any]]) -> str:
    """Hash the embedding model settings and ordered chunk content."""
    hasher = hashlib.sha256()
    hasher.update(b"all-MiniLM-L6-v2:normalized:v1")
    for chunk in chunks:
        data = [
            chunk["file_path"],
            chunk["first_character_index"],
            chunk["last_character_index"],
            chunk["text"]
        ]
        encoded = json.dumps(data, ensure_ascii=True).encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)

    return hasher.hexdigest()


def build_embeddings(
    index_type: str
) -> tuple[list[dict[str, Any]], Any]:
    """Load matching embeddings or encode and save the indexed chunks."""
    if index_type not in ("docs", "code", "combined"):
        raise ValueError("Invalid index type")

    base = Path("data/processed")

    with open(base / f"{index_type}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedding_path = base / f"{index_type}_embeddings.npy"
    fingerprint_path = base / f"{index_type}_embeddings.sha256"

    current_hash = chunks_fingerprint(chunks)

    valid = (
        embedding_path.exists()
        and fingerprint_path.exists()
        and fingerprint_path.read_text().strip() == current_hash
    )

    if valid:
        embeddings = np.load(embedding_path)

        valid = embeddings.shape == (
            len(chunks),
            model.get_sentence_embedding_dimension()
        )

    if not valid:
        print(f"Building {index_type} embeddings...")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True
        )

        np.save(embedding_path, embeddings)

        fingerprint_path.write_text(current_hash)

    return chunks, embeddings


def semantic_search(
    query: str, index_type: str, k: int = 5
) -> list[tuple[int, float]]:
    """Rank chunks by similarity to the normalized query vector."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    chunks, embeddings = build_embeddings(index_type)

    query_embedding = embed_query(query)

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:k]

    results = []

    for idx in top_indices:
        results.append((int(idx), float(scores[idx])))

    return results


def semantic_retrieve(
    question_id: str,
    query: str,
    index_type: str = "docs",
    k: int = 5
) -> MinimalSearchResults:
    """Return validated source locations from semantic search."""
    results = semantic_search(query, index_type, k)

    chunks, _ = build_embeddings(index_type)

    sources = get_sources(results, chunks)

    return MinimalSearchResults(
        question_id=question_id,
        question=query,
        retrieved_sources=sources
    )


if __name__ == "__main__":
    results = semantic_search(
        query="How does vLLM allocate GPU memory?",
        index_type="code",
        k=5
    )
    print(results)
