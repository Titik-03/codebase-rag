from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import pickle
from .retrieval import get_sources
from .models import MinimalSearchResults



model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

def embed_query(query: str):
    tokens = model.encode(
        query,
        normalize_embeddings=True
    )
    return tokens

def build_embeddings(index_type: str):

    with open(f"data/processed/{index_type}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    path = Path(f"data/processed/{index_type}_embeddings.npy")

    if not path.is_file():
        txt = [chunk["text"] for chunk in chunks]

        embeddings = model.encode(
            txt,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True
        )

        np.save(path, embeddings)

    embeddings = np.load(path)

    return chunks, embeddings

def semantic_search(query: str, index_type: str, k: int = 5):
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
):
    results = semantic_search(query, index_type, k)

    chunks, _ = build_embeddings(index_type)

    sources = get_sources(results, chunks)

    return MinimalSearchResults(
        question_id=question_id,
        question=query,
        retrieved_sources=sources
    )
if __name__ == "__main__":
    result = semantic_retrieve(
        question_id="test-docs-001",
        query="How does vLLM allocate KV cache blocks?",
        index_type="docs",
        k=5
    )

    print(result.model_dump_json(indent=2))