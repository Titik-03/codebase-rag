
from .embedding import semantic_search
from .combined_search import combined_search


def reciprocal_rank_fusion(
    bm25_results: list[tuple[int, float]],
    semantic_results: list[tuple[int, float]],
    k: int = 5,
) -> list[tuple[int, float]]:

    """Merge lexical and semantic rankings using reciprocal ranks."""
    scores = {}
    result = []
    for rank, (chunk_id, score) in enumerate(bm25_results, start=1):
        scores[chunk_id] = 1/(60 + rank)

    for rank, (chunk_id, score) in enumerate(semantic_results, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1/(60 + rank)

    for id, score in scores.items():
        result.append((id, score))

    result = sorted(result, key=lambda score: score[1], reverse=True)
    result = result[:k]
    return result


def hybrid_search(
    query: str,
    k: int = 5,
    candidate_k: int = 20,
) -> list[tuple[int, float]]:

    """Fuse BM25 and semantic candidates into one ranked result list."""
    if k <= 0 or candidate_k <= 0:
        raise ValueError("k and candidate_k must be positive")

    candidate_k = max(k, candidate_k)

    bm25_results = combined_search(query, k=candidate_k)

    semantic_results = semantic_search(
        query,
        "combined",
        k=candidate_k
    )

    return reciprocal_rank_fusion(
        bm25_results,
        semantic_results,
        k=k
    )


if __name__ == "__main__":
    query = "How does vLLM allocate GPU memory?"

    results = hybrid_search(query, k=5)

    for chunk_id, score in results:
        print(f"Chunk {chunk_id}: RRF score = {score:.6f}")
