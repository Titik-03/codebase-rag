from .models import MinimalSource
from .models import RagDataset, StudentSearchResults


def source_iou(
    expected: MinimalSource,
    retrieved: MinimalSource
) -> float:
    if expected.file_path != retrieved.file_path:
        return 0.0
    start = max(
        expected.first_character_index,
        retrieved.first_character_index
    )
    end = min(
        expected.last_character_index,
        retrieved.last_character_index
    )
    expected_length = (
        expected.last_character_index - expected.first_character_index
    )

    retrieved_length = (
        retrieved.last_character_index - retrieved.first_character_index
    )
    intersection = max(0, end - start)
    union = expected_length + retrieved_length - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def sources_match(
    expected: MinimalSource,
    retrieved: MinimalSource,
    threshold: float = 0.05
) -> bool:
    return source_iou(expected, retrieved) >= threshold


def recall_at_k(
    expected_sources: list[MinimalSource],
    retrieved_sources: list[MinimalSource],
    k: int
) -> float:

    if not expected_sources:
        return 0.0

    found = 0

    for expected in expected_sources:
        for retrieved in retrieved_sources[:k]:
            if sources_match(expected, retrieved):
                found += 1
                break

    return found / len(expected_sources)


def evaluate_dataset(
    reference: RagDataset,
    student: StudentSearchResults
) -> dict[str, float]:

    retrieved_by_id = {
        result.question_id: result
        for result in student.search_results
    }

    ks = [1, 3, 5]
    scores = {k: 0.0 for k in ks}
    evaluated = 0

    for question in reference.rag_questions:

        if not hasattr(question, "sources") or not question.sources:
            continue

        evaluated += 1

        result = retrieved_by_id.get(question.question_id)

        if result is None:
            retrieved_sources = []
        else:
            retrieved_sources = result.retrieved_sources

        for k in ks:
            scores[k] += recall_at_k(
                question.sources,
                retrieved_sources,
                k
            )

    if evaluated == 0:
        raise ValueError("No questions with reference sources")

    return {
        f"recall@{k}": scores[k] / evaluated
        for k in ks
    }


if __name__ == "__main__":
    expected = [
        MinimalSource(
            file_path="test.py",
            first_character_index=100,
            last_character_index=500
        ),
        MinimalSource(
            file_path="test.py",
            first_character_index=600,
            last_character_index=900
        )
    ]

    retrieved = [
        MinimalSource(
            file_path="other.py",
            first_character_index=100,
            last_character_index=500
        ),
        MinimalSource(
            file_path="test.py",
            first_character_index=200,
            last_character_index=500
        )
    ]

    print("Recall@1:", recall_at_k(expected, retrieved, 1))
    print("Recall@2:", recall_at_k(expected, retrieved, 2))
