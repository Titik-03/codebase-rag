import pickle
from .indexing import bm25_search, save_data
from pathlib import Path
from tqdm import tqdm
from .cache import (
    make_cache_key,
    get_index_version,
    load_cached_result,
    save_cached_result
)
from .models import (
    MinimalSource,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
)


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_sources(results, chunks) -> list[MinimalSource]:
    sources = []

    for chunk_id, _ in results:
        chunk = chunks[chunk_id]

        sources.append(
            MinimalSource(
                file_path=chunk["file_path"],
                first_character_index=chunk["first_character_index"],
                last_character_index=chunk["last_character_index"],
            )
        )

    return sources


def search(
    question_id,
    query,
    chunks,
    index_chunks,
    inverted,
    avgdl,
    k=5
):
    results = bm25_search(
        query,
        index_chunks,
        inverted,
        avgdl,
        top_k=k
    )

    sources = get_sources(results, chunks)

    return MinimalSearchResults(
        question_id=question_id,
        question=query,
        retrieved_sources=sources
    )


def search_dataset(
    dataset_path,
    chunks,
    index_chunks,
    inverted,
    avgdl,
    index_type: str,
    k=5,
    index_directory: str = "data/processed"
):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = f.read()

    dataset = RagDataset.model_validate_json(data)

    search_results = []

    index_version = get_index_version(index_type,
                                      index_directory)

    for question in tqdm(
        dataset.rag_questions,
        desc="Retrieving",
        unit="question"
    ):
        cache_key = make_cache_key(
            question.question,
            k,
            f"bm25:{index_type}:{index_version}"
        )

        cached = load_cached_result(cache_key)

        if cached is not None:
            result = MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=cached.retrieved_sources
            )

        else:
            result = search(
                question.question_id,
                question.question,
                chunks,
                index_chunks,
                inverted,
                avgdl,
                k
            )

            save_cached_result(cache_key, result)

        search_results.append(result)

    return StudentSearchResults(
        search_results=search_results,
        k=k
    )

def dump_json(result, output_path):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:
        f.write(result.model_dump_json(indent=2))


if __name__ == "__main__":

    # DOCS
    docs_chunks = load_data(
        "data/processed/docs_chunks.pkl"
    )
    docs_index = load_data(
        "data/processed/docs_index.pkl"
    )
    docs_inverted = load_data(
        "data/processed/docs_inverted_index.pkl"
    )
    docs_metadata = load_data(
        "data/processed/docs_metadata.pkl"
    )

    # CODE
    code_chunks = load_data(
        "data/processed/code_chunks.pkl"
    )
    code_index = load_data(
        "data/processed/code_index.pkl"
    )
    code_inverted = load_data(
        "data/processed/code_inverted_index.pkl"
    )
    code_metadata = load_data(
        "data/processed/code_metadata.pkl"
    )

    docs_results = search_dataset(
        "data/datasets/UnansweredQuestions/dataset_docs_public.json",
        docs_chunks,
        docs_index,
        docs_inverted,
        docs_metadata["avgdl"],
        k=5
    )

    code_results = search_dataset(
        "data/datasets/UnansweredQuestions/dataset_code_public.json",
        code_chunks,
        code_index,
        code_inverted,
        code_metadata["avgdl"],
        k=5
    )
    save_data(f"data/processed/docs_re.pkl", docs_results)
    save_data(f"data/processed/code_re.pkl", code_results)
    dump_json(
        docs_results,
        "data/output/search_results/docs_result.json"
    )

    dump_json(
        code_results,
        "data/output/search_results/code_result.json"
    )
