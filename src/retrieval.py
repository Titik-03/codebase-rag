import pickle
from indexing import bm25_search
from pathlib import Path
import json
from models import (
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
    ) -> MinimalSearchResults:

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

def search_dataset(chunks, index_chunks, inverted, metadata):
    with open("data/datasets/UnansweredQuestions/dataset_docs_public.json", "r") as f:
        data = f.read()
        dataset = RagDataset.model_validate_json(data)
        search_result = []
        for question in dataset.rag_questions:
            id = question.question_id
            query = question.question
            result = search(
                question_id=id,
                query=query,
                chunks=chunks,
                index_chunks=index_chunks,
                inverted=inverted,
                avgdl=metadata["avgdl"],
                k=5
            )
            search_result.append(result)
        return StudentSearchResults(
            search_results=search_result,
            k=5
        )
def dump_json(result):
    output_dir = Path("data/output/search_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "result.json", "w") as f:
        f.write(result.model_dump_json(indent=2))


if __name__ == "__main__":
    chunks = load_data("data/processed/chunks.pkl")
    index_chunks = load_data("data/processed/index.pkl")
    inverted = load_data("data/processed/inverted_index.pkl")
    metadata = load_data("data/processed/metadata.pkl")
    result = search_dataset(chunks, index_chunks, inverted, metadata)
    dump_json(result)
