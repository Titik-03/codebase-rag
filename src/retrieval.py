import pickle
from indexing import bm25_search


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_chunks(results, chunks):
    source = []
    for chunk_id, scores in results:
        source.append({
            "score": scores,
            "chunks": chunks[chunk_id]
        })
    return source
if __name__ == "__main__":
    chunks = load_data("data/chunks.pkl")
    index_chunks = load_data("data/index.pkl")
    inverted = load_data("data/inverted_index.pkl")
    metadata = load_data("data/metadata.pkl")
    query = "How does vLLM allocate GPU memory?"
    results = bm25_search(
        query,
        index_chunks,
        inverted,
        metadata["avgdl"],
        top_k=5
    )
    print(get_chunks(results, chunks))
    # print("Chunks:", len(chunks))
    # print("Index entries:", len(index_chunks))
    # print("Indexed tokens:", len(inverted))
    # print("Metadata:", metadata)