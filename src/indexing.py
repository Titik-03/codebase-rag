import re
from chunking import parse_nodes, chunk_markdown, add_source_to_chunks
from collections import Counter
import pickle
import math

def tokenize_text(text):
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

def tokenize(chunks, name):
    for c in chunks:
        tokens = tokenize_text(c["text"])

        if name == "code":
            tokens += tokenize_text(c["file_path"])

        c["tokens"] = tokens

def index(chunks):
    index = []

    for chunk_id, chunk in enumerate(chunks):
        c = {}

        c["chunk_id"] = chunk_id
        c["tokens"] = chunk["tokens"]
        c["tf"] = Counter(c["tokens"])
        c["length"] = len(c["tokens"])

        index.append(c)

    return index

def inverted_index(index):
    inverted = {}
    for chunk in index:
        chunk_id = chunk["chunk_id"]
        for token, freq in chunk["tf"].items():
            if token not in inverted:
                inverted[token] = {}
            inverted[token][chunk_id] = freq

    return inverted

def df(inverted):
    df = {}
    for token, chunks in inverted.items():
        df[token] = len(chunks)
    return df

def average_lenght(index):
    i = sum(chunk["length"] for chunk in index)
    j = len(index)
    return i/j

def bm25_idf(token, inverted, N):
    if token not in inverted:
        return 0

    df = len(inverted[token])

    idf = math.log(
        1 + (N - df + 0.5) / (df + 0.5)
    )

    return idf

def bm25_tf(tf, chunk_length, avgdl, k1=1.5, b=0.75):

    numerator = tf * (k1 + 1)

    length_normalization = (
        1 - b + b * (chunk_length / avgdl)
    )

    denominator = tf + k1 * length_normalization

    return numerator / denominator
def bm25_term_score(
    token,
    chunk,
    inverted,
    N,
    avgdl,
    k1=1.5,
    b=0.75
):
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
    query,
    index,
    inverted,
    avgdl,
    top_k=5,
    k1=1.5,
    b=0.75
):
    query_tokens = tokenize_text(query)

    scores = {}
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

def save_data(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def process_chunks(chunks, name):
    add_source_to_chunks(chunks)
    tokenize(chunks, name)

    index_chunks = index(chunks)
    inv = inverted_index(index_chunks)

    metadata = {
        "N": len(index_chunks),
        "avgdl": average_lenght(index_chunks)
    }

    save_data(f"data/processed/{name}_chunks.pkl", chunks)
    save_data(f"data/processed/{name}_index.pkl", index_chunks)
    save_data(f"data/processed/{name}_inverted_index.pkl", inv)
    save_data(f"data/processed/{name}_metadata.pkl", metadata)


if __name__ == "__main__":

    chunks_py = parse_nodes(
        "data/raw/vllm-0.10.1",
        max_chunk_size=2000
    )

    chunks_doc = chunk_markdown(
        "data/raw/vllm-0.10.1",
        max_chunk_size=2000
    )

    process_chunks(chunks_py, "code")
    process_chunks(chunks_doc, "docs")

    print("Indexing finished.")