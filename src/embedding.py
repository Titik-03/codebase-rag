from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import pickle

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

def bui

with open("data/processed/docs_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# chunks = chunks[:5]

txt = [i["text"] for i in chunks]

if not Path("data/processed/docs_embeddings.npy").is_file():
    embeddings = model.encode(
        txt,
        normalize_embeddings=True
    )
    np.save("data/processed/docs_embeddings.npy", embeddings)

embeddings = np.load("data/processed/docs_embeddings.npy")

query = "how does the vllm manages GPU memory"
query_embeded = embed_query(query)

scores = embeddings @ query_embeded
k = 5

top_indices = np.argsort(scores)[::-1][:k]

for idx in top_indices:
    print("Similarity:", scores[idx])
    print("File:", chunks[idx]["file_path"])
    print("Text:", chunks[idx]["text"][:300])
    print("-" * 50)