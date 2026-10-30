from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

def tokenize_query(query: str):
    tokens = model.encode(
        query,
        normalize_embeddings=True
    )
    return tokens


with open("data/processed/docs_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# chunks = chunks[:5]

txt = [i["text"] for i in chunks]

embeddings = model.encode(
    txt,
    normalize_embeddings=True
)

print("Shape:", embeddings.shape)
print("First embedding:", embeddings[0][:10])