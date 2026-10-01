from models import MinimalSource
from transformers import AutoTokenizer, AutoModelForCausalLM
from retrieval import load_data


MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 200

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
CONTEXT_LIMIT = model.config.max_position_embeddings

MAX_INPUT_TOKENS = CONTEXT_LIMIT - MAX_NEW_TOKENS


def get_context(sources: list[MinimalSource]) -> list[str]:
    """Read and extract the text represented by retrieved sources."""
    contexts = []

    for source in sources:
        with open(source.file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunk = text[
            source.first_character_index:
            source.last_character_index
        ]

        contexts.append(chunk)

    return contexts

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text."""
    tokens = tokenizer(text)
    return len(tokens["input_ids"])


def fit_context(
    contexts: list[str],
    question: str,
    max_input_tokens: int
) -> list[str]:
    """Select contexts that fit inside the model input token budget."""
    selected_chunks = []

    for context in contexts:
        candidate = selected_chunks + [context]

        prompt = build_prompt(
            question,
            candidate
        )

        token_count = count_tokens(prompt)

        if token_count <= max_input_tokens:
            selected_chunks.append(context)

    return selected_chunks

def build_prompt(question: str, contexts: list[str]) -> str:
    """Build a prompt containing retrieved context and the question."""
    context_text = "\n\n".join(contexts)

    prompt = f"""
Answer the question using only the provided context.

Context:
{context_text}

Question:
{question}

Answer:
"""

    return prompt


def generate_answer(
    question: str,
    sources: list[MinimalSource]
) -> str:
    """Generate an answer from a question and its retrieved sources."""

    contexts = get_context(sources)
    contexts = fit_context(
        contexts,
        question,
        MAX_INPUT_TOKENS
    )
    prompt = build_prompt(
        question,
        contexts
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS
    )

    input_length = inputs["input_ids"].shape[1]

    generated_ids = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return answer


if __name__ == "__main__":
    docs = load_data("data/processed/docs_re.pkl")

    result = docs.search_results[0]

    contexts = get_context(
        result.retrieved_sources
    )

    selected_contexts = fit_context(
        contexts,
        result.question,
        MAX_INPUT_TOKENS
    )

    print("\nQUESTION:")
    print(result.question)

    # print("\nRETRIEVED CONTEXT:")
    # for i, context in enumerate(selected_contexts, start=1):
    #     print(f"\n--- Context {i} ---")
    #     print(context)

    answer = generate_answer(
        result.question,
        result.retrieved_sources
    )

    print("\nGENERATED ANSWER:")
    print(answer)