import torch

from .models import (
    MinimalSource,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from .retrieval import load_data

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


def prepare_inputs(question: str, contexts: list[str]):
    prompt = build_prompt(question, contexts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical assistant for the vLLM repository. "
                "Answer using only the provided context. "
                "Be precise and concise."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True
    )

def fit_context(
    contexts: list[str],
    question: str,
    max_input_tokens: int
) -> list[str]:

    selected_chunks = []

    for context in contexts:
        candidate = selected_chunks + [context]

        inputs = prepare_inputs(question, candidate)

        token_count = inputs["input_ids"].shape[1]

        if token_count <= max_input_tokens:
            selected_chunks.append(context)

    return selected_chunks

def build_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n\n".join(contexts)

    return f"""Answer the question using only the provided context.

Rules:
- Give a direct and precise answer.
- Preserve exact function names, API endpoints, and parameters.
- Do not repeat the question or the answer.
- Do not invent information.
- If the context does not contain the answer, say "Insufficient context".

Context:
{context_text}

Question: {question}
"""

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

    if not contexts:
        return "Insufficient context"

    inputs = prepare_inputs(question, contexts)
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    generated_ids = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return answer.strip()

def generate_dataset(dataset: StudentSearchResults) -> StudentSearchResultsAndAnswer:
    answers = []
    for result in tqdm(dataset.search_results):
        answer = generate_answer(
            result.question,
            result.retrieved_sources
        )
        answers.append(
            MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=answer
            )
        )
    return StudentSearchResultsAndAnswer(
        search_results=answers,
        k=dataset.k
    )

def save_results(results: StudentSearchResultsAndAnswer, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(results.model_dump_json(indent=2))


# if __name__ == "__main__":
#     test_result = StudentSearchResultsAndAnswer(
#         search_results=[
#             MinimalAnswer(
#                 question_id="test1",
#                 question="What is vLLM?",
#                 retrieved_sources=[],
#                 answer="vLLM is an LLM inference engine."
#             )
#         ],
#         k=5
#     )

#     save_results(test_result, "test_answers.json")

#     print("JSON saved successfully!")