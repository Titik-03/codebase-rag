from models import MinimalSource
from transformers import AutoTokenizer, AutoModelForCausalLM
from retrieval import load_data

def getcontext(sources: list[MinimalSource]) -> list[str]:
    contexts = []
    for source in sources:
        with open(source.file_path, "r") as f:
            text = f.read()
            chunk = text[source.first_character_index:
                     source.last_character_index]
            contexts.append(chunk)
    return contexts

def build_prompt(question: str, contexts: list[str]) -> str:
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

MODEL_NAME = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
def generate_answer(
    question: str,
    sources: list[MinimalSource]
) -> str:

    contexts = getcontext(sources)

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
        max_new_tokens=200
    )

    input_length = inputs["input_ids"].shape[1]

    generated_ids = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return answer

docs = load_data("data/processed/docs_re.pkl")

result = docs.search_results[0]

answer = generate_answer(
    result.question,
    result.retrieved_sources
)

print("Question:", result.question)
print("Answer:", answer)