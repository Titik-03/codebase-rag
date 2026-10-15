import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "You answer technical questions using provided evidence. "
            "Give only the answer, without explanations."
        )
    },
    {
        "role": "user",
        "content": """
Context:
To dynamically load a LoRA adapter, send a POST request
to the /v1/load_lora_adapter endpoint.

Question:
What HTTP endpoint is used to dynamically load a LoRA adapter?
"""
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt"
)

inputs = inputs.to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        inputs,
        max_new_tokens=64,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

generated_ids = outputs[0][inputs.shape[1]:]

answer = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
)

print("GENERATED ANSWER:")
print(answer)