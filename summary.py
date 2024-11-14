import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Initialize the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def summarize_text(text, max_new_tokens=512):
    prompt = f"Summarize the following text in a detailed manner:\n{text}"
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    summarized_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return summarized_text

def process_transcription(transcription_file, summary_file_name):
    # Read the transcription file
    with open(transcription_file, "r", encoding="utf-8") as trans_file:
        transcription = trans_file.read()

    # Summarize the entire transcription
    summarized_text = summarize_text(transcription)

    # Write the summarized text to the summary file
    with open(summary_file_name, "w", encoding="utf-8") as summary_file:
        summary_file.write(summarized_text)