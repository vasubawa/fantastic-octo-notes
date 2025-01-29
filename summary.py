import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

model_name = "Qwen/Qwen2.5-Coder-3B"

# Initialize the model with optimized   
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    offload_folder="offload",
    offload_buffers=True,  # Enable buffer offloading
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def summarize_text(text, max_new_tokens=512):
    prompt = f"""
Create an exhaustive, detailed analysis of this transcript. Your task:

1. **Content Extraction (100% Coverage)**
   - Document EVERY concept mentioned
   - Include ALL numerical data and specifications 
   - Preserve ALL examples and comparisons
   - Maintain ALL relationships between concepts
   - Capture ALL implementation details

Use detailed sub-sections and clear hierarchy.
NEVER skip or summarize - document EVERYTHING.
You will use the MARKDOWN format from START to FINISH

Begin detailed analysis:\n{text}"""

    messages = [
        {"role": "system", "content": "You are an expert analyst creating exhaustive documentation."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens = max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    summarized_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return summarized_text

def process_transcription(transcription_file, summary_file_name, chunk_size=1000):  # Reduced chunk size
    if not os.path.exists(transcription_file):
        print(f"Error: The file {transcription_file} does not exist.")
        return

    with open(transcription_file, "r", encoding="utf-8") as trans_file:
        transcription = trans_file.read()

    # Split the transcription into chunks
    chunks = [transcription[i:i + chunk_size] for i in range(0, len(transcription), chunk_size)]
    
    summarized_text = ""
    start_time = time.time()
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        try:
            summarized_text += summarize_text(chunk) + "\n"
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue
        cleanup_gpu_memory()  # Clear GPU memory after processing each chunk
    end_time = time.time()

    with open(summary_file_name, "w", encoding="utf-8") as summary_file:
        summary_file.write(summarized_text)

    elapsed_time = end_time - start_time
    print(f"Summarization completed in {elapsed_time // 60:.0f} minutes and {elapsed_time % 60:.0f} seconds.")

cleanup_gpu_memory()