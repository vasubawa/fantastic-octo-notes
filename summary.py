from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Initialize the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def estimate_max_tokens_3b(text, tokenizer):
    # Count input tokens for logging
    input_tokens = len(tokenizer.encode(text))
    
    # 3B Model: 32K context window
    # Reserve ~half for input, half for generation
    max_tokens = min(16384, 32768 - input_tokens)
    
    # Set minimum reasonable summary length
    min_tokens = 2048
    
    tokens = max(min_tokens, max_tokens)
    print(f"[3B] Input tokens: {input_tokens}, Max generation tokens: {tokens}")
    return tokens

def estimate_max_tokens_7b(text, tokenizer):
    # Count input tokens for logging
    input_tokens = len(tokenizer.encode(text))
    
    # 7B Model: 131K context window
    # Reserve ~half for input, half for generation  
    max_tokens = min(65536, 131072 - input_tokens)
    
    # Set minimum reasonable summary length
    min_tokens = 2048
    
    tokens = max(min_tokens, max_tokens)
    print(f"[7B] Input tokens: {input_tokens}, Max generation tokens: {tokens}")
    return tokens

# Use this for current 3B model - remember to change the 7B to 3B
#estimate_max_tokens = estimate_max_tokens_3b
#model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

# Uncomment below and comment above to use 7B model - remeber to change the 3B to 7B
estimate_max_tokens = estimate_max_tokens_7b
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

def summarize_text(text):
    max_new_tokens = estimate_max_tokens(text, tokenizer)
    
    prompt = f"""
Create an exhaustive, detailed analysis of this transcript. Your task:

1. **Content Extraction (100% Coverage)**
   - Document EVERY concept mentioned
   - Include ALL numerical data and specifications 
   - Preserve ALL examples and comparisons
   - Maintain ALL relationships between concepts
   - Capture ALL implementation details

2. **Structural Analysis**
   - Break down EACH major topic
   - Show progression of ideas
   - Highlight connections between sections
   - Document ALL prerequisites and dependencies
   - Map knowledge flow and build-up

3. **Detail Preservation**
   - Start-to-finish coverage
   - No summarization - maintain ALL details
   - Keep ALL specific values and parameters
   - Preserve ALL procedural steps
   - Include ALL caveats and conditions

Use detailed sub-sections and clear hierarchy.
NEVER skip or summarize - document EVERYTHING.

Begin detailed analysis:\n{text}"""

    messages = [
        {"role": "system", "content": "You are an expert analyst creating exhaustive documentation. Your task is to capture and explain EVERY detail from the input, regardless of domain. Never summarize or omit information."},
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
    with open(transcription_file, "r", encoding="utf-8") as trans_file:
        transcription = trans_file.read()

    summarized_text = summarize_text(transcription)

    with open(summary_file_name, "w", encoding="utf-8") as summary_file:
        
        summary_file.write(summarized_text)