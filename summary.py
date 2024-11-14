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
    prompt = f"""
Summarize the following text in a detailed manner, focusing on key points and providing explanations for each section. Include specific examples and comparisons where applicable. The summary should cover:

1. Introduction and overview
2. Main concepts and ideas
3. Key takeaways and implications
4. Potential applications or real-world relevance

Organize your response into clear sections with appropriate headings. Use bullet points and numbered lists to enhance readability. Ensure that your summary is comprehensive yet concise, capturing the essence of the original text while adding value through analysis and explanation.

Please respond as if you were explaining this topic to someone who is unfamiliar with it but interested in learning more. Be clear, concise, and engaging in your writing style.

Format your response using Markdown syntax. This means:
- Use headers (e.g., # for main headers, ## for subheaders)
- Utilize bold (**text**) and italic (*text*) formatting
- Create lists using asterisks (*) for bullets or numbers
- Use code blocks (```)
:\n{text}"""
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

    # Write the summarized text to a Markdown file
    with open(summary_file_name, "w", encoding="utf-8") as summary_file:
        summary_file.write(summarized_text)

