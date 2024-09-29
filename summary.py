import nltk
from transformers import pipeline
import torch
from tqdm import tqdm


def chunk_text(text, max_chunk_length=1024):
    sentences = nltk.sent_tokenize(text)  # Split text into sentences
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(" ".join(current_chunk + [sentence])) <= max_chunk_length:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_text(text, max_chunk_length=1024):
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)  # Use a smaller model if needed
    chunks = chunk_text(text, max_chunk_length=max_chunk_length)
    summarized_chunks = []

    for chunk in chunks:
        input_length = len(chunk.split())
        max_length = min(150, input_length)  # Adjust max_length based on input length
        min_length = min(10, max_length - 1)  # Ensure min_length is less than max_length
        summarized_chunk = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summarized_chunks.append(summarized_chunk)

    summarized_text = " ".join(summarized_chunks)
    return summarized_text

def process_transcription(transcription_file, summary_file_name):
    # Read the transcription file
    with open(transcription_file, "r", encoding="utf-8") as trans_file:
        transcription = trans_file.read()

    # Split transcription into logical sections
    sections = transcription.split("\n\n")  # Assuming double newline separates sections in the original text

    summarized_text = ""

    # Process each section individually to maintain structure
    for i, section in enumerate(sections):
        if len(section.strip()) == 0:
            continue  # Skip empty sections
        section_header = f"## Section {i + 1}\n"
        summarized_section = summarize_text(section)  # Summarize each section
        summarized_text += section_header + summarized_section + "\n\n"

    # Write the summarized text to the summary file
    with open(summary_file_name, "w", encoding="utf-8") as summary_file:
        summary_file.write(summarized_text)

    print(f"Summary saved to {summary_file_name}")

    # Clear cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()