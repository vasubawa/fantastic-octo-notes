import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import nltk

def chunk_text(text, max_chunk_length=512):
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

def summarize_text(text, max_chunk_length=512):  # Reduce chunk length
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("text-generation", model="gpt2", device=device) 
    chunks = chunk_text(text, max_chunk_length=max_chunk_length)
    summarized_chunks = []

    for chunk in tqdm(chunks, desc="Summarizing chunks"):
        input_length = len(chunk.split())
        max_length = min(100, input_length)  # Reduce max_length
        min_length = min(10, max_length - 1)
        summarized_chunk = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['generated_text']
        summarized_chunks.append(summarized_chunk)

    summarized_text = " ".join(summarized_chunks)
    return summarized_text

def process_transcription(transcription_file, summary_file_name):
    # Read the transcription file
    with open(transcription_file, "r", encoding="utf-8") as trans_file:
        transcription = trans_file.read()

    # Split transcription into logical sections
    sections = transcription.split("\n\n")  

    summarized_text = ""

    # Process each section individually to maintain structure
    for i, section in enumerate(sections):
        if len(section.strip()) == 0:
            continue
        summarized_section = summarize_text(section)  # Summarize each section
        summarized_text += f"Section {i+1}:\n{summarized_section}\n\n"

    # Write the summarized text to the summary file
    with open(summary_file_name, "w", encoding="utf-8") as summary_file:
        summary_file.write(summarized_text)

# Example usage of AutoTokenizer and AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")