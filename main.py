import os
import sys
import torch
import warnings
from audio_extraction import extract_audio_from_video
from transcription import transcribe_audio
from summary import process_transcription
from notes_formatter import format_notes  # Import the PDF generator

# Suppress specific FutureWarning from transformers.tokenization_utils_base
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Function to print GPU information
def print_gpu_info():
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {gpu_info.name}")
    else:
        print("No GPU available, using CPU.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    audio_file = "createdFiles/output.wav"
    transcribed_file_name = "createdFiles/transcription.txt"
    summary_file_name = "createdFiles/summary.txt"
    pdf_file_name = "createdFiles/notes.pdf"  # New file for the PDF

    # Ensure directories exist
    os.makedirs(os.path.dirname(audio_file), exist_ok=True)
    os.makedirs(os.path.dirname(transcribed_file_name), exist_ok=True)
    os.makedirs(os.path.dirname(summary_file_name), exist_ok=True)
    os.makedirs(os.path.dirname(pdf_file_name), exist_ok=True)

    if not os.path.isfile(input_file):
        print(f"File {input_file} does not exist.")
        sys.exit(1)

    print("================ Press Ctrl+Z to release GPU resources in case of emergency. ==================")
    print_gpu_info()

    # Extract audio from video
    print("Starting audio extraction...")
    #extract_audio_from_video(input_file, audio_file)
    print("Audio extraction completed.")

    # Transcribe audio to text
    print("Starting transcription...")
    #transcribe_audio(audio_file, transcribed_file_name)
    print("Transcription completed.")

    # Process transcription
    print("Starting transcription processing...")
    process_transcription(transcribed_file_name, summary_file_name)
    print("Transcription processing completed.")

    # Generate PDF
    print("Starting PDF generation...")
    format_notes(summary_file_name, pdf_file_name)
    print("PDF generation completed.")

if __name__ == "__main__":
    main()