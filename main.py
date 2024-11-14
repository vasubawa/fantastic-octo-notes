import os
import sys
import torch
import warnings
import signal
import atexit
from audio_extraction import extract_audio_from_video
from transcription import transcribe_audio
from summary import process_transcription
from notes_formatter import format_notes

# Suppress specific FutureWarning from transformers.tokenization_utils_base
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Function to print GPU information
def print_gpu_info():
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {gpu_info.name}")
    else:
        print("No GPU available, using CPU.")

# Function to release resources
def cleanup():
    print("Cleaning up resources...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Add any other cleanup code here

# Register the cleanup function to be called on exit
atexit.register(cleanup)

# Handle SIGINT (Ctrl+C)
def signal_handler(sig, frame):
    print("Interrupt received, releasing resources...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def cleanup_created_files(files):
    for file in files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Removed file: {file}")
            except Exception as e:
                print(f"Error removing file {file}: {e}")

def main():

    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    audio_file = "createdFiles/output.wav"
    transcribed_file_name = "createdFiles/transcription.txt"
    summary_file_name = "createdFiles/summary.md"
    pdf_file_name = "createdFiles/notes.pdf"

    # Ensure directories exist
    os.makedirs(os.path.dirname(audio_file), exist_ok=True)
    os.makedirs(os.path.dirname(transcribed_file_name), exist_ok=True)
    os.makedirs(os.path.dirname(summary_file_name), exist_ok=True)
    os.makedirs(os.path.dirname(pdf_file_name), exist_ok=True)

    if not os.path.isfile(input_file):
        print(f"File {input_file} does not exist.")
        sys.exit(1)

    print("================ Press Ctrl+C to release GPU resources in case of emergency. ==================")
    print_gpu_info()

    # Extract audio from video
    print("Starting audio extraction...")
    extract_audio_from_video(input_file, audio_file)
    print(f"Audio extraction completed. {audio_file} Generated.")

    # Transcribe audio to text
    print("Starting transcription(Audio to Text)...")
    transcribe_audio(audio_file, transcribed_file_name)
    print(f"Transcription completed. {transcribed_file_name} Generated.")

    # Process transcription
    print("Starting transcription processing(Summary)...")
    process_transcription(transcribed_file_name, summary_file_name)
    print(f"Transcription processing completed. {summary_file_name} Generated.")

    # Generate PDF
    print("Starting PDF generation(Summary to PDF)...")
    format_notes(summary_file_name, pdf_file_name)
    print(f"PDF generation completed. {pdf_file_name} Generated.")

    # Clean up created files
    cleanup_created_files([audio_file, transcribed_file_name, summary_file_name])

if __name__ == "__main__":
    main()