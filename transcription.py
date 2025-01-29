from faster_whisper import WhisperModel
from tqdm import tqdm
import torch

def transcribe_audio(audio_file, transcribed_file_name):
    model_size = "small"  # Use a smaller model size
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    # Check if the model has the flatten_parameters method and call it
    if hasattr(model.model, 'flatten_parameters'):
        model.model.flatten_parameters()

    segments, info = model.transcribe(audio_file, beam_size=5, language="en", condition_on_previous_text=False)

    transcribed_text = ""
    segment_count = sum(1 for _ in segments)  # Count the number of segments

    # Reinitialize the generator
    segments, info = model.transcribe(audio_file, beam_size=5, language="en", condition_on_previous_text=False)

    # Initialize the progress bar
    pbar = tqdm(total=segment_count, desc="Transcribing", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} segments")

    for i, segment in enumerate(segments):
        transcribed_text += "\n" + segment.text
        pbar.update(1)  # Update the progress bar

    pbar.close()  # Close the progress bar when done

    with open(transcribed_file_name, "w", encoding="utf-8") as file:
        file.write(transcribed_text)

    # Allow memory growth
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, 0)  # Use GPU index 0