import subprocess
from tqdm import tqdm
import psutil

def extract_audio_from_video(video_file, audio_file):
    command = [
        'ffmpeg', '-y', '-i', video_file, '-q:a', '0', '-map', 'a', audio_file
    ]
    with tqdm(total=1, desc="Extracting Audio", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} steps") as pbar:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while process.poll() is None:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            pbar.set_postfix(CPU=f"{cpu_usage}%", Memory=f"{memory_usage}%")
        pbar.update(1)