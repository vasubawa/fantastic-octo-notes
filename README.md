# Audio Transcription and Summarization

This project extracts audio from a video file, transcribes the audio to text, summarizes the transcription, and generates a PDF of the summary.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed Python 3.6 or later.
- You have installed `pip` (Python package installer).
- You have installed `ffmpeg` for audio extraction.
- You have installed `wkhtmltopdf` for PDF generation.
- You have installed `CUDA` and `cuDNN` for GPU acceleration (optional but recommended).

## Installation

1. **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`
    ```

3. **Install Python dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Install `ffmpeg`**:
    - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and follow the installation instructions.
    - **macOS**: Install using Homebrew:
      ```sh
      brew install ffmpeg
      ```
    - **Linux**: Install using your package manager. For example, on Ubuntu:
      ```sh
      sudo apt-get install ffmpeg
      ```

5. **Install `wkhtmltopdf`**:
    - **Windows**: Download the installer from [wkhtmltopdf.org](https://wkhtmltopdf.org/downloads.html) and follow the installation instructions.
    - **macOS**: Install using Homebrew:
      ```sh
      brew install wkhtmltopdf
      ```
    - **Linux**: Install using your package manager. For example, on Ubuntu:
      ```sh
      sudo apt-get install wkhtmltopdf
      ```

6. **Install CUDA and cuDNN** (optional but recommended for GPU acceleration):
    - **Windows and Linux**: Follow the installation instructions from the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-toolkit).
		- **cuDNN Version**: 8.x.x.x  (cudnn-linux-x86_64-8.9.7.29_cuda12-archive) Exact one I used

## Usage

1. **Run the main script**:
    ```sh
    python main.py <input_file>
    ```

    Replace `<input_file>` with the path to your audio/video file.

2. **Output**:
    - The extracted audio will be saved as [output.wav]).
    - The transcription will be saved as [transcription.txt].
    - The summary will be saved as [summary.md].
    - The PDF of the summary will be saved as [notes.pdf].
