import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
from IPython.display import Audio, display

import librosa
import matplotlib.pyplot as plt
import librosa.display

import io
# import numpy as np
import os
import torch
from transformers import pipeline
import warnings


def write_displays(path):
    array, sampling_rate = librosa.load(path)
    plt.figure().set_figwidth(12)
    librosa.display.waveshow(array, sr=sampling_rate)
    plt.show()


def record_audio(filename="output.wav", duration=5, samplerate=16000):
    """
    Records audio from the microphone.
    Args:
        filename (str): Name of the file to save the audio.
        duration (int): Duration of the recording in seconds.
        samplerate (int): Sample rate of the audio (Hz).
    """
    print(f"Recording for {duration} seconds... Please speak clearly.")
    sd.default.samplerate = samplerate
    sd.default.channels = 1  # Mono
    myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()  # Wait until recording is finished
    write(filename, samplerate, myrecording)  # Save as WAV file
    print(f"Recording complete. Audio saved to {filename}")
    display(Audio(filename))  # Play back the recording
    write_displays(filename)
    return filename


# Record a 10-second audio clip
# audio_file = record_audio(duration=10)

# filename = "/Users/suchithkc/PycharmProjects/speakerdiarization/output.wav"
# write_displays(filename)

# --- SECTION 3: Speech-to-Text Conversion ---
# ------------------------------------------

print("\n--- Performing Speech-to-Text Conversion ---")


# Load a pre-trained ASR (Automatic Speech Recognition) model
# We'll use a smaller, faster model like 'facebook/wav2vec2-base-960h' or 'openai/whisper-tiny'
# For better accuracy, consider 'openai/whisper-small' or 'openai/whisper-base' but they are slower.
# 'openai/whisper-tiny' is good for quick demos.

def transcribe_local_audio(model_path, audio_path):
    """
    Loads a Hugging Face ASR model from a local path and transcribes an audio file.

    Args:
        model_path (str): The local directory containing the downloaded model files.
        audio_path (str): The full path to the local audio file (e.g., WAV, MP3).
    """
    # Check if the local model directory exists
    if not os.path.isdir(model_path):
        print(f"Error: Local model directory '{model_path}' not found.")
        print("Please ensure the model was downloaded correctly using `snapshot_download`.")
        return

    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        print("Please provide a valid path to your audio file.")
        return

    # Optional: Basic check for audio file validity
    try:
        with sf.SoundFile(audio_path, 'r') as f:
            print(f"Loading audio file: {audio_path} (Sample rate: {f.samplerate} Hz, Channels: {f.channels})")
    except Exception as e:
        print(
            f"Warning: Could not open audio file {audio_path} with soundfile. It might be corrupted or an unsupported format.")
        print(f"Error details: {e}")
        # Continue anyway, let the pipeline handle potential errors

    try:
        # Initialize the ASR pipeline, pointing to your local model directory
        # The 'model' argument now takes the local path instead of the Hugging Face ID
        print("LOCAL_MODEL_PATH -- > inside", model_path)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        print(f"ASR pipeline loaded successfully from local path: {model_path}")
        print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    except Exception as e:
        print(f"Error: Could not load ASR model from local path '{model_path}': {e}")
        print("Please ensure all model files are correctly present in the directory.")
        return

    print(f"\nTranscribing audio from: {audio_path}")
    try:
        result = pipe(audio_path)
        print("\nTranscription:")
        print(result["text"])
    except Exception as e:
        print(f"Error during transcription: {e}")
        print("Ensure your audio file is a supported format (e.g., WAV, MP3) and not corrupted.")


LOCAL_MODEL_PATH = "/Users/suchithkc/PycharmProjects/speakerdiarization/whisper-tiny-local"
AUDIO_FILE_PATH = "/Users/suchithkc/PycharmProjects/speakerdiarization/output.wav"

print("LOCAL_MODEL_PATH", LOCAL_MODEL_PATH)
try:
    # pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny",
    #                 device=0 if torch.cuda.is_available() else -1)
    # print("ASR model 'openai/whisper-tiny' loaded successfully.")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file_full_path = os.path.join(current_dir, AUDIO_FILE_PATH)
    transcribe_local_audio(LOCAL_MODEL_PATH, audio_file_full_path)
except Exception as e:
    print(f"Could not load Whisper Tiny model: {e}. Trying with 'facebook/wav2vec2-base-960h'.")
    # pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h",
    #                 device=0 if torch.cuda.is_available() else -1)

# # Perform speech-to-text
# try:
#     text_result = pipe(audio_file)
#     print("\n--- Transcription Result ---")
#     print(text_result["text"])
# except Exception as e:
#     print(f"Error during transcription: {e}")
#     print("Please ensure the audio file is valid and the model loaded correctly.")



###download 

from huggingface_hub import snapshot_download
import os

model_name = "openai/whisper-tiny"
local_dir = "./whisper-tiny-local" # Directory where the model will be saved

print(f"Downloading {model_name} to {local_dir}...")
try:
    snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"Successfully downloaded {model_name} to {local_dir}")
except Exception as e:
    print(f"Error downloading model: {e}")
    print("Please check your internet connection or try again later.")

# You can list the downloaded files to verify
if os.path.exists(local_dir):
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            print(os.path.join(root, file))
