import os
import torch
import streamlit as st

@st.cache_resource
def get_whisper_model(model_size="tiny"): # "tiny" is much faster than "base"
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_size, device=device)

def transcribe_audio(audio_path):
    """
    Transcribes audio to text using a cached Whisper model.
    """
    # Check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = get_whisper_model()
    result = model.transcribe(audio_path)
    return result["text"]
