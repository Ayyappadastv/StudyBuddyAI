import torch
import streamlit as st

@st.cache_resource
def get_summarizer_pipeline():
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    # Using a faster, smaller model for speed
    return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=device)

def summarize_text(text, max_length=150, min_length=50):
    """
    Summarizes text using a cached DistilBART pipeline.
    """
    summarizer = get_summarizer_pipeline()
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def summarize_chunks(chunks, model_name="facebook/bart-large-cnn"):
    """
    Summarizes multiple chunks and joins them.
    """
    summaries = []
    for chunk in chunks:
        if len(chunk.strip()) > 100: # Only summarize meaningful chunks
            summaries.append(summarize_text(chunk))
    return " ".join(summaries)
