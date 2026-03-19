import re
from collections import Counter

def summarize_text(text, max_sentences=5):
    """
    Lightweight extractive summarizer. No heavy model downloads needed.
    Works on any Python version and any cloud environment.
    """
    # Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 30]
    
    if not sentences:
        return text[:500]
    
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    
    # Score sentences by word frequency
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    stop_words = {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that',
                  'this', 'was', 'for', 'are', 'with', 'as', 'on', 'i', 'we', 'be',
                  'at', 'by', 'an', 'but', 'not', 'have', 'from', 'or', 'had', 'his',
                  'they', 'she', 'he', 'been', 'has', 'so', 'its', 'if', 'can', 'will'}
    filtered_words = [w for w in words if w not in stop_words]
    word_freq = Counter(filtered_words)
    
    # Score each sentence
    def score_sentence(s):
        s_words = re.findall(r'\b[a-z]{3,}\b', s.lower())
        return sum(word_freq.get(w, 0) for w in s_words) / (len(s_words) + 1)
    
    scored = sorted(enumerate(sentences), key=lambda x: score_sentence(x[1]), reverse=True)
    top_indices = sorted([i for i, _ in scored[:max_sentences]])
    
    return " ".join(sentences[i] for i in top_indices)


def summarize_chunks(chunks, model_name=None):
    """
    Summarizes multiple text chunks and joins them.
    Uses extractive summarization - works on any cloud environment.
    """
    summaries = []
    for chunk in chunks:
        if len(chunk.strip()) > 100:
            summaries.append(summarize_text(chunk, max_sentences=3))
    
    if not summaries:
        return "No substantial content found to summarize."
    
    combined = " ".join(summaries)
    # Final pass to get top highlights
    return summarize_text(combined, max_sentences=8)
