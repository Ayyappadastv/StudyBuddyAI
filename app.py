import streamlit as st
import os
import plotly.express as px
import pandas as pd
from utils.transcription import transcribe_audio
from utils.processing import split_text, create_vector_store, save_vector_store, load_vector_store
from utils.summarization import summarize_chunks
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub # Note: Using local or API? 
# For now, let's use a simple local model or just the vector store for retrieval
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Page Config & Styling ---
# --- Page Config ---
st.set_page_config(page_title="StudyBuddy AI", layout="wide", page_icon="🎓")

def local_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }
        .main-header {
            color: #1e3a8a !important;
            font-size: 3.5rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0;
            line-height: 1.2;
        }
        .brand-text {
            color: #3b82f6;
            font-weight: 700;
        }
        .welcome-card {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            text-align: center;
            border: 1px solid #e2e8f0;
            margin-top: 20px;
        }
        .summary-card {
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-left: 5px solid #3b82f6;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- App logic ---
st.markdown("<h1 class='main-header'>🎓 StudyBuddy AI</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Cached functions for speed ---
@st.cache_data(show_spinner=False)
def get_transcription(audio_path):
    return transcribe_audio(audio_path)

@st.cache_data(show_spinner=False)
def get_summary(text_chunks):
    return summarize_chunks(text_chunks[:5])

@st.cache_resource(show_spinner=False)
def get_vector_store(text_chunks):
    return create_vector_store(text_chunks)

# --- App logic ---

# Sidebar
with st.sidebar:
    st.header("Upload Lecture")
    audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"], help="Support for MP3, WAV, M4A, OGG")
    
    st.divider()
    st.header("Getting Started")
    st.info("1. Upload audio\n2. Wait for AI\n3. Start learning!")
    st.divider()
    st.markdown("---")
    st.markdown("Created with ❤️ by **Ayyappadas TV**")

if not audio_file:
    st.markdown("""
        <div class="welcome-card">
            <h1 style="color: #1e3a8a; margin-bottom: 15px;">🚀 Ready to Master Your Lectures?</h1>
            <p style="font-size: 1.2rem; color: #475569; max-width: 600px; margin: 0 auto 30px;">
                StudyBuddy AI turns your audio recordings into smart notes, flashcards, and instant answers.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div style="background: #eff6ff; padding: 25px; border-radius: 15px; border: 1px solid #dbeafe;">
                    <h3 style="color: #2563eb; margin: 0 0 10px;">📝 Lecture Notes</h3>
                    <p style="font-size: 0.9rem; color: #1e40af; margin: 0;">Get high-quality summaries of long lectures instantly.</p>
                </div>
                <div style="background: #fef2f2; padding: 25px; border-radius: 15px; border: 1px solid #fee2e2;">
                    <h3 style="color: #dc2626; margin: 0 0 10px;">❓ Ask AI</h3>
                    <p style="font-size: 0.9rem; color: #991b1b; margin: 0;">Ask anything about the professor's lecture and get answers.</p>
                </div>
                <div style="background: #f0fdf4; padding: 25px; border-radius: 15px; border: 1px solid #dcfce7;">
                    <h3 style="color: #16a34a; margin: 0 0 10px;">📊 Smart Insights</h3>
                    <p style="font-size: 0.9rem; color: #166534; margin: 0;">Visualize the most important keywords and topics.</p>
                </div>
                <div style="background: #fefce8; padding: 25px; border-radius: 15px; border: 1px solid #fef9c3;">
                    <h3 style="color: #ca8a04; margin: 0 0 10px;">🗂️ Flashcards</h3>
                    <p style="font-size: 0.9rem; color: #854d0e; margin: 0;">Automatically generated cards for active recall study.</p>
                </div>
            </div>
            <div style="margin-top: 40px; padding: 20px; background: #f8fafc; border-radius: 10px;">
                <p style="font-weight: 700; color: #1e3a8a; margin: 0;">👈 To begin, please upload an audio file in the sidebar!</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "❓ Q&A", "📊 Insights", "🗂️ Flashcards"])

if audio_file:
    # Save file temporarily
    temp_path = f"temp_{audio_file.name}"
    if not os.path.exists(temp_path):
        with open(temp_path, "wb") as f:
            f.write(audio_file.getbuffer())

    # --- Step 1: Transcription ---
    if 'transcription' not in st.session_state:
        with st.spinner("Transcribing audio with Whisper Tiny (Super Fast)..."):
            st.session_state.transcription = get_transcription(temp_path)
    
    full_text = st.session_state.transcription

    # --- Step 2: Chunking & Indexing ---
    if 'vector_store' not in st.session_state:
        with st.spinner("Processing text and building index..."):
            chunks = split_text(full_text)
            st.session_state.chunks = chunks
            st.session_state.vector_store = get_vector_store(chunks)
            st.session_state.summary = get_summary(chunks)

    with tab1:
        st.markdown("### 📝 Lecture Notes")
        st.markdown(f"<div class='summary-card'>{st.session_state.summary}</div>", unsafe_allow_html=True)
        
        with st.expander("🔍 View Full Transcription"):
            st.write(full_text)

    with tab2:
        st.markdown("### ❓ Lecture Q&A")
        query = st.text_input("Ask a question about the lecture:", placeholder="e.g., 'What are the main points?'")
        
        if query:
            with st.spinner("Finding answers..."):
                docs = st.session_state.vector_store.similarity_search(query, k=3)
                for i, doc in enumerate(docs):
                    st.info(f"**Source {i+1}:**\n{doc.page_content}")

    with tab3:
        st.markdown("### 📊 Analytics")
        # Real word frequency analysis
        words = full_text.lower().split()
        stop_words = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'this', 'was', 'for', 'are', 'with', 'as', 'on', 'i', 'we'])
        filtered_words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 3]
        
        if filtered_words:
            word_counts = pd.Series(filtered_words).value_counts().head(10).reset_index()
            word_counts.columns = ['Keyword', 'Occurrences']
            
            fig = px.bar(word_counts, x="Keyword", y="Occurrences", 
                         color="Occurrences", 
                         color_continuous_scale="Blues",
                         template="plotly_white")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 🚀 Resources")
            for keyword in word_counts['Keyword'].head(5):
                st.markdown(f"- [{keyword.capitalize()} on Wikipedia](https://en.wikipedia.org/wiki/{keyword})")
                st.markdown(f"- [{keyword.capitalize()} on YouTube](https://www.youtube.com/results?search_query={keyword}+lecture)")
        else:
            st.info("Not enough data for visualization yet.")

    with tab4:
        st.markdown("### 🗂️ Flashcards")
        if 'flashcards' not in st.session_state:
            # Generate simple flashcards
            flashcards = []
            for chunk in st.session_state.chunks[:5]:
                sentences = [s.strip() for s in chunk.split('.') if len(s.strip()) > 30]
                if len(sentences) >= 2:
                    flashcards.append({
                        "question": sentences[0] + "?",
                        "answer": sentences[1] + "."
                    })
            st.session_state.flashcards = flashcards

        if st.session_state.flashcards:
            for i, card in enumerate(st.session_state.flashcards):
                with st.expander(f"Question {i+1}"):
                    st.write(card['question'])
                    st.markdown("---")
                    st.success(f"**Answer:** {card['answer']}")
        else:
            st.info("Not enough content for flashcards.")

# No else block needed here as Welcome Card handles empty state
st.sidebar.markdown("---")
st.sidebar.markdown("Created by **Ayyappadas TV**")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #4b6584; font-size: 0.8rem;'>StudyBuddy AI - A project by <span class='brand-text'>Ayyappadas TV</span></div>", unsafe_allow_html=True)
