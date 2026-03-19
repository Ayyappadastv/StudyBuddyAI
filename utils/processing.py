from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import os

@st.cache_resource
def get_embeddings_model(model_name="all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

def split_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Splits text into manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)

def create_vector_store(chunks):
    """
    Creates a FAISS vector store using a cached embeddings model.
    """
    embeddings = get_embeddings_model()
    return FAISS.from_texts(chunks, embeddings)

def save_vector_store(vector_store, path="faiss_index"):
    """
    Saves the FAISS vector store to disk.
    """
    vector_store.save_local(path)

def load_vector_store(path="faiss_index", model_name="all-MiniLM-L6-v2"):
    """
    Loads a FAISS vector store from disk.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
