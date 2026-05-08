from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, VECTOR_STORE_PATH
import os
import streamlit as st

# Initialize embeddings (uses lighter model from config)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_vector_store():
    """Load FAISS vector store or create empty placeholder"""
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            return FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            st.info("Creating new vector store...")

    # Create empty placeholder if no index exists
    empty_docs = ["This is a placeholder document. Please ingest HR policy documents using Admin Controls."]
    metadatas = [{"source": "system", "page": 0}]
    
    vector_store = FAISS.from_texts(empty_docs, embeddings, metadatas=metadatas)
    return vector_store


def save_vector_store(vector_store):
    """Save vector store to disk"""
    try:
        vector_store.save_local(VECTOR_STORE_PATH)
        return True
    except Exception as e:
        st.error(f"Failed to save vector store: {e}")
        return False