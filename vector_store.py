from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, VECTOR_STORE_PATH
import os
import streamlit as st

# Use fast model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            return FAISS.load_local(
                VECTOR_STORE_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"Loading vector store failed: {e}. Creating new...")

    # Placeholder
    empty_docs = ["Placeholder for HR policies. Please re-ingest documents."]
    metadatas = [{"source": "system", "page": 0}]
    return FAISS.from_texts(empty_docs, embeddings, metadatas=metadatas)


def save_vector_store(vector_store):
    try:
        vector_store.save_local(VECTOR_STORE_PATH)
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False