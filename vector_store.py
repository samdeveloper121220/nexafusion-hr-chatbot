# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from config import EMBEDDING_MODEL
# import os

# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# VECTOR_STORE_PATH = "faiss_index"

# def get_vector_store():
#     if os.path.exists(VECTOR_STORE_PATH):
#         return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
#     # Initial empty store with better metadata
#     empty_docs = ["This is a placeholder document for HR policies."]
#     metadatas = [{"source": "system", "page": 0}]
#     return FAISS.from_texts(empty_docs, embeddings, metadatas=metadatas)

# def save_vector_store(vector_store):
#     vector_store.save_local(VECTOR_STORE_PATH)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL
import os

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
VECTOR_STORE_PATH = "faiss_index"

def get_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Initial empty store
    empty_docs = ["This is a placeholder document for HR policies."]
    metadatas = [{"source": "system", "page": 0}]
    return FAISS.from_texts(empty_docs, embeddings, metadatas=metadatas)

def save_vector_store(vector_store):
    vector_store.save_local(VECTOR_STORE_PATH)