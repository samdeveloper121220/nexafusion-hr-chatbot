import os
from dotenv import load_dotenv

load_dotenv()

# ====================== API KEYS ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# ====================== PATHS ======================
HR_POLICIES_DIR = "hr_policies"
VECTOR_STORE_PATH = "faiss_index"

# ====================== EMBEDDING SETTINGS ======================
# Changed to lighter & faster model for Streamlit Cloud
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Much faster than all-mpnet-base-v2

# ====================== CHUNKING SETTINGS ======================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ====================== LLM SETTINGS ======================
LLM_TEMPERATURE = 0.1
LLM_MODEL = "gemini-1.5-flash"

# ====================== RAG SETTINGS ======================
CONFIDENCE_THRESHOLD = 0.65   # Adjust between 0.5 - 0.75

# ====================== APP SETTINGS ======================
MAX_RETRIEVED_DOCS = 6