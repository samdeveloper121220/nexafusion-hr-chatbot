import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

HR_POLICIES_DIR = "hr_policies"
VECTOR_STORE_PATH = "faiss_index"

# Use fast & light model (Critical for Streamlit Cloud)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

LLM_TEMPERATURE = 0.1
CONFIDENCE_THRESHOLD = 0.65