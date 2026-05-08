import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

HR_POLICIES_DIR = "hr_policies"
VECTOR_STORE_PATH = "faiss_index"

# Fast & Lightweight Model (Best for Cloud)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

LLM_TEMPERATURE = 0.1
CONFIDENCE_THRESHOLD = 0.65