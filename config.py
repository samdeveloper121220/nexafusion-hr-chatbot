import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HR_POLICIES_DIR = "hr_policies"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_TEMPERATURE = 0.0
CONFIDENCE_THRESHOLD = 0.6