# Configuration loading 
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000") # Added for CLI later

# Validate essential variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.") 