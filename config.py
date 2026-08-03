import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# API Models Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# Audio Processing Settings
AUDIO_CHUNK_MINUTES = int(os.getenv("AUDIO_CHUNK_MINUTES", "1"))
SARVAM_PIECE_SECONDS = int(os.getenv("SARVAM_PIECE_SECONDS", "25"))
AUDIO_SAMPLE_RATE = 16000

# RAG & Vector Store Settings
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
