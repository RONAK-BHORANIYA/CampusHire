import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DOCS_DIR = ROOT / "data" / "docs"            # drop new PDFs / .md / .txt here
COMPANIES_FILE = ROOT / "data" / "companies.json"
INDEX_DIR = ROOT / "data" / "index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 150
CANDIDATES_PER_RETRIEVER = 20
RRF_K = 60
TOP_K_FINAL = 5
MIN_RERANK_SCORE = -4.0  # below this, chunks are treated as irrelevant

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
