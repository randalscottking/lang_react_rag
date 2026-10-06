import os
from pathlib import Path

DOCS_DIR = Path(os.getenv("DOCS_DIR", "/data"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ragapp")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

SUPPORTED_EXTENSIONS = {".txt", ".md"}
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]
