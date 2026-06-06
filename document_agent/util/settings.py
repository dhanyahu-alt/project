import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # points to document_agent/

MODEL_FLASH           = "gemini-2.0-flash"
MODEL_PRO             = "gemini-2.0-pro"
EMBEDDING_MODEL       = "models/text-embedding-004"

CONFIDENCE_HIGH       = 0.90
CONFIDENCE_MEDIUM     = 0.70

DB_PATH               = str(BASE_DIR / "storage" / "documents.db")
SESSION_DB_PATH       = str(BASE_DIR / "storage" / "sessions.db")
VECTOR_DB_PATH        = str(BASE_DIR / "storage" / "vector_db")
ARCHIVE_PATH          = str(BASE_DIR / "storage" / "archive")