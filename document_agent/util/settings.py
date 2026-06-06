import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

MODEL_FLASH           = "gemini-2.5-flash"
MODEL_FLASH_LITE      = "gemini-2.5-flash-lite"
EMBEDDING_MODEL       = "models/text-embedding-004"

RATE_LIMIT_RPM        = 10
RATE_LIMIT_DELAY_SEC  = 6
MAX_LOOP_ITERATIONS   = 2
CONFIDENCE_HIGH       = 0.90
CONFIDENCE_MEDIUM     = 0.70

DB_PATH               = str(BASE_DIR / "storage" / "documents.db")
SESSION_DB_PATH       = str(BASE_DIR / "storage" / "sessions.db")
VECTOR_DB_PATH        = str(BASE_DIR / "storage" / "vector_db")
ARCHIVE_PATH          = str(BASE_DIR / "storage" / "archive")