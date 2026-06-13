import json
import sqlite3
from datetime import datetime
from pathlib import Path
from google.adk.sessions import DatabaseSessionService
from ..util.settings import DB_PATH, SESSION_DB_PATH

def get_session_service() -> DatabaseSessionService:
    """Returns the ADK DatabaseSessionService for persistent session storage.

    Uses aiosqlite async driver so ADK's async session management works
    correctly. Sessions survive adk web restarts -- required for
    Human-in-the-Loop workflows where humans may respond hours later.

    Returns:
        DatabaseSessionService: Configured with sqlite+aiosqlite driver.
    """
    print(f"Initializing DatabaseSessionService -- "
          f"path: {SESSION_DB_PATH}")
    Path(SESSION_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return DatabaseSessionService(
        db_url=f"sqlite+aiosqlite:///{SESSION_DB_PATH}"
    )

def get_connection() -> sqlite3.Connection:
    """Opens and returns a sqlite3 connection to the document database.

    Creates the parent directory if it does not exist.
    Always use as a context manager:
        with get_connection() as conn:
            conn.execute(...)
            conn.commit()

    Returns:
        sqlite3.Connection: Connection to DB_PATH with row_factory set
                            so rows are returned as dicts.
    """
    print(f"Opening connection to: {DB_PATH}")
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row
    return conn
    
def init_database() -> bool:
    """Creates all database tables if they do not already exist.

    Safe to call multiple times -- uses CREATE TABLE IF NOT EXISTS.
    Called automatically at module import so tables are always ready
    before any tool function runs.

    Tables created:
        documents          -- one row per document version
        processing_log     -- one row per processing action
        audit_trail        -- one row per agent/tool/human event
        archived_documents -- one row per archived approved document

    Returns:
        bool: True if initialisation succeeded, False on error.
    """
    print(f"Initializing database at: {DB_PATH}")

    create_documents = """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id           TEXT    PRIMARY KEY,
            file_path        TEXT,
            file_name        TEXT    NOT NULL,
            doc_type         TEXT,
            raw_text         TEXT,
            extracted_data   TEXT,
            confidence_score REAL,
            status           TEXT    DEFAULT 'PENDING',
            version          INTEGER NOT NULL DEFAULT 1,
            is_latest        INTEGER NOT NULL DEFAULT 1,
            previous_doc_id  TEXT,
            processed_at     TEXT,
            approved_at      TEXT,
            approved_by      TEXT
        )
    """
    # doc_id format  : {file_name_without_ext}_v{version}
    #                  e.g. LoA1_v1, LoA1_v2, notice_1_v3
    # version        : starts at 1, increments on each re-upload
    # is_latest      : 1 = current version, 0 = older version
    # previous_doc_id: doc_id of previous version (NULL for v1)

    create_processing_log = """
        CREATE TABLE IF NOT EXISTS processing_log (
            log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            doc_id      TEXT,
            agent_name  TEXT,
            action      TEXT,
            status      TEXT,
            timestamp   TEXT,
            details     TEXT
        )
    """
    # status  : SUCCESS / FAILURE / PENDING
    # details : JSON blob of relevant context

    create_audit_trail = """
        CREATE TABLE IF NOT EXISTS audit_trail (
            audit_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     TEXT,
            user_id        TEXT,
            agent_name     TEXT,
            event_type     TEXT,
            timestamp      TEXT,
            input_summary  TEXT,
            output_summary TEXT,
            state_snapshot TEXT,
            duration_ms    INTEGER
        )
    """
    # event_type     : AGENT_START / AGENT_END / TOOL_CALL /
    #                  TOOL_RESULT / STATE_CHANGE / HUMAN_DECISION
    # state_snapshot : JSON of relevant state keys at time of event

    create_archived_documents = """
        CREATE TABLE IF NOT EXISTS archived_documents (
            archive_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id        TEXT,
            original_path TEXT,
            archive_path  TEXT,
            archived_at   TEXT,
            metadata_path TEXT
        )
    """
    # metadata_path : path to {doc_id}_metadata.json sidecar file

    try:
        with get_connection() as conn:
            conn.execute(create_documents)
            conn.execute(create_processing_log)
            conn.execute(create_audit_trail)
            conn.execute(create_archived_documents)
            conn.commit()
            print(f"Tables created/verified: documents, "
                  f"processing_log, audit_trail, archived_documents")
            return True

    except sqlite3.Error as e:
        print(f"ERROR initializing tables: "
              f"{type(e).__name__}: {e}")
        return False

def generate_doc_id(file_name: str, version: int) -> str:
    """Generates a versioned document ID from file name and version number.

    Strips the file extension and appends _v{version}.

    Args:
        file_name: Filename e.g. "LoA1.pdf"
        version  : Version number e.g. 1, 2, 3

    Returns:
        str: e.g. "LoA1_v1", "LoA1_v2", "notice_1_v3"

    Examples:
        generate_doc_id("LoA1.pdf", 1)      -> "LoA1_v1"
        generate_doc_id("LoA1.pdf", 2)      -> "LoA1_v2"
        generate_doc_id("notice_1.pdf", 1)  -> "notice_1_v1"
    """
    stem   = Path(file_name).stem   # removes extension
    doc_id = f"{stem}_v{version}"
    print(f"Generated doc_id: {doc_id} "
          f"(file: {file_name}, version: {version})")
    return doc_id


def get_next_version(file_name: str) -> tuple:
    """Checks if a document with this file_name already exists and
    returns the next version number plus the current latest doc_id.

    Args:
        file_name: Filename to check e.g. "LoA1.pdf"

    Returns:
        tuple: (next_version: int, previous_doc_id: str or None)
            next_version    = 1        if file never processed before
                            = max + 1  if file was processed before
            previous_doc_id = None     if first upload
                            = doc_id of current latest version

    Examples:
        First upload of LoA1.pdf:
            -> (1, None)
        Second upload (LoA1_v1 exists):
            -> (2, "LoA1_v1")
        Third upload (LoA1_v2 is latest):
            -> (3, "LoA1_v2")
    """
    print(f"Checking existing versions for: {file_name}")
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT doc_id, version
                FROM   documents
                WHERE  file_name = ?
                AND    is_latest  = 1
                """,
                (file_name,)
            ).fetchone()

            if row is None:
                print(f"No existing record -- "
                      f"this is a new document (version 1)")
                return 1, None
            else:
                current_version = row["version"]
                current_doc_id  = row["doc_id"]
                next_version    = current_version + 1
                print(f"Existing latest: version={current_version} "
                      f"doc_id={current_doc_id} -- "
                      f"new version will be: {next_version}")
                return next_version, current_doc_id

    except sqlite3.Error as e:
        print(f"ERROR in get_next_version: "
              f"{type(e).__name__}: {e}")
        return 1, None


def mark_previous_version_outdated(file_name: str) -> bool:
    """Sets is_latest = 0 on the current latest version of a document.

    Called before inserting a new version so only one record
    per file_name has is_latest = 1 at any time.

    Args:
        file_name: Filename whose current latest should be marked old.

    Returns:
        bool: True if update succeeded, False on error.
    """
    print(f"Marking previous version outdated for: {file_name}")
    try:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET    is_latest = 0
                WHERE  file_name = ?
                AND    is_latest  = 1
                """,
                (file_name,)
            )
            conn.commit()
            print(f"Previous version marked is_latest=0 "
                  f"for: {file_name}")
            return True

    except sqlite3.Error as e:
        print(f"ERROR marking previous version outdated: "
              f"{type(e).__name__}: {e}")
        return False


print("DB Module loaded -- running init_database() ")
init_database()