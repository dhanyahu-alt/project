import json
import sqlite3
from datetime import datetime

from ..storage.database import (
    get_connection,
    generate_doc_id,
    get_next_version,
    mark_previous_version_outdated,
)

def save_document_to_db(document_data: dict) -> dict:
    """Saves a processed and validated document record to the database.

    Handles both new documents and re-uploads automatically.
    For new documents: creates version 1.
    For re-uploads: increments version, marks old version as not latest.
    For Human-in-the-Loop gate to be added.

    Args:
        document_data: Dict containing at minimum:
            file_path        (str)  : absolute path to the source file
            file_name        (str)  : filename e.g. "LoA1.pdf"
            doc_type         (str)  : "LOA" / "NOTICE" / "BUSINESS" / "UNKNOWN"
            raw_text         (str)  : full extracted text
            extracted_data   (dict) : typed extraction result as dict
            confidence_score (float): extraction confidence 0.0 to 1.0
            session_id       (str)  : ADK session ID

    Returns:
        dict with keys:
            is_success      (bool) : True if save succeeded
            doc_id          (str)  : generated versioned ID e.g. "LoA1_v2"
            version         (int)  : version number assigned
            is_reupload     (bool) : True if this file was processed before
            previous_doc_id (str)  : doc_id of previous version or None
            error           (str)  : error message if is_success is False
    """
    print(f"save_document_to_db called")

    result = {
        "is_success":      False,
        "doc_id":          None,
        "version":         None,
        "is_reupload":     False,
        "previous_doc_id": None,
        "error":           None,
    }

    # Validate required fields 
    file_name = document_data.get("file_name")
    file_path = document_data.get("file_path")

    if not file_name:
        result["error"] = "document_data missing required field: file_name"
        print(f"ERROR -- {result['error']}")
        return result

    if not file_path:
        result["error"] = "document_data missing required field: file_path"
        print(f"ERROR -- {result['error']}")
        return result

    print(f"Processing save for file: {file_name}")

    try: 
        next_version, previous_doc_id = get_next_version(file_name)
        is_reupload = previous_doc_id is not None

        print(f"Version: {next_version} | "
              f"Is reupload: {is_reupload} | "
              f"Previous doc_id: {previous_doc_id}")

        doc_id = generate_doc_id(file_name, next_version)
        print(f"Generated doc_id: {doc_id}")

        if is_reupload:
            print(f"Re-upload detected -- "
                  f"marking previous version outdated ...")
            mark_previous_version_outdated(file_name)

        now = datetime.utcnow().isoformat()

        extracted_data_json = json.dumps(
            document_data.get("extracted_data", {})
        )

        print(f" Inserting document record into database")

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, file_path, file_name, doc_type,
                    raw_text, extracted_data, confidence_score,
                    status, version, is_latest, previous_doc_id,
                    processed_at
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?
                )
                """,
                (
                    doc_id,
                    file_path,
                    file_name,
                    document_data.get("doc_type",         "UNKNOWN"),
                    document_data.get("raw_text",         ""),
                    extracted_data_json,
                    document_data.get("confidence_score", 0.0),
                    "PENDING",
                    next_version,
                    1,                  # is_latest = 1 for new record
                    previous_doc_id,
                    now,
                )
            )
            conn.commit()

        print(f"Document record saved successfully")

        print(f"Insert processing log entry")

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO processing_log (
                    session_id, doc_id, agent_name,
                    action, status, timestamp, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_data.get("session_id", ""),
                    doc_id,
                    "document_processing_manager",
                    "SAVE_DOCUMENT",
                    "SUCCESS",
                    now,
                    json.dumps({
                        "version":     next_version,
                        "is_reupload": is_reupload,
                        "doc_type":    document_data.get("doc_type", "UNKNOWN"),
                    })
                )
            )
            conn.commit()

        print(f"Processing log entry saved")
        result["is_success"]      = True
        result["doc_id"]          = doc_id
        result["version"]         = next_version
        result["is_reupload"]     = is_reupload
        result["previous_doc_id"] = previous_doc_id

        print(f"save_document_to_db complete"
              f"doc_id: {doc_id} | version: {next_version} | "
              f"reupload: {is_reupload}")
        return result

    except sqlite3.Error as e:
        print(f"ERROR -- SQLite error: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Database error: {str(e)}"
        return result

    except Exception as e:
        print(f"ERROR -- Unexpected: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Unexpected error: {str(e)}"
        return result

# QUERY DOCUMENTS
def query_documents(query: str, doc_type: str = None) -> dict:
    """Queries the document database for processed records.

    Searches file_name and doc_type fields. Returns only latest
    versions by default. Optionally filters by document type.

    Args:
        query   : Search keyword to match against file_name or doc_type
        doc_type: Optional filter e.g. "LOA" / "NOTICE" / "BUSINESS"

    Returns:
        dict with keys:
            is_success (bool)       : True if query succeeded
            results    (List[dict]) : matching document records
            count      (int)        : number of records returned
            error      (str)        : error message if is_success is False
    """
    print(f"[database_tool] query_documents called -- "
          f"query: '{query}' | doc_type filter: {doc_type}")

    result = {
        "is_success": False,
        "results":    [],
        "count":      0,
        "error":      None,
    }

    try:
        with get_connection() as conn:
            if doc_type:
                rows = conn.execute(
                    """
                    SELECT doc_id, file_name, doc_type, status,
                           version, is_latest, confidence_score,
                           processed_at, approved_at, approved_by
                    FROM   documents
                    WHERE  is_latest = 1
                    AND    doc_type  = ?
                    AND    (file_name LIKE ? OR doc_type LIKE ?)
                    ORDER  BY processed_at DESC
                    """,
                    (doc_type, f"%{query}%", f"%{query}%")
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT doc_id, file_name, doc_type, status,
                           version, is_latest, confidence_score,
                           processed_at, approved_at, approved_by
                    FROM   documents
                    WHERE  is_latest = 1
                    AND    (file_name LIKE ? OR doc_type LIKE ?)
                    ORDER  BY processed_at DESC
                    """,
                    (f"%{query}%", f"%{query}%")
                ).fetchall()

        records = [dict(row) for row in rows]

        result["is_success"] = True
        result["results"]    = records
        result["count"]      = len(records)

        print(f"[database_tool] query_documents -- {len(records)} records found")
        return result

    except sqlite3.Error as e:
        print(f" ERROR in query_documents: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Database error: {str(e)}"
        return result

def get_document_by_id(doc_id: str) -> dict:
    """Retrieves a specific document record by its ID.

    Fetches the full record including raw_text and extracted_data
    for the exact doc_id provided e.g. "LoA1_v2".

    Args:
        doc_id: Versioned document ID e.g. "LoA1_v1", "LoA1_v2"

    Returns:
        dict with keys:
            is_success (bool) : True if record found
            document   (dict) : full document record
            error      (str)  : error message if is_success is False
    """
    print(f"get_document_by_id called -- doc_id: {doc_id}")

    result = {
        "is_success": False,
        "document":   None,
        "error":      None,
    }

    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM   documents
                WHERE  doc_id = ?
                """,
                (doc_id,)
            ).fetchone()

        if row is None:
            result["error"] = f"No document found with doc_id: {doc_id}"
            print(f" No record found for doc_id: {doc_id}")
            return result

        record = dict(row)

        # Parse extracted_data JSON back to dict
        if record.get("extracted_data"):
            try:
                record["extracted_data"] = json.loads(record["extracted_data"])
            except json.JSONDecodeError:
                pass    # leave as string if JSON parse fails

        result["is_success"] = True
        result["document"]   = record

        print(f"Found doc_id: {doc_id} | "
              f"version: {record.get('version')} | "
              f"status: {record.get('status')}")
        return result

    except sqlite3.Error as e:
        print(f"ERROR in get_document_by_id: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Database error: {str(e)}"
        return result

def get_document_versions(file_name: str) -> dict:
    """Retrieves all versions of a document by its file name.

    Returns all historical versions sorted oldest to newest.
    Useful for reviewing the full processing history of a document.

    Args:
        file_name: Filename e.g. "LoA1.pdf"

    Returns:
        dict with keys:
            is_success     (bool)       : True if query succeeded
            file_name      (str)        : the file_name queried
            total_versions (int)        : total number of versions
            versions       (List[dict]) : all versions oldest first
            error          (str)        : error message if is_success is False
    """
    print(f" get_document_versions called -- "
          f"file_name: {file_name}")

    result = {
        "is_success":     False,
        "file_name":      file_name,
        "total_versions": 0,
        "versions":       [],
        "error":          None,
    }

    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, file_name, doc_type, status,
                       version, is_latest, confidence_score,
                       processed_at, approved_at, approved_by,
                       previous_doc_id
                FROM   documents
                WHERE  file_name = ?
                ORDER  BY version ASC
                """,
                (file_name,)
            ).fetchall()

        versions = [dict(row) for row in rows]

        result["is_success"]     = True
        result["total_versions"] = len(versions)
        result["versions"]       = versions

        print(f"Found {len(versions)} version(s) "
              f"for file: {file_name}")
        for v in versions:
            print(f"  -> v{v['version']} | doc_id: {v['doc_id']} | "
                  f"is_latest: {v['is_latest']} | "
                  f"status: {v['status']}")
        return result

    except sqlite3.Error as e:
        print(f"ERROR in get_document_versions: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Database error: {str(e)}"
        return result

def get_latest_document(file_name: str) -> dict:
    """Retrieves the latest version of a document by file name.

    Returns only the record with is_latest = 1 for this file_name.
    Used by the manager agent to check if a document was previously
    processed before deciding the version number.

    Args:
        file_name: Filename e.g. "LoA1.pdf"

    Returns:
        dict with keys:
            is_success (bool) : True if query succeeded
            found      (bool) : True if a record exists for this file_name
            document   (dict) : latest version record or None if not found
            version    (int)  : current latest version number or 0
            error      (str)  : error message if is_success is False
    """
    print(f"get_latest_document called -- "
          f"file_name: {file_name}")

    result = {
        "is_success": False,
        "found":      False,
        "document":   None,
        "version":    0,
        "error":      None,
    }

    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM   documents
                WHERE  file_name = ?
                AND    is_latest  = 1
                """,
                (file_name,)
            ).fetchone()

        result["is_success"] = True

        if row is None:
            result["found"]   = False
            result["version"] = 0
            print(f" No existing record for: {file_name} "
                  f"-- this is a new document")
        else:
            record = dict(row)
            if record.get("extracted_data"):
                try:
                    record["extracted_data"] = json.loads(
                        record["extracted_data"]
                    )
                except json.JSONDecodeError:
                    pass

            result["found"]    = True
            result["document"] = record
            result["version"]  = record.get("version", 0)
            print(f"Latest version found: "
                  f"v{result['version']} | "
                  f"doc_id: {record.get('doc_id')} | "
                  f"status: {record.get('status')}")

        return result

    except sqlite3.Error as e:
        print(f" ERROR in get_latest_document: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Database error: {str(e)}"
        return result

def get_processing_history(session_id: str) -> dict:
    """Retrieves the processing history for a session.

    Returns all processing_log entries for the given session_id
    sorted chronologically. Useful for debugging and audit reviews.

    Args:
        session_id: ADK session ID to retrieve history for

    Returns:
        dict with keys:
            is_success (bool)       : True if query succeeded
            session_id (str)        : the session_id queried
            history    (List[dict]) : log entries oldest first
            count      (int)        : number of entries
            error      (str)        : error message if is_success is False
    """
    print(f"get_processing_history called -- "
          f"session_id: {session_id}")

    result = {
        "is_success": False,
        "session_id": session_id,
        "history":    [],
        "count":      0,
        "error":      None,
    }

    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM   processing_log
                WHERE  session_id = ?
                ORDER  BY timestamp ASC
                """,
                (session_id,)
            ).fetchall()

        history = []
        for row in rows:
            entry = dict(row)
            # Parse details JSON back to dict
            if entry.get("details"):
                try:
                    entry["details"] = json.loads(entry["details"])
                except json.JSONDecodeError:
                    pass
            history.append(entry)

        result["is_success"] = True
        result["history"]    = history
        result["count"]      = len(history)

        print(f"Processing history -- "
              f"{len(history)} entries for session: {session_id}")
        return result

    except sqlite3.Error as e:
        print(f" ERROR in get_processing_history: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Database error: {str(e)}"
        return result