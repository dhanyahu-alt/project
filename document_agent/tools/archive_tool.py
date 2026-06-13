import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from google.adk.tools import ToolContext

from ..storage.database import get_connection
from ..util.settings import ARCHIVE_PATH


def archive_document(doc_id:       str,
                     file_path:    str,
                     metadata:     dict,
                     tool_context: ToolContext) -> dict:
    """Archives an approved document with its metadata to the archive store.

    Copies the original document file to the archive folder and writes
    a metadata JSON sidecar file. Updates the archived_documents table
    and sets the document status to ARCHIVED in the documents table.
    Call this tool after save_document_to_db succeeds on the APPROVE path.

    Args:
        doc_id      : Versioned document ID e.g. LoA1_v1, LoA1_v2
        file_path   : Absolute path to the original document file
        metadata    : Dict containing document metadata to write to sidecar.
                      Should include: doc_type, confidence_score,
                      extracted_data, session_id. Additional fields
                      from session state are added automatically.
        tool_context: ADK ToolContext for session state access

    Returns:
        dict with keys:
            is_success    (bool) : True if archiving succeeded
            doc_id        (str)  : The doc_id archived
            archive_path  (str)  : Path to the archived document file
            metadata_path (str)  : Path to the metadata JSON sidecar file
            archive_dir   (str)  : Path to the archive directory
            error         (str)  : Error message if is_success is False
    """
    print(f"archive_document called -- doc_id: {doc_id}")

    result = {
        "is_success":    False,
        "doc_id":        doc_id,
        "archive_path":  "",
        "metadata_path": "",
        "archive_dir":   "",
        "error":         None,
    }

    # -- Validate inputs --
    if not doc_id:
        result["error"] = "doc_id is required"
        print(f"ERROR -- {result['error']}")
        return result

    if not file_path:
        result["error"] = "file_path is required"
        print(f"ERROR -- {result['error']}")
        return result

    source_path = Path(file_path)
    if not source_path.exists():
        result["error"] = f"Source file not found: {file_path}"
        print(f"ERROR -- {result['error']}")
        return result

    if not source_path.is_file():
        result["error"] = f"Source path is not a file: {file_path}"
        print(f"ERROR -- {result['error']}")
        return result

    try:
        now      = datetime.utcnow()
        year     = now.strftime("%Y")
        month    = now.strftime("%m")

        # -- Step A: Build archive directory path 
        archive_dir = Path(ARCHIVE_PATH) / year / month / doc_id
        archive_dir.mkdir(parents=True, exist_ok=True)

        print(f"Archive directory: {archive_dir}")

        # -- Step B: Copy original file to archive 
        dest_file_path = archive_dir / source_path.name
        shutil.copy2(source_path, dest_file_path)

        print(f"File copied: "
              f"{source_path.name} -> {dest_file_path}")

        # -- Step C: Build full metadata for sidecar file 
        session_id = ""
        user_id    = ""
        try:
            session_id = str(tool_context.session.id)
        except Exception:
            pass
        try:
            user_id = str(tool_context.session.user_id)
        except Exception:
            pass

        # Enrich metadata with session state values
        full_metadata = dict(metadata or {})
        full_metadata.update({
            "doc_id":             doc_id,
            "original_file_path": str(source_path),
            "archive_file_path":  str(dest_file_path),
            "archived_at":        now.isoformat(),
            "session_id":         session_id,
            "approved_by":        tool_context.state.get(
                                      "approved_by", user_id or "user"
                                  ),
            "approval_timestamp": tool_context.state.get(
                                      "approval_timestamp",
                                      now.isoformat()
                                  ),
            "archive_year":       year,
            "archive_month":      month,
        })

        metadata_file_name = f"{doc_id}_metadata.json"
        metadata_path      = archive_dir / metadata_file_name

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, indent=2, default=str)

        print(f"Metadata written: {metadata_file_name}")

        # -- Step E: Insert into archived_documents table
        archived_at = now.isoformat()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO archived_documents (
                    doc_id, original_path, archive_path,
                    archived_at, metadata_path
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    str(source_path),
                    str(dest_file_path),
                    archived_at,
                    str(metadata_path),
                )
            )
            conn.commit()

        print(f"archived_documents table updated")

        # -- Step F: Update documents table status to ARCHIVED 
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET    status      = 'ARCHIVED',
                       approved_at = ?,
                       approved_by = ?
                WHERE  doc_id      = ?
                """,
                (
                    archived_at,
                    tool_context.state.get("approved_by", user_id or "user"),
                    doc_id,
                )
            )
            conn.commit()

        print(f"documents table: status set to ARCHIVED "
              f"for doc_id: {doc_id}")

        # -- Populate result 
        result["is_success"]    = True
        result["archive_path"]  = str(dest_file_path)
        result["metadata_path"] = str(metadata_path)
        result["archive_dir"]   = str(archive_dir)

        print(f"Archive complete -- "
              f"doc_id: {doc_id} | "
              f"archive: {archive_dir}")
        return result

    except shutil.Error as e:
        print(f"ERROR -- File copy failed: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"File copy error: {str(e)}"
        return result

    except sqlite3.Error as e:
        print(f"ERROR -- Database error: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Database error: {str(e)}"
        return result

    except Exception as e:
        print(f"ERROR -- Unexpected: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Unexpected error: {str(e)}"
        return result