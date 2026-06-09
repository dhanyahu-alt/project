import json
import sqlite3
from datetime import datetime

from google.adk.tools import ToolContext

from ..storage.database import get_connection

def _write_decision_audit(session_id:    str,
                           user_id:       str,
                           decision:      str,
                           doc_id:        str = "",
                           file_name:     str = "") -> None:
    """Writes a HUMAN_DECISION audit record to the audit_trail table.

    Records who made the decision, when, and what they decided.
    Called by process_human_decision on both APPROVE and REJECT.

    Args:
        session_id : ADK session identifier
        user_id    : ADK user identifier
        decision   : "APPROVED" or "REJECTED"
        doc_id     : Document ID being reviewed (from session state)
        file_name  : File name being reviewed (from session state)
    """
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_trail (
                    session_id, user_id, agent_name, event_type,
                    timestamp, input_summary, output_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    "approval_tool",
                    "HUMAN_DECISION",
                    datetime.utcnow().isoformat(),
                    f"doc_id: {doc_id} | file: {file_name}",
                    decision,
                )
            )
            conn.commit()
        print(f" Audit written: HUMAN_DECISION = {decision}")
    except sqlite3.Error as e:
        print(f" WARNING -- audit write failed: "
              f"{type(e).__name__}: {e}")
    except Exception as e:
        print(f" WARNING -- unexpected audit error: "
              f"{type(e).__name__}: {e}")


def process_human_decision(decision:     str,
                            tool_context: ToolContext) -> dict:
    """Processes the human APPROVE or REJECT decision for document saving.

    Call this tool after presenting the extracted document data to the
    user for review. Pass the user's exact response as the decision
    parameter. This tool updates the session state so the HITL callback
    in human_review_callback.py knows whether to allow or block the
    save_document_to_db tool call.

    Valid decision values:
        APPROVE -- user confirms the extracted data is correct
                   and approves saving to the database
        REJECT  -- user rejects the document and it will NOT be saved

    After calling this tool:
        On APPROVE: call save_document_to_db to complete the save.
                    The HITL callback will verify approval and allow it.
        On REJECT:  call update_processing_stage with REJECTED.
                    Inform the user the document was not saved.

    Args:
        decision    : The human's decision -- must be APPROVE or REJECT
                      (case-insensitive)
        tool_context: ADK ToolContext for session state access

    Returns:
        dict with keys:
            is_success  (bool) : True if decision was processed
            approved    (bool) : True if APPROVE, False if REJECT
            decision    (str)  : The normalised decision (APPROVE/REJECT)
            message     (str)  : Confirmation message for the user
            error       (str)  : Error message if is_success is False
    """
    print(f" process_human_decision called -- "
          f"decision: '{decision}'")

    result = {
        "is_success": False,
        "approved":   False,
        "decision":   "",
        "message":    "",
        "error":      None,
    }

    # -- Validate decision input ---------------------------------------------
    if not decision or not decision.strip():
        result["error"] = (
            "Decision cannot be empty. "
            "Please type APPROVE to save or REJECT to discard."
        )
        print(f" ERROR -- empty decision received")
        return result

    decision_upper = decision.strip().upper()

    if decision_upper not in ("APPROVE", "REJECT"):
        result["error"] = (
            f"Invalid decision: '{decision}'. "
            "Please type APPROVE to save the document "
            "or REJECT to discard it."
        )
        print(f" ERROR -- invalid decision: '{decision}'")
        return result

    # -- Read session identifiers --------------------------------------------
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

    # -- Read current doc info from state ------------------------------------
    doc_id    = tool_context.state.get("current_doc_id",  "UNKNOWN")
    file_name = ""
    try:
        file_path = tool_context.state.get("current_doc_path", "")
        if file_path:
            from pathlib import Path
            file_name = Path(file_path).name
    except Exception:
        pass

    if decision_upper == "APPROVE":
        print(f" Processing APPROVE decision for: {doc_id}")

        try:
            # Set human_approved = True in session state
            # The HITL callback reads this on the next save_document_to_db call
            tool_context.state["human_approved"]    = True
            tool_context.state["pending_review"]    = False
            tool_context.state["approval_timestamp"] = datetime.utcnow().isoformat()
            tool_context.state["approved_by"]        = user_id or "user"

            print(f" State updated: human_approved = True")

            # Write audit record
            _write_decision_audit(
                session_id = session_id,
                user_id    = user_id,
                decision   = "APPROVED",
                doc_id     = doc_id,
                file_name  = file_name,
            )

            result["is_success"] = True
            result["approved"]   = True
            result["decision"]   = "APPROVE"
            result["message"]    = (
                "Approval recorded successfully. "
                "Proceeding to save the document to the database."
            )

            print(f" APPROVED -- doc_id: {doc_id} | "
                  f"user: {user_id or 'user'}")
            return result

        except Exception as e:
            print(f" ERROR during APPROVE: "
                  f"{type(e).__name__}: {e}")
            result["error"] = f"Failed to record approval: {str(e)}"
            return result

    if decision_upper == "REJECT":
        print(f" Processing REJECT decision for: {doc_id}")

        try:
            # Set state to reflect rejection
            tool_context.state["human_approved"]   = False
            tool_context.state["pending_review"]   = False
            tool_context.state["needs_correction"] = True
            tool_context.state["rejected_at"]      = datetime.utcnow().isoformat()
            tool_context.state["rejected_by"]      = user_id or "user"

            print(f" State updated: "
                  f"human_approved=False, needs_correction=True")

            # Write audit record
            _write_decision_audit(
                session_id = session_id,
                user_id    = user_id,
                decision   = "REJECTED",
                doc_id     = doc_id,
                file_name  = file_name,
            )

            result["is_success"] = True
            result["approved"]   = False
            result["decision"]   = "REJECT"
            result["message"]    = (
                "Document rejected. "
                "The document will not be saved to the database. "
                "You can re-process the document or discard it."
            )

            print(f" REJECTED -- doc_id: {doc_id} | "
                  f"user: {user_id or 'user'}")
            return result

        except Exception as e:
            print(f" ERROR during REJECT: "
                  f"{type(e).__name__}: {e}")
            result["error"] = f"Failed to record rejection: {str(e)}"
            return result

    # Should never reach here given the validation above
    result["error"] = "Unexpected error in process_human_decision"
    return result