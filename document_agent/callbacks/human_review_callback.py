import json
from datetime import datetime
from typing import Optional

from google.adk.tools import ToolContext

from ..storage.database import get_connection


def _write_hitl_audit(session_id:    str,
                      user_id:       str,
                      event_type:    str,
                      input_summary: str = "",
                      output_summary:str = "") -> None:
    """Writes a Human-in-the-Loop specific audit record.

    Records HUMAN_DECISION_REQUESTED and HUMAN_DECISION events
    in the audit_trail table. These are distinct from the regular
    TOOL_CALL / TOOL_RESULT events written by audit_callback.py.

    Args:
        session_id    : ADK session identifier
        user_id       : ADK user identifier
        event_type    : HUMAN_DECISION_REQUESTED or HUMAN_DECISION
        input_summary : Data presented to the human for review
        output_summary: Human's decision (APPROVED or REJECTED)
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
                    "human_review_callback",
                    event_type,
                    datetime.utcnow().isoformat(),
                    input_summary[:300]  if input_summary  else "",
                    output_summary[:300] if output_summary else "",
                )
            )
            conn.commit()
        print(f"[hitl_callback] Audit written: {event_type}")
    except Exception as e:
        print(f"[hitl_callback] WARNING -- audit write failed: "
              f"{type(e).__name__}: {e}")

def before_save_to_db_callback(tool_context: ToolContext,
                                args: dict) -> Optional[dict]:
    """Human-in-the-Loop gate -- intercepts save_document_to_db.

    Fires as a before_tool_callback. Checks whether a human has
    approved the document save. If not approved, skips the tool
    and returns a review request to the user. If approved, allows
    the tool to execute normally.

    CRITICAL: This callback is ONLY meant for the save_document_to_db
    tool. The combined_before_tool wrapper in manager.py routes calls
    here only when tool_name == 'save_document_to_db'.

    Args:
        tool_context: ADK ToolContext with state access
        args        : Arguments being passed to save_document_to_db

    Returns:
        dict  -- if human has NOT approved yet (SKIPS the tool)
        None  -- if human HAS approved (ALLOWS the tool to execute)
    """
    print(f"[hitl_callback] before_save_to_db_callback fired")

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

    try:
        # -- Step A: Check if human has already approved ------------------
        human_approved = tool_context.state.get("human_approved", False)
        print(f"[hitl_callback] human_approved = {human_approved}")

        # -- Step B: If NOT approved -- intercept and block the save ------
        if not human_approved:
            print(f"[hitl_callback] Save blocked -- awaiting human approval")

            # Store pending data in session state for retrieval later
            tool_context.state["pending_review"] = True
            tool_context.state["pending_data"]   = json.dumps(
                _sanitize_args_for_review(args)
            )
            tool_context.state["processing_stage"] = "AWAITING_APPROVAL"

            # Build the review summary to present to the user
            review_summary = _build_review_summary(args)

            # Write audit record
            _write_hitl_audit(
                session_id    = session_id,
                user_id       = user_id,
                event_type    = "HUMAN_DECISION_REQUESTED",
                input_summary = review_summary[:300],
            )

            # RETURN A DICT -- this SKIPS the save_document_to_db tool
            # The manager receives this dict as the tool result instead
            return {
                "is_success":     False,
                "status":         "PENDING_HUMAN_APPROVAL",
                "message":        (
                    "Document has been processed and validated. "
                    "Please review the extracted data below and "
                    "type APPROVE to save to the database, "
                    "or REJECT to discard this document."
                ),
                "review_summary": review_summary,
                "doc_type":       args.get("doc_type",   "UNKNOWN"),
                "file_name":      args.get("file_name",  ""),
                "confidence":     args.get("confidence_score", 0.0),
                "instructions":   (
                    "Call process_human_decision with APPROVE or REJECT."
                ),
            }

        # -- Step C: If approved -- allow the save to proceed -------------
        print(f"[hitl_callback] Human approved -- allowing save to proceed")

        # Write audit record for the approval
        _write_hitl_audit(
            session_id     = session_id,
            user_id        = user_id,
            event_type     = "HUMAN_APPROVED",
            output_summary = "Human approved -- save_document_to_db executing",
        )

        # Reset approval state for the next document
        tool_context.state["human_approved"] = False
        tool_context.state["pending_review"] = False
        tool_context.state["pending_data"]   = None

        # RETURN None -- this ALLOWS the tool to execute normally
        return None

    except Exception as e:
        print(f"[hitl_callback] ERROR in before_save_to_db_callback: "
              f"{type(e).__name__}: {e}")
        # On unexpected error -- block the save to be safe
        return {
            "is_success": False,
            "status":     "CALLBACK_ERROR",
            "message":    (
                f"Human review callback encountered an error: {str(e)}. "
                "Please try again."
            ),
            "error": str(e),
        }

def _sanitize_args_for_review(args: dict) -> dict:
    """Prepares tool args for storage in session state as pending_data.

    Truncates large text fields so session state does not bloat.
    raw_text is truncated to 500 chars -- enough for context but
    not the full document.

    Args:
        args: The tool arguments dict from save_document_to_db

    Returns:
        dict: Sanitized copy safe for session state storage
    """
    sanitized = {}
    try:
        for k, v in (args or {}).items():
            if k == "raw_text" and isinstance(v, str) and len(v) > 500:
                sanitized[k] = v[:500] + "... [truncated for review]"
            elif k == "extracted_data" and isinstance(v, dict):
                sanitized[k] = v   # keep full extracted data for review
            else:
                sanitized[k] = v
    except Exception:
        sanitized = args or {}
    return sanitized


def _build_review_summary(args: dict) -> str:
    """Builds a human-readable summary of the document for review.

    Creates a clear, concise text summary of the key fields so the
    human reviewer can quickly assess the extraction quality.

    Args:
        args: The tool arguments dict from save_document_to_db

    Returns:
        str: Formatted review summary
    """
    try:
        lines = []
        lines.append("=" * 50)
        lines.append("DOCUMENT REVIEW SUMMARY")
        lines.append("=" * 50)
        lines.append(f"File          : {args.get('file_name',  'N/A')}")
        lines.append(f"Document Type : {args.get('doc_type',   'N/A')}")
        lines.append(f"Confidence    : {args.get('confidence_score', 0.0):.2f}")
        lines.append(f"Session ID    : {args.get('session_id', 'N/A')}")
        lines.append("-" * 50)

        extracted = args.get("extracted_data", {})
        if extracted and isinstance(extracted, dict):
            lines.append("EXTRACTED FIELDS:")
            for field, value in extracted.items():
                if value is not None:
                    val_str = str(value)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    lines.append(f"  {field:<25}: {val_str}")
                else:
                    lines.append(f"  {field:<25}: [NOT FOUND]")

        lines.append("=" * 50)
        lines.append("Type APPROVE to save or REJECT to discard.")
        return "\n".join(lines)

    except Exception as e:
        return f"Review summary unavailable: {str(e)}"