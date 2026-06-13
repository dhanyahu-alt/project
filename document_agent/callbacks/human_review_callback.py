import json
from datetime import datetime
from typing import Optional

from google.adk.tools import ToolContext

from ..storage.database import get_connection


# Confidence threshold for auto-approval decision
CONFIDENCE_THRESHOLD = 0.90

def _write_hitl_audit(session_id:     str,
                      user_id:        str,
                      event_type:     str,
                      input_summary:  str = "",
                      output_summary: str = "") -> None:
    """Writes a Human-in-the-Loop specific audit record.

    Records HUMAN_DECISION_REQUESTED, HUMAN_APPROVED, and AUTO_DECISION
    events in the audit_trail table. These are distinct from the regular
    TOOL_CALL and TOOL_RESULT events written by audit_callback.py.

    Args:
        session_id    : ADK session identifier
        user_id       : ADK user identifier
        event_type    : HUMAN_DECISION_REQUESTED / HUMAN_APPROVED /
                        AUTO_DECISION
        input_summary : Data presented for review or timeout details
        output_summary: Decision made (APPROVED / AUTO_APPROVED /
                        AUTO_REJECTED)
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
        print(f"Audit written: {event_type}")
    except Exception as e:
        print(f"WARNING -- audit write failed: "
              f"{type(e).__name__}: {e}")

def _is_review_timed_out(tool_context: ToolContext) -> bool:
    """Checks whether the human review timeout period has expired.

    Reads review_started_at and review_timeout_mins from session state.
    Returns True if the elapsed time since the review was requested
    exceeds the configured timeout. Returns False if the timeout has
    not expired or if the review has not started yet.

    Args:
        tool_context: ADK ToolContext with session state access

    Returns:
        bool: True if timeout has expired, False otherwise
    """
    try:
        started_at   = tool_context.state.get("review_started_at",   None)
        timeout_mins = tool_context.state.get("review_timeout_mins", 60)

        if not started_at:
            print(f"review_started_at not set -- "
                  f"timeout check skipped")
            return False

        start_dt = datetime.fromisoformat(started_at)
        elapsed  = datetime.utcnow() - start_dt
        elapsed_mins = elapsed.total_seconds() / 60

        print(f"Review elapsed: {elapsed_mins:.1f} min "
              f"(timeout: {timeout_mins} min)")

        if elapsed_mins > timeout_mins:
            print(f"TIMEOUT -- review period expired after "
                  f"{elapsed_mins:.1f} minutes")
            return True

        print(f"Review still within timeout period")
        return False

    except Exception as e:
        print(f"WARNING -- timeout check error: "
              f"{type(e).__name__}: {e}")
        return False

def _handle_auto_decision(tool_context: ToolContext,
                           args:        dict,
                           session_id:  str,
                           user_id:     str) -> Optional[dict]:
    """Fires the auto-decision after the review timeout expires.

    Checks the document's confidence_score and either auto-approves
    (confidence >= 0.90) or auto-rejects (confidence < 0.90).
    Records the decision in audit_trail as AUTO_DECISION.

    This function is called only once per document -- the
    auto_decision_fired flag in session state prevents double-firing
    if save_document_to_db is called again after the auto-decision.

    Args:
        tool_context: ADK ToolContext with session state access
        args        : Arguments passed to save_document_to_db
        session_id  : ADK session identifier for audit record
        user_id     : ADK user identifier for audit record

    Returns:
        None  -- if auto-approved (allows save_document_to_db to execute)
        dict  -- if auto-rejected (blocks save_document_to_db)
    """
    print(f"_handle_auto_decision called")

    # -- Prevent double-firing -----------------------------------------------
    if tool_context.state.get("auto_decision_fired", False):
        print(f"auto_decision already fired -- skipping")
        # If previously auto-approved, human_approved is True -- allow save
        if tool_context.state.get("human_approved", False):
            return None
        # If previously auto-rejected, block the save
        return {
            "is_success": False,
            "status":     "AUTO_REJECTED",
            "message":    (
                "This document was previously auto-rejected due to review "
                "timeout and low confidence. Please re-process if needed."
            ),
        }

    # Mark as fired to prevent double-firing
    tool_context.state["auto_decision_fired"] = True

    # -- Read confidence score from args -------------------------------------
    confidence = 0.0
    try:
        confidence = float(args.get("confidence_score", 0.0))
    except (ValueError, TypeError):
        confidence = 0.0

    now = datetime.utcnow().isoformat()
    print(f"Auto-decision -- confidence: {confidence} | "
          f"threshold: {CONFIDENCE_THRESHOLD}")

    if confidence >= CONFIDENCE_THRESHOLD:
        print(f"AUTO-APPROVED -- "
              f"confidence {confidence} >= {CONFIDENCE_THRESHOLD}")

        # Update session state to reflect auto-approval
        tool_context.state["human_approved"]     = True
        tool_context.state["approved_by"]        = "auto_timeout"
        tool_context.state["approval_timestamp"] = now

        # Write AUTO_DECISION audit record
        _write_hitl_audit(
            session_id     = session_id,
            user_id        = user_id,
            event_type     = "AUTO_DECISION",
            input_summary  = (
                f"Timeout expired | "
                f"confidence: {confidence} | "
                f"threshold: {CONFIDENCE_THRESHOLD}"
            ),
            output_summary = (
                f"AUTO_APPROVED -- "
                f"confidence {confidence} >= {CONFIDENCE_THRESHOLD}"
            ),
        )

        # Return None to ALLOW save_document_to_db to execute
        return None

    else:
        print(f"AUTO-REJECTED -- "
              f"confidence {confidence} < {CONFIDENCE_THRESHOLD}")

        # Update session state to reflect auto-rejection
        tool_context.state["human_approved"]   = False
        tool_context.state["needs_correction"] = True
        tool_context.state["rejected_at"]      = now
        tool_context.state["rejected_by"]      = "auto_timeout"

        # Write AUTO_DECISION audit record
        _write_hitl_audit(
            session_id     = session_id,
            user_id        = user_id,
            event_type     = "AUTO_DECISION",
            input_summary  = (
                f"Timeout expired | "
                f"confidence: {confidence} | "
                f"threshold: {CONFIDENCE_THRESHOLD}"
            ),
            output_summary = (
                f"AUTO_REJECTED -- "
                f"confidence {confidence} < {CONFIDENCE_THRESHOLD}"
            ),
        )

        # Return a DICT to BLOCK save_document_to_db from executing
        return {
            "is_success": False,
            "status":     "AUTO_REJECTED",
            "message":    (
                f"Review timeout of 60 minutes has expired. "
                f"Document auto-rejected because confidence score "
                f"{confidence:.2f} is below the threshold "
                f"{CONFIDENCE_THRESHOLD}. "
                "Please re-process the document if needed."
            ),
            "confidence": confidence,
            "threshold":  CONFIDENCE_THRESHOLD,
        }

def before_save_to_db_callback(tool_context: ToolContext,
                                args: dict) -> Optional[dict]:
    """Human-in-the-Loop gate -- intercepts save_document_to_db.

    Fires as a before_tool_callback via combined_before_tool in manager.py.
    Checks whether a human has approved the document save. If the review
    timeout has expired, makes an automatic decision. If neither condition
    is met, blocks the save and requests human review.

    LOGIC ORDER:
      1. human_approved = True  -> HUMAN_APPROVED  -> return None (allow)
      2. timeout expired        -> AUTO_DECISION    -> return None or dict
      3. neither                -> HUMAN_DECISION_REQUESTED -> return dict

    Args:
        tool_context: ADK ToolContext with session state access
        args        : Arguments being passed to save_document_to_db

    Returns:
        None -- if approved (human or auto) -- ALLOWS tool to execute
        dict -- if blocked (pending or auto-rejected) -- SKIPS tool
    """
    print(f"before_save_to_db_callback fired")

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
        #Has the human already approved?
        human_approved = tool_context.state.get("human_approved", False)
        print(f"human_approved = {human_approved}")

        if human_approved:
            print(f"Human approved -- allowing save")

            _write_hitl_audit(
                session_id     = session_id,
                user_id        = user_id,
                event_type     = "HUMAN_APPROVED",
                output_summary = "Human approved -- save_document_to_db executing",
            )

            # Reset for next document
            tool_context.state["human_approved"] = False
            tool_context.state["pending_review"] = False
            tool_context.state["pending_data"]   = None

            # RETURN None -- ALLOWS the tool to execute
            return None

        #Has the review timeout expired?
        if _is_review_timed_out(tool_context):
            print(f"Timeout expired -- firing auto-decision")
            return _handle_auto_decision(
                tool_context = tool_context,
                args         = args,
                session_id   = session_id,
                user_id      = user_id,
            )

        print(f"Save blocked -- awaiting human approval")

        # Store pending data in state
        tool_context.state["pending_review"] = True
        tool_context.state["pending_data"]   = json.dumps(
            _sanitize_args_for_review(args)
        )
        tool_context.state["processing_stage"] = "AWAITING_APPROVAL"

        review_summary = _build_review_summary(args)

        _write_hitl_audit(
            session_id    = session_id,
            user_id       = user_id,
            event_type    = "HUMAN_DECISION_REQUESTED",
            input_summary = review_summary[:300],
        )

        # RETURN a DICT -- SKIPS the save_document_to_db tool
        return {
            "is_success":     False,
            "status":         "PENDING_HUMAN_APPROVAL",
            "message":        (
                "Document has been processed and validated. "
                "Please review the extracted data below and "
                "type APPROVE to save to the database, "
                "or REJECT to discard this document. "
                "If no response is received within 60 minutes, "
                "the system will automatically approve documents "
                "with confidence 0.90 or above and reject those below."
            ),
            "review_summary": review_summary,
            "doc_type":       args.get("doc_type",         "UNKNOWN"),
            "file_name":      args.get("file_name",        ""),
            "confidence":     args.get("confidence_score", 0.0),
            "instructions":   (
                "Call process_human_decision with APPROVE or REJECT."
            ),
        }

    except Exception as e:
        print(f"ERROR in before_save_to_db_callback: "
              f"{type(e).__name__}: {e}")
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
                sanitized[k] = v
            else:
                sanitized[k] = v
    except Exception:
        sanitized = args or {}
    return sanitized


def _build_review_summary(args: dict) -> str:
    """Builds a human-readable summary of the document for review.

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
        lines.append(
            "Note: If no response within 60 minutes, the system will "
            "auto-decide based on confidence score."
        )
        return "\n".join(lines)

    except Exception as e:
        return f"Review summary unavailable: {str(e)}"