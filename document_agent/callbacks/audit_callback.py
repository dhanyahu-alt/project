import json
import sqlite3
from datetime import datetime
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext

from ..storage.database import get_connection


def _write_audit_record(
    session_id:     str,
    user_id:        str,
    agent_name:     str,
    event_type:     str,
    input_summary:  str  = "",
    output_summary: str  = "",
    state_snapshot: str  = "",
    duration_ms:    int  = 0,
) -> None:
    """Writes a single record to the audit_trail table.

    Internal helper used by all four callback functions.
    Silently swallows exceptions so audit failures never crash
    the main processing pipeline.

    Args:
        session_id     : ADK session identifier
        user_id        : ADK user identifier
        agent_name     : Name of the agent or tool being audited
        event_type     : AGENT_START / AGENT_END / TOOL_CALL / TOOL_RESULT
        input_summary  : First 300 chars of input (sanitized)
        output_summary : First 300 chars of output (sanitized)
        state_snapshot : JSON string of relevant state keys
        duration_ms    : Duration in milliseconds (AGENT_END only)
    """
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_trail (
                    session_id, user_id, agent_name, event_type,
                    timestamp, input_summary, output_summary,
                    state_snapshot, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    agent_name,
                    event_type,
                    datetime.utcnow().isoformat(),
                    input_summary[:300]  if input_summary  else "",
                    output_summary[:300] if output_summary else "",
                    state_snapshot,
                    duration_ms,
                )
            )
            conn.commit()
    except sqlite3.Error as e:
        print(f"WARNING -- audit write failed: "
              f"{type(e).__name__}: {e}")
    except Exception as e:
        print(f"WARNING -- unexpected audit error: "
              f"{type(e).__name__}: {e}")


def _get_state_snapshot(context) -> str:
    """Extracts relevant state keys and returns as a JSON string.

    Captures the current values of the key processing state keys
    for the audit record. Returns empty JSON if state is unavailable.

    Args:
        context: CallbackContext or ToolContext with access to state

    Returns:
        str: JSON string of relevant state keys
    """
    try:
        state = context.state if hasattr(context, 'state') else {}
        snapshot = {
            "processing_stage":    state.get("processing_stage",    "NOT_SET"),
            "current_doc_path":    state.get("current_doc_path",    "NOT_SET"),
            "current_doc_id":      state.get("current_doc_id",      "NOT_SET"),
            "current_doc_version": state.get("current_doc_version", "NOT_SET"),
            "human_approved":      state.get("human_approved",       False),
            "is_reupload":         state.get("is_reupload",          False),
        }
        return json.dumps(snapshot)
    except Exception:
        return "{}"


def _safe_str(value, max_len: int = 300) -> str:
    """Safely converts any value to a truncated string for audit logging.

    Args:
        value  : Any value to convert
        max_len: Maximum length of the returned string

    Returns:
        str: String representation truncated to max_len
    """
    try:
        if isinstance(value, str):
            return value[:max_len]
        return str(value)[:max_len]
    except Exception:
        return ""

def before_agent_audit(callback_context: CallbackContext) -> None:
    """Audit callback that fires before any agent starts processing.

    Records an AGENT_START event in the audit_trail table and stores
    the current timestamp in session state so after_agent_audit can
    calculate the total duration.

    Attach to any LlmAgent as:
        before_agent_callback = before_agent_audit

    Args:
        callback_context: ADK CallbackContext with agent info and state

    Returns:
        None -- always. Returning anything else would skip the agent.
    """
    try:
        agent_name = getattr(callback_context, 'agent_name', 'unknown')
        session_id = ""
        user_id    = ""

        try:
            session_id = str(callback_context.session.id)
        except Exception:
            pass
        try:
            user_id = str(callback_context.session.user_id)
        except Exception:
            pass

        input_summary = ""
        try:
            messages = callback_context.session.events
            if messages:
                last = messages[-1]
                content = getattr(last, 'content', None)
                if content:
                    input_summary = _safe_str(str(content))
        except Exception:
            pass

        # Capture state snapshot
        state_snapshot = _get_state_snapshot(callback_context)

        # Store start time in state for duration calculation
        start_time_key = f"temp:agent_start_time_{agent_name}"
        try:
            callback_context.state[start_time_key] = datetime.utcnow().isoformat()
        except Exception:
            pass

        print(f"AGENT_START -- {agent_name} | "
              f"session: {session_id[:8] if session_id else 'N/A'}...")

        _write_audit_record(
            session_id     = session_id,
            user_id        = user_id,
            agent_name     = agent_name,
            event_type     = "AGENT_START",
            input_summary  = input_summary,
            state_snapshot = state_snapshot,
        )

    except Exception as e:
        print(f"WARNING -- before_agent_audit error: "
              f"{type(e).__name__}: {e}")

    return None

def after_agent_audit(callback_context: CallbackContext) -> None:
    """Audit callback that fires after any agent completes processing.

    Records an AGENT_END event with duration and output summary
    in the audit_trail table.

    Attach to any LlmAgent as:
        after_agent_callback = after_agent_audit

    Args:
        callback_context: ADK CallbackContext with agent info and state

    Returns:
        None -- always. Returning anything else would replace agent output.
    """
    try:
        agent_name = getattr(callback_context, 'agent_name', 'unknown')
        session_id = ""
        user_id    = ""

        try:
            session_id = str(callback_context.session.id)
        except Exception:
            pass
        try:
            user_id = str(callback_context.session.user_id)
        except Exception:
            pass

        # Calculate duration from stored start time
        duration_ms = 0
        start_time_key = f"temp:agent_start_time_{agent_name}"
        try:
            start_str = callback_context.state.get(start_time_key)
            if start_str:
                start_dt = datetime.fromisoformat(start_str)
                end_dt   = datetime.utcnow()
                duration_ms = int(
                    (end_dt - start_dt).total_seconds() * 1000
                )
        except Exception:
            pass

        # Build output summary from agent response
        output_summary = ""
        try:
            messages = callback_context.session.events
            if messages:
                last = messages[-1]
                content = getattr(last, 'content', None)
                if content:
                    output_summary = _safe_str(str(content))
        except Exception:
            pass

        state_snapshot = _get_state_snapshot(callback_context)

        print(f"AGENT_END -- {agent_name} | "
              f"duration: {duration_ms}ms")

        _write_audit_record(
            session_id     = session_id,
            user_id        = user_id,
            agent_name     = agent_name,
            event_type     = "AGENT_END",
            output_summary = output_summary,
            state_snapshot = state_snapshot,
            duration_ms    = duration_ms,
        )

    except Exception as e:
        print(f"WARNING -- after_agent_audit error: "
              f"{type(e).__name__}: {e}")

    return None

def before_tool_audit(tool_context: ToolContext, args: dict) -> None:
    """Audit callback that fires before any tool is called.

    Records a TOOL_CALL event with the tool name and sanitized
    arguments in the audit_trail table.

    NOTE: This callback is for audit only. The Human-in-the-Loop
    gate for save_document_to_db is in a SEPARATE callback in
    human_review_callback.py. Both callbacks can be combined using
    a wrapper function -- see manager.py Step 5 update.

    Attach to any LlmAgent as:
        before_tool_callback = before_tool_audit

    Args:
        tool_context: ADK ToolContext with tool name and state access
        args        : Dict of arguments being passed to the tool

    Returns:
        None -- always. Returning a dict would SKIP the tool.
    """
    try:
        tool_name  = getattr(tool_context, 'tool_name', 'unknown_tool')
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

        safe_args = {}
        try:
            for k, v in (args or {}).items():
                if k == "raw_text" and isinstance(v, str) and len(v) > 200:
                    safe_args[k] = v[:200] + "... [truncated]"
                elif k == "image_bytes":
                    safe_args[k] = f"<bytes: {len(v)} bytes>" if v else None
                else:
                    safe_args[k] = v
        except Exception:
            safe_args = {}

        input_summary  = json.dumps(safe_args)[:300]
        state_snapshot = _get_state_snapshot(tool_context)

        print(f"TOOL_CALL -- {tool_name}")

        _write_audit_record(
            session_id     = session_id,
            user_id        = user_id,
            agent_name     = tool_name,
            event_type     = "TOOL_CALL",
            input_summary  = input_summary,
            state_snapshot = state_snapshot,
        )

    except Exception as e:
        print(f"WARNING -- before_tool_audit error: "
              f"{type(e).__name__}: {e}")

    return None


def after_tool_audit(tool_context: ToolContext,
                     result: dict) -> None:
    """Audit callback that fires after any tool returns its result.

    Records a TOOL_RESULT event with the tool name and sanitized
    result in the audit_trail table.

    Attach to any LlmAgent as:
        after_tool_callback = after_tool_audit

    Args:
        tool_context: ADK ToolContext with tool name and state access
        result      : Dict returned by the tool

    Returns:
        None -- always. Returning a dict would REPLACE the tool result.
    """
    try:
        tool_name  = getattr(tool_context, 'tool_name', 'unknown_tool')
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

        # Sanitize result for logging
        safe_result = {}
        try:
            for k, v in (result or {}).items():
                if k == "text" and isinstance(v, str) and len(v) > 200:
                    safe_result[k] = v[:200] + "... [truncated]"
                elif k == "image_bytes":
                    safe_result[k] = f"<bytes>" if v else None
                elif k == "pages" and isinstance(v, list):
                    safe_result[k] = f"<{len(v)} pages>"
                else:
                    safe_result[k] = v
        except Exception:
            safe_result = {}

        output_summary = json.dumps(safe_result)[:300]
        is_success     = result.get("is_success", "unknown") if result else "unknown"

        print(f"TOOL_RESULT -- {tool_name} | "
              f"is_success: {is_success}")

        _write_audit_record(
            session_id     = session_id,
            user_id        = user_id,
            agent_name     = tool_name,
            event_type     = "TOOL_RESULT",
            output_summary = output_summary,
        )

    except Exception as e:
        print(f"WARNING -- after_tool_audit error: "
              f"{type(e).__name__}: {e}")

    return None