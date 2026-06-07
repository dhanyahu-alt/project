from datetime import datetime
from google.adk.tools import ToolContext
VALID_STAGES = [
    "LOADING",
    "CLASSIFYING",
    "EXTRACTING",
    "VALIDATING",
    "AWAITING_APPROVAL",
    "SAVING",
    "INDEXED",
    "ARCHIVED",
    "COMPLETE",
    "REJECTED",
    "ERROR",
]

def update_processing_stage(stage: str,
                             tool_context: ToolContext) -> dict:
    """Updates the current processing stage in session state.

    Sets the processing_stage key in session state so the manager
    agent always knows where in the workflow it is. The stage value
    is injected into the manager instruction via {processing_stage}.

    Valid stages:
        LOADING          -> reading the PDF file
        CLASSIFYING      -> classification agent running
        EXTRACTING       -> extraction agent running
        VALIDATING       -> validation agent running
        AWAITING_APPROVAL -> waiting for human APPROVE / REJECT
        SAVING           -> saving to SQLite database
        INDEXED          -> indexed in ChromaDB vector db
        ARCHIVED         -> document archived to storage/archive/
        COMPLETE         -> all steps done successfully
        REJECTED         -> human rejected the document
        ERROR            -> an error occurred during processing

    Args:
        stage       : One of the valid stage strings listed above
        tool_context: ADK ToolContext -- provides access to session state

    Returns:
        dict with keys:
            is_success (bool) : True if stage was updated
            stage      (str)  : the stage that was set
            previous   (str)  : the stage that was set before this call
            timestamp  (str)  : ISO timestamp of the update
            message    (str)  : confirmation message
            error      (str)  : error message if is_success is False
    """
    print(f"update_processing_stage called -- stage: {stage}")

    result = {
        "is_success": False,
        "stage":      stage,
        "previous":   None,
        "timestamp":  None,
        "message":    None,
        "error":      None,
    }

    # -- Validate stage value ------------------------------------------------
    stage_upper = stage.upper().strip()

    if stage_upper not in VALID_STAGES:
        result["error"] = (
            f"Invalid stage: '{stage}'. "
            f"Valid stages are: {', '.join(VALID_STAGES)}"
        )
        print(f"ERROR -- {result['error']}")
        return result

    try:
        # -- Read current stage before updating -
        previous_stage = tool_context.state.get("processing_stage", "NOT_SET")
        result["previous"] = previous_stage

        # -- Update state keys 
        now = datetime.utcnow().isoformat()

        tool_context.state["processing_stage"] = stage_upper
        tool_context.state["stage_updated_at"] = now

        result["is_success"] = True
        result["stage"]      = stage_upper
        result["timestamp"]  = now
        result["message"]    = (
            f"Stage updated: {previous_stage} -> {stage_upper}"
        )

        print(f"Stage updated: "
              f"{previous_stage} -> {stage_upper} | "
              f"at: {now}")
        return result

    except Exception as e:
        print(f"ERROR in update_processing_stage: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"State update error: {str(e)}"
        return result

def set_current_document(file_path:   str,
                          version:     int,
                          doc_id:      str,
                          is_reupload: bool,
                          tool_context: ToolContext) -> dict:
    """Sets the current document and version details in session state.

    Stores all document-related state keys in one call so the manager
    agent has full context about the document being processed. These
    values are injected into the manager instruction via template syntax:
        {current_doc_path}
        {current_doc_version}
        {current_doc_id}
        {is_reupload}

    Args:
        file_path   : Absolute path to the document file
        version     : Version number assigned e.g. 1, 2, 3
        doc_id      : Versioned doc_id e.g. "LoA1_v2"
        is_reupload : True if this file was previously processed
        tool_context: ADK ToolContext -- provides access to session state

    Returns:
        dict with keys:
            is_success   (bool) : True if state was updated
            file_path    (str)  : the file_path that was set
            version      (int)  : the version that was set
            doc_id       (str)  : the doc_id that was set
            is_reupload  (bool) : the is_reupload flag that was set
            message      (str)  : confirmation message
            error        (str)  : error message if is_success is False
    """
    print(f"set_current_document called -- "
          f"file_path: {file_path} | "
          f"version: {version} | "
          f"doc_id: {doc_id} | "
          f"is_reupload: {is_reupload}")

    result = {
        "is_success":  False,
        "file_path":   file_path,
        "version":     version,
        "doc_id":      doc_id,
        "is_reupload": is_reupload,
        "message":     None,
        "error":       None,
    }

    # -- Validate required fields ----
    if not file_path:
        result["error"] = "file_path is required"
        print(f"ERROR -- {result['error']}")
        return result

    if not doc_id:
        result["error"] = "doc_id is required"
        print(f"ERROR -- {result['error']}")
        return result

    try:
        now = datetime.utcnow().isoformat()

        # -- Set all document state keys in one call -------------------------
        tool_context.state["current_doc_path"]    = file_path
        tool_context.state["current_doc_version"] = int(version)
        tool_context.state["current_doc_id"]      = doc_id
        tool_context.state["is_reupload"]         = bool(is_reupload)
        tool_context.state["doc_state_set_at"]    = now

        tool_context.state["human_approved"]  = False
        tool_context.state["pending_review"]  = False
        tool_context.state["pending_data"]    = None

        result["is_success"] = True
        result["message"]    = (
            f"Document state set: {doc_id} | "
            f"v{version} | "
            f"reupload: {is_reupload}"
        )

        print(f"Document state set -- "
              f"doc_id: {doc_id} | "
              f"version: {version} | "
              f"is_reupload: {is_reupload}")
        print(f" Approval state reset: "
              f"human_approved=False, pending_review=False")
        return result

    except Exception as e:
        print(f"ERROR in set_current_document: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"State update error: {str(e)}"
        return result