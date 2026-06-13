import json
from datetime import datetime
from pathlib import Path
from typing import List

from google.adk.tools import ToolContext

def get_documents_in_folder(folder_path: str) -> dict:
    """Scans a folder and returns all PDF file paths found.

    Use this tool when the user provides a folder path instead of a
    single document file path. Returns a list of absolute paths to all
    PDF files found in the folder. Does not scan sub-folders.

    Args:
        folder_path: Absolute or relative path to the folder to scan.

    Returns:
        dict with keys:
            is_success  (bool)       : True if scan completed
            folder_path (str)        : the folder path that was scanned
            pdf_files   (List[str])  : absolute paths to all PDF files found
            count       (int)        : total number of PDF files found
            error       (str)        : error message if is_success is False
    """
    print(f"get_documents_in_folder called "
          f"folder: {folder_path}")

    result = {
        "is_success":  False,
        "folder_path": folder_path,
        "pdf_files":   [],
        "count":       0,
        "error":       None,
    }

    if not folder_path or not folder_path.strip():
        result["error"] = "folder_path is required"
        print(f"ERROR - {result['error']}")
        return result

    folder = Path(folder_path)

    if not folder.exists():
        result["error"] = f"Folder not found: {folder_path}"
        print(f"ERROR -{result['error']}")
        return result

    if not folder.is_dir():
        result["error"] = f"Path is not a folder: {folder_path}"
        print(f"ERROR - {result['error']}")
        return result

    try:
        pdf_files = []

        for item in sorted(folder.iterdir()):
            if item.is_file() and item.suffix.lower() == ".pdf":
                pdf_files.append(str(item.resolve()))

        print(f"Scan complete -"
              f"found {len(pdf_files)} PDF file(s) in: {folder_path}")

        for pdf in pdf_files:
            print(f"  -> {Path(pdf).name}")

        result["is_success"] = True
        result["pdf_files"]  = pdf_files
        result["count"]      = len(pdf_files)

        if len(pdf_files) == 0:
            print(f"WARNING : no PDF files found in folder")

        return result

    except PermissionError as e:
        result["error"] = f"Permission denied reading folder: {str(e)}"
        print(f"ERROR - {result['error']}")
        return result

    except Exception as e:
        result["error"] = f"Unexpected error scanning folder: {str(e)}"
        print(f"ERROR - {type(e).__name__}: {e}")
        return result

def generate_batch_summary(tool_context: ToolContext) -> dict:
    """Generates a summary report of batch document processing results.

    Reads the batch results stored in session state under the key
    app:batch_results and produces a consolidated summary. Call this
    tool after all documents in a batch have been processed.

    The app:batch_results state key should contain a list of dicts,
    where each dict represents one processed document with at minimum:
        file_name        (str)   : name of the processed file
        is_success       (bool)  : whether processing succeeded
        doc_type         (str)   : classified document type
        confidence_score (float) : validation confidence score
        doc_id           (str)   : generated document ID
        error            (str)   : error message if failed

    Returns:
        dict with keys:
            is_success       (bool)  : True if summary generated
            total_processed  (int)   : total documents in batch
            successful       (int)   : documents processed successfully
            failed           (int)   : documents that failed processing
            by_doc_type      (dict)  : count per document type
                                       e.g. {"LOA": 2, "NOTICE": 1}
            avg_confidence   (float) : average confidence across successes
            results          (list)  : the full batch_results list
            generated_at     (str)   : ISO timestamp of summary generation
            error            (str)   : error message if is_success is False
    """
    print(f"generate_batch_summary called")

    result = {
        "is_success":      False,
        "total_processed": 0,
        "successful":      0,
        "failed":          0,
        "by_doc_type":     {},
        "avg_confidence":  0.0,
        "results":         [],
        "generated_at":    datetime.utcnow().isoformat(),
        "error":           None,
    }

    try:
        raw = tool_context.state.get("app:batch_results", None)

        if raw is None:
            result["error"] = (
                "No batch results found in state. "
                "Key app:batch_results is not set. "
                "Ensure documents were processed before calling this tool."
            )
            print(f"ERROR - {result['error']}")
            return result

        if isinstance(raw, str):
            try:
                batch_results = json.loads(raw)
            except json.JSONDecodeError as e:
                result["error"] = f"Could not parse batch_results JSON: {str(e)}"
                print(f"ERROR - {result['error']}")
                return result
        elif isinstance(raw, list):
            batch_results = raw
        else:
            result["error"] = (
                f"app:batch_results has unexpected type: {type(raw).__name__}. "
                "Expected list or JSON string."
            )
            print(f"ERROR -- {result['error']}")
            return result

        if not batch_results:
            print(f"WARNING -- batch_results list is empty")
            result["is_success"]      = True
            result["total_processed"] = 0
            return result

        total       = len(batch_results)
        successful  = 0
        failed      = 0
        by_doc_type = {}
        confidence_scores = []

        for doc_result in batch_results:
            if not isinstance(doc_result, dict):
                continue

            # Count successes and failures
            if doc_result.get("is_success", False):
                successful += 1

                # Count by doc_type
                doc_type = doc_result.get("doc_type", "UNKNOWN")
                by_doc_type[doc_type] = by_doc_type.get(doc_type, 0) + 1

                # Collect confidence scores
                confidence = doc_result.get("confidence_score", 0.0)
                if isinstance(confidence, (int, float)) and confidence > 0:
                    confidence_scores.append(float(confidence))
            else:
                failed += 1

        # Calculate average confidence
        avg_confidence = 0.0
        if confidence_scores:
            avg_confidence = round(
                sum(confidence_scores) / len(confidence_scores), 4
            )

        result["is_success"]      = True
        result["total_processed"] = total
        result["successful"]      = successful
        result["failed"]          = failed
        result["by_doc_type"]     = by_doc_type
        result["avg_confidence"]  = avg_confidence
        result["results"]         = batch_results

        print(f"Batch summary generated:")
        print(f"  Total processed : {total}")
        print(f"  Successful      : {successful}")
        print(f"  Failed          : {failed}")
        print(f"  By doc type     : {by_doc_type}")
        print(f"  Avg confidence  : {avg_confidence}")

        return result

    except Exception as e:
        result["error"] = f"Unexpected error generating summary: {str(e)}"
        print(f"ERROR - {type(e).__name__}: {e}")
        return result