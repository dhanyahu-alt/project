import json
from typing import Optional

from ..storage.vector_db import (
    add_document_to_vector_db,
    mark_old_chunks_outdated,
    search_similar_documents,
    find_by_document_type,
    chunk_text,
    generate_embedding,
)

def index_document_in_vector_db(doc_id:   str,
                                 text:     str,
                                 metadata: dict) -> dict:
    """Indexes a document in the vector db for semantic search.

    Chunks the document text, generates embeddings using Gemini
    text-embedding-004 (free, no daily limit), and stores all chunks
    in ChromaDB with version metadata. For re-uploads, marks old
    chunks outdated before indexing the new version.

    Args:
        doc_id  : Versioned document ID e.g. "LoA1_v2"
        text    : Full extracted text of the document
        metadata: Dict containing at minimum:
                    doc_type    (str)  : "LOA" / "NOTICE" / "BUSINESS"
                    file_name   (str)  : e.g. "LoA1.pdf"
                    version     (int)  : e.g. 2
                    is_latest   (int)  : 1 = current, 0 = older version
                    is_reupload (bool) : True if file was processed before

    Returns:
        dict with keys:
            is_success     (bool) : True if indexing succeeded
            doc_id         (str)  : the doc_id indexed
            version        (int)  : version number indexed
            chunks_indexed (int)  : number of chunks stored in ChromaDB
            is_reupload    (bool) : True if old chunks were marked outdated
            error          (str)  : error message if is_success is False
    """
    print(f" index_document_in_vector_db called -- "
          f"doc_id: {doc_id}")

    result = {
        "is_success":     False,
        "doc_id":         doc_id,
        "version":        metadata.get("version",   1),
        "chunks_indexed": 0,
        "is_reupload":    False,
        "error":          None,
    }

    # -- Validate inputs 
    if not doc_id:
        result["error"] = "doc_id is required"
        print(f" ERROR -- {result['error']}")
        return result

    if not text or not text.strip():
        result["error"] = "text is empty -- nothing to index"
        print(f" ERROR -- {result['error']}")
        return result

    if not metadata.get("file_name"):
        result["error"] = "metadata missing required field: file_name"
        print(f" ERROR -- {result['error']}")
        return result

    try:
        file_name   = metadata.get("file_name")
        version     = int(metadata.get("version",    1))
        is_reupload = bool(metadata.get("is_reupload", False))

        print(f" file_name: {file_name} | "
              f"version: {version} | is_reupload: {is_reupload}")

        # -- Step 1: Mark old chunks outdated if re-upload 
        if is_reupload:
            print(f" Re-upload detected -- "
                  f"marking old chunks outdated for: {file_name}")
            mark_old_chunks_outdated(file_name)
            result["is_reupload"] = True

        # -- Step 2: Build clean metadata for ChromaDB 
        # ChromaDB metadata values must be str, int, or float -- NOT bool
        clean_metadata = {
            "doc_type":  str(metadata.get("doc_type",  "UNKNOWN")),
            "file_name": str(file_name),
            "version":   int(version),
            "is_latest": int(metadata.get("is_latest", 1)),
        }

        # -- Step 3: Add document to vector db
        print(f" Indexing document: {doc_id} ...")
        success = add_document_to_vector_db(
            doc_id=doc_id,
            text=text,
            metadata=clean_metadata,
        )

        if not success:
            result["error"] = f"add_document_to_vector_db returned False"
            print(f" ERROR -- indexing failed for: {doc_id}")
            return result

        # -- Step 4: Count chunks indexed 
        chunks         = chunk_text(text)
        chunks_indexed = len(chunks)
        result["chunks_indexed"] = chunks_indexed

        result["is_success"] = True
        print(f"Indexing complete -- "
              f"doc_id: {doc_id} | "
              f"chunks: {chunks_indexed} | "
              f"version: {version}")
        return result

    except Exception as e:
        print(f" ERROR in index_document_in_vector_db: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Indexing error: {str(e)}"
        return result

def find_similar_documents(query: str, n_results: int = 5) -> dict:
    """Finds semantically similar documents using vector search.

    Generates a query embedding and searches ChromaDB for the nearest
    neighbour chunks. Returns only latest versions by default.
    Useful for finding related documents before processing a new one.

    Args:
        query    : Natural language search query
                   e.g. "authorization to act on behalf of company"
        n_results: Number of results to return (default: 5)

    Returns:
        dict with keys:
            is_success (bool)       : True if search succeeded
            query      (str)        : the query that was searched
            results    (List[dict]) : ranked list of matching chunks
                each result contains:
                    chunk_id     (str)  : unique chunk identifier
                    doc_id       (str)  : e.g. "LoA1_v2"
                    file_name    (str)  : e.g. "LoA1.pdf"
                    doc_type     (str)  : e.g. "LOA"
                    version      (int)  : version number
                    chunk_index  (int)  : which chunk matched
                    text_preview (str)  : first 200 chars of matching chunk
                    distance     (float): cosine distance (lower = more similar)
            count      (int)        : number of results returned
            error      (str)        : error message if is_success is False
    """
    print(f" find_similar_documents called -- "
          f"query: '{query[:80]}' | n_results: {n_results}")

    result = {
        "is_success": False,
        "query":      query,
        "results":    [],
        "count":      0,
        "error":      None,
    }

    if not query or not query.strip():
        result["error"] = "query is empty"
        print(f" ERROR -- {result['error']}")
        return result

    try:
        matches = search_similar_documents(
            query_text=query,
            n_results=n_results,
            latest_only=True,
        )

        result["is_success"] = True
        result["results"]    = matches
        result["count"]      = len(matches)

        print(f"find_similar_documents -- "
              f"{len(matches)} results returned")
        return result

    except Exception as e:
        print(f" ERROR in find_similar_documents: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Search error: {str(e)}"
        return result

def check_duplicate_document(text: str,
                              threshold: float = 0.95) -> dict:
    """Checks if a near-identical document has already been processed.

    Generates an embedding for the first chunk of the provided text
    and searches ChromaDB for documents with cosine similarity above
    the threshold. Used before processing to detect duplicates that
    may have a different filename but identical content.

    Args:
        text     : Extracted text of the document to check
        threshold: Cosine similarity threshold (default: 0.95)
                   Documents above this threshold are flagged as duplicate
                   0.95 = 95% similar -- catches near-identical documents
                   Lower value = more sensitive (more false positives)
                   Higher value = less sensitive (may miss duplicates)

    Returns:
        dict with keys:
            is_success      (bool)  : True if check completed
            is_duplicate    (bool)  : True if a near-identical doc found
            similar_doc_id  (str)   : doc_id of similar document or None
            similar_file    (str)   : file_name of similar document or None
            score           (float) : highest similarity score (0.0 to 1.0)
            error           (str)   : error message if is_success is False
    """
    print(f" check_duplicate_document called -- "
          f"threshold: {threshold}")

    result = {
        "is_success":     False,
        "is_duplicate":   False,
        "similar_doc_id": None,
        "similar_file":   None,
        "score":          0.0,
        "error":          None,
    }

    if not text or not text.strip():
        result["error"] = "text is empty -- cannot check for duplicates"
        print(f" ERROR -- {result['error']}")
        return result

    try:
        # Use first 400 words as representative sample for comparison
        words       = text.split()
        sample_text = " ".join(words[:400])

        print(f" Checking for duplicates using "
              f"first {min(len(words), 400)} words as sample ...")

        # Search for similar documents
        matches = search_similar_documents(
            query_text=sample_text,
            n_results=3,
            latest_only=True,
        )

        result["is_success"] = True

        if not matches:
            print(f" No existing documents in vector db -- "
                  f"not a duplicate")
            return result

        # Check top result against threshold
        # ChromaDB cosine distance: 0.0 = identical, 1.0 = completely different
        # Convert distance to similarity: similarity = 1.0 - distance
        top_match  = matches[0]
        distance   = top_match.get("distance", 1.0)
        similarity = round(1.0 - distance, 4)

        print(f" Top match: {top_match.get('doc_id')} | "
              f"distance: {distance} | similarity: {similarity}")

        if similarity >= threshold:
            result["is_duplicate"]   = True
            result["similar_doc_id"] = top_match.get("doc_id")
            result["similar_file"]   = top_match.get("file_name")
            result["score"]          = similarity
            print(f" DUPLICATE DETECTED -- "
                  f"similar_doc_id: {result['similar_doc_id']} | "
                  f"score: {similarity}")
        else:
            result["is_duplicate"] = False
            result["score"]        = similarity
            print(f" Not a duplicate -- "
                  f"highest similarity: {similarity} "
                  f"(below threshold: {threshold})")

        return result

    except Exception as e:
        print(f" ERROR in check_duplicate_document: "
              f"{type(e).__name__}: {e}")
        result["error"] = f"Duplicate check error: {str(e)}"
        return result