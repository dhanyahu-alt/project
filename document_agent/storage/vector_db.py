import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import chromadb
import google.genai as genai

from ..util.settings import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    RATE_LIMIT_DELAY_SEC,
)

# INITIALISE CHROMADB
_client     = None
_collection = None


def initialize_vector_db() -> chromadb.Collection:
    """Initializes the ChromaDB persistent client and document collection.

    Creates the vector db directory if it does not exist.
    Uses a persistent client so embeddings survive adk web restarts.
    Safe to call multiple times -- returns existing collection if already
    initialized.

    Returns:
        chromadb.Collection: The document_embeddings collection.
    """
    global _client, _collection

    if _collection is not None:
        print(f"Vector Db Collection already initialized -- reusing")
        return _collection

    print(f"Initializing ChromaDB at: {VECTOR_DB_PATH}")
    Path(VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

    try:
        _client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))

        _collection = _client.get_or_create_collection(
            name="document_embeddings",
            metadata={
                "hnsw:space": "cosine",
                # cosine similarity: score of 1.0 = identical
                #                    score of 0.0 = completely different
            }
        )

        print(f"Vector Db Collection 'document_embeddings' ready -- "
              f"existing documents: {_collection.count()}")
        return _collection

    except Exception as e:
        print(f"ERROR initializing ChromaDB: "
              f"{type(e).__name__}: {e}")
        raise


def get_collection() -> chromadb.Collection:
    """Returns the collection, initializing if needed.

    Returns:
        chromadb.Collection
    """
    global _collection
    if _collection is None:
        initialize_vector_db()
    return _collection

#---- TEXT CHUNKING

def chunk_text(text: str,
               chunk_size: int = 400,
               overlap: int = 50) -> List[str]:
    """Splits document text into overlapping word-based chunks.

    Uses word-level splitting so chunks never cut mid-word.
    Overlap ensures context is preserved across chunk boundaries
    so the LLM can find information that spans two chunks.

    Args:
        text      : Full document text to chunk.
        chunk_size: Number of words per chunk (default: 400).
        overlap   : Number of words to overlap between chunks (default: 50).

    Returns:
        List[str]: List of text chunks. Empty list if text is empty.

    Examples:
        text with 500 words, chunk_size=400, overlap=50:
            chunk_0: words   0-399  (400 words)
            chunk_1: words 350-499  (150 words -- last chunk may be smaller)
    """
    print(f"Vector Db Chunking text -- "
          f"total chars: {len(text)}, "
          f"chunk_size: {chunk_size} words, overlap: {overlap} words")

    if not text or not text.strip():
        print(f"Vector Db WARNING -- empty text provided, no chunks created")
        return []

    words  = text.split()
    total  = len(words)
    chunks = []
    start  = 0

    while start < total:
        end         = min(start + chunk_size, total)
        chunk_words = words[start:end]
        chunk_text_ = " ".join(chunk_words)
        chunks.append(chunk_text_)

        # Move start forward by (chunk_size - overlap)
        # so next chunk begins overlap words before current end
        if end == total:
            break
        start += (chunk_size - overlap)

    print(f"Vector Db Chunking complete -- "
          f"{len(chunks)} chunks created from {total} words")
    return chunks

#----EMBEDDING GENERATION

def generate_embedding(text: str, task_type: str) -> List[float]:
    """Generates a text embedding vector using Gemini text-embedding-004.

    Args:
        text     : Text to embed. Should be a single chunk (not full doc).
        task_type: "RETRIEVAL_DOCUMENT" when indexing (storing)
                   "RETRIEVAL_QUERY"    when searching (querying)
                   Using the correct task_type improves search accuracy.

    Returns:
        List[float]: Embedding vector. Dimensionality is 768 for
                     text-embedding-004.

    Raises:
        Exception: If the Gemini API call fails.
    """
    try:
        client = genai.Client()
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config={"task_type": task_type}
        )
        embedding = response.embeddings[0].values
        return embedding

    except Exception as e:
        print(f"vector Db ERROR generating embedding: "
              f"{type(e).__name__}: {e}")
        raise

#-- ADD DOCUMENT TO VECTOR db

def add_document_to_vector_db(doc_id:   str,
                              text:     str,
                              metadata: dict) -> bool:
    """Chunks a document, generates embeddings, and stores in ChromaDB.

    Each chunk is stored as a separate ChromaDB entry with the same
    doc_id but a unique chunk_id. Metadata includes version and
    is_latest so searches can filter by version.

    Args:
        doc_id  : Versioned document ID e.g. "LoA1_v2"
        text    : Full extracted text of the document
        metadata: Dict containing at minimum:
                    doc_type  (str) : "LOA" / "NOTICE" / "BUSINESS"
                    file_name (str) : e.g. "LoA1.pdf"
                    version   (int) : e.g. 2
                    is_latest (int) : 1 = current, 0 = older version

    Returns:
        bool: True if all chunks were indexed successfully, False on error.
    """
    print(f"Adding document to vector db: {doc_id}")

    if not text or not text.strip():
        print(f"vector db ERROR -- empty text for doc_id: {doc_id}")
        return False

    try:
        collection = get_collection()

        # -- Step 1: Chunk the text ------------------------------------------
        chunks = chunk_text(text)
        if not chunks:
            print(f"vector Db ERROR -- no chunks produced for: {doc_id}")
            return False

        total_chunks = len(chunks)

        # -- Step 2: Generate embeddings and collect for batch upsert --------
        embeddings = []
        documents  = []
        metadatas  = []
        ids        = []

        for idx, chunk in enumerate(chunks):
            print(f"Vector Db Generating embedding for chunk "
                  f"{idx + 1} of {total_chunks} ...")

            embedding = generate_embedding(
                text=chunk,
                task_type="RETRIEVAL_DOCUMENT"
            )

            chunk_id = f"{doc_id}_chunk_{idx}"

            # Build per-chunk metadata
            chunk_metadata = {
                "doc_id":       doc_id,
                "doc_type":     str(metadata.get("doc_type",  "UNKNOWN")),
                "file_name":    str(metadata.get("file_name", "")),
                "version":      int(metadata.get("version",   1)),
                "is_latest":    int(metadata.get("is_latest", 1)),
                "chunk_index":  idx,
                "total_chunks": total_chunks,
                "processed_at": datetime.utcnow().isoformat(),
            }

            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append(chunk_metadata)
            ids.append(chunk_id)

            # Small delay to stay within Gemini embedding rate limits
            if idx < total_chunks - 1:
                time.sleep(RATE_LIMIT_DELAY_SEC / 10)
                # embedding model allows 1500 RPM so delay is minimal

        # -- Step 3: Batch upsert to ChromaDB --------------------------------
        print(f"Vector Db Upserting {total_chunks} chunks to ChromaDB")
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"vector Db Successfully indexed {total_chunks} chunks "
              f"for doc_id: {doc_id} | "
              f"version: {metadata.get('version', 1)} | "
              f"is_latest: {metadata.get('is_latest', 1)}")
        return True

    except Exception as e:
        print(f"vector db ERROR adding document {doc_id}: "
              f"{type(e).__name__}: {e}")
        return False

#---- UPDATE IS_LATEST FLAG FOR RE-UPLOADS

def mark_old_chunks_outdated(file_name: str) -> bool:
    """Sets is_latest = 0 on all existing chunks for a file_name.

    Called before indexing a new version so searches default to
    returning only the latest version chunks.

    Args:
        file_name: e.g. "LoA1.pdf" -- all chunks with this file_name
                   and is_latest=1 will be updated to is_latest=0.

    Returns:
        bool: True if update succeeded, False on error.
    """
    print(f"vector Db Marking old chunks outdated for: {file_name}")
    try:
        collection = get_collection()

        # Get all chunk IDs for this file_name where is_latest = 1
        results = collection.get(
            where={
                "$and": [
                    {"file_name": {"$eq": file_name}},
                    {"is_latest": {"$eq": 1}},
                ]
            }
        )

        if not results["ids"]:
            print(f"vector db No existing chunks found for: {file_name}")
            return True

        # Update metadata for each old chunk
        old_ids = results["ids"]
        print(f"vector db Updating {len(old_ids)} old chunks "
              f"to is_latest=0 ...")

        updated_metadatas = []
        for meta in results["metadatas"]:
            updated_meta = dict(meta)
            updated_meta["is_latest"] = 0
            updated_metadatas.append(updated_meta)

        collection.update(
            ids=old_ids,
            metadatas=updated_metadatas,
        )

        print(f"vector db {len(old_ids)} old chunks marked is_latest=0 "
              f"for: {file_name}")
        return True

    except Exception as e:
        print(f"vector db ERROR marking old chunks outdated: "
              f"{type(e).__name__}: {e}")
        return False

#----------SEARCH documents

def search_similar_documents(query_text:  str,
                              n_results:   int  = 5,
                              latest_only: bool = True) -> List[dict]:
    """Searches for semantically similar documents using cosine similarity.

    Generates a query embedding and finds the nearest neighbours
    in ChromaDB. By default only searches latest versions.

    Args:
        query_text  : The search query e.g. "authorization to sign contracts"
        n_results   : Number of results to return (default: 5)
        latest_only : If True, only returns latest versions (default: True)

    Returns:
        List[dict]: List of results, each containing:
            chunk_id      (str)  : unique chunk identifier
            doc_id        (str)  : e.g. "LoA1_v2"
            file_name     (str)  : e.g. "LoA1.pdf"
            doc_type      (str)  : e.g. "LOA"
            version       (int)  : e.g. 2
            is_latest     (int)  : 1 = latest version
            chunk_index   (int)  : which chunk matched
            text_preview  (str)  : first 200 chars of matching chunk
            distance      (float): cosine distance (lower = more similar)
    """
    print(f"vector db Searching for: '{query_text[:80]}' "
          f"(n_results={n_results}, latest_only={latest_only})")

    try:
        collection = get_collection()

        # Generate query embedding
        query_embedding = generate_embedding(
            text=query_text,
            task_type="RETRIEVAL_QUERY"
        )

        # Build where filter
        where_filter = {"is_latest": {"$eq": 1}} if latest_only else None

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        ids        = results["ids"][0]
        documents  = results["documents"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]

        for i, chunk_id in enumerate(ids):
            formatted.append({
                "chunk_id":     chunk_id,
                "doc_id":       metadatas[i].get("doc_id",      ""),
                "file_name":    metadatas[i].get("file_name",   ""),
                "doc_type":     metadatas[i].get("doc_type",    ""),
                "version":      metadatas[i].get("version",     1),
                "is_latest":    metadatas[i].get("is_latest",   1),
                "chunk_index":  metadatas[i].get("chunk_index", 0),
                "text_preview": documents[i][:200] if documents[i] else "",
                "distance":     round(distances[i], 4),
            })

        print(f"vector db Search complete -- "
              f"{len(formatted)} results returned")
        for r in formatted:
            print(f"  -> {r['doc_id']} | type: {r['doc_type']} | "
                  f"distance: {r['distance']}")

        return formatted

    except Exception as e:
        print(f"vector db ERROR searching: {type(e).__name__}: {e}")
        return []


def find_by_document_type(doc_type:     str,
                           n_results:   int  = 10,
                           latest_only: bool = True) -> List[dict]:
    """Retrieves documents filtered by document type.

    Args:
        doc_type    : "LOA" / "NOTICE" / "BUSINESS" / "UNKNOWN"
        n_results   : Maximum results to return (default: 10)
        latest_only : If True, only latest versions (default: True)

    Returns:
        List[dict]: Matching document chunks with metadata.
    """
    print(f"vector Finding documents of type: {doc_type} "
          f"(latest_only={latest_only})")
    try:
        collection = get_collection()

        where_filter = {
            "$and": [
                {"doc_type":  {"$eq": doc_type}},
                {"is_latest": {"$eq": 1}},
            ]
        } if latest_only else {"doc_type": {"$eq": doc_type}}

        results = collection.get(
            where=where_filter,
            limit=n_results,
            include=["documents", "metadatas"],
        )

        formatted = []
        for i, chunk_id in enumerate(results["ids"]):
            formatted.append({
                "chunk_id":  chunk_id,
                "doc_id":    results["metadatas"][i].get("doc_id",    ""),
                "file_name": results["metadatas"][i].get("file_name", ""),
                "doc_type":  results["metadatas"][i].get("doc_type",  ""),
                "version":   results["metadatas"][i].get("version",   1),
            })

        print(f"vector db Found {len(formatted)} chunks "
              f"of type: {doc_type}")
        return formatted

    except Exception as e:
        print(f"vector db ERROR finding by type: "
              f"{type(e).__name__}: {e}")
        return []

print("vector db Module loaded -- initializing vector db ...")
initialize_vector_db()