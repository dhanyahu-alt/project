from datetime import datetime
from pathlib import Path

import pymupdf


def load_pdf(file_path: str) -> dict:
    """Loads and extracts text content from a PDF document file.

    Opens the PDF at the given path, extracts full text from every page,
    reads document metadata, and detects whether the document is scanned
    (image-based with no selectable text). Returns a structured dict with
    all extracted content. Never raises exceptions — errors are returned
    inside the dict under the 'error' key.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        dict with keys:
            is_success  (bool)  : True if file was read without errors
            is_scanned  (bool)  : True if text content is too sparse
                                  (likely a scanned / image-only PDF)
            file_path   (str)   : The original path provided
            file_name   (str)   : The filename only (e.g. LoA1.pdf)
            page_count  (int)   : Total number of pages (0 on failure)
            pages       (list)  : Per-page text list (one entry per page)
            text        (str)   : Full extracted text from all pages
            keydata     (dict)  : Title, author, creation date, etc.
            error       (str|None): Error message if is_success is False
    """
    print(f"inside pdf loader with path: {file_path}")

    result = {
        "is_success": False,
        "is_scanned": False,
        "file_path":  file_path,
        "file_name":  Path(file_path).name,
        "page_count": 0,
        "pages":      [],
        "text":       "",
        "keydata":    {},
        "error":      None,
    }

    # -- Cases to check for file validity ------------------------------------
    path = Path(file_path)

    if not path.exists():
        result["error"] = f"File not found: {file_path}"
        return result

    if not path.is_file():
        result["error"] = f"Path is not a file: {file_path}"
        return result

    if path.suffix.lower() not in (".pdf",):
        result["error"] = f"File is not a PDF (extension: {path.suffix}): {file_path}"
        return result

    # -- Open and process the PDF --------------------------------------------
    doc = None
    try:
        print(f"Opening PDF: {file_path}")
        doc = pymupdf.open(file_path)

        page_count = len(doc)
        result["page_count"] = page_count
        print(f"PDF opened successfully — Pages: {page_count}")

        if page_count == 0:
            result["error"] = "PDF has no pages."
            return result

        # -- Extract text page by page ---------------------------------------
        pages_text = []
        full_text_parts = []

        for page_index in range(page_count):
            print(f"PDF Extracting text from page {page_index + 1} of {page_count} ...")
            page = doc[page_index]
            page_text = page.get_text("text").strip()
            pages_text.append(page_text)
            if page_text:
                full_text_parts.append(f"--- Page {page_index + 1} ---\n{page_text}")

        full_text = "\n\n".join(full_text_parts)
        result["pages"] = pages_text
        result["text"]  = full_text

        total_chars = sum(len(p) for p in pages_text)
        print(f"PDF Text extraction complete — Total characters: {total_chars}")

        # -- Detect scanned / image-based PDF --------------------------------
        avg_chars_per_page = total_chars / page_count if page_count > 0 else 0
        result["is_scanned"] = avg_chars_per_page < 50
        print(f"PDF Avg chars/page: {avg_chars_per_page:.1f} — "
              f"Is scanned: {result['is_scanned']}")

        # -- Extract document keydata ----------------------------------------
        print(f"PDF Reading document for fetching keydata ...")
        raw_meta = doc.metadata or {}

        result["keydata"] = {
            "title":           raw_meta.get("title",        "").strip() or None,
            "author":          raw_meta.get("author",       "").strip() or None,
            "subject":         raw_meta.get("subject",      "").strip() or None,
            "creator":         raw_meta.get("creator",      "").strip() or None,
            "producer":        raw_meta.get("producer",     "").strip() or None,
            "creation_date":   raw_meta.get("creationDate", "").strip() or None,
            "mod_date":        raw_meta.get("modDate",      "").strip() or None,
            "page_count":      page_count,
            "file_size_bytes": path.stat().st_size,
            "extracted_at":    datetime.utcnow().isoformat(),
        }

        print(f"PDF keydata — "
              f"Title: {result['keydata']['title']} | "
              f"Author: {result['keydata']['author']} | "
              f"Size: {result['keydata']['file_size_bytes']} bytes")

        result["is_success"] = True
        print(f"PDF loaded successfully: {result['file_name']}")
        return result

    except pymupdf.FileDataError as e:
        print(f"ERROR loading PDF - corrupted or unreadable: {e}")
        result["error"] = f"PDF is corrupted or unreadable: {str(e)}"
        return result

    except PermissionError as e:
        print(f"PDF ERROR — Permission denied: {e}")
        result["error"] = f"Permission denied reading file: {str(e)}"
        return result

    except Exception as e:
        print(f"PDF ERROR — Unexpected: {type(e).__name__}: {e}")
        result["error"] = f"Unexpected error reading PDF: {type(e).__name__}: {str(e)}"
        return result

    finally:
        if doc is not None:
            try:
                doc.close()
                print(f"PDF Document handle closed.")
            except Exception:
                pass


def render_page_as_image(file_path: str, page_index: int = 0,
                         dpi: int = 150) -> dict:
    """Renders a single PDF page as a PNG image bytes object.

    Used as a helper when the PDF is detected as scanned (is_scanned=True).
    The returned image bytes can be passed directly to the OCR tool
    (extract_text_from_image) for Gemini Vision processing.

    Args:
        file_path : Path to the PDF file.
        page_index: Zero-based page index to render (default: 0).
        dpi       : Resolution for rendering (default: 150 DPI).
                    Use 200+ for better OCR accuracy on dense text.

    Returns:
        dict with keys:
            is_success  (bool)  : True if rendering succeeded
            file_name   (str)   : The filename only (e.g. LoA1.pdf)
            image_bytes (bytes) : PNG image bytes (None on failure)
            page_index  (int)   : The page that was rendered
            width_px    (int)   : Image width in pixels
            height_px   (int)   : Image height in pixels
            error       (str|None): Error message if is_success is False
    """
    path = Path(file_path)
    if not path.exists():
        return {
            "is_success":  False,
            "file_name":   path.name,
            "image_bytes": None,
            "page_index":  page_index,
            "width_px":    0,
            "height_px":   0,
            "error":       f"File not found: {file_path}",
        }

    result = {
        "is_success":  False,
        "file_name":   path.name,
        "image_bytes": None,
        "page_index":  page_index,
        "width_px":    0,
        "height_px":   0,
        "error":       None,
    }

    doc = None
    try:
        # path already defined above — removed duplicate Path() call -
        print(f"PDF Rendering page {page_index} of: {file_path} at {dpi} DPI ...")
        doc = pymupdf.open(file_path)

        if page_index >= len(doc) or page_index < 0:
            result["error"] = (
                f"Page index {page_index} out of range "
                f"(document has {len(doc)} pages)"
            )
            return result

        page = doc[page_index]

        # Scale matrix for the requested DPI (72 DPI is PDF native)
        zoom   = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)

        # Render the page to a Pixmap (RGB colour space)
        pixmap = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB)

        # Convert pixmap to PNG bytes
        image_bytes = pixmap.tobytes("png")

        result["is_success"]  = True
        result["image_bytes"] = image_bytes
        result["width_px"]    = pixmap.width
        result["height_px"]   = pixmap.height

        print(f"[pdf_loader] Page rendered — "
              f"{pixmap.width}x{pixmap.height}px | "
              f"{len(image_bytes)} bytes")
        return result

    except Exception as e:
        print(f"PDF ERROR — Render failed: {type(e).__name__}: {e}")
        result["error"] = f"Error rendering page: {type(e).__name__}: {str(e)}"
        return result

    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass