import base64
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import io
from ..util.settings import MODEL_FLASH

def extract_text_from_image(image_path: str) -> dict:
    """Extracts text from scanned document images using Gemini Vision OCR.

    Sends the image at the given path to Gemini Vision and returns all
    text found in the document, preserving original structure and layout.
    Use this when load_pdf returns is_scanned=True on a document.

    Args:
        image_path: Absolute or relative path to the image file.
                    Supported formats: PNG, JPEG, BMP, TIFF, WEBP.

    Returns:
        dict with keys:
            is_success   (bool)  : True if OCR completed without errors
            image_path   (str)   : The original path provided
            file_name    (str)   : The filename only (e.g. page_1.png)
            extracted_text (str) : Full text extracted from the image
            char_count   (int)   : Number of characters extracted
            error        (str|None): Error message if is_success is False
    """
    print(f"Inside extract_text_from_image with path: {image_path}")

    result = {
        "is_success":      False,
        "image_path":      image_path,
        "file_name":       Path(image_path).name,
        "extracted_text":  "",
        "char_count":      0,
        "error":           None,
    }

    # -- Validate the image path ---------------------------------------------
    path = Path(image_path)

    if not path.exists():
        result["error"] = f"Image file not found: {image_path}"
        return result

    if not path.is_file():
        result["error"] = f"Path is not a file: {image_path}"
        return result

    supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")
    if path.suffix.lower() not in supported_formats:
        result["error"] = (
            f"Unsupported image format: {path.suffix}. "
            f"Supported: {', '.join(supported_formats)}"
        )
        return result

    try:
        print(f"[ocr_tool] Loading image: {image_path}")

        # -- Load image using Pillow and convert to bytes --------------------
        with Image.open(image_path) as img:
            # Convert to RGB if image is in RGBA or palette mode
            if img.mode not in ("RGB", "L"):
                print(f"Inside OCR Converting image mode {img.mode} -> RGB ...")
                img = img.convert("RGB")

            # Save to bytes buffer as JPEG for Gemini
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            image_bytes = buffer.getvalue()

        print(f"Inside OCR Image loaded — Size: {len(image_bytes)} bytes")

        # -- Encode image to base64 for Gemini API ---------------------------
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        print(f"Inside OCR Image encoded to base64 successfully")

        # -- Build Gemini Vision request -------------------------------------
        print(f"Inside OCR Sending image to Gemini Vision ({MODEL_FLASH}) ...")

        model = genai.GenerativeModel(model_name=MODEL_FLASH)

        prompt = (
            "You are a document OCR specialist. "
            "Extract ALL text from this document image exactly as it appears. "
            "Preserve the original structure, layout, headings, paragraphs, "
            "bullet points, tables, dates, names, and numbers. "
            "Do not summarize, interpret, or add any commentary. "
            "Return only the raw extracted text."
        )

        response = model.generate_content(
            contents=[
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_b64,
                            }
                        },
                        {
                            "text": prompt
                        },
                    ]
                }
            ]
        )

        # -- Extract text from response --------------------------------------
        extracted_text = response.text.strip() if response.text else ""
        char_count = len(extracted_text)

        print(f"OCR complete — Characters extracted: {char_count}")

        if char_count == 0:
            print(f"OCR WARNING — No text extracted from image")

        result["is_success"]     = True
        result["extracted_text"] = extracted_text
        result["char_count"]     = char_count

        print(f"OCR Text extracted successfully from: {result['file_name']}")
        return result

    except Exception as e:
        print(f"ERROR — OCR failed: {type(e).__name__}: {e}")
        result["error"] = f"OCR failed: {type(e).__name__}: {str(e)}"
        return result


def extract_text_from_image_bytes(image_bytes: bytes,
                                   file_name: str = "page.png") -> dict:
    """Extracts text from image bytes using Gemini Vision OCR.

    Used when image bytes come directly from render_page_as_image()
    in pdf_loader_tool.py — no file path needed, works with in-memory
    PNG bytes from pymupdf page rendering.

    Args:
        image_bytes: Raw PNG image bytes (from render_page_as_image).
        file_name  : Optional label for logging (default: page.png).

    Returns:
        dict with keys:
            is_success     (bool) : True if OCR completed without errors
            file_name      (str)  : The label provided
            extracted_text (str)  : Full text extracted from the image
            char_count     (int)  : Number of characters extracted
            error          (str|None): Error message if is_success is False
    """
    print(f"Inside extract_text_from_image_bytes for: {file_name}")

    result = {
        "is_success":      False,
        "file_name":       file_name,
        "extracted_text":  "",
        "char_count":      0,
        "error":           None,
    }

    if not image_bytes:
        result["error"] = "image_bytes is empty or None"
        return result

    try:
        print(f"OCr Encoding {len(image_bytes)} bytes to base64 ...")
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        print(f"OCR Sending image bytes to Gemini Vision ({MODEL_FLASH}) ...")

        model = genai.GenerativeModel(model_name=MODEL_FLASH)

        prompt = (
            "You are a document OCR specialist. "
            "Extract ALL text from this document image exactly as it appears. "
            "Preserve the original structure, layout, headings, paragraphs, "
            "bullet points, tables, dates, names, and numbers. "
            "Do not summarize, interpret, or add any commentary. "
            "Return only the raw extracted text."
        )

        response = model.generate_content(
            contents=[
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64,
                            }
                        },
                        {
                            "text": prompt
                        },
                    ]
                }
            ]
        )

        extracted_text = response.text.strip() if response.text else ""
        char_count = len(extracted_text)

        print(f"OCR complete — Characters extracted: {char_count}")

        if char_count == 0:
            print(f"OCR WARNING — No text extracted from image bytes")

        result["is_success"]     = True
        result["extracted_text"] = extracted_text
        result["char_count"]     = char_count

        print(f"OCR Text extracted successfully from bytes: {file_name}")
        return result

    except Exception as e:
        print(f"OCR ERROR — OCR failed: {type(e).__name__}: {e}")
        result["error"] = f"OCR failed: {type(e).__name__}: {str(e)}"
        return result


def analyze_document_layout(image_path: str) -> dict:
    """Analyzes the visual layout and structure of a document image.

    Sends the image to Gemini Vision and returns a structural analysis
    identifying document type indicators, signatures, tables, headers,
    logos, and other layout elements. Helps the classification agent
    make more accurate decisions on scanned documents.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        dict with keys:
            is_success        (bool) : True if analysis completed
            image_path        (str)  : The original path provided
            file_name         (str)  : The filename only
            document_type_hint (str) : Likely document type from layout
            has_signature     (bool) : Signature block detected
            has_tables        (bool) : Tables detected
            has_letterhead    (bool) : Company letterhead detected
            has_logo          (bool) : Logo or seal detected
            layout_summary    (str)  : Full layout description
            error             (str|None): Error message if is_success is False
    """
    print(f"inside analyze_document_layout with path: {image_path}")

    result = {
        "is_success":          False,
        "image_path":          image_path,
        "file_name":           Path(image_path).name,
        "document_type_hint":  "UNKNOWN",
        "has_signature":       False,
        "has_tables":          False,
        "has_letterhead":      False,
        "has_logo":            False,
        "layout_summary":      "",
        "error":               None,
    }

    # -- Validate path -------------------------------------------------------
    path = Path(image_path)
    if not path.exists():
        result["error"] = f"Image file not found: {image_path}"
        return result

    if not path.is_file():
        result["error"] = f"Path is not a file: {image_path}"
        return result

    try:
        print(f"OCR Loading image for layout analysis: {image_path}")

        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            image_bytes = buffer.getvalue()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        print(f"OCR Sending image to Gemini Vision for layout analysis ...")

        model = genai.GenerativeModel(model_name=MODEL_FLASH)

        prompt = (
            "Analyze the layout and visual structure of this document image. "
            "Identify and report the following in your response:\n"
            "1. DOCUMENT_TYPE: What type of document is this? "
            "   Choose from: LOA (Letter of Authorization), NOTICE, "
            "   BUSINESS (contract/agreement), or UNKNOWN\n"
            "2. HAS_SIGNATURE: Is there a signature block or actual signature? "
            "   Answer YES or NO\n"
            "3. HAS_TABLES: Are there any tables or grid structures? "
            "   Answer YES or NO\n"
            "4. HAS_LETTERHEAD: Is there a company letterhead at the top? "
            "   Answer YES or NO\n"
            "5. HAS_LOGO: Is there a company logo, seal, or emblem? "
            "   Answer YES or NO\n"
            "6. LAYOUT_SUMMARY: Provide a 2-3 sentence description of the "
            "   document's visual layout and key structural elements.\n"
            "Format your response exactly as:\n"
            "DOCUMENT_TYPE: <value>\n"
            "HAS_SIGNATURE: <YES/NO>\n"
            "HAS_TABLES: <YES/NO>\n"
            "HAS_LETTERHEAD: <YES/NO>\n"
            "HAS_LOGO: <YES/NO>\n"
            "LAYOUT_SUMMARY: <description>"
        )

        response = model.generate_content(
            contents=[
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_b64,
                            }
                        },
                        {
                            "text": prompt
                        },
                    ]
                }
            ]
        )

        # -- Parse structured response ---------------------------------------
        response_text = response.text.strip() if response.text else ""
        print(f"OCR Layout analysis response received — parsing ")

        lines = response_text.split("\n")
        parsed = {}
        for line in lines:
            if ":" in line:
                key, _, value = line.partition(":")
                parsed[key.strip()] = value.strip()

        result["document_type_hint"] = parsed.get("DOCUMENT_TYPE", "UNKNOWN").upper()
        result["has_signature"]      = parsed.get("HAS_SIGNATURE",  "NO").upper() == "YES"
        result["has_tables"]         = parsed.get("HAS_TABLES",     "NO").upper() == "YES"
        result["has_letterhead"]     = parsed.get("HAS_LETTERHEAD", "NO").upper() == "YES"
        result["has_logo"]           = parsed.get("HAS_LOGO",       "NO").upper() == "YES"
        result["layout_summary"]     = parsed.get("LAYOUT_SUMMARY", response_text)

        print(f"OCR Layout analysis — "
              f"Type hint: {result['document_type_hint']} | "
              f"Signature: {result['has_signature']} | "
              f"Tables: {result['has_tables']} | "
              f"Letterhead: {result['has_letterhead']}")

        result["is_success"] = True
        print(f"OCR Layout analysis complete: {result['file_name']}")
        return result

    except Exception as e:
        print(f"OCR ERROR — Layout analysis failed: {type(e).__name__}: {e}")
        result["error"] = f"Layout analysis failed: {type(e).__name__}: {str(e)}"
        return result