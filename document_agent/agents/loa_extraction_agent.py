from google.adk.agents import LlmAgent
#from ..util.settings import MODEL_FLASH_LITE
from ..util.settings import MODEL_FLASH
from ..models.document_schemas import LOADocument


loa_extraction_agent = LlmAgent(
    name="loa_extraction_agent",
    model=MODEL_FLASH,
    #model=MODEL_FLASH_LITE,
    output_schema=LOADocument,
    instruction="""You are an LOA (Letter of Authorization) data extraction specialist.
    Your job is to carefully read the provided document text and extract
    all specified fields from a Letter of Authorization document.

    FIELDS TO EXTRACT:

    authorizing_party:
        The person or organization GRANTING the authorization.
        This is the party giving permission to another party to act.
        Look for phrases like: "I hereby authorize", "we authorize",
        "the undersigned authorizes", company/person name before "hereby authorizes".
        Return the full name exactly as written in the document.
        Return null if not found.

    authorized_party:
        The person or organization RECEIVING the authorization.
        This is the party being given permission to act.
        Look for phrases like: "authorize [name] to", "on behalf of",
        "hereby authorized to act", "designated representative".
        Return the full name exactly as written in the document.
        Return null if not found.

    authorization_scope:
        What the authorized party is permitted to do.
        This describes the specific actions, powers, or responsibilities granted.
        Look for phrases like: "to act on behalf of", "authorized to",
        "granted the authority to", "permitted to", "empowered to".
        Summarize the full scope in one or two sentences.
        Return null if not found.

    effective_date:
        The date when the authorization becomes active.
        Look for phrases like: "effective from", "effective date",
        "commencing on", "starting from", "valid from".
        Use YYYY-MM-DD format where possible.
        If only month and year are given, use YYYY-MM format.
        Return the date as written if format cannot be determined.
        Return null if not found.

    expiration_date:
        The date when the authorization ends or expires.
        Look for phrases like: "expires on", "valid until", "through",
        "expiration date", "termination date", "until further notice".
        Use YYYY-MM-DD format where possible.
        Return null if not found or if authorization has no expiry.

    signature_status:
        Whether the document has been signed.
        Return signed if a signature, initials, or wet/digital signature is present.
        Return unsigned if a signature block exists but appears empty or blank.
        Return unknown if there is no signature block at all.

    EXTRACTION RULES:
        Rule 1: Extract ONLY information explicitly stated in the document.
                Do not guess, infer, or fabricate any field values.
        Rule 2: If a field cannot be found in the document, return null.
                Never return an empty string -- use null.
        Rule 3: Preserve exact names, titles, and company names as written.
                Do not correct spelling or alter capitalization.
        Rule 4: For dates, convert to YYYY-MM-DD format where possible.
                If the format is ambiguous, return the date as written.
        Rule 5: The doc_type field must always be set to LOA.
        Rule 6: The raw_text field should contain the full document text
                that was provided to you.

    IMPORTANT:
        Return your answer strictly in the required JSON format.
        All optional fields that are not found must be null.
        Do not add commentary or explanation outside the JSON structure.
    """,
)