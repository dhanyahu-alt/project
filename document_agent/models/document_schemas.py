from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


# ============================================================================
# ENUMS
# ============================================================================

class DocumentType(str, Enum):
    """Supported document types for classification."""
    LOA      = "LOA"       # Letter of Authorization
    NOTICE   = "NOTICE"    # Legal / Compliance Notice
    BUSINESS = "BUSINESS"  # Business Agreement / Contract
    UNKNOWN  = "UNKNOWN"   # Could not be classified


class SignatureStatus(str, Enum):
    """Signature detection status on a document."""
    SIGNED   = "signed"
    UNSIGNED = "unsigned"
    UNKNOWN  = "unknown"


# ============================================================================
# BASE DOCUMENT
# ============================================================================

class DocumentBase(BaseModel):
    """Base fields shared across all document types."""

    doc_id:       str            = Field(...,  description="Unique document identifier")
    file_path:    str            = Field(...,  description="Absolute path to the source file")
    file_name:    str            = Field(...,  description="Filename only e.g. LoA1.pdf")
    doc_type:     DocumentType   = Field(...,  description="Classified document type")
    raw_text:     str            = Field("",   description="Full extracted text from PDF")
    page_count:   int            = Field(0,    description="Total number of pages")
    processed_at: datetime       = Field(default_factory=datetime.utcnow,
                                         description="UTC timestamp of processing")
    session_id:   str            = Field("",   description="ADK session ID for traceability")


# ============================================================================
# DOCUMENT TYPE SCHEMAS
# ============================================================================

class LOADocument(DocumentBase):
    """Extracted fields from a Letter of Authorization document."""

    authorizing_party:    Optional[str] = Field(
        None,
        description="The party granting the authorization (person or organization)"
    )
    authorized_party:     Optional[str] = Field(
        None,
        description="The party receiving the authorization (person or organization)"
    )
    authorization_scope:  Optional[str] = Field(
        None,
        description="What the authorized party is permitted to do"
    )
    effective_date:       Optional[str] = Field(
        None,
        description="Date when the authorization becomes effective (YYYY-MM-DD if possible)"
    )
    expiration_date:      Optional[str] = Field(
        None,
        description="Date when the authorization expires (YYYY-MM-DD if possible)"
    )
    signature_status:     Optional[SignatureStatus] = Field(
        SignatureStatus.UNKNOWN,
        description="Whether the document has been signed"
    )


class NoticeDocument(DocumentBase):
    """Extracted fields from a Notice or legal notification document."""

    notice_type:      Optional[str]       = Field(
        None,
        description="Type of notice e.g. Cease and Desist, Compliance, Legal Notice"
    )
    recipient:        Optional[str]       = Field(
        None,
        description="Name or entity the notice is addressed to"
    )
    subject:          Optional[str]       = Field(
        None,
        description="Subject or title of the notice"
    )
    important_dates:  Optional[List[str]] = Field(
        None,
        description="All significant dates mentioned in the notice with context"
    )
    action_required:  Optional[str]       = Field(
        None,
        description="Summary of what action the recipient must take"
    )
    deadline:         Optional[str]       = Field(
        None,
        description="Most critical deadline date for required action"
    )


class BusinessDocument(DocumentBase):
    """Extracted fields from a Business Agreement or Contract document."""

    parties_involved:  Optional[List[str]] = Field(
        None,
        description="All named parties in the agreement (companies and individuals)"
    )
    key_terms:         Optional[List[str]] = Field(
        None,
        description="Key contractual terms, obligations, or conditions"
    )
    financial_amounts: Optional[List[str]] = Field(
        None,
        description="All financial amounts with currency and context"
    )
    reference_numbers: Optional[List[str]] = Field(
        None,
        description="Contract IDs, invoice numbers, PO numbers, reference codes"
    )


# ============================================================================
# CLASSIFICATION RESULT
# ============================================================================

class ClassificationResult(BaseModel):
    """Result returned by the Classification Agent."""

    doc_type:                   DocumentType = Field(
        ...,
        description="The identified document type"
    )
    confidence_score:           float        = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning:                  str          = Field(
        ...,
        description="Explanation of why this classification was chosen"
    )
    suggested_extraction_agent: str          = Field(
        ...,
        description=(
            "Name of the extraction agent to call next: "
            "loa_extraction_agent / notice_extraction_agent / "
            "business_doc_agent / none"
        )
    )


# ============================================================================
# VALIDATION RESULT
# ============================================================================

class ValidationResult(BaseModel):
    """Result returned by the Validation Agent."""

    is_valid:         bool             = Field(
        ...,
        description="True if all required fields are present and consistent"
    )
    confidence_score: float            = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall extraction confidence score between 0.0 and 1.0"
    )
    missing_fields:   List[str]        = Field(
        default_factory=list,
        description="List of required fields that were not found in the document"
    )
    review_required:  bool             = Field(
        ...,
        description=(
            "True if human review is needed. "
            "Triggered when confidence < 0.90 or required fields are missing"
        )
    )
    review_reason:    Optional[str]    = Field(
        None,
        description="Explanation of why human review is required"
    )
    recommendations:  Optional[List[str]] = Field(
        None,
        description="Suggested improvements or fields to re-examine"
    )