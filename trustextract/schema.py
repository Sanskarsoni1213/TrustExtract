"""
TrustExtract Core Output Schema
Follows Azure AI Document Intelligence naming conventions and strict target JSON schema.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    OK = "OK"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    UNPROCESSABLE = "UNPROCESSABLE"


class FieldType(str, Enum):
    IDENTIFIER = "IDENTIFIER"
    FREE_TEXT = "FREE_TEXT"
    NUMERIC = "NUMERIC"
    DATE = "DATE"


class ExtractionAgreement(str, Enum):
    OCR_EQ_VISION = "ocr==vision"
    OCR_NEAR_MATCH = "ocr_near_match"
    OCR_NEQ_VISION = "ocr!=vision"
    SINGLE_SOURCE = "single_source"


class FieldSource(BaseModel):
    page: int
    bbox: List[float] = Field(description="[x, y, width, height] normalized or pixel coordinates")
    crop_ref: str = Field(description="Encrypted or secure reference identifier to source image crop")


class ConfidenceBreakdown(BaseModel):
    """Per-field signal components feeding into the fused confidence score."""
    native_ocr: float = Field(description="Raw confidence from the winning extraction path")
    agreement_signal: float = Field(description="1.0=both paths agree, 0.85=near match, 0.8=single-source, 0.6=disagree")
    consistency_signal: Optional[float] = Field(default=None, description="Fraction of multi-pass OCR runs that produced the same value, or None if single-source")
    validation_signal: float = Field(description="1.0=all business-logic checks pass, 0.0=any check failed")
    validation_flags: List[str] = Field(default_factory=list, description="Named results of each validator (e.g. ARITHMETIC_OK, DATE_FAIL)")
    final: float = Field(description="Fused confidence after weighting and validation caps")


class FieldValue(BaseModel):
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    source: Optional[FieldSource] = None
    extraction_agreement: ExtractionAgreement = ExtractionAgreement.SINGLE_SOURCE
    validation: str = "PENDING"
    confidence_breakdown: Optional[ConfidenceBreakdown] = None


class TableData(BaseModel):
    rows: List[List[Any]] = Field(default_factory=list)
    confidence_per_cell: List[List[float]] = Field(default_factory=list)


class CrossDocumentConflict(BaseModel):
    entity_key: str
    field_name: str
    document_ids: List[str]
    conflicting_values: List[Dict[str, Any]]
    resolution_status: str = "UNRESOLVED"


class TrustExtractResult(BaseModel):
    document_id: str
    document_status: DocumentStatus
    unprocessable_reason: Optional[str] = None
    integrity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    authenticity_flags: List[str] = Field(default_factory=list)
    # Explicit list of every reason document_status was escalated to NEEDS_REVIEW.
    # Each entry is a human-readable string naming the rule and the triggering value.
    # Empty list means document_status is OK (or UNPROCESSABLE for a different reason).
    escalation_reasons: List[str] = Field(default_factory=list)
    fields: Dict[str, FieldValue] = Field(default_factory=dict)
    tables: List[TableData] = Field(default_factory=list)
    cross_document_conflicts: List[CrossDocumentConflict] = Field(default_factory=list)

    def to_output_json(self) -> Dict[str, Any]:
        """Convert result to clean dictionary matching exact target specification."""
        return self.model_dump(mode="json")
