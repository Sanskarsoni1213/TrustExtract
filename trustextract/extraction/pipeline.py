"""
Dual-Path Extraction Pipeline for TrustExtract (Loop 2)
Coordinates Path A (OCR Engine) and Path B (Vision Model), extracts tables,
generates encrypted crops for provenance, and reconciles field extractions into the target schema.
"""

from typing import Any, Dict, List, Optional, Tuple
from trustextract.schema import (
    DocumentStatus,
    FieldType,
    ExtractionAgreement,
    FieldSource,
    FieldValue,
    TableData,
    TrustExtractResult,
)
from trustextract.security.audit import audit_logger
from trustextract.security.crypto import KMSEncryptionInterface, default_kms_service
from trustextract.ingestion.normalizer import NormalizedDocument
from trustextract.extraction.path_a_ocr import PathAOCRExtractor, RawFieldExtraction
from trustextract.extraction.path_b_vision import VisionExtractorInterface, ProductionVisionExtractor, IndependentVisionModel
from trustextract.extraction.table import TableStructureExtractor
from trustextract.extraction.crop import CropManager
from trustextract.extraction.validators import DocumentValidator, DocumentValidationReport
from trustextract.extraction.confidence import ConfidencePipeline, DocumentEscalationEngine


import re
from datetime import datetime


FIELD_TYPE_REGISTRY: Dict[str, FieldType] = {
    "InvoiceId": FieldType.IDENTIFIER,
    "DocumentNumber": FieldType.IDENTIFIER,
    "PurchaseOrder": FieldType.IDENTIFIER,
    "TaxId": FieldType.IDENTIFIER,
    "AccountNumber": FieldType.IDENTIFIER,
    "TrackingNumber": FieldType.IDENTIFIER,
    "InvoiceDate": FieldType.DATE,
    "DueDate": FieldType.DATE,
    "DeliveryDate": FieldType.DATE,
    "Subtotal": FieldType.NUMERIC,
    "TotalTax": FieldType.NUMERIC,
    "InvoiceTotal": FieldType.NUMERIC,
    "UnitPrice": FieldType.NUMERIC,
    "Quantity": FieldType.NUMERIC,
    "TaxRate": FieldType.NUMERIC,
    "GrossPay": FieldType.NUMERIC,
    "TotalDeductions": FieldType.NUMERIC,
    "NetPay": FieldType.NUMERIC,
    "EmployeeId": FieldType.IDENTIFIER,
    "EmployeeName": FieldType.FREE_TEXT,
    "Department": FieldType.FREE_TEXT,
    "PayPeriodStart": FieldType.DATE,
    "PayPeriodEnd": FieldType.DATE,
    "PayDate": FieldType.DATE,
    "CustomerName": FieldType.FREE_TEXT,
    "VendorName": FieldType.FREE_TEXT,
    "CustomerAddress": FieldType.FREE_TEXT,
    "VendorAddress": FieldType.FREE_TEXT,
    "Description": FieldType.FREE_TEXT,
}


def classify_field_type(field_name: str) -> FieldType:
    """Classify a field name into IDENTIFIER, FREE_TEXT, NUMERIC, or DATE."""
    if field_name in FIELD_TYPE_REGISTRY:
        return FIELD_TYPE_REGISTRY[field_name]
    fname_lower = field_name.lower()
    if any(k in fname_lower for k in ["id", "number", "num", "code", "ref", "ssn", "ein", "po"]):
        return FieldType.IDENTIFIER
    if any(k in fname_lower for k in ["date", "time"]):
        return FieldType.DATE
    if any(k in fname_lower for k in ["total", "subtotal", "tax", "amount", "price", "rate", "qty", "quantity"]):
        return FieldType.NUMERIC
    return FieldType.FREE_TEXT


def _compute_levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _compute_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _try_parse_numeric(val: Any) -> Optional[float]:
    """Clean currency/formatting symbols and parse as float if possible."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = re.sub(r'[\$,€£\s]', '', val)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _try_parse_date(val: Any) -> Optional[str]:
    """Parse standard date strings to canonical YYYY-MM-DD."""
    if not isinstance(val, str):
        return None
    val_str = val.strip()
    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y",
        "%d-%m-%Y", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y",
        "%d %B %Y", "%d %b %Y"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(val_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def determine_field_agreement(val_a: Any, val_b: Any, field_name: str = "") -> Tuple[ExtractionAgreement, float]:
    """
    Evaluates semantic agreement between Path A and Path B extractions based on field type.

    Rules:
      1. IDENTIFIER fields (InvoiceId, AccountNumber, PO):
         - Strict exact match only after trimming leading/trailing whitespace.
         - NO case-folding, NO character collapsing, and NO Levenshtein near-match leniency.
         - Any mismatch (even a single character like 1 vs I) is OCR_NEQ_VISION (penalty 0.90).
      2. NUMERIC fields:
         - Clean currency symbols and compare numerical float value (|a-b| < 0.01).
      3. DATE fields:
         - Canonical parsed date comparison (YYYY-MM-DD).
      4. FREE_TEXT fields:
         - Exact match -> OCR_EQ_VISION (1.0).
         - Whitespace/case/punctuation normalized match -> OCR_EQ_VISION (1.0).
         - Near match (edit distance <= 2, ratio >= 0.85) -> OCR_NEAR_MATCH (penalty 0.95).
         - Divergent text -> OCR_NEQ_VISION (penalty 0.90).

    Returns:
        Tuple of (ExtractionAgreement, confidence_penalty_factor).
    """
    field_type = classify_field_type(field_name)

    # 1. IDENTIFIER: Strict exact match only
    if field_type == FieldType.IDENTIFIER:
        str_a = str(val_a).strip()
        str_b = str(val_b).strip()
        if str_a == str_b:
            return ExtractionAgreement.OCR_EQ_VISION, 1.0
        else:
            return ExtractionAgreement.OCR_NEQ_VISION, 0.90

    # 2. NUMERIC / Currency comparison
    if field_type == FieldType.NUMERIC:
        num_a = _try_parse_numeric(val_a)
        num_b = _try_parse_numeric(val_b)
        if num_a is not None and num_b is not None:
            if abs(num_a - num_b) < 0.01:
                return ExtractionAgreement.OCR_EQ_VISION, 1.0
            elif abs(num_a - num_b) <= 1.0:
                return ExtractionAgreement.OCR_NEAR_MATCH, 0.95
            else:
                return ExtractionAgreement.OCR_NEQ_VISION, 0.90

    # 3. DATE comparison
    if field_type == FieldType.DATE:
        date_a = _try_parse_date(val_a)
        date_b = _try_parse_date(val_b)
        if date_a is not None and date_b is not None:
            if date_a == date_b:
                return ExtractionAgreement.OCR_EQ_VISION, 1.0
            else:
                return ExtractionAgreement.OCR_NEQ_VISION, 0.90

    # 4. FREE_TEXT: Normalization + near-match leniency allowed
    str_a = str(val_a).strip()
    str_b = str(val_b).strip()

    if str_a == str_b:
        return ExtractionAgreement.OCR_EQ_VISION, 1.0

    # Normalize whitespace, case, and punctuation for text comparison
    norm_a = re.sub(r'[^a-zA-Z0-9]', '', str_a.lower())
    norm_b = re.sub(r'[^a-zA-Z0-9]', '', str_b.lower())

    if norm_a == norm_b and len(norm_a) > 0:
        return ExtractionAgreement.OCR_EQ_VISION, 1.0

    # Near match check: edit distance and similarity ratio
    max_len = max(len(norm_a), len(norm_b))
    if max_len > 0:
        dist = _compute_levenshtein(norm_a, norm_b)
        sim_ratio = (max_len - dist) / max_len
        if (dist <= 2 and max_len >= 5) or sim_ratio >= 0.85:
            return ExtractionAgreement.OCR_NEAR_MATCH, 0.95

    return ExtractionAgreement.OCR_NEQ_VISION, 0.90


class DualPathExtractor:
    """Orchestrates Path A (OCR) and Path B (Vision) extractions and table recognition."""

    def __init__(
        self,
        kms_service: Optional[KMSEncryptionInterface] = None,
        vision_model: Optional[VisionExtractorInterface] = None,
        storage_dir: str = "encrypted_store",
        n_consistency_passes: int = 3,
    ):
        self.kms_service = kms_service or default_kms_service
        self.crop_manager = CropManager(kms_service=self.kms_service, storage_dir=storage_dir)
        self.ocr_extractor = PathAOCRExtractor()
        self.vision_extractor = vision_model or ProductionVisionExtractor()
        self.table_extractor = TableStructureExtractor()
        self.validator = DocumentValidator()
        self.confidence_pipeline = ConfidencePipeline(n_consistency_passes=n_consistency_passes)
        self.escalation_engine = DocumentEscalationEngine()

    def process_normalized_document(
        self,
        norm_doc: NormalizedDocument,
        actor_id: str = "dual_path_worker"
    ) -> Tuple[TrustExtractResult, Dict[str, Any], DocumentValidationReport]:
        """
        Execute Loop 2 Dual-Path Extraction + Loop 3 Confidence Scoring:
        1. Run Path A (RapidOCR) on each page.
        2. Run Path B (Vision Model) on each page.
        3. Extract tabular structure from page tokens.
        4. Reconcile both paths into target schema with provenance crops and agreement flags.
        5. [Loop 3] Run business-logic validators (arithmetic, date, format).
        6. [Loop 3] Run self-consistency sampling (3 perturbed OCR passes).
        7. [Loop 3] Fuse all signals into final per-field confidence scores.
        Returns:
            Tuple of:
              - TrustExtractResult (with enriched ConfidenceBreakdown per field)
              - raw side-by-side comparison dict (Loop 2 inspection)
              - DocumentValidationReport (Loop 3 validation results)
        """
        audit_logger.log_event(
            action="DUAL_PATH_START",
            actor_id=actor_id,
            document_id=norm_doc.document_id,
            details={"page_count": len(norm_doc.pages)}
        )

        all_path_a_fields: Dict[str, RawFieldExtraction] = {}
        all_path_b_fields: Dict[str, RawFieldExtraction] = {}
        all_tables: List[TableData] = []
        final_fields: Dict[str, FieldValue] = {}

        for page in norm_doc.pages:
            # 1. Path A: OCR engine
            tokens, ocr_fields = self.ocr_extractor.extract_page_ocr(page.image, page.page_number)
            all_path_a_fields.update(ocr_fields)

            # 2. Path B: Vision model
            vision_fields = self.vision_extractor.extract_fields(page.image, page.page_number, norm_doc.document_id)
            all_path_b_fields.update(vision_fields)

            # 3. Table extraction
            page_tables = self.table_extractor.extract_tables(tokens)
            all_tables.extend(page_tables)

            # Reconcile fields for this page
            all_field_names = set(ocr_fields.keys()).union(vision_fields.keys())

            for fname in all_field_names:
                fa = ocr_fields.get(fname)
                fb = vision_fields.get(fname)

                # Determine agreement & selected value
                if fa and fb:
                    agreement, penalty_factor = determine_field_agreement(fa.value, fb.value, fname)
                    primary = fa if fa.confidence >= fb.confidence else fb
                    chosen_bbox = primary.bbox
                    reconciled_confidence = round(primary.confidence * penalty_factor, 3)
                elif fa:
                    agreement = ExtractionAgreement.SINGLE_SOURCE
                    primary = fa
                    chosen_bbox = fa.bbox
                    reconciled_confidence = primary.confidence
                else:
                    assert fb is not None
                    agreement = ExtractionAgreement.SINGLE_SOURCE
                    primary = fb
                    chosen_bbox = fb.bbox
                    reconciled_confidence = primary.confidence

                # Generate encrypted provenance crop
                crop_ref, norm_bbox = self.crop_manager.generate_crop_ref(
                    page_image=page.image,
                    bbox=chosen_bbox,
                    page_number=page.page_number,
                    document_id=norm_doc.document_id
                )

                final_fields[fname] = FieldValue(
                    value=primary.value,
                    confidence=reconciled_confidence,
                    source=FieldSource(page=page.page_number, bbox=norm_bbox, crop_ref=crop_ref),
                    extraction_agreement=agreement,
                    validation="PENDING"
                )

        audit_logger.log_event(
            action="DUAL_PATH_COMPLETE",
            actor_id=actor_id,
            document_id=norm_doc.document_id,
            details={
                "extracted_fields_count": len(final_fields),
                "tables_count": len(all_tables),
                "agreement_breakdown": {
                    "ocr==vision": sum(1 for f in final_fields.values() if f.extraction_agreement == ExtractionAgreement.OCR_EQ_VISION),
                    "ocr_near_match": sum(1 for f in final_fields.values() if f.extraction_agreement == ExtractionAgreement.OCR_NEAR_MATCH),
                    "ocr!=vision": sum(1 for f in final_fields.values() if f.extraction_agreement == ExtractionAgreement.OCR_NEQ_VISION),
                    "single_source": sum(1 for f in final_fields.values() if f.extraction_agreement == ExtractionAgreement.SINGLE_SOURCE),
                }
            }
        )

        result = TrustExtractResult(
            document_id=norm_doc.document_id,
            document_status=DocumentStatus.OK,
            unprocessable_reason=None,
            integrity_score=None,
            authenticity_flags=[],
            fields=final_fields,
            tables=all_tables,
            cross_document_conflicts=[]
        )

        # ------------------------------------------------------------------
        # Loop 3: Business-logic validation + self-consistency + confidence fusion
        # ------------------------------------------------------------------
        # Collect raw field values (not FieldValue objects) for the validator
        raw_values: Dict[str, Any] = {fname: fv.value for fname, fv in final_fields.items()}
        table_rows: List[List[Any]] = all_tables[0].rows if all_tables else []
        validation_report = self.validator.validate(raw_values, table_rows)

        # Run confidence pipeline on each page (we only have one page in this demo;
        # for multi-page docs, per-page fields are already merged into final_fields)
        for page in norm_doc.pages:
            # Re-use the already-extracted Path A fields for consistency re-runs
            page_a_fields = {
                fname: ref
                for fname, ref in all_path_a_fields.items()
                if ref.page == page.page_number
            }
            self.confidence_pipeline.score(
                result_fields=final_fields,
                page_image=page.image,
                page_number=page.page_number,
                path_a_fields=page_a_fields,
                validation_report=validation_report,
            )

        # Propagate document-level validation flags (arithmetic / date / format)
        if validation_report.summary_flags:
            result.authenticity_flags.extend([
                f"VALIDATION:{flag}" for flag in validation_report.summary_flags
            ])
            if "ARITHMETIC_INCONSISTENCY" in validation_report.summary_flags:
                reason = "Arithmetic inconsistency: Subtotal + TotalTax != InvoiceTotal"
                result.escalation_reasons.append(reason)
                result.document_status = DocumentStatus.NEEDS_REVIEW

        # Post-fusion escalation: confidence threshold + path-disagreement rules
        # Runs AFTER confidence fusion so it sees final fused scores.
        self.escalation_engine.evaluate(result)

        audit_logger.log_event(
            action="ESCALATION_EVAL",
            actor_id=actor_id,
            document_id=norm_doc.document_id,
            details={
                "document_status": result.document_status.value,
                "escalation_count": len(result.escalation_reasons),
                "escalation_reasons": result.escalation_reasons,
            }
        )

        # Build raw side-by-side dictionary for Loop 2 inspection
        side_by_side = {}
        for fname in sorted(set(all_path_a_fields.keys()).union(all_path_b_fields.keys())):
            side_by_side[fname] = {
                "path_a_ocr": {
                    "value": all_path_a_fields[fname].value if fname in all_path_a_fields else None,
                    "confidence": all_path_a_fields[fname].confidence if fname in all_path_a_fields else None,
                    "raw_text": all_path_a_fields[fname].raw_text if fname in all_path_a_fields else None,
                    "bbox": all_path_a_fields[fname].bbox if fname in all_path_a_fields else None,
                },
                "path_b_vision": {
                    "value": all_path_b_fields[fname].value if fname in all_path_b_fields else None,
                    "confidence": all_path_b_fields[fname].confidence if fname in all_path_b_fields else None,
                    "raw_text": all_path_b_fields[fname].raw_text if fname in all_path_b_fields else None,
                    "bbox": all_path_b_fields[fname].bbox if fname in all_path_b_fields else None,
                },
                "agreement": final_fields[fname].extraction_agreement.value if fname in final_fields else None
            }

        return result, side_by_side, validation_report
