"""
Integrity & Tamper Pipeline for TrustExtract (Loop 4)
Orchestrates ELA, PDF metadata forensics, font consistency analysis,
cryptographic & perceptual duplicate detection, and AI synthesis checks.
Computes the document-level integrity_score (0.0 - 1.0) and populates authenticity_flags[].
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional
from PIL import Image

from trustextract.schema import DocumentStatus, TrustExtractResult
from trustextract.security.audit import audit_logger
from trustextract.ingestion.normalizer import NormalizedDocument
from trustextract.integrity.ela import ErrorLevelAnalyzer, ELAResult
from trustextract.integrity.pdf_forensics import PDFMetadataForensicDetector, PDFMetadataResult
from trustextract.integrity.font_analysis import FontConsistencyDetector, FontConsistencyResult
from trustextract.integrity.duplicate_detector import DocumentDeduplicator, DuplicateCheckResult
from trustextract.integrity.ai_detector import AIGenerationDetector, AISynthesisResult


@dataclass
class IntegrityReport:
    document_id: str
    integrity_score: float
    authenticity_flags: List[str]
    ela_result: Optional[ELAResult] = None
    pdf_result: Optional[PDFMetadataResult] = None
    font_result: Optional[FontConsistencyResult] = None
    duplicate_result: Optional[DuplicateCheckResult] = None
    ai_result: Optional[AISynthesisResult] = None
    details: Dict[str, Any] = field(default_factory=dict)


# Penalty weightings for document-level integrity score calculation
INTEGRITY_FLAG_PENALTIES: Dict[str, float] = {
    "DUPLICATE_EXACT_MATCH": 0.50,
    "DUPLICATE_NEAR_MATCH": 0.35,
    "ELA_ANOMALY_DETECTED": 0.30,
    "FONT_INCONSISTENCY": 0.25,
    "METADATA_SUSPICIOUS_EDIT_HISTORY": 0.20,
    "AI_GENERATION_SUSPECTED": 0.35,
    "METADATA_INCONSISTENT_CAPTURE_ORIGIN": 0.15,
}

# Threshold below which document integrity escalates document_status to NEEDS_REVIEW
INTEGRITY_ESCALATION_THRESHOLD = 0.70


class IntegrityPipeline:
    """
    Loop 4 Integrity and Tamper Detection Orchestrator.
    Evaluates documents across multiple independent physical and forensic dimensions:
      1. Error Level Analysis (differential recompression artifacting)
      2. PDF Metadata Forensics (creation vs modification timelines, editing software)
      3. Typographical & Font Consistency (character pitch, bounding box sizing, baseline jitter)
      4. Exact (SHA-256) and Near-Duplicate (pHash) Deduplication
      5. AI-Generation & Capture Claim Metadata Consistency
    """

    def __init__(
        self,
        deduplicator: Optional[DocumentDeduplicator] = None,
        ela_quality: int = 90,
        time_delta_hours: float = 2.0,
        font_deviation_sigma: float = 2.8,
    ):
        self.ela_analyzer = ErrorLevelAnalyzer(quality=ela_quality)
        self.pdf_forensics = PDFMetadataForensicDetector(suspicious_time_delta_hours=time_delta_hours)
        self.font_analyzer = FontConsistencyDetector(deviation_sigma_threshold=font_deviation_sigma)
        self.deduplicator = deduplicator or DocumentDeduplicator()
        self.ai_detector = AIGenerationDetector()

    def evaluate_document(
        self,
        norm_doc: NormalizedDocument,
        raw_file_bytes: Optional[bytes] = None,
        file_path: Optional[str] = None,
        all_tokens: Optional[List[Dict[str, Any]]] = None,
        extracted_fields: Optional[Dict[str, Any]] = None,
        claimed_capture_method: Optional[str] = None,
        actor_id: str = "integrity_worker"
    ) -> IntegrityReport:
        """
        Run all integrity checks on a document and return an IntegrityReport.
        """
        doc_id = norm_doc.document_id
        page_images = [p.image for p in norm_doc.pages]
        primary_image = page_images[0] if page_images else Image.new("RGB", (100, 100))

        audit_logger.log_event(
            action="INTEGRITY_EVAL_START",
            actor_id=actor_id,
            document_id=doc_id,
            details={"page_count": len(page_images), "original_format": norm_doc.original_format}
        )

        all_flags: List[str] = []
        details: Dict[str, Any] = {}

        # 1. Error Level Analysis (ELA) on page images
        ela_res = self.ela_analyzer.analyze(primary_image)
        if ela_res.has_anomaly:
            for flag in ela_res.flags:
                if flag not in all_flags:
                    all_flags.append(flag)
            details["ela"] = {
                "peak_error": ela_res.peak_error,
                "mean_error": ela_res.mean_error,
                "anomalous_regions": [
                    {"bbox": r.bbox, "anomaly_score": r.anomaly_score, "description": r.description}
                    for r in ela_res.anomalous_regions
                ]
            }

        # 2. PDF Metadata Forensics
        pdf_res = None
        if norm_doc.original_format.lower() == "pdf" or (file_path and file_path.lower().endswith(".pdf")):
            if file_path and os.path.exists(file_path):
                pdf_res = self.pdf_forensics.analyze_file(file_path)
            else:
                pdf_res = self.pdf_forensics.analyze_metadata(norm_doc.metadata)

            if pdf_res.has_anomaly:
                for flag in pdf_res.flags:
                    if flag not in all_flags:
                        all_flags.append(flag)
                details["pdf_metadata"] = pdf_res.details

        # 3. Font and Typographical Consistency Analysis
        font_res = None
        if all_tokens and extracted_fields:
            font_res = self.font_analyzer.analyze_tokens_and_fields(all_tokens, extracted_fields)
            if font_res.has_anomaly:
                for flag in font_res.flags:
                    if flag not in all_flags:
                        all_flags.append(flag)
                details["font_inconsistency"] = [
                    {
                        "field_name": af.field_name,
                        "token_text": af.token_text,
                        "char_width": af.char_width,
                        "token_height": af.token_height,
                        "deviation_score": af.deviation_score,
                        "reason": af.reason,
                        "bbox": af.bbox
                    }
                    for af in font_res.anomalous_fields
                ]

        # 4. Deduplication & Perceptual Hashing Check
        dup_res = None
        if raw_file_bytes is not None or file_path is not None:
            data_bytes = raw_file_bytes
            if data_bytes is None and file_path and os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    data_bytes = f.read()

            if data_bytes:
                dup_res = self.deduplicator.check_and_register(
                    document_id=doc_id,
                    file_bytes=data_bytes,
                    page_images=page_images
                )
                if dup_res.is_duplicate:
                    for flag in dup_res.flags:
                        if flag not in all_flags:
                            all_flags.append(flag)
                    details["duplicate_matches"] = [
                        {
                            "matched_document_id": m.matched_document_id,
                            "match_type": m.match_type,
                            "similarity_percentage": m.similarity_percentage,
                            "description": m.description
                        }
                        for m in dup_res.matches
                    ]

        # 5. AI Generation & Capture Claim Metadata Consistency
        ai_res = self.ai_detector.analyze_image_and_metadata(
            image=primary_image,
            doc_metadata=norm_doc.metadata,
            claimed_capture_method=claimed_capture_method
        )
        if ai_res.is_suspicious:
            for flag in ai_res.flags:
                if flag not in all_flags:
                    all_flags.append(flag)
            details["ai_generation"] = ai_res.details

        # 6. Calculate document-level integrity score (0.0 to 1.0)
        total_penalty = sum(INTEGRITY_FLAG_PENALTIES.get(fl, 0.20) for fl in all_flags)
        integrity_score = round(max(0.0, min(1.0, 1.0 - total_penalty)), 3)

        audit_logger.log_event(
            action="INTEGRITY_EVAL_COMPLETE",
            actor_id=actor_id,
            document_id=doc_id,
            details={
                "integrity_score": integrity_score,
                "flags_count": len(all_flags),
                "authenticity_flags": all_flags
            }
        )

        return IntegrityReport(
            document_id=doc_id,
            integrity_score=integrity_score,
            authenticity_flags=all_flags,
            ela_result=ela_res,
            pdf_result=pdf_res,
            font_result=font_res,
            duplicate_result=dup_res,
            ai_result=ai_res,
            details=details
        )

    def attach_to_result(
        self,
        result: TrustExtractResult,
        report: IntegrityReport
    ) -> None:
        """
        Attaches integrity score and authenticity flags to TrustExtractResult,
        and escalates document_status if integrity thresholds are breached.
        """
        result.integrity_score = report.integrity_score
        result.authenticity_flags = report.authenticity_flags

        # If integrity indicates tampering, duplicate submission, or severe anomaly:
        if report.integrity_score < INTEGRITY_ESCALATION_THRESHOLD or len(report.authenticity_flags) > 0:
            result.document_status = DocumentStatus.NEEDS_REVIEW
