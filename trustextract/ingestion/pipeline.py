"""
Ingestion & Quality Gate Pipeline for TrustExtract (Loop 1)
Orchestrates untrusted input validation, format normalization, security checks, and quality gates.
"""

import hashlib
import os
from io import BytesIO
from typing import Optional, Tuple
from trustextract.schema import DocumentStatus, TrustExtractResult
from trustextract.security.audit import audit_logger
from trustextract.security.crypto import KMSEncryptionInterface, default_kms_service
from trustextract.security.sandbox import SafeParserBoundary, SecurityValidationError
from trustextract.ingestion.normalizer import DocumentNormalizer, NormalizedDocument
from trustextract.ingestion.quality import QualityChecker, QualityThresholds, QualityReport


class IngestionPipeline:
    """Ingestion & Format Normalization Pipeline with Quality Gates and Security Isolation."""

    def __init__(
        self,
        quality_thresholds: Optional[QualityThresholds] = None,
        kms_service: Optional[KMSEncryptionInterface] = None,
        storage_dir: str = "encrypted_store"
    ):
        self.quality_checker = QualityChecker(quality_thresholds or QualityThresholds())
        self.kms_service: KMSEncryptionInterface = kms_service or default_kms_service
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def process_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        actor_id: str = "system_ingestion_worker"
    ) -> Tuple[TrustExtractResult, Optional[NormalizedDocument]]:
        """
        Execute Loop 1 Ingestion:
        1. Audit log ingestion initiation.
        2. Sandbox & validate untrusted input format/size.
        3. Normalize input to standardized page images + text layers.
        4. Run quality checks (resolution, blur, text density, brightness).
        5. Return unprocessable state immediately if any check fails, or OK with normalized doc.
        """
        if not os.path.exists(file_path):
            doc_id = document_id or "unknown_doc"
            audit_logger.log_event(
                action="INGEST_FILE_NOT_FOUND",
                actor_id=actor_id,
                document_id=doc_id,
                details={"file_path_hash": hashlib.sha256(file_path.encode()).hexdigest()[:16]}
            )
            return TrustExtractResult(
                document_id=doc_id,
                document_status=DocumentStatus.UNPROCESSABLE,
                unprocessable_reason=f"Input file not found at path: {os.path.basename(file_path)}"
            ), None

        # Compute document ID if not provided (SHA-256 of file content)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        doc_id = document_id or f"doc_{file_hash[:16]}"

        audit_logger.log_event(
            action="INGEST_START",
            actor_id=actor_id,
            document_id=doc_id,
            details={"file_size_bytes": len(file_bytes), "file_sha256": file_hash}
        )

        # Step 1: Sandbox & Untrusted Input Validation
        try:
            detected_format = SafeParserBoundary.inspect_untrusted_input(file_path)
        except (SecurityValidationError, Exception) as exc:
            reason = f"Security or format validation failed: {str(exc)}"
            audit_logger.log_event(
                action="INGEST_SECURITY_REJECTED",
                actor_id=actor_id,
                document_id=doc_id,
                details={"rejection_reason": reason}
            )
            return TrustExtractResult(
                document_id=doc_id,
                document_status=DocumentStatus.UNPROCESSABLE,
                unprocessable_reason=reason
            ), None

        # Step 2: Format Normalization
        try:
            normalized_doc = DocumentNormalizer.normalize(file_path, detected_format, doc_id)
            # Ensure full cryptographic SHA-256 hash is reliably attached to document metadata
            normalized_doc.metadata["file_sha256"] = file_hash
        except Exception as exc:
            reason = f"Document normalization error: {str(exc)}"
            audit_logger.log_event(
                action="INGEST_NORMALIZATION_FAILED",
                actor_id=actor_id,
                document_id=doc_id,
                details={"error": str(exc)}
            )
            return TrustExtractResult(
                document_id=doc_id,
                document_status=DocumentStatus.UNPROCESSABLE,
                unprocessable_reason=reason
            ), None

        if not normalized_doc.pages:
            reason = "Document contains no readable pages."
            return TrustExtractResult(
                document_id=doc_id,
                document_status=DocumentStatus.UNPROCESSABLE,
                unprocessable_reason=reason
            ), None

        # Step 3: Quality Check on Normalized Pages
        for page in normalized_doc.pages:
            report: QualityReport = self.quality_checker.assess_page_image(page.image)
            if not report.passed:
                reason = f"Page {page.page_number} quality failure: " + "; ".join(report.failure_reasons)
                audit_logger.log_event(
                    action="INGEST_QUALITY_REJECTED",
                    actor_id=actor_id,
                    document_id=doc_id,
                    details={"page_number": page.page_number, "metrics": report.metrics, "failure_reasons": report.failure_reasons}
                )
                return TrustExtractResult(
                    document_id=doc_id,
                    document_status=DocumentStatus.UNPROCESSABLE,
                    unprocessable_reason=reason
                ), None

        # Step 4: Encrypt Normalized Pages at Rest
        for page in normalized_doc.pages:
            buf = BytesIO()
            page.image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            encrypted_bytes = self.kms_service.encrypt(
                img_bytes, context={"doc_id": doc_id, "page": str(page.page_number)}
            )
            enc_file_path = os.path.join(self.storage_dir, f"{doc_id}_page_{page.page_number}.enc")
            with open(enc_file_path, "wb") as f_enc:
                f_enc.write(encrypted_bytes)

        audit_logger.log_event(
            action="INGEST_SUCCESS",
            actor_id=actor_id,
            document_id=doc_id,
            details={"page_count": len(normalized_doc.pages), "detected_format": detected_format}
        )

        result = TrustExtractResult(
            document_id=doc_id,
            document_status=DocumentStatus.OK,
            unprocessable_reason=None
        )
        return result, normalized_doc
