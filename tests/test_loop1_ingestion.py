"""
Unit & Integration Tests for Loop 1: Ingestion & Format Normalization
Tests security boundary, format normalization, quality checks, and output schema.
"""

import json
import os
import pytest
from trustextract.schema import DocumentStatus, TrustExtractResult
from trustextract.security.sandbox import SafeParserBoundary, SecurityValidationError
from trustextract.security.crypto import AESGCMKmsEncryptionService
from trustextract.security.audit import AuditLogger
from trustextract.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def pipeline():
    return IngestionPipeline(storage_dir="tests_encrypted_store")


def test_clean_pdf_proceeds(pipeline):
    """Clean PDF invoice must pass quality checks and proceed with status OK."""
    result, norm_doc = pipeline.process_document(
        "sample_data/clean_invoice.pdf",
        document_id="doc_clean_invoice_001",
        actor_id="test_worker_1"
    )

    assert result.document_status == DocumentStatus.OK
    assert result.unprocessable_reason is None
    assert result.document_id == "doc_clean_invoice_001"
    assert norm_doc is not None
    assert len(norm_doc.pages) == 1
    assert norm_doc.original_format == "pdf"
    assert norm_doc.pages[0].text_layer is not None
    assert "VendorName" in norm_doc.pages[0].text_layer

    # Check output JSON schema validity
    out_json = result.to_output_json()
    assert out_json["document_status"] == "OK"
    assert out_json["unprocessable_reason"] is None
    assert "fields" in out_json
    assert "tables" in out_json
    assert "cross_document_conflicts" in out_json


def test_deliberately_blurry_sample_returns_unprocessable(pipeline):
    """Deliberately blurry document must fail blur quality check and return UNPROCESSABLE with reason."""
    result, norm_doc = pipeline.process_document(
        "sample_data/deliberately_blurry_sample.jpg",
        document_id="doc_blurry_001",
        actor_id="test_worker_1"
    )

    assert result.document_status == DocumentStatus.UNPROCESSABLE
    assert result.unprocessable_reason is not None
    assert any(q in result.unprocessable_reason for q in ["Blur check failed", "Text density check failed", "quality failure"])
    assert norm_doc is None

    out_json = result.to_output_json()
    assert out_json["document_status"] == "UNPROCESSABLE"
    assert isinstance(out_json["unprocessable_reason"], str)


def test_low_res_sample_returns_unprocessable(pipeline):
    """Tiny thumbnail image must fail resolution check and return UNPROCESSABLE with reason."""
    result, norm_doc = pipeline.process_document(
        "sample_data/low_res_sample.png",
        document_id="doc_lowres_001",
        actor_id="test_worker_1"
    )

    assert result.document_status == DocumentStatus.UNPROCESSABLE
    assert result.unprocessable_reason is not None
    assert "Resolution check failed" in result.unprocessable_reason
    assert norm_doc is None


def test_extension_spoofing_rejected(pipeline, tmp_path):
    """An executable or plain text file disguised as .pdf must be rejected by magic bytes."""
    fake_pdf = tmp_path / "malicious.pdf"
    fake_pdf.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00FAKE_EXECUTABLE_PAYLOAD")

    result, norm_doc = pipeline.process_document(str(fake_pdf), actor_id="test_worker_1")
    assert result.document_status == DocumentStatus.UNPROCESSABLE
    assert "magic bytes did not match" in result.unprocessable_reason


def test_docx_normalization_and_processing(pipeline):
    """Clean DOCX document must be normalized and pass quality checks."""
    result, norm_doc = pipeline.process_document(
        "sample_data/clean_agreement.docx",
        document_id="doc_clean_docx_001",
        actor_id="test_worker_1"
    )

    assert result.document_status == DocumentStatus.OK
    assert result.unprocessable_reason is None
    assert norm_doc is not None
    assert len(norm_doc.pages) == 1
    assert norm_doc.original_format == "docx"
    assert norm_doc.pages[0].text_layer is not None
    assert "VendorName: Apex Technologies" in norm_doc.pages[0].text_layer


def test_encryption_at_rest():
    """Verify AES-256-GCM encryption service protects data at rest."""
    kms = AESGCMKmsEncryptionService()
    sensitive_data = b"CONFIDENTIAL_CUSTOMER_SSN_999-00-1111"
    encrypted = kms.encrypt(sensitive_data, context={"doc_id": "test_123"})

    assert encrypted != sensitive_data
    # Nonce (12 bytes) + Ciphertext + Tag (16 bytes)
    assert len(encrypted) >= len(sensitive_data) + 28

    decrypted = kms.decrypt(encrypted, context={"doc_id": "test_123"})
    assert decrypted == sensitive_data

    # Decryption fails if authenticated context is tampered
    with pytest.raises(Exception):
        kms.decrypt(encrypted, context={"doc_id": "tampered_id"})


def test_zero_pii_audit_logging(tmp_path):
    """Ensure audit logger strictly suppresses and hashes sensitive PII values."""
    audit_file = tmp_path / "audit.log"
    logger = AuditLogger(log_file=str(audit_file))

    logger.log_event(
        action="TEST_ACTION",
        actor_id="actor_42",
        document_id="doc_999",
        details={
            "status": "PROCESSED",
            "name": "Jane Doe",
            "extracted_text": "Sensitive Contract Body Text",
            "ssn": "000-12-3456"
        }
    )

    with open(audit_file, "r") as f:
        log_content = f.read()

    assert "Jane Doe" not in log_content
    assert "Sensitive Contract Body Text" not in log_content
    assert "000-12-3456" not in log_content
    assert "[REDACTED:sha256=" in log_content
    assert "actor_42" in log_content


def test_borderline_readable_samples_pass(pipeline):
    """Slightly blurry, skewed, uneven lighting, and compressed samples should pass if readable."""
    for sample in [
        "sample_data/audit_samples/borderline_slight_blur.png",
        "sample_data/audit_samples/borderline_skew_6deg.png",
        "sample_data/audit_samples/borderline_uneven_lighting.png",
        "sample_data/audit_samples/borderline_jpeg_q25.jpg",
        "sample_data/audit_samples/borderline_jpeg_q10.jpg",
    ]:
        res, norm = pipeline.process_document(sample, actor_id="test_worker_1")
        assert res.document_status == DocumentStatus.OK, f"Failed for {sample}: {res.unprocessable_reason}"
        assert norm is not None


def test_scanned_pdf_no_text_layer(pipeline):
    """Scanned PDF with image only must be normalized without text layer shortcut."""
    res, norm = pipeline.process_document("sample_data/audit_samples/scanned_image_only.pdf", actor_id="test_worker_1")
    assert res.document_status == DocumentStatus.OK
    assert norm is not None
    assert norm.pages[0].text_layer is None


def test_truncated_pdf_unprocessable(pipeline):
    """Truncated/corrupt PDF must fail cleanly with UNPROCESSABLE."""
    res, norm = pipeline.process_document("sample_data/audit_samples/truncated_corrupt.pdf", actor_id="test_worker_1")
    assert res.document_status == DocumentStatus.UNPROCESSABLE
    assert "normalization error" in res.unprocessable_reason.lower()
    assert norm is None


def test_password_protected_pdf_unprocessable(pipeline):
    """Password-protected PDF must fail cleanly with explicit permission/encryption error."""
    res, norm = pipeline.process_document("sample_data/audit_samples/password_protected.pdf", actor_id="test_worker_1")
    assert res.document_status == DocumentStatus.UNPROCESSABLE
    assert "encrypted or password-protected" in res.unprocessable_reason
    assert norm is None


def test_file_hash_full_sha256_persisted(pipeline):
    """Full 64-hex-character SHA-256 hash must be recorded in normalized_doc metadata."""
    res, norm = pipeline.process_document("sample_data/clean_receipt.png", actor_id="test_worker_1")
    assert res.document_status == DocumentStatus.OK
    assert norm is not None
    file_sha256 = norm.metadata.get("file_sha256")
    assert file_sha256 is not None
    assert len(file_sha256) == 64
    assert all(c in "0123456789abcdefABCDEF" for c in file_sha256)


def test_kms_dependency_injection(tmp_path):
    """Verify IngestionPipeline works seamlessly with any KMSEncryptionInterface implementation."""
    from trustextract.security.crypto import KMSEncryptionInterface

    class MockCloudKmsService(KMSEncryptionInterface):
        def __init__(self):
            self.calls = []

        def encrypt(self, plaintext: bytes, context=None) -> bytes:
            self.calls.append(("encrypt", context))
            return b"CLOUD_KMS_CIPHERTEXT::" + plaintext[:16]

        def decrypt(self, ciphertext: bytes, context=None) -> bytes:
            self.calls.append(("decrypt", context))
            return b"DECRYPTED_MOCK"

    mock_kms = MockCloudKmsService()
    custom_pipeline = IngestionPipeline(kms_service=mock_kms, storage_dir=str(tmp_path / "enc"))

    res, norm = custom_pipeline.process_document("sample_data/clean_receipt.png", actor_id="test_worker_1")
    assert res.document_status == DocumentStatus.OK
    assert len(mock_kms.calls) > 0
    assert mock_kms.calls[0][0] == "encrypt"
