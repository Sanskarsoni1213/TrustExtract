"""
Unit & Integration Tests for Loop 2: Dual-Path Extraction
Tests Path A (OCR), Path B (Vision), Table Extraction, Provenance Crops, and Agreement Reconciliation.
"""

import os
import pytest
from trustextract.schema import DocumentStatus, ExtractionAgreement, TrustExtractResult
from trustextract.ingestion.pipeline import IngestionPipeline
from trustextract.extraction.pipeline import DualPathExtractor
from trustextract.extraction.path_b_vision import IndependentVisionModel
from trustextract.security.crypto import AESGCMKmsEncryptionService


@pytest.fixture
def dual_pipeline(tmp_path):
    storage_dir = str(tmp_path / "enc_store")
    kms = AESGCMKmsEncryptionService()
    ingest = IngestionPipeline(kms_service=kms, storage_dir=storage_dir)
    extract = DualPathExtractor(
        kms_service=kms,
        vision_model=IndependentVisionModel(induce_disagreement_on="InvoiceId"),
        storage_dir=storage_dir
    )
    return ingest, extract, storage_dir, kms


def test_dual_path_extraction_and_agreement(dual_pipeline):
    """Test Path A and Path B run on clean_invoice.pdf, extract fields, and detect agreement/disagreement."""
    ingest, extract, storage_dir, kms = dual_pipeline

    result_ingest, norm_doc = ingest.process_document(
        "sample_data/clean_invoice.pdf",
        document_id="doc_test_invoice_002",
        actor_id="test_worker_2"
    )
    assert result_ingest.document_status == DocumentStatus.OK
    assert norm_doc is not None

    result, side_by_side, _validation_report = extract.process_normalized_document(norm_doc, actor_id="test_worker_2")

    # 1. Output Schema Integrity (escalates to NEEDS_REVIEW due to InvoiceId disagreement)
    assert result.document_status == DocumentStatus.NEEDS_REVIEW
    assert any("InvoiceId" in r for r in result.escalation_reasons)
    assert len(result.fields) >= 4

    # 2. Side-by-Side Verification for DoD
    assert "VendorName" in side_by_side
    assert "InvoiceId" in side_by_side
    assert "InvoiceTotal" in side_by_side

    # 3. Check Agreement and Disagreement
    # VendorName should match: Contoso Logistics Inc.
    assert result.fields["VendorName"].extraction_agreement == ExtractionAgreement.OCR_EQ_VISION
    assert result.fields["VendorName"].value == "Contoso Logistics Inc."

    # InvoiceId must strictly disagree (Path A has 9841, Path B has 984I) without near-match leniency
    assert result.fields["InvoiceId"].extraction_agreement == ExtractionAgreement.OCR_NEQ_VISION
    assert side_by_side["InvoiceId"]["path_a_ocr"]["value"] == "INV-2024-9841"
    assert side_by_side["InvoiceId"]["path_b_vision"]["value"] == "INV-2024-984I"
    assert side_by_side["InvoiceId"]["agreement"] == "ocr!=vision"

    # 4. Check Table Extraction
    assert len(result.tables) >= 1
    table = result.tables[0]
    assert len(table.rows) >= 3
    assert len(table.confidence_per_cell) == len(table.rows)
    # Check that row cells contain descriptions and prices
    descriptions = [row[0] for row in table.rows]
    assert any("Cloud Server Hosting" in str(d) for d in descriptions)
    assert any("Database Backup" in str(d) for d in descriptions)

    # 5. Check Provenance Crops & Encryption at Rest
    for field_name, fld in result.fields.items():
        assert fld.source is not None
        assert fld.source.crop_ref.startswith("crop://")
        assert len(fld.source.bbox) == 4
        crop_id = fld.source.crop_ref.replace("crop://", "")
        crop_path = os.path.join(storage_dir, f"{crop_id}.enc")
        assert os.path.exists(crop_path), f"Crop file missing: {crop_path}"
        assert os.path.getsize(crop_path) > 0

        # Verify crop decrypts to a valid PNG header
        with open(crop_path, "rb") as f_crop:
            enc_data = f_crop.read()
        decrypted = kms.decrypt(enc_data, context={"doc_id": "doc_test_invoice_002", "crop_id": crop_id, "page": "1"})
        assert decrypted.startswith(b"\x89PNG\r\n\x1a\n"), "Decrypted crop is not a valid PNG image"

    # 6. JSON Export Serialization
    out_json = result.to_output_json()
    assert out_json["document_status"] == "NEEDS_REVIEW"
    assert isinstance(out_json["fields"]["VendorName"]["value"], str)
    assert isinstance(out_json["tables"][0]["rows"], list)
