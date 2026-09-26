"""
Loop 4 Unit and Integration Tests: Integrity, Tamper Forensics, and Deduplication
Tests:
  1. Error Level Analysis (ELA) on pristine vs spliced images
  2. PDF metadata forensics on clean vs edited PDF timelines
  3. Font/character-pitch consistency analyzer on aligned vs tampered field values
  4. Exact (SHA-256) and Perceptual (pHash) Deduplication
  5. AI generator metadata signatures and capture claim validation
  6. Full end-to-end IntegrityPipeline integration (Definition of Done test cases A, B, C)
"""

import io
import os
import tempfile
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf as fitz

from trustextract.schema import DocumentStatus, TrustExtractResult, FieldValue, FieldSource
from trustextract.ingestion.normalizer import NormalizedDocument, NormalizedPage
from trustextract.integrity import (
    ErrorLevelAnalyzer,
    PDFMetadataForensicDetector,
    FontConsistencyDetector,
    DocumentDeduplicator,
    AIGenerationDetector,
    IntegrityPipeline,
    compute_image_phash,
    hamming_distance,
)


# ─── 1. Error Level Analysis (ELA) Tests ──────────────────────────────────────

def test_ela_on_clean_image():
    """Uniformly compressed image should produce no ELA anomaly."""
    img = Image.new("RGB", (300, 300), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Invoice #1001", fill=(20, 20, 20))
    draw.text((50, 100), "Amount: $500.00", fill=(20, 20, 20))

    # Save to JPEG once so it has uniform compression history
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    clean_jpeg = Image.open(buf)

    analyzer = ErrorLevelAnalyzer(quality=90)
    result = analyzer.analyze(clean_jpeg)

    assert not result.has_anomaly
    assert "ELA_ANOMALY_DETECTED" not in result.flags
    assert len(result.anomalous_regions) == 0


def test_ela_on_spliced_tampered_image():
    """Pasting uncompressed or differently compressed content onto a JPEG must trigger ELA anomaly."""
    # Background JPEG compressed at lower quality (Q=60)
    bg = Image.new("RGB", (400, 400), color=(240, 240, 240))
    draw = ImageDraw.Draw(bg)
    draw.rectangle([20, 20, 380, 380], fill=(245, 245, 245), outline=(200, 200, 200))
    draw.text((50, 50), "Standard Document Body Text Line 1", fill=(30, 30, 30))
    draw.text((50, 100), "Standard Document Body Text Line 2", fill=(30, 30, 30))

    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=60)
    buf.seek(0)
    bg_jpeg = Image.open(buf).convert("RGB")

    # Tampered patch: sharp, high-contrast, uncompressed noise patch pasted into center
    patch = Image.new("RGB", (100, 60), color=(255, 255, 255))
    patch_draw = ImageDraw.Draw(patch)
    # Dense high-frequency alteration
    for i in range(0, 100, 4):
        patch_draw.line([(i, 0), (i, 60)], fill=(220, 20, 20), width=2)
    patch_draw.text((10, 20), "$99,999.00", fill=(0, 0, 0))

    bg_jpeg.paste(patch, (150, 150))

    analyzer = ErrorLevelAnalyzer(quality=90)
    result = analyzer.analyze(bg_jpeg)

    assert result.has_anomaly
    assert "ELA_ANOMALY_DETECTED" in result.flags
    assert len(result.anomalous_regions) >= 1
    # Check that identified region overlaps with the spliced patch at (150, 150)
    r_bbox = result.anomalous_regions[0].bbox
    assert r_bbox[0] <= 250 and r_bbox[1] <= 250


# ─── 2. PDF Metadata Forensics Tests ──────────────────────────────────────────

def test_pdf_metadata_on_clean_pdf(tmp_path):
    """Clean PDF with matching creation/mod dates and standard producer produces no flags."""
    pdf_path = str(tmp_path / "clean.pdf")
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.set_metadata({
        "creator": "ERP System v2.1",
        "producer": "PyMuPDF PDF Engine",
        "creationDate": "D:20240315100000Z",
        "modDate": "D:20240315100005Z"
    })
    doc.save(pdf_path)
    doc.close()

    detector = PDFMetadataForensicDetector()
    result = detector.analyze_file(pdf_path)

    assert not result.has_anomaly
    assert "METADATA_SUSPICIOUS_EDIT_HISTORY" not in result.flags


def test_pdf_metadata_on_suspicious_editor_tool(tmp_path):
    """PDF created in ERP then modified in Photoshop / editor tool must trigger suspicious edit flag."""
    pdf_path = str(tmp_path / "tampered_meta.pdf")
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.set_metadata({
        "creator": "ERP Billing Service",
        "producer": "Adobe Photoshop 24.0 (Windows)",
        "creationDate": "D:20240315100000Z",
        "modDate": "D:20240320153000Z"  # 5 days later
    })
    doc.save(pdf_path)
    doc.close()

    detector = PDFMetadataForensicDetector(suspicious_time_delta_hours=2.0)
    result = detector.analyze_file(pdf_path)

    assert result.has_anomaly
    assert "METADATA_SUSPICIOUS_EDIT_HISTORY" in result.flags
    assert result.suspicious_tool_detected == "photoshop"


# ─── 3. Font and Spacing Consistency Tests ───────────────────────────────────

def test_font_consistency_on_uniform_document():
    """Document with uniform font sizing and character pitch across all fields passes."""
    # Synthetic OCR tokens with consistent 10px pitch and 18px height
    tokens = [
        {"text": "Invoice Number:", "bbox": [50, 50, 150, 18], "confidence": 0.95},
        {"text": "INV-2024-001", "bbox": [210, 50, 120, 18], "confidence": 0.95},
        {"text": "Subtotal:", "bbox": [50, 100, 80, 18], "confidence": 0.95},
        {"text": "$500.00", "bbox": [210, 100, 70, 18], "confidence": 0.95},
        {"text": "Total:", "bbox": [50, 130, 50, 18], "confidence": 0.95},
        {"text": "$550.00", "bbox": [210, 130, 70, 18], "confidence": 0.95},
    ]
    fields = {
        "InvoiceTotal": FieldValue(value="$550.00", confidence=0.95, source=FieldSource(page=1, bbox=[210, 130, 70, 18], crop_ref="crop://1")),
        "Subtotal": FieldValue(value="$500.00", confidence=0.95, source=FieldSource(page=1, bbox=[210, 100, 70, 18], crop_ref="crop://2")),
    }

    detector = FontConsistencyDetector()
    result = detector.analyze_tokens_and_fields(tokens, fields)

    assert not result.has_anomaly
    assert "FONT_INCONSISTENCY" not in result.flags


def test_font_consistency_on_tampered_field():
    """Field pasted with mismatched height / character pitch must trigger FONT_INCONSISTENCY."""
    # Body tokens at height=16, char_width=9
    tokens = [
        {"text": "Cloud Server Hosting March", "bbox": [50, 100, 230, 16], "confidence": 0.95},
        {"text": "Database Backup Storage", "bbox": [50, 130, 210, 16], "confidence": 0.95},
        {"text": "Subtotal Amount Due", "bbox": [50, 200, 180, 16], "confidence": 0.95},
        {"text": "Tax Total Calculated", "bbox": [50, 230, 180, 16], "confidence": 0.95},
        # Tampered InvoiceTotal pasted at height=38 (2.4x median) and char_width=25 (2.8x median)
        {"text": "$9,999.00", "bbox": [250, 270, 225, 38], "confidence": 0.92},
    ]
    fields = {
        "InvoiceTotal": FieldValue(value="$9,999.00", confidence=0.92, source=FieldSource(page=1, bbox=[250, 270, 225, 38], crop_ref="crop://tampered"))
    }

    detector = FontConsistencyDetector()
    result = detector.analyze_tokens_and_fields(tokens, fields)

    assert result.has_anomaly
    assert "FONT_INCONSISTENCY" in result.flags
    assert any(af.field_name == "InvoiceTotal" for af in result.anomalous_fields)


# ─── 4. Exact and Near-Duplicate Deduplication Tests ──────────────────────────

def test_deduplicator_exact_and_near_match():
    """Exact byte resubmission triggers DUPLICATE_EXACT_MATCH; modified image triggers DUPLICATE_NEAR_MATCH."""
    dedup = DocumentDeduplicator()

    # Document 1 (Original)
    img1 = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img1)
    draw.rectangle([20, 20, 180, 180], fill=(230, 230, 230), outline=(50, 50, 50), width=3)
    draw.text((40, 40), "ORIGINAL INVOICE 001", fill=(0, 0, 0))

    buf1 = io.BytesIO()
    img1.save(buf1, format="PNG")
    bytes1 = buf1.getvalue()

    # First submission: clean registration
    res1 = dedup.check_and_register("doc_original_001", bytes1, [img1])
    assert not res1.is_duplicate
    assert len(res1.flags) == 0

    # Second submission: exact byte duplicate of Document 1
    res_exact = dedup.check_and_register("doc_resubmission_exact", bytes1, [img1])
    assert res_exact.is_duplicate
    assert res_exact.is_exact_match
    assert "DUPLICATE_EXACT_MATCH" in res_exact.flags

    # Third submission: near-duplicate (lightly recompressed at Q=75 and slightly cropped)
    buf_mod = io.BytesIO()
    img1.crop((2, 2, 198, 198)).resize((200, 200)).save(buf_mod, format="JPEG", quality=75)
    bytes_mod = buf_mod.getvalue()
    img_mod = Image.open(io.BytesIO(bytes_mod))

    res_near = dedup.check_and_register("doc_resubmission_near", bytes_mod, [img_mod], auto_index=False)
    assert res_near.is_duplicate
    assert res_near.is_near_match
    assert "DUPLICATE_NEAR_MATCH" in res_near.flags


# ─── 5. AI Generation & Metadata Detection Tests ─────────────────────────────

def test_ai_generator_metadata_detection():
    """Document with generative AI producer in metadata must trigger AI_GENERATION_SUSPECTED."""
    detector = AIGenerationDetector()
    img = Image.new("RGB", (100, 100), color=(250, 250, 250))
    meta = {
        "creator": "Midjourney v6.0 Prompt Generator",
        "producer": "Stable Diffusion WebUI",
    }

    res = detector.analyze_image_and_metadata(img, doc_metadata=meta)
    assert res.is_suspicious
    assert "AI_GENERATION_SUSPECTED" in res.flags


# ─── 6. Full DoD Integration Test Cases ──────────────────────────────────────

def test_dod_case_a_untouched_clean_invoice():
    """DoD Case A: Pristine clean_invoice.pdf must score high integrity (1.0) and zero flags."""
    pipeline = IntegrityPipeline()
    doc_path = "sample_data/clean_invoice.pdf"
    assert os.path.exists(doc_path)

    doc = fitz.open(doc_path)
    pix = doc[0].get_pixmap(dpi=150)
    page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc_meta = doc.metadata or {}
    doc.close()

    with open(doc_path, "rb") as f:
        raw_bytes = f.read()

    norm_doc = NormalizedDocument(
        document_id="clean_invoice_dod",
        original_format="pdf",
        pages=[NormalizedPage(page_number=1, image=page_img)],
        metadata=doc_meta
    )

    report = pipeline.evaluate_document(norm_doc, raw_file_bytes=raw_bytes, file_path=doc_path)

    assert report.integrity_score >= 0.95
    assert len(report.authenticity_flags) == 0


def test_dod_case_b_tampered_document(tmp_path):
    """DoD Case B: Manually altered document triggers specific tamper flags with region identified."""
    pipeline = IntegrityPipeline()
    
    # Create a base page
    base_img = Image.new("RGB", (600, 800), color=(248, 248, 248))
    draw = ImageDraw.Draw(base_img)
    draw.text((50, 50), "ACME INVOICE #9841", fill=(20, 20, 20))
    draw.text((50, 100), "Subtotal: $900.00", fill=(20, 20, 20))
    draw.text((50, 140), "Tax: $72.00", fill=(20, 20, 20))

    # Save to JPEG baseline
    buf = io.BytesIO()
    base_img.save(buf, format="JPEG", quality=70)
    buf.seek(0)
    tampered_img = Image.open(buf).convert("RGB")

    # Spliced tamper: Paste a high-contrast modified amount at (300, 200)
    patch = Image.new("RGB", (140, 50), color=(255, 255, 255))
    pdraw = ImageDraw.Draw(patch)
    for i in range(0, 140, 3):
        pdraw.line([(i, 0), (i, 50)], fill=(10, 10, 10), width=1)
    pdraw.text((10, 15), "$9,999.00", fill=(10, 10, 10))
    tampered_img.paste(patch, (300, 200))

    # Tokens with divergent font height on InvoiceTotal
    tokens = [
        {"text": "ACME INVOICE #9841", "bbox": [50, 50, 220, 16], "confidence": 0.95},
        {"text": "Subtotal: $900.00", "bbox": [50, 100, 180, 16], "confidence": 0.95},
        {"text": "Tax: $72.00", "bbox": [50, 140, 120, 16], "confidence": 0.95},
        {"text": "$9,999.00", "bbox": [300, 200, 140, 50], "confidence": 0.90},  # 50px height vs 16px
    ]
    fields = {
        "InvoiceTotal": FieldValue(value="$9,999.00", confidence=0.90, source=FieldSource(page=1, bbox=[300, 200, 140, 50], crop_ref="crop://t1"))
    }

    norm_doc = NormalizedDocument(
        document_id="tampered_invoice_dod",
        original_format="png",
        pages=[NormalizedPage(page_number=1, image=tampered_img)]
    )

    report = pipeline.evaluate_document(
        norm_doc,
        all_tokens=tokens,
        extracted_fields=fields
    )

    # Must trigger at least one specific flag and lower integrity score
    assert report.integrity_score < 0.80
    assert len(report.authenticity_flags) > 0
    assert any(f in report.authenticity_flags for f in ["ELA_ANOMALY_DETECTED", "FONT_INCONSISTENCY"])


def test_dod_case_c_duplicate_resubmission():
    """DoD Case C: Duplicate submission triggers exact and near-duplicate flags."""
    dedup = DocumentDeduplicator()
    pipeline = IntegrityPipeline(deduplicator=dedup)

    doc_path = "sample_data/clean_invoice.pdf"
    with open(doc_path, "rb") as f:
        raw_bytes = f.read()

    doc = fitz.open(doc_path)
    pix = doc[0].get_pixmap(dpi=150)
    page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    norm_doc1 = NormalizedDocument(
        document_id="doc_submission_1",
        original_format="pdf",
        pages=[NormalizedPage(page_number=1, image=page_img)]
    )

    # Submission 1: Indexes document
    report1 = pipeline.evaluate_document(norm_doc1, raw_file_bytes=raw_bytes)
    assert "DUPLICATE_EXACT_MATCH" not in report1.authenticity_flags

    # Submission 2: Exact byte duplicate
    norm_doc2 = NormalizedDocument(
        document_id="doc_submission_2_exact",
        original_format="pdf",
        pages=[NormalizedPage(page_number=1, image=page_img)]
    )
    report2 = pipeline.evaluate_document(norm_doc2, raw_file_bytes=raw_bytes)
    assert "DUPLICATE_EXACT_MATCH" in report2.authenticity_flags
    assert report2.integrity_score <= 0.50


def test_subtle_size_matched_tamper_isolation():
    """
    Evaluates ELA on a subtle size-matched text edit (same font size/layout).
    Verifies that ELA anomaly score in isolation does not trigger on low-contrast
    text-only changes where 8x8 DCT errors remain within normal text variance,
    documenting the physical sensitivity boundary.
    """
    base = Image.new("RGB", (650, 850), color=(250, 250, 250))
    d = ImageDraw.Draw(base)
    d.text((50, 50), "ACME INVOICE #9841", fill=(20, 20, 20))
    d.text((50, 90), "Subtotal: $900.00", fill=(20, 20, 20))
    d.text((50, 125), "Tax: $72.00", fill=(20, 20, 20))
    d.text((50, 160), "Total: $972.00", fill=(20, 20, 20))

    buf = io.BytesIO()
    base.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    jpeg_base = Image.open(buf).convert("RGB")

    # Subtle edit: replace '$972.00' with '$9,972.00' matching the 16px body font size
    tampered_subtle = jpeg_base.copy()
    dt = ImageDraw.Draw(tampered_subtle)
    dt.rectangle([100, 155, 250, 175], fill=(250, 250, 250))
    dt.text((100, 160), "$9,972.00", fill=(20, 20, 20))

    ela = ErrorLevelAnalyzer(quality=90)
    res = ela.analyze(tampered_subtle)

    # Isolated ELA on flat white size-matched text has error within normal document text variance
    assert not res.has_anomaly
    assert len(res.anomalous_regions) == 0


def test_dual_loop3_and_loop4_simultaneous_escalation():
    """
    Verifies distinct, independent visibility when a document has both a Loop 3
    escalation (low confidence / arithmetic mismatch) and a Loop 4 authenticity flag (duplicate).
    """
    from trustextract.schema import ExtractionAgreement, FieldType, ConfidenceBreakdown
    from trustextract.extraction.confidence import DocumentEscalationEngine

    dedup = DocumentDeduplicator()
    dummy_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    dedup.check_and_register("DOC-ORIGINAL-001", b"PREVIOUS_BINARY_CONTENT", [dummy_img])

    norm_doc = NormalizedDocument(
        document_id="DOC-2024-DUAL-FLAG",
        original_format="pdf",
        pages=[NormalizedPage(page_number=1, image=dummy_img)]
    )

    fields = {
        "VendorName": FieldValue(value="Contoso Logistics Inc.", confidence=0.97, field_type=FieldType.FREE_TEXT),
        "InvoiceId": FieldValue(
            value="INV-2024-9841",
            confidence=0.842,
            field_type=FieldType.IDENTIFIER,
            extraction_agreement=ExtractionAgreement.OCR_NEQ_VISION,
            confidence_breakdown=ConfidenceBreakdown(
                native_ocr=0.96,
                agreement_signal=0.60,
                consistency_signal=0.98,
                validation_signal=1.0,
                validation_flags=["PASSED"],
                final=0.842
            )
        ),
        "Subtotal": FieldValue(value="$900.00", confidence=0.96, field_type=FieldType.NUMERIC),
        "Tax": FieldValue(value="$72.00", confidence=0.95, field_type=FieldType.NUMERIC),
        "InvoiceTotal": FieldValue(value="$9,999.00", confidence=0.50, validation="FAILED", field_type=FieldType.NUMERIC),
    }

    result = TrustExtractResult(
        document_id=norm_doc.document_id,
        fields=fields,
        overall_confidence=0.84,
        document_status=DocumentStatus.OK,
        escalation_reasons=["ARITHMETIC_INCONSISTENCY: Subtotal (900.00) + Tax (72.00) != InvoiceTotal (9999.00)"]
    )

    escalation_engine = DocumentEscalationEngine()
    escalation_engine.evaluate(result)

    pipeline = IntegrityPipeline(deduplicator=dedup)
    report = pipeline.evaluate_document(norm_doc, raw_file_bytes=b"PREVIOUS_BINARY_CONTENT")
    pipeline.attach_to_result(result, report)

    # Both lists must be populated independently
    assert len(result.escalation_reasons) >= 3
    assert any("ARITHMETIC_INCONSISTENCY" in r for r in result.escalation_reasons)
    assert any("InvoiceId" in r and "ocr!=vision" in r for r in result.escalation_reasons)
    assert "DUPLICATE_EXACT_MATCH" in result.authenticity_flags
    assert result.document_status == DocumentStatus.NEEDS_REVIEW
    assert result.integrity_score <= 0.50


def test_worst_case_subtle_tamper_boundary_finding():
    """
    Worst-case boundary test: A subtle, size-matched edit on a non-arithmetic identifier
    (InvoiceId altered from INV-2024-9841 to INV-2024-9999) with matching typography on a JPEG.
    
    Demonstrates the exact architectural boundary where single-document image/metadata
    forensics reach their physical limit and pass the document as OK/high-confidence,
    motivating Loop 5 (cross-document reconciliation against POs/contracts).

    IMPORTANT CAVEAT:
    Cross-document reconciliation is a partial mitigation conditional on bundle composition,
    not a complete closure of the single-document tamper boundary. It only functions when a
    second, independent document in the bundle asserts the same fact and was not altered by
    the same attacker. If a single tampered document is submitted alone with no corroborating
    document, this gap remains open.
    """
    from trustextract.schema import ExtractionAgreement, FieldType, ConfidenceBreakdown
    from trustextract.extraction.confidence import DocumentEscalationEngine

    base_id_doc = Image.new("RGB", (650, 850), color=(250, 250, 250))
    di = ImageDraw.Draw(base_id_doc)
    di.text((50, 50), "Customer Receipt / Order Reference", fill=(20, 20, 20))
    di.text((50, 90), "AccountID: ACC-88129", fill=(20, 20, 20))
    di.text((50, 130), "InvoiceId: INV-2024-9841", fill=(20, 20, 20))
    di.text((50, 170), "Date: 2024-03-15", fill=(20, 20, 20))

    buf_id = io.BytesIO()
    base_id_doc.save(buf_id, format="JPEG", quality=80)
    buf_id.seek(0)
    jpeg_id_base = Image.open(buf_id).convert("RGB")

    # Attacker alters ONLY the InvoiceId digits: 'INV-2024-9999' using same 16px font and layout
    tampered_id_img = jpeg_id_base.copy()
    di_t = ImageDraw.Draw(tampered_id_img)
    di_t.rectangle([130, 125, 290, 148], fill=(250, 250, 250))
    di_t.text((130, 130), "INV-2024-9999", fill=(20, 20, 20))

    buf_final = io.BytesIO()
    tampered_id_img.save(buf_final, format="JPEG", quality=80)
    final_bytes = buf_final.getvalue()
    final_img = Image.open(io.BytesIO(final_bytes))

    worst_tokens = [
        {"text": "Customer Receipt / Order Reference", "bbox": [50, 50, 290, 16], "confidence": 0.97},
        {"text": "AccountID: ACC-88129", "bbox": [50, 90, 190, 16], "confidence": 0.96},
        {"text": "InvoiceId:", "bbox": [50, 130, 75, 16], "confidence": 0.96},
        {"text": "INV-2024-9999", "bbox": [130, 130, 125, 16], "confidence": 0.95},
        {"text": "Date: 2024-03-15", "bbox": [50, 170, 140, 16], "confidence": 0.96},
    ]

    worst_fields = {
        "InvoiceId": FieldValue(
            value="INV-2024-9999",
            confidence=0.95,
            field_type=FieldType.IDENTIFIER,
            extraction_agreement=ExtractionAgreement.OCR_EQ_VISION,
            confidence_breakdown=ConfidenceBreakdown(
                native_ocr=0.95,
                agreement_signal=1.0,
                consistency_signal=1.0,
                validation_signal=1.0,
                validation_flags=["PASSED"],
                final=0.968
            ),
            source=FieldSource(page=1, bbox=[130, 130, 125, 16], crop_ref="crop://worst_1")
        ),
        "AccountID": FieldValue(value="ACC-88129", confidence=0.96, field_type=FieldType.IDENTIFIER),
        "Date": FieldValue(value="2024-03-15", confidence=0.96, field_type=FieldType.DATE),
    }

    norm_doc_worst = NormalizedDocument(
        document_id="DOC-WORST-CASE-SUBTLE-TAMPER",
        original_format="jpeg",
        pages=[NormalizedPage(page_number=1, image=final_img)]
    )

    result_worst = TrustExtractResult(
        document_id=norm_doc_worst.document_id,
        fields=worst_fields,
        overall_confidence=0.96,
        document_status=DocumentStatus.OK,
        escalation_reasons=[]
    )

    esc_engine = DocumentEscalationEngine()
    esc_engine.evaluate(result_worst)

    pipeline = IntegrityPipeline()
    report_worst = pipeline.evaluate_document(
        norm_doc_worst,
        raw_file_bytes=final_bytes,
        all_tokens=worst_tokens,
        extracted_fields=worst_fields
    )
    pipeline.attach_to_result(result_worst, report_worst)

    # Documented boundary: this subtle isolated edit has no detectable statistical footprint on a single document
    assert result_worst.integrity_score == 1.0
    assert len(result_worst.authenticity_flags) == 0
    assert len(result_worst.escalation_reasons) == 0
    assert result_worst.document_status == DocumentStatus.OK


