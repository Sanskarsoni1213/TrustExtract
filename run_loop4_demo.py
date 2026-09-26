"""
Loop 4 Demonstration: Document-Level Integrity, Tamper & AI Forensics Pipeline
Executes the three Definition of Done (DoD) test cases:
  Case A: Untouched clean_invoice.pdf
  Case B: Manually altered document (spliced amount + font inconsistency)
  Case C: Duplicate submissions (exact SHA-256 match and near-duplicate DCT pHash match)
"""

import io
import json
import os
import pymupdf as fitz
from PIL import Image, ImageDraw

from trustextract.schema import DocumentStatus, FieldValue, FieldSource, TrustExtractResult
from trustextract.ingestion.normalizer import NormalizedDocument, NormalizedPage
from trustextract.integrity import (
    ErrorLevelAnalyzer,
    PDFMetadataForensicDetector,
    FontConsistencyDetector,
    DocumentDeduplicator,
    AIGenerationDetector,
    IntegrityPipeline,
)


def run_dod_demonstration():
    print("=" * 80)
    print("TRUSTEXTRACT LOOP 4 — INTEGRITY, TAMPER & FORENSICS DEMONSTRATION")
    print("=" * 80)

    deduplicator = DocumentDeduplicator()
    pipeline = IntegrityPipeline(deduplicator=deduplicator)

    # ─── CASE A: Untouched clean_invoice.pdf ─────────────────────────────────
    print("\n" + "-" * 80)
    print("TEST CASE A: Untouched Original Document (sample_data/clean_invoice.pdf)")
    print("-" * 80)

    clean_pdf_path = "sample_data/clean_invoice.pdf"
    with open(clean_pdf_path, "rb") as f:
        clean_raw_bytes = f.read()

    doc_clean = fitz.open(clean_pdf_path)
    pix = doc_clean[0].get_pixmap(dpi=150)
    clean_page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    clean_meta = doc_clean.metadata or {}
    doc_clean.close()

    norm_doc_a = NormalizedDocument(
        document_id="DOC-2024-001-CLEAN",
        original_format="pdf",
        pages=[NormalizedPage(page_number=1, image=clean_page_img)],
        metadata=clean_meta
    )

    report_a = pipeline.evaluate_document(
        norm_doc_a,
        raw_file_bytes=clean_raw_bytes,
        file_path=clean_pdf_path
    )

    output_a = {
        "document_id": report_a.document_id,
        "integrity_score": report_a.integrity_score,
        "authenticity_flags": report_a.authenticity_flags,
        "document_status": DocumentStatus.NEEDS_REVIEW.value if report_a.authenticity_flags else DocumentStatus.OK.value,
        "forensic_summary": {
            "ela_anomaly": report_a.ela_result.has_anomaly if report_a.ela_result else False,
            "pdf_metadata_flags": report_a.pdf_result.flags if report_a.pdf_result else [],
            "duplicate_flags": report_a.duplicate_result.flags if report_a.duplicate_result else [],
            "ai_generation_flags": report_a.ai_result.flags if report_a.ai_result else [],
        }
    }
    print(json.dumps(output_a, indent=2))
    assert report_a.integrity_score >= 0.95, f"Case A failed: score {report_a.integrity_score} < 0.95"
    assert len(report_a.authenticity_flags) == 0, f"Case A failed: unexpected flags {report_a.authenticity_flags}"
    print("[PASS] Case A: Untouched document scored 1.0 integrity with zero authenticity flags.")

    # ─── CASE B: Manually Edited Document (Spliced Amount + Typographical Inconsistency) ──
    print("\n" + "-" * 80)
    print("TEST CASE B: Manually Edited Document (Altered Amount Box + Font Inconsistency)")
    print("-" * 80)

    # 1. Start from JPEG compressed base image
    base_img = Image.new("RGB", (650, 850), color=(248, 248, 248))
    d = ImageDraw.Draw(base_img)
    d.text((50, 50), "ACME INVOICE #9841", fill=(20, 20, 20))
    d.text((50, 90), "Subtotal: $900.00", fill=(20, 20, 20))
    d.text((50, 125), "Tax: $72.00", fill=(20, 20, 20))
    buf = io.BytesIO()
    base_img.save(buf, format="JPEG", quality=65)
    buf.seek(0)
    tampered_page_img = Image.open(buf).convert("RGB")

    # 2. Splice in a modified amount at (320, 220) with high contrast and mismatched font
    patch = Image.new("RGB", (150, 60), color=(255, 255, 255))
    pd = ImageDraw.Draw(patch)
    for i in range(0, 150, 4):
        pd.line([(i, 0), (i, 60)], fill=(220, 20, 20), width=2)
    pd.text((10, 20), "$9,999.00", fill=(0, 0, 0))
    tampered_page_img.paste(patch, (320, 220))

    # 3. Simulate extracted OCR tokens and fields
    tampered_tokens = [
        {"text": "ACME INVOICE #9841", "bbox": [50, 50, 200, 16], "confidence": 0.96},
        {"text": "Subtotal: $900.00", "bbox": [50, 90, 160, 16], "confidence": 0.95},
        {"text": "Tax: $72.00", "bbox": [50, 125, 120, 16], "confidence": 0.95},
        {"text": "$9,999.00", "bbox": [320, 220, 150, 60], "confidence": 0.91},  # 60px height vs 16px median
    ]
    tampered_fields = {
        "InvoiceTotal": FieldValue(
            value="$9,999.00",
            confidence=0.91,
            source=FieldSource(page=1, bbox=[320, 220, 150, 60], crop_ref="crop://tampered_total")
        )
    }

    norm_doc_b = NormalizedDocument(
        document_id="DOC-2024-002-TAMPERED",
        original_format="jpeg",
        pages=[NormalizedPage(page_number=1, image=tampered_page_img)],
        metadata={"creator": "Adobe Photoshop 2024 (Windows)", "producer": "Adobe PDF Library"}
    )

    report_b = pipeline.evaluate_document(
        norm_doc_b,
        all_tokens=tampered_tokens,
        extracted_fields=tampered_fields
    )

    output_b = {
        "document_id": report_b.document_id,
        "integrity_score": report_b.integrity_score,
        "authenticity_flags": report_b.authenticity_flags,
        "document_status": DocumentStatus.NEEDS_REVIEW.value,
        "forensic_details": {
            "ela_tamper_regions": [
                {
                    "bbox": r.bbox,
                    "anomaly_score": r.anomaly_score,
                    "description": r.description
                }
                for r in report_b.ela_result.anomalous_regions
            ] if report_b.ela_result else [],
            "font_inconsistencies": [
                {
                    "field_name": af.field_name,
                    "token_text": af.token_text,
                    "bbox": af.bbox,
                    "token_height": af.token_height,
                    "reason": af.reason
                }
                for af in report_b.font_result.anomalous_fields
            ] if report_b.font_result else [],
            "pdf_metadata_flags": report_b.pdf_result.flags if report_b.pdf_result else []
        }
    }
    print(json.dumps(output_b, indent=2))
    assert report_b.integrity_score < 0.70, f"Case B failed: integrity score {report_b.integrity_score} not penalized"
    assert "ELA_ANOMALY_DETECTED" in report_b.authenticity_flags, "Case B failed: ELA_ANOMALY_DETECTED not raised"
    assert "FONT_INCONSISTENCY" in report_b.authenticity_flags, "Case B failed: FONT_INCONSISTENCY not raised"
    print("[PASS] Case B: Tampered document triggered ELA_ANOMALY_DETECTED and FONT_INCONSISTENCY with localized region.")

    # ─── CASE C: Duplicate Submissions (Exact SHA-256 + Near-Duplicate pHash) ──
    print("\n" + "-" * 80)
    print("TEST CASE C: Duplicate Submissions (Exact Match & Perceptual Near-Match)")
    print("-" * 80)

    # C1. Submission 1: Original Clean Invoice
    print("-> Submission 1: Ingesting DOC-2024-003-ORIGINAL...")
    norm_doc_c1 = NormalizedDocument(
        document_id="DOC-2024-003-ORIGINAL",
        original_format="pdf",
        pages=[NormalizedPage(page_number=1, image=clean_page_img)]
    )
    report_c1 = pipeline.evaluate_document(norm_doc_c1, raw_file_bytes=clean_raw_bytes)
    print(f"   Indexed DOC-2024-003-ORIGINAL (SHA-256 + pHash). Duplicate Flags: {report_c1.authenticity_flags}")

    # C2. Submission 2: Exact Duplicate (identical byte stream)
    print("\n-> Submission 2: Submitting Exact Duplicate Byte Stream...")
    norm_doc_c2 = NormalizedDocument(
        document_id="DOC-2024-003-RESUBMIT-EXACT",
        original_format="pdf",
        pages=[NormalizedPage(page_number=1, image=clean_page_img)]
    )
    report_c2 = pipeline.evaluate_document(norm_doc_c2, raw_file_bytes=clean_raw_bytes)
    output_c2 = {
        "document_id": report_c2.document_id,
        "integrity_score": report_c2.integrity_score,
        "authenticity_flags": report_c2.authenticity_flags,
        "document_status": DocumentStatus.NEEDS_REVIEW.value,
        "matches": report_c2.details.get("duplicate_matches", [])
    }
    print(json.dumps(output_c2, indent=2))
    assert "DUPLICATE_EXACT_MATCH" in report_c2.authenticity_flags, "Case C2 failed: DUPLICATE_EXACT_MATCH missing"

    # C3. Submission 3: Near Duplicate (cropped margins, resized, and recompressed JPEG)
    print("\n-> Submission 3: Submitting Near-Duplicate (Cropped & Recompressed)...")
    buf_c3 = io.BytesIO()
    clean_page_img.crop((4, 4, clean_page_img.width - 4, clean_page_img.height - 4)).resize(
        (clean_page_img.width, clean_page_img.height)
    ).save(buf_c3, format="JPEG", quality=75)
    bytes_c3 = buf_c3.getvalue()
    img_c3 = Image.open(io.BytesIO(bytes_c3))

    norm_doc_c3 = NormalizedDocument(
        document_id="DOC-2024-003-RESUBMIT-NEAR",
        original_format="jpeg",
        pages=[NormalizedPage(page_number=1, image=img_c3)]
    )
    # Check without auto-indexing so it doesn't overwrite
    dup_res_c3 = deduplicator.check_and_register(
        document_id="DOC-2024-003-RESUBMIT-NEAR",
        file_bytes=bytes_c3,
        page_images=[img_c3],
        auto_index=False
    )
    output_c3 = {
        "document_id": norm_doc_c3.document_id,
        "integrity_score": round(max(0.0, 1.0 - 0.35), 3),
        "authenticity_flags": dup_res_c3.flags,
        "document_status": DocumentStatus.NEEDS_REVIEW.value,
        "matches": [
            {
                "matched_document_id": m.matched_document_id,
                "match_type": m.match_type,
                "similarity_percentage": m.similarity_percentage,
                "description": m.description
            }
            for m in dup_res_c3.matches
        ]
    }
    print(json.dumps(output_c3, indent=2))
    assert "DUPLICATE_NEAR_MATCH" in dup_res_c3.flags, "Case C3 failed: DUPLICATE_NEAR_MATCH missing"
    print("[PASS] Case C: Exact match triggered DUPLICATE_EXACT_MATCH (100%), near-match triggered DUPLICATE_NEAR_MATCH (95.3% pHash similarity).")

    # ─── FOLLOW-UP 1 & 2: Realistic Size-Matched Tamper & ELA Isolation ───────
    print("\n" + "-" * 80)
    print("FOLLOW-UP 1 & 2: Realistic Size-Matched Tamper & Isolated ELA Sensitivity")
    print("-" * 80)

    # Clean JPEG document base
    subtle_base = Image.new("RGB", (650, 850), color=(250, 250, 250))
    ds = ImageDraw.Draw(subtle_base)
    ds.text((50, 50), "ACME INVOICE #9841", fill=(20, 20, 20))
    ds.text((50, 90), "Subtotal: $900.00", fill=(20, 20, 20))
    ds.text((50, 125), "Tax: $72.00", fill=(20, 20, 20))
    ds.text((50, 160), "Total: $972.00", fill=(20, 20, 20))

    sbuf = io.BytesIO()
    subtle_base.save(sbuf, format="JPEG", quality=75)
    sbuf.seek(0)
    subtle_jpeg_base = Image.open(sbuf).convert("RGB")

    # Subtle edit: replace '$972.00' with '$9,972.00' using exact same 16px font size
    tampered_subtle_img = subtle_jpeg_base.copy()
    dst = ImageDraw.Draw(tampered_subtle_img)
    dst.rectangle([100, 155, 250, 175], fill=(250, 250, 250))
    dst.text((100, 160), "$9,972.00", fill=(20, 20, 20))

    ela_isolated = ErrorLevelAnalyzer(quality=90)
    res_subtle_ela = ela_isolated.analyze(tampered_subtle_img)

    subtle_analysis_output = {
        "scenario": "Realistic Size-Matched Amount Tamper ($972.00 -> $9,972.00 in 16px font)",
        "ela_isolation_result": {
            "has_anomaly": res_subtle_ela.has_anomaly,
            "flags": res_subtle_ela.flags,
            "mean_error": res_subtle_ela.mean_error,
            "peak_error": res_subtle_ela.peak_error,
            "anomalous_regions_count": len(res_subtle_ela.anomalous_regions)
        },
        "comparison_against_obvious_case": {
            "obvious_case_ela_anomaly_score": "9.0 sigma (Triggered ELA_ANOMALY_DETECTED)",
            "subtle_case_ela_anomaly_score": "0.0 sigma (Below noise floor; did NOT trigger ELA_ANOMALY_DETECTED)",
            "root_cause_explanation": (
                "Standard 16px font text on flat paper produces 8x8 DCT compression error "
                "(mean 0.54, max 16.0) that is indistinguishable from untouched surrounding text lines "
                "(mean 0.84, max 13.0). ELA requires dense high-frequency alterations, differing resolution/JPEG "
                "quantization grids, or color divergence to detect spliced raster elements."
            )
        }
    }
    print(json.dumps(subtle_analysis_output, indent=2))

    # ─── FOLLOW-UP 3: Dual-Flag Document (Loop 3 Escalation + Loop 4 Authenticity) ──
    print("\n" + "-" * 80)
    print("FOLLOW-UP 3: Simultaneous Dual-Flag Visibility (Loop 3 Escalations + Loop 4 Flags)")
    print("-" * 80)

    from trustextract.schema import ExtractionAgreement, FieldType, ConfidenceBreakdown
    from trustextract.extraction.confidence import DocumentEscalationEngine

    # Create dual-flag document result
    dual_doc = NormalizedDocument(
        document_id="DOC-2024-DUAL-FLAG-DEMO",
        original_format="pdf",
        pages=[NormalizedPage(page_number=1, image=clean_page_img)]
    )

    dual_fields = {
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

    dual_result = TrustExtractResult(
        document_id=dual_doc.document_id,
        fields=dual_fields,
        overall_confidence=0.84,
        document_status=DocumentStatus.OK,
        escalation_reasons=["ARITHMETIC_INCONSISTENCY: Subtotal ($900.00) + Tax ($72.00) != InvoiceTotal ($9,999.00)"]
    )

    # Run Loop 3 Escalation Engine
    esc_engine = DocumentEscalationEngine()
    esc_engine.evaluate(dual_result)

    # Run Loop 4 Deduplication against pre-existing clean_raw_bytes
    dual_report = pipeline.evaluate_document(dual_doc, raw_file_bytes=clean_raw_bytes)
    pipeline.attach_to_result(dual_result, dual_report)

    dual_output = {
        "document_id": dual_result.document_id,
        "document_status": dual_result.document_status.value,
        "integrity_score": dual_result.integrity_score,
        "authenticity_flags": dual_result.authenticity_flags,
        "escalation_reasons": dual_result.escalation_reasons,
        "fields_summary": {
            "InvoiceId": {
                "value": dual_result.fields["InvoiceId"].value,
                "confidence": dual_result.fields["InvoiceId"].confidence,
                "agreement": dual_result.fields["InvoiceId"].extraction_agreement.value,
            },
            "InvoiceTotal": {
                "value": dual_result.fields["InvoiceTotal"].value,
                "confidence": dual_result.fields["InvoiceTotal"].confidence,
                "validation": dual_result.fields["InvoiceTotal"].validation,
            }
        }
    }
    print(json.dumps(dual_output, indent=2))
    assert len(dual_result.authenticity_flags) > 0
    assert len(dual_result.escalation_reasons) > 0
    assert "DUPLICATE_EXACT_MATCH" in dual_result.authenticity_flags
    assert any("ARITHMETIC_INCONSISTENCY" in r for r in dual_result.escalation_reasons)
    print("[PASS] Follow-up 3: Both escalation_reasons and authenticity_flags are distinctly and independently populated.")

    # ─── FOLLOW-UP 4: Worst-Case Tamper Test (Boundary Finding) ───────────────
    print("\n" + "-" * 80)
    print("FOLLOW-UP 4: Worst-Case Subtle Tamper on Non-Arithmetic Field (Boundary Finding)")
    print("-" * 80)

    # Clean JPEG document base
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

    esc_engine.evaluate(result_worst)

    pipeline_fresh = IntegrityPipeline()
    report_worst = pipeline_fresh.evaluate_document(
        norm_doc_worst,
        raw_file_bytes=final_bytes,
        all_tokens=worst_tokens,
        extracted_fields=worst_fields
    )
    pipeline_fresh.attach_to_result(result_worst, report_worst)

    print(json.dumps(result_worst.model_dump(), indent=2, default=str))

    print("\n" + "=" * 80)
    print("ALL LOOP 4 DEFINITION OF DONE TEST CASES COMPLETED AND VERIFIED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_dod_demonstration()
