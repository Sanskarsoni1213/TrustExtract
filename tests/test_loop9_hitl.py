"""
Loop 9 Tests: Human-in-the-Loop Review Queue & Active Learning Retrain Demonstration
=====================================================================================
Tests:
  1. Human correction logging, schema validation, and audit trail generation.
  2. Active learning alias canonicalization and spurious field suppression.
  3. End-to-end before/after evaluation on held-out test documents and fields.
"""

import os
import json
import pytest

from trustextract.evaluation.human_feedback import (
    HumanCorrection,
    ReviewFeedbackStore,
    ActiveFeedbackCalibrator,
)
from trustextract.evaluation.evaluate import evaluate_dataset, DocumentEvaluator
from trustextract.extraction.pipeline import DualPathExtractor
from trustextract.extraction.path_b_vision import LocalVisionLayoutExtractor
from trustextract.schema import FieldType


def test_human_correction_logging(tmp_path):
    """Verify logging of human corrections with audit records."""
    store_file = os.path.join(tmp_path, "test_corrections.json")
    store = ReviewFeedbackStore(storage_path=store_file)

    c1 = HumanCorrection(
        document_id="clean_receipt.png",
        field_name="MerchantName",
        original_extracted_value="STARBUCKS COFFEE #1042",
        corrected_value="Starbucks Coffee",
        evidence_bbox=[140.0, 52.0, 480.0, 45.0],
        reviewer_id="human_reviewer_42",
        notes="Normalized store branch string to canonical merchant brand name."
    )
    store.record_correction(c1)

    loaded_store = ReviewFeedbackStore(storage_path=store_file)
    corrections = loaded_store.get_corrections()
    assert len(corrections) == 1
    assert corrections[0].field_name == "MerchantName"
    assert corrections[0].corrected_value == "Starbucks Coffee"


def test_active_feedback_calibrator_alias_and_suppression():
    """Verify that the calibrator learns alias mappings and spurious field suppressions."""
    store = ReviewFeedbackStore(storage_path=None)

    # 1. Alias correction
    store.record_correction(HumanCorrection(
        document_id="clean_receipt.png",
        field_name="MerchantName",
        original_extracted_value="STARBUCKS COFFEE #1042",
        corrected_value="Starbucks Coffee",
    ))

    # 2. Spurious field suppression
    store.record_correction(HumanCorrection(
        document_id="clean_payslip.png",
        field_name="TotalTax",
        original_extracted_value=160.0,
        corrected_value=None,
        is_spurious_fp=True,
    ))

    calibrator = ActiveFeedbackCalibrator(feedback_store=store)
    training_summary = calibrator.train_on_feedback()

    assert training_summary["total_corrections"] == 2
    assert training_summary["learned_aliases"] == 1
    assert training_summary["learned_suppressions"] == 1

    # Test post-processing
    raw_fields = {
        "MerchantName": "STARBUCKS COFFEE #1042",
        "TotalTax": 160.0,
        "EmployeeName": "Sarah Chen",
    }
    processed = calibrator.apply_post_processing("clean_payslip.png", raw_fields)
    assert "TotalTax" not in processed  # Suppressed
    assert processed["MerchantName"] == "Starbucks Coffee"  # Canonicalized
    assert processed["EmployeeName"] == "Sarah Chen"  # Untouched


def test_held_out_evaluation_before_and_after():
    """
    Verify evaluation on held-out documents (clean_invoice.pdf, clean_po.pdf, clean_agreement.docx)
    before and after active feedback training.
    """
    held_out_manifest = [
        ("sample_data/clean_invoice.pdf", "sample_data/ground_truth/clean_invoice.json"),
        ("sample_data/clean_po.pdf", "sample_data/ground_truth/clean_po.json"),
        ("sample_data/clean_agreement.docx", "sample_data/ground_truth/clean_agreement.json"),
    ]

    extractor = DualPathExtractor(
        vision_model=LocalVisionLayoutExtractor(),
        n_consistency_passes=1
    )

    report_before = evaluate_dataset(held_out_manifest, dual_extractor=extractor)
    assert report_before.overall_precision == 1.0000
    assert report_before.overall_recall == 1.0000
    assert report_before.overall_f1 == 1.0000
