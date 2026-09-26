"""
Loop 8 Unit & Integration Tests: Evaluation Engine & Confidence Calibration
Tests:
  1. Semantic equivalence matching across field types (IDENTIFIER, NUMERIC, DATE, FREE_TEXT).
  2. DocumentEvaluator TP, FP, FN calculation logic.
  3. CalibrationBucket stats (accuracy, mean confidence, calibration gap).
  4. End-to-end evaluate_dataset over ground truth reference samples.
"""

import pytest
from trustextract.schema import FieldType, FieldValue, ExtractionAgreement
from trustextract.evaluation.evaluate import (
    DocumentEvaluator,
    CalibrationBucket,
    FieldEvaluationMetric,
    evaluate_dataset,
)


def test_semantic_values_match():
    """Verify semantic matching logic across data types."""
    evaluator = DocumentEvaluator(currency_tolerance=0.02)

    # 1. IDENTIFIER: Strict match only
    assert evaluator.values_match("INV-2024-9841", "INV-2024-9841", "InvoiceId") is True
    assert evaluator.values_match("INV-2024-984I", "INV-2024-9841", "InvoiceId") is False
    assert evaluator.values_match("EMP-2024-0471", "EMP-2024-0471", "EmployeeId") is True

    # 2. NUMERIC: Float currency tolerance
    assert evaluator.values_match("$972.00", 972.00, "InvoiceTotal") is True
    assert evaluator.values_match(972.01, 972.00, "InvoiceTotal") is True
    assert evaluator.values_match(973.00, 972.00, "InvoiceTotal") is False

    # 3. DATE: Canonical parsed date comparison
    assert evaluator.values_match("2024-03-15", "2024/03/15", "InvoiceDate") is True
    assert evaluator.values_match("03/15/2024", "2024-03-15", "InvoiceDate") is True
    assert evaluator.values_match("2024-03-16", "2024-03-15", "InvoiceDate") is False

    # 4. FREE_TEXT: Normalized string comparison
    assert evaluator.values_match("Contoso Logistics Inc.", "contoso logistics, inc", "VendorName") is True
    assert evaluator.values_match("Sarah Chen", "sarah  chen", "EmployeeName") is True
    assert evaluator.values_match("Acme Corp", "Contoso Logistics", "VendorName") is False


def test_document_evaluator_metrics():
    """Verify TP, FP, FN classification for a sample document."""
    evaluator = DocumentEvaluator()

    extracted = {
        "VendorName": FieldValue(value="Contoso Logistics Inc.", confidence=0.98),
        "InvoiceTotal": FieldValue(value=972.00, confidence=0.97),
        "InvoiceId": FieldValue(value="INV-2024-984I", confidence=0.84),  # Wrong value -> FP
        "ExtraField": FieldValue(value="Unexpected", confidence=0.90),      # Not in GT -> FP
    }

    ground_truth = {
        "VendorName": "Contoso Logistics Inc.",
        "InvoiceTotal": 972.00,
        "InvoiceId": "INV-2024-9841",
        "DueDate": "2024-04-15",                                          # Missed -> FN
    }

    res = evaluator.evaluate_sample(extracted, ground_truth, doc_id="test_doc")
    metrics = res["metrics"]

    assert metrics["tp"] == 2  # VendorName, InvoiceTotal
    assert metrics["fp"] == 2  # InvoiceId (wrong), ExtraField (unwanted)
    assert metrics["fn"] == 1  # DueDate (missed)

    fres = res["field_results"]
    assert fres["VendorName"]["status"] == "TP"
    assert fres["InvoiceTotal"]["status"] == "TP"
    assert fres["InvoiceId"]["status"] == "FP"
    assert fres["ExtraField"]["status"] == "FP"
    assert fres["DueDate"]["status"] == "FN"


def test_calibration_bucket_calculation():
    """Verify confidence bucket accuracy and calibration gap computation."""
    bucket = CalibrationBucket("High ([0.90, 1.00])", 0.90, 1.00)
    bucket.total_count = 10
    bucket.correct_count = 9
    bucket.sum_confidence = 9.50

    assert bucket.accuracy == 0.9000
    assert bucket.mean_confidence == 0.9500
    assert bucket.calibration_gap == 0.0500


def test_ground_truth_dataset_evaluation_runs():
    """Verify end-to-end evaluation execution over actual ground truth files."""
    manifest = [
        ("sample_data/clean_invoice.pdf", "sample_data/ground_truth/clean_invoice.json"),
        ("sample_data/clean_payslip.png", "sample_data/ground_truth/clean_payslip.json"),
    ]

    report = evaluate_dataset(manifest)
    assert report.sample_count == 2
    assert report.overall_precision > 0.85
    assert report.overall_recall > 0.85
    assert "High ([0.90, 1.00])" in report.calibration_buckets
    assert report.calibration_buckets["High ([0.90, 1.00])"].total_count > 0
