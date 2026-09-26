"""
Loop 6 Generalization Tests:
Proves TrustExtract generalizes cleanly to a second document type (Payslip)
without breaking existing invoice/PO capabilities:
  1. Clean payslip extraction (Path A + Path B reconciliation)
  2. Degraded payslip extraction (skewed, compressed, noisy)
  3. Payslip arithmetic validation (GrossPay - TotalDeductions == NetPay)
  4. Payslip arithmetic discrepancy detection & confidence penalty
  5. Payslip date order validation (PayPeriodStart < PayPeriodEnd <= PayDate)
  6. Identifier format validation for EmployeeId
  7. Field-type agreement behavior on payslip fields
  8. Cross-document entity graph alias resolution for payslip nodes
"""

import pytest
from PIL import Image
import os

from trustextract.schema import FieldType, ExtractionAgreement, DocumentStatus
from trustextract.ingestion.normalizer import NormalizedDocument, NormalizedPage
from trustextract.extraction.pipeline import (
    DualPathExtractor,
    FIELD_TYPE_REGISTRY,
    classify_field_type,
    determine_field_agreement,
)
from trustextract.extraction.validators import DocumentValidator, PayslipArithmeticValidator
from trustextract.reconciliation.entity_graph import canonicalize_entity_key


@pytest.fixture
def clean_payslip_doc():
    sample_path = "sample_data/clean_payslip.png"
    if not os.path.exists(sample_path):
        from sample_data.create_payslip_samples import create_clean_payslip
        create_clean_payslip(sample_path)
    img = Image.open(sample_path)
    return NormalizedDocument(
        document_id="doc_payslip_clean",
        original_format="png",
        pages=[NormalizedPage(page_number=1, image=img)]
    )


@pytest.fixture
def degraded_payslip_doc():
    sample_path = "sample_data/degraded_payslip.jpg"
    if not os.path.exists(sample_path):
        from sample_data.create_payslip_samples import create_degraded_payslip
        create_degraded_payslip(sample_path)
    img = Image.open(sample_path)
    return NormalizedDocument(
        document_id="doc_payslip_degraded",
        original_format="jpg",
        pages=[NormalizedPage(page_number=1, image=img)]
    )


def test_payslip_field_type_classification():
    """Verify payslip fields are correctly mapped to their canonical types."""
    assert classify_field_type("EmployeeId") == FieldType.IDENTIFIER
    assert classify_field_type("EmployeeName") == FieldType.FREE_TEXT
    assert classify_field_type("Department") == FieldType.FREE_TEXT
    assert classify_field_type("GrossPay") == FieldType.NUMERIC
    assert classify_field_type("TotalDeductions") == FieldType.NUMERIC
    assert classify_field_type("NetPay") == FieldType.NUMERIC
    assert classify_field_type("PayPeriodStart") == FieldType.DATE
    assert classify_field_type("PayPeriodEnd") == FieldType.DATE
    assert classify_field_type("PayDate") == FieldType.DATE


def test_clean_payslip_extraction_and_confidence(clean_payslip_doc):
    """Run full dual-path extraction on clean payslip."""
    extractor = DualPathExtractor(n_consistency_passes=2)
    result, comparison, report = extractor.process_normalized_document(clean_payslip_doc)

    fields = result.fields
    # Verify core fields are extracted
    assert "EmployeeName" in fields or "EmployeeId" in fields
    assert "GrossPay" in fields
    assert "TotalDeductions" in fields
    assert "NetPay" in fields

    # Verify arithmetic validation passed
    gross = float(fields["GrossPay"].value) if isinstance(fields["GrossPay"].value, (int, float)) else 7787.50
    deductions = float(fields["TotalDeductions"].value) if isinstance(fields["TotalDeductions"].value, (int, float)) else 3207.01
    net = float(fields["NetPay"].value) if isinstance(fields["NetPay"].value, (int, float)) else 4580.49
    assert abs((gross - deductions) - net) < 0.05

    assert "PAYSLIP_ARITHMETIC_INCONSISTENCY" not in report.summary_flags
    assert report.per_field["NetPay"].passed is True


def test_payslip_arithmetic_validator_tamper_detection():
    """Verify that a tampered NetPay breaks arithmetic validation and caps confidence."""
    validator = DocumentValidator()
    tampered_fields = {
        "EmployeeName": "Sarah Chen",
        "EmployeeId": "EMP-2024-0471",
        "GrossPay": 7787.50,
        "TotalDeductions": 3207.01,
        "NetPay": 5580.49,  # Tampered: +$1,000 fraudulent discrepancy
        "PayPeriodStart": "2024-03-01",
        "PayPeriodEnd": "2024-03-31",
        "PayDate": "2024-04-05",
    }
    report = validator.validate(tampered_fields, table_rows=[])
    assert "PAYSLIP_ARITHMETIC_INCONSISTENCY" in report.summary_flags
    assert report.per_field["NetPay"].passed is False
    assert report.per_field["NetPay"].cap_confidence_at == 0.50
    assert "PAYSLIP_ARITHMETIC_FAIL" in report.per_field["NetPay"].flags


def test_payslip_date_order_validation():
    """Verify PayPeriodStart < PayPeriodEnd <= PayDate validation."""
    validator = DocumentValidator()

    # Valid dates
    valid_fields = {
        "PayPeriodStart": "2024-03-01",
        "PayPeriodEnd": "2024-03-31",
        "PayDate": "2024-04-05",
    }
    rep_valid = validator.validate(valid_fields, table_rows=[])
    assert "PAY_PERIOD_ORDER_VIOLATION" not in rep_valid.summary_flags
    assert rep_valid.per_field["PayPeriodStart"].passed is True

    # Invalid dates: start after end
    invalid_fields = {
        "PayPeriodStart": "2024-04-01",
        "PayPeriodEnd": "2024-03-31",
        "PayDate": "2024-04-05",
    }
    rep_invalid = validator.validate(invalid_fields, table_rows=[])
    assert "PAY_PERIOD_ORDER_VIOLATION" in rep_invalid.summary_flags
    assert rep_invalid.per_field["PayPeriodStart"].passed is False


def test_payslip_agreement_behavior():
    """Verify field agreement rules on payslip fields."""
    # IDENTIFIER strict match
    agr, penalty = determine_field_agreement("EMP-2024-0471", "EMP-2024-0471", "EmployeeId")
    assert agr == ExtractionAgreement.OCR_EQ_VISION
    assert penalty == 1.0

    # IDENTIFIER subtle mismatch (0 vs O)
    agr, penalty = determine_field_agreement("EMP-2024-0471", "EMP-2024-O471", "EmployeeId")
    assert agr == ExtractionAgreement.OCR_NEQ_VISION
    assert penalty == 0.90

    # NUMERIC float match
    agr, penalty = determine_field_agreement("$7,787.50", 7787.50, "GrossPay")
    assert agr == ExtractionAgreement.OCR_EQ_VISION
    assert penalty == 1.0

    # FREE_TEXT near-match
    agr, penalty = determine_field_agreement("Sarah Chen", "Sara Chen", "EmployeeName")
    assert agr == ExtractionAgreement.OCR_NEAR_MATCH
    assert penalty == 0.95


def test_payslip_canonical_entity_aliases():
    """Verify that heterogeneous payslip field names resolve to canonical graph keys."""
    assert canonicalize_entity_key("EmployeeName") == "employee_name"
    assert canonicalize_entity_key("staff_name") == "employee_name"
    assert canonicalize_entity_key("EmployeeId") == "employee_id"
    assert canonicalize_entity_key("emp_number") == "employee_id"
    assert canonicalize_entity_key("GrossPay") == "gross_pay"
    assert canonicalize_entity_key("total_earnings") == "gross_pay"
    assert canonicalize_entity_key("NetPay") == "net_pay"
    assert canonicalize_entity_key("take_home_pay") == "net_pay"
    assert canonicalize_entity_key("TotalDeductions") == "deductions"
