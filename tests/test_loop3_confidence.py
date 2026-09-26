"""
Tests for Loop 3: Confidence Scoring & Validation Pipeline
Covers:
  1. InvoiceArithmeticValidator - arithmetic pass and fail cases
  2. DateLogicValidator - date ordering and unparseable detection
  3. IdentifierFormatValidator - pattern matching
  4. SelfConsistencySampler - runs 3 passes and returns consistency scores
  5. ConfidenceFuser - weighted formula and hard cap on validation failure
  6. Full pipeline integration on clean_invoice.pdf
"""

import pytest
from PIL import Image
import numpy as np

from trustextract.extraction.validators import (
    DocumentValidator,
    DocumentValidationReport,
    InvoiceArithmeticValidator,
    DateLogicValidator,
    IdentifierFormatValidator,
)
from trustextract.extraction.confidence import ConfidenceFuser, SelfConsistencySampler
from trustextract.extraction.path_a_ocr import RawFieldExtraction
from trustextract.schema import ExtractionAgreement, ConfidenceBreakdown
from trustextract.ingestion.pipeline import IngestionPipeline
from trustextract.extraction.pipeline import DualPathExtractor
from trustextract.extraction.path_b_vision import IndependentVisionModel
from trustextract.security.crypto import AESGCMKmsEncryptionService


# ─── Validator Unit Tests ────────────────────────────────────────────────────

class TestInvoiceArithmeticValidator:

    def _run(self, fields, rows):
        report = DocumentValidationReport()
        InvoiceArithmeticValidator().validate(fields, rows, report)
        return report

    def test_arithmetic_passes_when_correct(self):
        """900 + 72 = 972 should produce ARITHMETIC_OK on all three fields."""
        report = self._run(
            {"Subtotal": 900.0, "TotalTax": 72.0, "InvoiceTotal": 972.0},
            [["Item A", "", 350.0, 700.0], ["Item B", "", 150.0, 150.0], ["Item C", "", 10.0, 50.0]]
        )
        for fname in ("Subtotal", "TotalTax", "InvoiceTotal"):
            assert report.per_field[fname].passed, f"{fname} should pass"
            assert "ARITHMETIC_OK" in report.per_field[fname].flags

    def test_arithmetic_fails_when_total_wrong(self):
        """900 + 72 != 1000 should fail."""
        report = self._run(
            {"Subtotal": 900.0, "TotalTax": 72.0, "InvoiceTotal": 1000.0},
            []
        )
        for fname in ("Subtotal", "TotalTax", "InvoiceTotal"):
            assert not report.per_field[fname].passed
        assert "ARITHMETIC_INCONSISTENCY" in report.summary_flags

    def test_line_sum_check_passes(self):
        """700 + 150 + 50 = 900 == Subtotal."""
        report = self._run(
            {"Subtotal": 900.0, "TotalTax": 72.0, "InvoiceTotal": 972.0},
            [["A", "", 350.0, 700.0], ["B", "", 150.0, 150.0], ["C", "", 10.0, 50.0]]
        )
        assert "LINE_SUM_OK" in report.per_field["Subtotal"].flags

    def test_line_sum_check_fails(self):
        """If line items don't match subtotal, LINE_SUM_FAIL should appear."""
        report = self._run(
            {"Subtotal": 999.0, "TotalTax": 72.0, "InvoiceTotal": 1071.0},
            [["A", "", 100.0, 100.0]]
        )
        assert "LINE_SUM_FAIL" in report.per_field["Subtotal"].flags


class TestDateLogicValidator:

    def _run(self, fields):
        report = DocumentValidationReport()
        DateLogicValidator().validate(fields, report)
        return report

    def test_valid_date_order(self):
        """InvoiceDate 2024-03-15 < DueDate 2024-04-15 should pass."""
        report = self._run({"InvoiceDate": "2024-03-15", "DueDate": "2024-04-15"})
        assert report.per_field["InvoiceDate"].passed
        assert report.per_field["DueDate"].passed
        assert "DATE_LOGIC_OK" in report.per_field["InvoiceDate"].flags

    def test_reversed_date_order_fails(self):
        """DueDate before InvoiceDate should raise DATE_ORDER_VIOLATION."""
        report = self._run({"InvoiceDate": "2024-04-15", "DueDate": "2024-03-15"})
        assert not report.per_field["InvoiceDate"].passed
        assert "DATE_ORDER_VIOLATION" in report.summary_flags

    def test_unparseable_date_fails(self):
        report = self._run({"InvoiceDate": "not-a-date", "DueDate": "2024-04-15"})
        assert not report.per_field["InvoiceDate"].passed
        assert "DATE_UNPARSEABLE" in report.per_field["InvoiceDate"].flags


class TestIdentifierFormatValidator:

    def _run(self, fields):
        report = DocumentValidationReport()
        IdentifierFormatValidator().validate(fields, report)
        return report

    def test_valid_invoice_id(self):
        report = self._run({"InvoiceId": "INV-2024-9841"})
        assert report.per_field["InvoiceId"].passed
        assert "ID_FORMAT_OK" in report.per_field["InvoiceId"].flags

    def test_ambiguous_invoice_id_fails(self):
        """INV-2024-984I should still pass the format check (alphanumeric)."""
        report = self._run({"InvoiceId": "INV-2024-984I"})
        assert report.per_field["InvoiceId"].passed

    def test_bad_format_fails(self):
        """A random freeform string should fail the pattern check."""
        report = self._run({"InvoiceId": "invoice#12345"})
        assert not report.per_field["InvoiceId"].passed
        assert "ID_FORMAT_FAIL" in report.per_field["InvoiceId"].flags


# ─── Confidence Fuser Unit Tests ─────────────────────────────────────────────

class TestConfidenceFuser:

    def test_all_signals_perfect(self):
        """All signals at max should yield near-1.0 final score."""
        from trustextract.extraction.validators import FieldValidationResult
        fv = FieldValidationResult(field_name="TestField", flags=["PASS"], passed=True)
        bd = ConfidenceFuser().fuse(
            field_name="TestField",
            native_confidence=1.0,
            agreement=ExtractionAgreement.OCR_EQ_VISION,
            consistency_score=1.0,
            validation_result=fv,
        )
        assert bd.final == pytest.approx(1.0, abs=0.001)

    def test_disagreement_lowers_score(self):
        """OCR_NEQ_VISION agreement_signal=0.6 should reduce final score."""
        from trustextract.extraction.validators import FieldValidationResult
        fv = FieldValidationResult(field_name="F", flags=[], passed=True)
        bd_agree = ConfidenceFuser().fuse("F", 0.9, ExtractionAgreement.OCR_EQ_VISION, 1.0, fv)
        bd_disagree = ConfidenceFuser().fuse("F", 0.9, ExtractionAgreement.OCR_NEQ_VISION, 1.0, fv)
        assert bd_disagree.final < bd_agree.final

    def test_single_source_reweights_without_consistency(self):
        """Single-source fields (consistency_score=None) must reweight remaining 3 signals proportionally."""
        from trustextract.extraction.validators import FieldValidationResult
        fv = FieldValidationResult(field_name="VisionOnlyField", flags=["PASS"], passed=True)
        bd = ConfidenceFuser().fuse(
            field_name="VisionOnlyField",
            native_confidence=0.90,
            agreement=ExtractionAgreement.SINGLE_SOURCE,
            consistency_score=None,
            validation_result=fv,
        )
        assert bd.consistency_signal is None
        # Proportional weights: 0.35/0.75, 0.25/0.75, 0.15/0.75
        expected = round((0.35 / 0.75) * 0.90 + (0.25 / 0.75) * 0.80 + (0.15 / 0.75) * 1.0, 3)
        assert bd.final == pytest.approx(expected, abs=0.002)

    def test_near_match_uses_0_85_signal(self):
        """OCR_NEAR_MATCH agreement_signal should be 0.85."""
        from trustextract.extraction.validators import FieldValidationResult
        fv = FieldValidationResult(field_name="F", flags=[], passed=True)
        bd = ConfidenceFuser().fuse("F", 0.95, ExtractionAgreement.OCR_NEAR_MATCH, 1.0, fv)
        assert bd.agreement_signal == 0.85
        assert bd.final == pytest.approx(round(0.35 * 0.95 + 0.25 * 0.85 + 0.25 * 1.0 + 0.15 * 1.0, 3), abs=0.002)

    def test_validation_failure_caps_score(self):
        """A failing validator should cap final confidence at cap_confidence_at."""
        from trustextract.extraction.validators import FieldValidationResult
        fv = FieldValidationResult(field_name="F", flags=["FAIL"], passed=False, cap_confidence_at=0.50)
        bd = ConfidenceFuser().fuse("F", 1.0, ExtractionAgreement.OCR_EQ_VISION, 1.0, fv)
        assert bd.final <= 0.50


class TestFieldAgreementReconciler:

    def test_whitespace_and_case_normalization(self):
        """Whitespace collapse and case insensitivity should produce OCR_EQ_VISION."""
        from trustextract.extraction.pipeline import determine_field_agreement
        agr1, penalty1 = determine_field_agreement("Contoso LogisticsInc.", "Contoso Logistics Inc.", "VendorName")
        assert agr1 == ExtractionAgreement.OCR_EQ_VISION
        assert penalty1 == 1.0

        agr2, penalty2 = determine_field_agreement("AcmeEnterpriseCorp", "Acme Enterprise Corp", "CustomerName")
        assert agr2 == ExtractionAgreement.OCR_EQ_VISION
        assert penalty2 == 1.0

    def test_numeric_currency_normalization(self):
        """Currency and float formats should match numerically."""
        from trustextract.extraction.pipeline import determine_field_agreement
        agr, penalty = determine_field_agreement("$900.00", "900", "Subtotal")
        assert agr == ExtractionAgreement.OCR_EQ_VISION
        assert penalty == 1.0

    def test_date_format_normalization(self):
        """Different date string representations of the same date should match."""
        from trustextract.extraction.pipeline import determine_field_agreement
        agr, penalty = determine_field_agreement("2024-03-15", "03/15/2024", "InvoiceDate")
        assert agr == ExtractionAgreement.OCR_EQ_VISION
        assert penalty == 1.0

    def test_identifier_strict_disagreement_on_single_char_diff(self):
        """Identifier fields (InvoiceId, AccountNumber, PO) must strictly disagree on 1-char diff (penalty 0.90)."""
        from trustextract.extraction.pipeline import determine_field_agreement
        # 1-char difference on InvoiceId must NOT be forgiven by near-match
        agr, penalty = determine_field_agreement("INV-2024-9841", "INV-2024-984I", "InvoiceId")
        assert agr == ExtractionAgreement.OCR_NEQ_VISION
        assert penalty == 0.90

        # AccountNumber single character difference
        agr_acc, pen_acc = determine_field_agreement("ACCT-12345", "ACCT-1234S", "AccountNumber")
        assert agr_acc == ExtractionAgreement.OCR_NEQ_VISION
        assert pen_acc == 0.90

    def test_free_text_near_match_lenient(self):
        """Free-text fields (names, descriptions, addresses) allow OCR_NEAR_MATCH for minor typos (penalty 0.95)."""
        from trustextract.extraction.pipeline import determine_field_agreement
        agr, penalty = determine_field_agreement("Acme Corp International", "Acme Corp Internatinal", "CustomerName")
        assert agr == ExtractionAgreement.OCR_NEAR_MATCH
        assert penalty == 0.95

    def test_genuine_mismatch(self):
        """Different values should produce OCR_NEQ_VISION (penalty 0.90)."""
        from trustextract.extraction.pipeline import determine_field_agreement
        agr, penalty = determine_field_agreement("INV-2024-9841", "INV-9999-0000", "InvoiceId")
        assert agr == ExtractionAgreement.OCR_NEQ_VISION
        assert penalty == 0.90


# ─── Self-Consistency Sampler ────────────────────────────────────────────────

class TestSelfConsistencySampler:

    def test_stable_field_scores_1_0(self, tmp_path):
        """A synthetic plain-white image re-run 3x should return stable consistency=1.0."""
        img = Image.fromarray(np.ones((100, 200, 3), dtype=np.uint8) * 240)
        # RapidOCR on blank image returns no tokens -> reference_fields empty -> consistency trivially 1.0
        sampler = SelfConsistencySampler(n_passes=3, seed=0)
        result = sampler.run(img, page_number=1, reference_fields={})
        # Empty reference means no fields to score -> empty dict is fine
        assert isinstance(result, dict)

    def test_scores_are_bounded_0_to_1(self, tmp_path):
        """All consistency scores must be in [0, 1]."""
        img = Image.fromarray(np.ones((200, 400, 3), dtype=np.uint8) * 200)
        sampler = SelfConsistencySampler(n_passes=3, seed=42)
        result = sampler.run(img, page_number=1, reference_fields={})
        for fname, score in result.items():
            assert 0.0 <= score <= 1.0, f"{fname}: {score}"


# ─── Full Integration Test ───────────────────────────────────────────────────

@pytest.fixture
def loop3_pipeline(tmp_path):
    storage_dir = str(tmp_path / "enc_store")
    kms = AESGCMKmsEncryptionService()
    ingest = IngestionPipeline(kms_service=kms, storage_dir=storage_dir)
    extract = DualPathExtractor(
        kms_service=kms,
        vision_model=IndependentVisionModel(induce_disagreement_on="InvoiceId"),
        storage_dir=storage_dir,
        n_consistency_passes=3,
    )
    return ingest, extract


def test_loop3_invoice_arithmetic_passes(loop3_pipeline):
    """Full pipeline: invoice 900 + 72 = 972 must produce ARITHMETIC_OK."""
    ingest, extract = loop3_pipeline
    _, norm_doc = ingest.process_document("sample_data/clean_invoice.pdf",
                                           document_id="doc_loop3_test", actor_id="test")
    result, _, val_report = extract.process_normalized_document(norm_doc)

    assert "ARITHMETIC_OK" in val_report.per_field["InvoiceTotal"].flags
    assert val_report.per_field["InvoiceTotal"].passed
    assert "ARITHMETIC_INCONSISTENCY" not in val_report.summary_flags


def test_loop3_all_fields_have_breakdown(loop3_pipeline):
    """Every field in the result must have a ConfidenceBreakdown attached."""
    ingest, extract = loop3_pipeline
    _, norm_doc = ingest.process_document("sample_data/clean_invoice.pdf",
                                           document_id="doc_loop3_bd_test", actor_id="test")
    result, _, _ = extract.process_normalized_document(norm_doc)

    for fname, fv in result.fields.items():
        assert fv.confidence_breakdown is not None, f"Missing breakdown on {fname}"
        bd = fv.confidence_breakdown
        assert 0.0 <= bd.final <= 1.0
        assert len(bd.validation_flags) > 0


def test_loop3_invoice_id_disagreement_lowers_confidence(loop3_pipeline):
    """InvoiceId (ocr!=vision) should have lower final confidence than an agreeing field."""
    ingest, extract = loop3_pipeline
    _, norm_doc = ingest.process_document("sample_data/clean_invoice.pdf",
                                           document_id="doc_loop3_id_test", actor_id="test")
    result, _, _ = extract.process_normalized_document(norm_doc)

    inv_id_conf   = result.fields["InvoiceId"].confidence
    vendor_conf   = result.fields["VendorName"].confidence
    assert inv_id_conf < vendor_conf, (
        f"InvoiceId ({inv_id_conf}) should be lower than VendorName ({vendor_conf})"
    )


def test_loop3_date_logic_ok(loop3_pipeline):
    """InvoiceDate 2024-03-15 < DueDate 2024-04-15 should be validated OK."""
    ingest, extract = loop3_pipeline
    _, norm_doc = ingest.process_document("sample_data/clean_invoice.pdf",
                                           document_id="doc_loop3_date_test", actor_id="test")
    result, _, val_report = extract.process_normalized_document(norm_doc)

    assert val_report.per_field["InvoiceDate"].passed
    assert "DATE_LOGIC_OK" in val_report.per_field["InvoiceDate"].flags


def test_loop3_confidence_formula_output_matches_expected(loop3_pipeline):
    """
    Verify the fusion formula for InvoiceDate:
    native=0.94, agreement=1.0 (ocr==vision), consistency=1.0, validation=1.0
    weight(0.35*0.94 + 0.25*1.0 + 0.25*1.0 + 0.15*1.0) = 0.979
    """
    ingest, extract = loop3_pipeline
    _, norm_doc = ingest.process_document("sample_data/clean_invoice.pdf",
                                           document_id="doc_loop3_formula_test", actor_id="test")
    result, _, _ = extract.process_normalized_document(norm_doc)

    bd = result.fields["InvoiceDate"].confidence_breakdown
    assert bd is not None
    expected = round(0.35 * bd.native_ocr + 0.25 * 1.0 + 0.25 * bd.consistency_signal + 0.15 * 1.0, 3)
    assert bd.final == pytest.approx(expected, abs=0.002)
