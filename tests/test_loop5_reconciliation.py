"""
Loop 5 Unit & Integration Tests: Cross-Document Reconciliation & Entity Graphs
Tests:
  1. Entity graph building and canonical key mapping across document schemas.
  2. Matching bundle with zero conflicts (Invoice + PO agree on all fields).
  3. Field-type-aware reconciliation: Free-text near match & numeric/date format invariance.
  4. Field-type-aware reconciliation: Identifier strict mismatch detection.
  5. The Worst-Case DoD Test: Subtle InvoiceId tamper missed by Loops 1-4 caught by Loop 5 bundle reconciliation.
  6. Full bundle escalation and schema output validation.
"""

import pytest
from trustextract.schema import (
    CrossDocumentConflict,
    DocumentStatus,
    ExtractionAgreement,
    FieldSource,
    FieldType,
    FieldValue,
    ConfidenceBreakdown,
    TrustExtractResult,
)
from trustextract.reconciliation import (
    CrossDocumentReconciler,
    EntityClaim,
    EntityGraph,
    EntityNode,
    canonicalize_entity_key,
)


def test_canonicalize_entity_key():
    """Verify alias mapping to shared canonical entity keys."""
    assert canonicalize_entity_key("InvoiceId") == "invoice_id"
    assert canonicalize_entity_key("InvoiceNumber") == "invoice_id"
    assert canonicalize_entity_key("invoice_ref") == "invoice_id"
    assert canonicalize_entity_key("PurchaseOrderNumber") == "po_number"
    assert canonicalize_entity_key("PO_Number") == "po_number"
    assert canonicalize_entity_key("SupplierName") == "vendor_name"
    assert canonicalize_entity_key("CompanyName") == "vendor_name"
    assert canonicalize_entity_key("OrderAmount") == "invoice_total"
    assert canonicalize_entity_key("OrderDate") == "invoice_date"


def test_matching_bundle_zero_conflicts():
    """A bundle where Invoice and Purchase Order agree on all shared entities produces zero conflicts."""
    doc_inv = TrustExtractResult(
        document_id="INV-001",
        document_status=DocumentStatus.OK,
        fields={
            "VendorName": FieldValue(value="Contoso Logistics Inc.", confidence=0.98, field_type=FieldType.FREE_TEXT),
            "InvoiceId": FieldValue(value="INV-2024-9841", confidence=0.97, field_type=FieldType.IDENTIFIER),
            "PONumber": FieldValue(value="PO-88210", confidence=0.96, field_type=FieldType.IDENTIFIER),
            "InvoiceTotal": FieldValue(value="$972.00", confidence=0.95, field_type=FieldType.NUMERIC),
            "InvoiceDate": FieldValue(value="2024-03-15", confidence=0.97, field_type=FieldType.DATE),
        }
    )

    doc_po = TrustExtractResult(
        document_id="PO-001",
        document_status=DocumentStatus.OK,
        fields={
            "SupplierName": FieldValue(value="Contoso Logistics Inc.", confidence=0.99, field_type=FieldType.FREE_TEXT),
            "InvoiceRef": FieldValue(value="INV-2024-9841", confidence=0.98, field_type=FieldType.IDENTIFIER),
            "PurchaseOrderNumber": FieldValue(value="PO-88210", confidence=0.99, field_type=FieldType.IDENTIFIER),
            "OrderTotal": FieldValue(value="972.00", confidence=0.98, field_type=FieldType.NUMERIC),
            "OrderDate": FieldValue(value="March 15, 2024", confidence=0.96, field_type=FieldType.DATE),
        }
    )

    reconciler = CrossDocumentReconciler()
    report = reconciler.reconcile_bundle([doc_inv, doc_po])

    assert report.conflicting_entities_count == 0
    assert len(report.conflicts) == 0
    assert doc_inv.document_status == DocumentStatus.OK
    assert doc_po.document_status == DocumentStatus.OK
    assert len(doc_inv.cross_document_conflicts) == 0
    assert len(doc_po.cross_document_conflicts) == 0


def test_free_text_near_match_tolerance_across_docs():
    """Free text variations (e.g. Inc. vs Inc) do not trigger false cross-document conflicts."""
    doc_a = TrustExtractResult(
        document_id="DOC-A",
        document_status=DocumentStatus.OK,
        fields={
            "VendorName": FieldValue(value="Contoso Logistics Inc.", confidence=0.98, field_type=FieldType.FREE_TEXT),
        }
    )
    doc_b = TrustExtractResult(
        document_id="DOC-B",
        document_status=DocumentStatus.OK,
        fields={
            "SupplierName": FieldValue(value="Contoso Logistics, Inc", confidence=0.96, field_type=FieldType.FREE_TEXT),
        }
    )

    reconciler = CrossDocumentReconciler()
    report = reconciler.reconcile_bundle([doc_a, doc_b])

    assert len(report.conflicts) == 0
    assert doc_a.document_status == DocumentStatus.OK


def test_numeric_and_date_format_invariance_across_docs():
    """Numeric ($972.00 vs 972) and dates (2024-03-15 vs March 15, 2024) reconcile without conflict."""
    doc_a = TrustExtractResult(
        document_id="DOC-A",
        document_status=DocumentStatus.OK,
        fields={
            "InvoiceTotal": FieldValue(value="$972.00", confidence=0.95, field_type=FieldType.NUMERIC),
            "InvoiceDate": FieldValue(value="2024-03-15", confidence=0.95, field_type=FieldType.DATE),
        }
    )
    doc_b = TrustExtractResult(
        document_id="DOC-B",
        document_status=DocumentStatus.OK,
        fields={
            "OrderAmount": FieldValue(value="972.0", confidence=0.95, field_type=FieldType.NUMERIC),
            "DocumentDate": FieldValue(value="2024/03/15", confidence=0.95, field_type=FieldType.DATE),
        }
    )

    reconciler = CrossDocumentReconciler()
    report = reconciler.reconcile_bundle([doc_a, doc_b])

    assert len(report.conflicts) == 0


def test_dod_worst_case_tampered_invoice_vs_genuine_po():
    """
    DoD Test: Subtle InvoiceId tamper ('INV-2024-9999' on Invoice vs 'INV-2024-9841' on PO).
    
    Loops 1-4 passed the single invoice as OK (0 flags, 1.0 integrity score).
    Loop 5 bundle reconciliation must catch the conflict, populate cross_document_conflicts[],
    and escalate document_status to NEEDS_REVIEW with full crop & value references.
    """
    # 1. Tampered Invoice (passed Loops 1-4 with status OK and high confidence)
    tampered_invoice = TrustExtractResult(
        document_id="DOC-INVOICE-TAMPERED-001",
        document_status=DocumentStatus.OK,
        integrity_score=1.0,
        authenticity_flags=[],
        escalation_reasons=[],
        fields={
            "InvoiceId": FieldValue(
                value="INV-2024-9999",  # Tampered digit!
                confidence=0.95,
                field_type=FieldType.IDENTIFIER,
                source=FieldSource(page=1, bbox=[130, 130, 125, 16], crop_ref="crop://invoice_tampered_id")
            ),
            "VendorName": FieldValue(
                value="Contoso Logistics Inc.",
                confidence=0.98,
                field_type=FieldType.FREE_TEXT,
                source=FieldSource(page=1, bbox=[50, 80, 200, 16], crop_ref="crop://vendor_name")
            ),
            "InvoiceTotal": FieldValue(
                value="$972.00",
                confidence=0.95,
                field_type=FieldType.NUMERIC,
                source=FieldSource(page=1, bbox=[400, 430, 120, 16], crop_ref="crop://invoice_total")
            )
        }
    )

    # 2. Corroborating Genuine Purchase Order (stating the original true ID)
    corroborating_po = TrustExtractResult(
        document_id="DOC-PO-GENUINE-8821",
        document_status=DocumentStatus.OK,
        integrity_score=1.0,
        authenticity_flags=[],
        escalation_reasons=[],
        fields={
            "InvoiceNumber": FieldValue(
                value="INV-2024-9841",  # True original ID
                confidence=0.98,
                field_type=FieldType.IDENTIFIER,
                source=FieldSource(page=1, bbox=[140, 120, 120, 16], crop_ref="crop://po_invoice_ref")
            ),
            "SupplierName": FieldValue(
                value="Contoso Logistics Inc.",
                confidence=0.99,
                field_type=FieldType.FREE_TEXT,
                source=FieldSource(page=1, bbox=[60, 90, 200, 16], crop_ref="crop://po_supplier")
            ),
            "OrderAmount": FieldValue(
                value="$972.00",
                confidence=0.97,
                field_type=FieldType.NUMERIC,
                source=FieldSource(page=1, bbox=[410, 440, 120, 16], crop_ref="crop://po_total")
            )
        }
    )

    # 3. Execute Loop 5 Cross-Document Reconciliation Pass
    reconciler = CrossDocumentReconciler()
    report = reconciler.reconcile_bundle([tampered_invoice, corroborating_po])

    # 4. Verify reconciliation outputs
    assert report.conflicting_entities_count == 1
    assert len(report.conflicts) == 1

    conflict = report.conflicts[0]
    assert conflict.entity_key == "invoice_id"
    assert "DOC-INVOICE-TAMPERED-001" in conflict.document_ids
    assert "DOC-PO-GENUINE-8821" in conflict.document_ids
    assert conflict.resolution_status == "UNRESOLVED"

    # Check that both conflicting values and source crops are explicitly captured
    val_map = {cv["document_id"]: cv for cv in conflict.conflicting_values}
    assert val_map["DOC-INVOICE-TAMPERED-001"]["value"] == "INV-2024-9999"
    assert val_map["DOC-INVOICE-TAMPERED-001"]["crop_ref"] == "crop://invoice_tampered_id"
    assert val_map["DOC-PO-GENUINE-8821"]["value"] == "INV-2024-9841"
    assert val_map["DOC-PO-GENUINE-8821"]["crop_ref"] == "crop://po_invoice_ref"

    # Verify that document_status was escalated to NEEDS_REVIEW on both participating documents
    assert tampered_invoice.document_status == DocumentStatus.NEEDS_REVIEW
    assert corroborating_po.document_status == DocumentStatus.NEEDS_REVIEW

    # Verify explicit escalation reasons
    assert any("CROSS_DOCUMENT_CONFLICT" in r and "invoice_id" in r for r in tampered_invoice.escalation_reasons)
    assert any("CROSS_DOCUMENT_CONFLICT" in r and "invoice_id" in r for r in corroborating_po.escalation_reasons)

    # Verify cross_document_conflicts array on the document schema
    assert len(tampered_invoice.cross_document_conflicts) == 1
    assert tampered_invoice.cross_document_conflicts[0].entity_key == "invoice_id"
