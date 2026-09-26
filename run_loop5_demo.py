"""
Loop 5 Demonstration: Cross-Document Reconciliation & Entity Graph Engine
Demonstrates the Definition of Done (DoD) scenario:
  - Document 1: Invoice with subtle tampered InvoiceId ('INV-2024-9999' vs true 'INV-2024-9841')
    that passed Loops 1-4 with status OK and 1.0 integrity score.
  - Document 2: Corroborating genuine Purchase Order stating the true 'INV-2024-9841'.
  - Executes CrossDocumentReconciler and outputs the final JSON with cross_document_conflicts[].
"""

import json
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
from trustextract.reconciliation import CrossDocumentReconciler


def run_loop5_demonstration():
    print("=" * 80)
    print("TRUSTEXTRACT LOOP 5 — CROSS-DOCUMENT RECONCILIATION DEMONSTRATION")
    print("=" * 80)

    # ─── 1. Document 1: Invoice with subtle tampered InvoiceId (passed Loops 1-4) ───
    print("\n" + "-" * 80)
    print("DOCUMENT 1: Submitted Invoice (Contains subtle single-document tamper on InvoiceId)")
    print("-" * 80)
    invoice_doc = TrustExtractResult(
        document_id="DOC-INVOICE-2024-001",
        document_status=DocumentStatus.OK,
        integrity_score=1.0,
        authenticity_flags=[],
        escalation_reasons=[],
        fields={
            "InvoiceId": FieldValue(
                value="INV-2024-9999",  # Tampered value!
                confidence=0.95,
                field_type=FieldType.IDENTIFIER,
                extraction_agreement=ExtractionAgreement.OCR_EQ_VISION,
                validation="PASSED",
                confidence_breakdown=ConfidenceBreakdown(
                    native_ocr=0.95,
                    agreement_signal=1.0,
                    consistency_signal=1.0,
                    validation_signal=1.0,
                    validation_flags=["PASSED"],
                    final=0.968
                ),
                source=FieldSource(page=1, bbox=[400, 47, 136, 16], crop_ref="crop://inv_001_id")
            ),
            "VendorName": FieldValue(
                value="Contoso Logistics Inc.",
                confidence=0.98,
                field_type=FieldType.FREE_TEXT,
                source=FieldSource(page=1, bbox=[50, 82, 195, 16], crop_ref="crop://inv_001_vendor")
            ),
            "InvoiceTotal": FieldValue(
                value="$972.00",
                confidence=0.96,
                field_type=FieldType.NUMERIC,
                source=FieldSource(page=1, bbox=[400, 431, 124, 18], crop_ref="crop://inv_001_total")
            ),
            "InvoiceDate": FieldValue(
                value="2024-03-15",
                confidence=0.97,
                field_type=FieldType.DATE,
                source=FieldSource(page=1, bbox=[400, 69, 109, 14], crop_ref="crop://inv_001_date")
            )
        }
    )
    print(f"Pre-reconciliation Status: {invoice_doc.document_status.value} (Integrity Score: {invoice_doc.integrity_score})")

    # ─── 2. Document 2: Corroborating Genuine Purchase Order ───────────────────
    print("\n" + "-" * 80)
    print("DOCUMENT 2: Corroborating Purchase Order (States true original ID: INV-2024-9841)")
    print("-" * 80)
    po_doc = TrustExtractResult(
        document_id="DOC-PO-2024-8821",
        document_status=DocumentStatus.OK,
        integrity_score=1.0,
        authenticity_flags=[],
        escalation_reasons=[],
        fields={
            "InvoiceNumber": FieldValue(
                value="INV-2024-9841",  # True original ID
                confidence=0.98,
                field_type=FieldType.IDENTIFIER,
                extraction_agreement=ExtractionAgreement.OCR_EQ_VISION,
                validation="PASSED",
                confidence_breakdown=ConfidenceBreakdown(
                    native_ocr=0.98,
                    agreement_signal=1.0,
                    consistency_signal=1.0,
                    validation_signal=1.0,
                    validation_flags=["PASSED"],
                    final=0.987
                ),
                source=FieldSource(page=1, bbox=[145, 120, 118, 16], crop_ref="crop://po_8821_inv_ref")
            ),
            "SupplierName": FieldValue(
                value="Contoso Logistics Inc.",
                confidence=0.99,
                field_type=FieldType.FREE_TEXT,
                source=FieldSource(page=1, bbox=[60, 90, 195, 16], crop_ref="crop://po_8821_supplier")
            ),
            "OrderAmount": FieldValue(
                value="$972.00",
                confidence=0.97,
                field_type=FieldType.NUMERIC,
                source=FieldSource(page=1, bbox=[410, 440, 120, 16], crop_ref="crop://po_8821_total")
            ),
            "OrderDate": FieldValue(
                value="March 15, 2024",
                confidence=0.96,
                field_type=FieldType.DATE,
                source=FieldSource(page=1, bbox=[410, 75, 115, 14], crop_ref="crop://po_8821_date")
            )
        }
    )
    print(f"Pre-reconciliation Status: {po_doc.document_status.value} (Integrity Score: {po_doc.integrity_score})")

    # ─── 3. Run Loop 5 Cross-Document Reconciliation ──────────────────────────
    print("\n" + "=" * 80)
    print("RUNNING BUNDLE CROSS-DOCUMENT RECONCILIATION PASS...")
    print("=" * 80)

    reconciler = CrossDocumentReconciler()
    report = reconciler.reconcile_bundle([invoice_doc, po_doc])

    print(f"Bundle Reconciliation Report Summary:")
    print(f"  - Total Reconciled Entities: {report.reconciled_entities_count}")
    print(f"  - Conflicting Entities Detected: {report.conflicting_entities_count}")
    print(f"  - Participating Documents: {report.participating_document_ids}")

    # ─── 4. Final Output JSON for the Tampered Invoice in the Reconciled Bundle ───
    print("\n" + "-" * 80)
    print("FINAL JSON OUTPUT: DOC-INVOICE-2024-001 (Post-Reconciliation)")
    print("-" * 80)
    print(json.dumps(invoice_doc.model_dump(), indent=2, default=str))

    # ─── 5. Final Output JSON for the Corroborating PO in the Reconciled Bundle ───
    print("\n" + "-" * 80)
    print("FINAL JSON OUTPUT: DOC-PO-2024-8821 (Post-Reconciliation)")
    print("-" * 80)
    print(json.dumps(po_doc.model_dump(), indent=2, default=str))

    # Verification assertions
    assert len(invoice_doc.cross_document_conflicts) == 1
    assert invoice_doc.document_status == DocumentStatus.NEEDS_REVIEW
    assert po_doc.document_status == DocumentStatus.NEEDS_REVIEW
    assert any("CROSS_DOCUMENT_CONFLICT" in r for r in invoice_doc.escalation_reasons)

    print("\n" + "=" * 80)
    print("LOOP 5 DEFINITION OF DONE DEMONSTRATION VERIFIED AND LOCKED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_loop5_demonstration()
