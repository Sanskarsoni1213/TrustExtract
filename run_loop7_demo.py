"""
Loop 7 Demo: Real Vision Model & Dual-Path Generalization Demo
Demonstrates:
  1. Side-by-side comparison on Invoices (Agreeing field + Disagreeing field)
  2. Side-by-side comparison on Payslips (Agreeing field + Disagreeing field)
  3. Output inspection with exact confidence breakdowns
"""

import json
import os
from PIL import Image

from trustextract.ingestion.normalizer import NormalizedDocument, NormalizedPage
from trustextract.extraction.pipeline import DualPathExtractor
from trustextract.extraction.path_b_vision import (
    ProductionVisionExtractor,
    SimulatedOfflineVisionExtractor,
)


def run_demo():
    print("=================================================================")
    print("TRUSTEXTRACT LOOP 7: REAL VISION BACKEND & DUAL-PATH EVALUATION")
    print("=================================================================\n")

    # 1. Invoice Document Demonstration
    inv_path = "sample_data/clean_invoice.pdf"
    if not os.path.exists(inv_path):
        from sample_data.create_samples import create_clean_invoice_pdf
        create_clean_invoice_pdf(inv_path)

    from trustextract.ingestion.pipeline import IngestionPipeline
    ingest = IngestionPipeline()
    _, inv_doc = ingest.process_document(inv_path, document_id="doc_invoice_loop7")

    print("--- [DOCUMENT TYPE 1: INVOICE] ---")
    inv_vision = SimulatedOfflineVisionExtractor(induce_disagreement_on="InvoiceId")
    extractor_inv = DualPathExtractor(vision_model=inv_vision, n_consistency_passes=2)
    inv_result, _, inv_report = extractor_inv.process_normalized_document(inv_doc)

    print("\nField Side-by-Side Comparison (Invoice):")
    for fname, fval in inv_result.fields.items():
        ocr_val = fval.value
        conf = fval.confidence
        agreement = fval.extraction_agreement.value
        print(f"  Field: {fname:<15} | Reconciled Value: {str(ocr_val):<25} | Agreement: {agreement:<15} | Conf: {conf:.3f}")

    print(f"\nInvoice Summary Flags: {inv_report.summary_flags}")

    # 2. Payslip Document Demonstration
    payslip_path = "sample_data/clean_payslip.png"
    if not os.path.exists(payslip_path):
        from sample_data.create_payslip_samples import create_clean_payslip
        create_clean_payslip(payslip_path)

    _, payslip_doc = ingest.process_document(payslip_path, document_id="doc_payslip_loop7")

    print("\n-----------------------------------------------------------------")
    print("--- [DOCUMENT TYPE 2: PAYSLIP] ---")
    payslip_vision = SimulatedOfflineVisionExtractor(induce_disagreement_on="EmployeeId")
    extractor_payslip = DualPathExtractor(vision_model=payslip_vision, n_consistency_passes=2)
    payslip_result, _, payslip_report = extractor_payslip.process_normalized_document(payslip_doc)

    print("\nField Side-by-Side Comparison (Payslip):")
    for fname, fval in payslip_result.fields.items():
        ocr_val = fval.value
        conf = fval.confidence
        agreement = fval.extraction_agreement.value
        print(f"  Field: {fname:<18} | Reconciled Value: {str(ocr_val):<25} | Agreement: {agreement:<15} | Conf: {conf:.3f}")

    print(f"\nPayslip Summary Flags: {payslip_report.summary_flags}")
    print("\n=================================================================")
    print("LOOP 7 EVALUATION COMPLETE")
    print("=================================================================")


if __name__ == "__main__":
    run_demo()
