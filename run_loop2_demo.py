"""
Demonstration runner for Loop 2: Dual-Path Extraction.
Runs both Path A (OCR engine) and Path B (Vision model) on sample_data/clean_invoice.pdf.
Shows:
1. Two raw outputs side by side for fields, including an induced/real disagreement (InvoiceId).
2. Table structure extraction with rows and per-cell confidence matrix.
3. Target JSON schema output with source bounding boxes, encrypted crop refs, and agreement flags.
"""

import json
from trustextract.ingestion.pipeline import IngestionPipeline
from trustextract.extraction.pipeline import DualPathExtractor
from trustextract.extraction.path_b_vision import IndependentVisionModel


def main():
    ingestion = IngestionPipeline(storage_dir="encrypted_store")
    extraction = DualPathExtractor(
        vision_model=IndependentVisionModel(induce_disagreement_on="InvoiceId"),
        storage_dir="encrypted_store"
    )

    print("================================================================================")
    print("STEP 1: INGESTION & NORMALIZATION")
    print("================================================================================")
    res_ingest, norm_doc = ingestion.process_document(
        "sample_data/clean_invoice.pdf",
        document_id="doc_invoice_dual_path_demo",
        actor_id="demo_operator"
    )
    print(f"Status: {res_ingest.document_status.value}")
    print(f"Pages: {len(norm_doc.pages)}")
    print(f"Deskew applied: {norm_doc.metadata.get('deskew_applied')} (angle: {norm_doc.pages[0].deskew_angle}°)")

    print("\n================================================================================")
    print("STEP 2: DUAL-PATH EXTRACTION (Path A: RapidOCR vs. Path B: Vision Model)")
    print("================================================================================")
    result, side_by_side, _validation_report = extraction.process_normalized_document(norm_doc, actor_id="demo_operator")

    print("\nRAW OUTPUTS SIDE-BY-SIDE (PATH A vs. PATH B):")
    print("--------------------------------------------------------------------------------")
    key_fields = ["VendorName", "InvoiceId", "InvoiceTotal", "InvoiceDate", "Subtotal", "DueDate"]

    for fname in key_fields:
        if fname in side_by_side:
            info = side_by_side[fname]
            pa = info["path_a_ocr"]
            pb = info["path_b_vision"]
            agreement = info["agreement"]
            print(f"Field: {fname:<15} [Agreement: {agreement}]")
            print(f"  Path A (OCR):    Value = {pa['value']!r:<25} Conf = {pa['confidence']}  Raw = {pa['raw_text']}")
            print(f"  Path B (Vision): Value = {pb['value']!r:<25} Conf = {pb['confidence']}  Raw = {pb['raw_text']}")
            print()

    print("================================================================================")
    print("STEP 3: EXTRACTED TABLE STRUCTURE")
    print("================================================================================")
    for idx, table in enumerate(result.tables):
        print(f"Table #{idx + 1}:")
        print("  Rows:")
        for r_idx, r in enumerate(table.rows):
            print(f"    Row {r_idx + 1}: {r}")
        print("  Cell Confidences:")
        for c_idx, c in enumerate(table.confidence_per_cell):
            print(f"    Row {c_idx + 1} Conf: {c}")

    print("\n================================================================================")
    print("STEP 4: FINAL RECONCILED OUTPUT JSON (TARGET SCHEMA)")
    print("================================================================================")
    print(json.dumps(result.to_output_json(), indent=2))


if __name__ == "__main__":
    main()
