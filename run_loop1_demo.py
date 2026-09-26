"""
Demonstration runner for Loop 1: Ingestion & Format Normalization.
Runs the pipeline on:
1. Clean document sample (clean_invoice.pdf) -> proceeds with status OK
2. Deliberately degraded / blurry photograph sample (deliberately_blurry_sample.jpg) -> returns UNPROCESSABLE with reason
Prints the exact target JSON schema output for both.
"""

import json
from trustextract.ingestion.pipeline import IngestionPipeline


def main():
    pipeline = IngestionPipeline(storage_dir="encrypted_store")

    print("================================================================================")
    print("TEST RUN 1: CLEAN SAMPLE (sample_data/clean_invoice.pdf)")
    print("================================================================================")
    result_clean, norm_doc = pipeline.process_document(
        "sample_data/clean_invoice.pdf",
        document_id="doc_clean_invoice_2024",
        actor_id="demo_operator"
    )
    clean_json = result_clean.to_output_json()
    print(json.dumps(clean_json, indent=2))
    if norm_doc:
        print(f"\n[Normalization Result]: {len(norm_doc.pages)} page(s) normalized and encrypted at rest.")
        print(f"[Format]: {norm_doc.original_format}")
        print(f"[Digital Text Extracted Preview]: {len(norm_doc.pages[0].text_layer or '')} chars")

    print("\n================================================================================")
    print("TEST RUN 2: DELIBERATELY LOW-QUALITY SAMPLE (sample_data/deliberately_blurry_sample.jpg)")
    print("================================================================================")
    result_blurry, norm_doc_blurry = pipeline.process_document(
        "sample_data/deliberately_blurry_sample.jpg",
        document_id="doc_blurry_photo_2024",
        actor_id="demo_operator"
    )
    blurry_json = result_blurry.to_output_json()
    print(json.dumps(blurry_json, indent=2))
    print(f"\n[Normalized Pages]: {norm_doc_blurry}")


if __name__ == "__main__":
    main()
