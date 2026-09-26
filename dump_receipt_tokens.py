import json
from trustextract.extraction.path_a_ocr import PathAOCRExtractor
from trustextract.ingestion.pipeline import IngestionPipeline

ingest = IngestionPipeline()
_, norm_doc = ingest.process_document('sample_data/clean_receipt.png', document_id='clean_receipt.png')
extractor = PathAOCRExtractor()
tokens, fields = extractor.extract_page_ocr(norm_doc.pages[0].image, 1)

print("--- OCR TOKENS FOR clean_receipt.png ---")
for i, t in enumerate(tokens):
    print(f"[{i:02d}] text={t['text']!r:<32} conf={t['confidence']:.4f} bbox={[round(x,1) for x in t['bbox']]}")

print("\n--- EXTRACTED FIELDS (Path A) ---")
for k, v in fields.items():
    print(f"{k:<18}: {v.value!r:<25} conf={v.confidence:.3f} raw={v.raw_text!r}")
