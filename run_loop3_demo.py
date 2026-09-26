"""
Loop 3 Demo: Confidence Scoring & Validation Pipeline
Shows, for every field:
  - Individual signal values (agreement, self-consistency, validation)
  - Business-logic check results (arithmetic, date, format)
  - Final fused confidence score
Specifically confirms whether the invoice arithmetic checks out.
"""

import json
import sys
import io

# Force UTF-8 stdout on Windows so the demo prints cleanly
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from trustextract.ingestion.pipeline import IngestionPipeline
from trustextract.extraction.pipeline import DualPathExtractor
from trustextract.extraction.path_b_vision import IndependentVisionModel

SEP = "=" * 80
DASH = "-" * 80


def main():
    ingestion = IngestionPipeline(storage_dir="encrypted_store")
    extraction = DualPathExtractor(
        vision_model=IndependentVisionModel(induce_disagreement_on="InvoiceId"),
        storage_dir="encrypted_store",
        n_consistency_passes=3,
    )

    print(SEP)
    print("LOOP 3 DEMO: CONFIDENCE SCORING & VALIDATION PIPELINE")
    print(SEP)

    # --- Step 1: Ingestion ---
    print("\n[STEP 1] Ingestion & Normalization")
    res_ingest, norm_doc = ingestion.process_document(
        "sample_data/clean_invoice.pdf",
        document_id="doc_invoice_loop3_demo",
        actor_id="demo_operator"
    )
    print(f"  Status : {res_ingest.document_status.value}")
    print(f"  Pages  : {len(norm_doc.pages)}")

    # --- Step 2: Dual-path extraction + Loop 3 confidence scoring ---
    print("\n[STEP 2] Dual-Path Extraction + Self-Consistency (3 passes) + Validation + Fusion")
    result, side_by_side, validation_report = extraction.process_normalized_document(
        norm_doc, actor_id="demo_operator"
    )

    # --- Step 3: Arithmetic check spotlight ---
    print("\n" + SEP)
    print("ARITHMETIC CHECK SPOTLIGHT")
    print(SEP)
    subtotal  = result.fields.get("Subtotal")
    tax       = result.fields.get("TotalTax")
    inv_total = result.fields.get("InvoiceTotal")

    sub_v  = subtotal.value  if subtotal   else "N/A"
    tax_v  = tax.value       if tax        else "N/A"
    itot_v = inv_total.value if inv_total  else "N/A"

    try:
        computed = round(float(sub_v) + float(tax_v), 2)
        match = abs(computed - float(itot_v)) <= 0.02
        result_str = "PASS" if match else "FAIL"
    except (TypeError, ValueError):
        computed   = "N/A"
        result_str = "CANNOT COMPUTE"

    print(f"  Subtotal     = {sub_v}")
    print(f"  TotalTax     = {tax_v}")
    print(f"  " + "-" * 29)
    print(f"  Computed     = {computed}  ({sub_v} + {tax_v})")
    print(f"  InvoiceTotal = {itot_v}")
    print(f"  Arithmetic   : {result_str}")

    print("\n  Validation flags per financial field:")
    for fname in ("Subtotal", "TotalTax", "InvoiceTotal"):
        vr = validation_report.per_field.get(fname)
        flags  = vr.flags  if vr else ["N/A"]
        passed = vr.passed if vr else True
        print(f"    {fname:<15}: {'OK' if passed else 'FAILED'}  {flags}")

    if validation_report.summary_flags:
        print(f"\n  Document-level flags: {validation_report.summary_flags}")
    else:
        print("\n  No document-level validation flags raised.")

    # --- Step 4: Date check ---
    print("\n" + SEP)
    print("DATE LOGIC CHECK")
    print(SEP)
    for fname in ("InvoiceDate", "DueDate"):
        vr = validation_report.per_field.get(fname)
        if vr:
            print(f"  {fname:<15}: {'OK' if vr.passed else 'FAILED'}  {vr.flags}")

    # --- Step 5: Identifier format check ---
    print("\n" + SEP)
    print("IDENTIFIER FORMAT CHECK")
    print(SEP)
    for fname in ("InvoiceId", "DocumentNumber"):
        vr = validation_report.per_field.get(fname)
        if vr:
            print(f"  {fname:<15}: {'OK' if vr.passed else 'FAILED'}  {vr.flags}  value={result.fields[fname].value!r}")

    # --- Step 6: Per-field signal table ---
    print("\n" + SEP)
    print("PER-FIELD CONFIDENCE SIGNAL BREAKDOWN")
    print(SEP)
    hdr = f"  {'Field':<17} {'NativeOCR':>10} {'Agreement':>10} {'Consist.':>10} {'Validate':>10} {'Final':>8}  Flags"
    print(hdr)
    print("  " + "-" * 80)

    for fname in sorted(result.fields):
        fv = result.fields[fname]
        bd = fv.confidence_breakdown
        if bd:
            print(
                f"  {fname:<17}"
                f" {bd.native_ocr:>10.3f}"
                f" {bd.agreement_signal:>10.3f}"
                f" {bd.consistency_signal:>10.3f}"
                f" {bd.validation_signal:>10.3f}"
                f" {bd.final:>8.3f}"
                f"  {', '.join(bd.validation_flags)}"
            )
        else:
            print(f"  {fname:<17}  [no breakdown]  final={fv.confidence:.3f}")

    # --- Step 7: Document status + escalation reasons ---
    print("\n" + SEP)
    print("DOCUMENT STATUS & ESCALATION")
    print(SEP)
    print(f"  document_status   : {result.document_status.value}")
    print(f"  authenticity_flags: {result.authenticity_flags or []}")
    if result.escalation_reasons:
        print(f"\n  escalation_reasons ({len(result.escalation_reasons)} rule(s) fired):")
        for r in result.escalation_reasons:
            print(f"    - {r}")
    else:
        print("\n  escalation_reasons: [] (no rules fired)")


    # --- Step 8: Full JSON with confidence_breakdown ---
    print("\n" + SEP)
    print("FULL OUTPUT JSON (with confidence_breakdown per field)")
    print(SEP)
    print(json.dumps(result.to_output_json(), indent=2))


if __name__ == "__main__":
    main()
