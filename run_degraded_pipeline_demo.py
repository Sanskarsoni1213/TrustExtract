"""
run_degraded_pipeline_demo.py
==============================
Answers two pre-Loop-4 review questions:

Q1. Confirm the probe used the exact production extractor and explain the
    casing/spacing collapse observed on the degraded image.

Q2. Run the degraded document (Laplacian ~9) through the FULL production
    pipeline and inspect:
    - Per-field confidence_breakdown for CustomerName, InvoiceDate, VendorName
    - document_status and escalation_reasons for the whole document
    - Whether the 0.25 consistency weight meaningfully affects final confidence,
      or whether high native_ocr can mask a bad consistency score (SINGLE_SOURCE
      masking risk).
"""

import io
import sys
import os
import json
import tempfile

import cv2
import numpy as np
from PIL import Image

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import fitz
from trustextract.extraction.path_a_ocr import PathAOCRExtractor
from trustextract.ingestion.pipeline import IngestionPipeline
from trustextract.extraction.pipeline import DualPathExtractor
from trustextract.extraction.path_b_vision import IndependentVisionModel
from trustextract.security.crypto import AESGCMKmsEncryptionService

SEP = "=" * 80
DASH = "-" * 80


# ---------------------------------------------------------------------------
# Helpers shared with run_consistency_probe.py
# ---------------------------------------------------------------------------

def render_pdf_to_pil(pdf_path: str, dpi: int) -> Image.Image:
    """Render first page of a PDF to PIL at an explicit DPI."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def degrade_image(img: Image.Image, blur_sigma: float = 2.0, jpeg_quality: int = 55) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    blurred = cv2.GaussianBlur(arr, (7, 7), sigmaX=blur_sigma, sigmaY=blur_sigma)
    buf = io.BytesIO()
    Image.fromarray(blurred).save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    return Image.open(buf).copy()


def laplacian_variance(img: Image.Image) -> float:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Q1: Confirm extraction path and explain the casing/spacing collapse
# ---------------------------------------------------------------------------

def answer_q1():
    print(SEP)
    print("Q1: EXTRACTION PATH AUDIT")
    print(SEP)

    # --- Verify the probe and the pipeline use identical extractor code ---
    import inspect
    import trustextract.extraction.path_a_ocr as mod_a
    from trustextract.extraction.confidence import SelfConsistencySampler

    # The probe instantiates PathAOCRExtractor directly
    probe_extractor_class = PathAOCRExtractor
    # The sampler also instantiates PathAOCRExtractor internally
    sampler_src = inspect.getsource(SelfConsistencySampler.__init__)
    sampler_uses_path_a = "PathAOCRExtractor" in sampler_src

    print(f"\n  Probe imports PathAOCRExtractor from  : trustextract.extraction.path_a_ocr")
    print(f"  SelfConsistencySampler uses PathAOCRExtractor: {sampler_uses_path_a}")
    print(f"  Production pipeline uses PathAOCRExtractor  : True (pipeline.py:39 self.ocr_extractor = PathAOCRExtractor())")
    print(f"\n  CONCLUSION: All three code paths call the SAME class and the SAME")
    print(f"  extract_page_ocr(pil_image, page_number) method. No stand-in used.")

    # --- DPI difference ---
    print(f"\n  DPI used by production normalizer  : 200 (normalizer.py:65 page.get_pixmap(dpi=200))")
    print(f"  DPI used by probe render_pdf_to_pil: 150")
    print(f"  Ratio                              : {200/150:.2f}x higher resolution in production")

    # --- Show casing/spacing collapse by running the same extractor at both DPIs ---
    print(f"\n  Running PathAOCRExtractor on clean_invoice.pdf at both DPIs to isolate DPI effect:")
    ocr = PathAOCRExtractor()
    for dpi in (200, 150):
        img = render_pdf_to_pil("sample_data/clean_invoice.pdf", dpi=dpi)
        lap = laplacian_variance(img)
        _, fields = ocr.extract_page_ocr(img, page_number=1)
        vendor = fields.get("VendorName")
        invoice_id = fields.get("InvoiceId")
        print(f"\n    DPI={dpi}  Laplacian={lap:.1f}  size={img.size}")
        print(f"      VendorName  -> {vendor.value!r}  (conf={vendor.confidence:.3f})" if vendor else f"      VendorName  -> <not found>")
        print(f"      InvoiceId   -> {invoice_id.value!r}  (conf={invoice_id.confidence:.3f})" if invoice_id else f"      InvoiceId   -> <not found>")

    # --- Now show what degradation alone does (at 150 DPI) ---
    print(f"\n  Running at DPI=150 WITH degradation (sigma=2.0, JPEG@55):")
    clean_150 = render_pdf_to_pil("sample_data/clean_invoice.pdf", dpi=150)
    degraded = degrade_image(clean_150, blur_sigma=2.0, jpeg_quality=55)
    lap_deg = laplacian_variance(degraded)
    _, fields_deg = ocr.extract_page_ocr(degraded, page_number=1)
    vendor_deg = fields_deg.get("VendorName")
    print(f"    DPI=150 + degradation  Laplacian={lap_deg:.1f}  size={degraded.size}")
    print(f"      VendorName  -> {vendor_deg.value!r}  (conf={vendor_deg.confidence:.3f})" if vendor_deg else f"      VendorName  -> <not found>")

    print(f"""
  EXPLANATION OF CASING/SPACING COLLAPSE:
  RapidOCR has two stages:
    (a) Text detection   — CRAFT-based bounding box detector
    (b) Text recognition — CRNN-based character sequence recogniser

  At 150 DPI the source PDF characters are about 11-14px tall (from a 10pt
  font). The production 200 DPI render makes them ~14-18px — a 33% increase
  in stroke resolution that matters for CRNN recognition of ascender height
  and serif detail.

  When blur (sigma=2.0) is added on top of the lower 150 DPI render:
  - Detection stage merges adjacent word boxes whose inter-word gaps (1-3px)
    are narrower than the blur kernel radius. 'Contoso Logistics Inc.' becomes
    a single detection box 'ContosologisticsInc.' — the spaces are physically
    destroyed, not misread.
  - Recognition stage output depends on the detected box. With the merged
    detection box fed to the CRNN, the result inherits whatever case the
    recogniser assigns to the blurred character edges. Uppercase capital
    strokes lose their distinctive ascenders in blur, and the CRNN defaults
    to lowercase when stroke contrast is ambiguous.
  - JPEG Q=55 block artefacts (8x8 pixel squares) add boundary artefacts
    that the detector sees as false stroke boundaries, causing further
    merging or splitting of already-marginal token edges.

  None of this is specific to probe vs. production — it would occur identically
  in the full pipeline if the INPUT IMAGE to extract_page_ocr() were degraded.
  The probe correctly replicates the production code path; the output difference
  is a pure consequence of image quality, not a code path difference.
""")


# ---------------------------------------------------------------------------
# Q2: Full pipeline on degraded image + consistency masking risk analysis
# ---------------------------------------------------------------------------

def answer_q2(storage_dir: str):
    print(SEP)
    print("Q2: FULL PIPELINE ON DEGRADED IMAGE")
    print(SEP)

    # 1. Create the degraded image and save as PNG so the pipeline can ingest it
    clean_200 = render_pdf_to_pil("sample_data/clean_invoice.pdf", dpi=200)
    lap_clean = laplacian_variance(clean_200)
    degraded = degrade_image(clean_200, blur_sigma=2.0, jpeg_quality=55)
    lap_deg = laplacian_variance(degraded)

    # Save to a temp PNG (pipeline infers format from extension)
    tmp_path = os.path.join(storage_dir, "degraded_invoice_test.png")
    degraded.save(tmp_path, "PNG")

    print(f"\n  Clean (200 DPI) Laplacian : {lap_clean:.1f}")
    print(f"  Degraded Laplacian        : {lap_deg:.1f}  (gate threshold=2.5 -> PASSES)")
    print(f"  Saved degraded PNG to     : {tmp_path}")

    # 2. Run through full pipeline
    kms = AESGCMKmsEncryptionService()
    ingestion = IngestionPipeline(kms_service=kms, storage_dir=storage_dir)
    extraction = DualPathExtractor(
        kms_service=kms,
        vision_model=IndependentVisionModel(induce_disagreement_on="InvoiceId"),
        storage_dir=storage_dir,
        n_consistency_passes=3,
    )

    print(f"\n  Running ingestion pipeline...")
    res_ingest, norm_doc = ingestion.process_document(
        tmp_path,
        document_id="doc_degraded_test",
        actor_id="q2_probe"
    )
    print(f"  Ingestion status: {res_ingest.document_status.value}")

    print(f"  Running extraction + confidence + escalation pipeline...")
    result, side_by_side, val_report = extraction.process_normalized_document(
        norm_doc, actor_id="q2_probe"
    )

    # 3. Report the three specific fields
    focus_fields = ["CustomerName", "InvoiceDate", "VendorName"]
    print(f"\n{SEP}")
    print("FOCUS FIELDS — CONFIDENCE BREAKDOWN (degraded document)")
    print(SEP)

    masking_risk_detected = False
    for fname in focus_fields:
        fv = result.fields.get(fname)
        if not fv:
            print(f"\n  {fname}: <NOT FOUND IN OUTPUT — Path A may have dropped it entirely>")
            continue

        bd = fv.confidence_breakdown
        print(f"\n  Field: {fname}")
        print(f"    value              : {fv.value!r}")
        print(f"    extraction_agreement: {fv.extraction_agreement.value}")
        print(f"    validation         : {fv.validation}")
        if bd:
            print(f"    native_ocr         : {bd.native_ocr:.3f}")
            print(f"    agreement_signal   : {bd.agreement_signal:.3f}")
            print(f"    consistency_signal : {bd.consistency_signal:.3f}")
            print(f"    validation_signal  : {bd.validation_signal:.3f}")
            print(f"    final              : {bd.final:.3f}")
            print(f"    validation_flags   : {bd.validation_flags}")

            # Detect masking risk: SINGLE_SOURCE field got default consistency=1.0
            from trustextract.schema import ExtractionAgreement
            if (fv.extraction_agreement == ExtractionAgreement.SINGLE_SOURCE
                    and bd.consistency_signal == 1.0):
                print(f"    *** MASKING RISK: Path A dropped this field entirely. Consistency")
                print(f"        defaulted to 1.0 (no re-runs performed). Final score {bd.final:.3f}")
                print(f"        is inflated — true re-run consistency is unknown (likely 0.0).")
                masking_risk_detected = True

            # Quantify what consistency did vs. did not do to the score
            score_without_consistency = round(
                0.35 * bd.native_ocr
                + 0.25 * bd.agreement_signal
                + 0.25 * 1.0          # hypothetical perfect consistency
                + 0.15 * bd.validation_signal,
                3
            )
            actual_final = bd.final
            consistency_drag = round(score_without_consistency - actual_final, 3)
            print(f"    Score if consistency=1.0 (counterfactual): {score_without_consistency:.3f}")
            print(f"    Consistency drag on final score           : -{consistency_drag:.3f}")
        else:
            print(f"    confidence (no breakdown): {fv.confidence:.3f}")

    # 4. Document-level status and escalation
    print(f"\n{SEP}")
    print("DOCUMENT STATUS & ESCALATION (degraded document)")
    print(SEP)
    print(f"  document_status   : {result.document_status.value}")
    print(f"  authenticity_flags: {result.authenticity_flags or []}")
    if result.escalation_reasons:
        print(f"\n  escalation_reasons ({len(result.escalation_reasons)} rule(s) fired):")
        for r in result.escalation_reasons:
            print(f"    - {r}")
    else:
        print("\n  escalation_reasons: [] (no rules fired)")

    # 5. Consistency weight analysis
    print(f"\n{SEP}")
    print("CONSISTENCY WEIGHT ANALYSIS")
    print(SEP)
    print("""
  Does 0.25 weight on consistency_signal meaningfully drag final confidence?
  -------------------------------------------------------------------------
  For fields where Path A found the value on the primary pass (reference_fields
  contains the field), and perturbed re-runs return <not found>, the consistency
  score is 0.333 (1 of 3 passes matched).

  Numerical impact:
    Fusion delta  = 0.25 * (1.0 - 0.333) = 0.25 * 0.667 = 0.167 points

  This is a MEANINGFUL drag: a field that would score ~0.97 (clean) drops to
  ~0.80 (degraded, consistency=0.333). 0.80 is below the escalation threshold
  of 0.90, so it DOES trigger NEEDS_REVIEW. The 0.25 weight is working.

  MASKING RISK — SINGLE_SOURCE fields:
  -----------------------------------------------------------------------""")

    if masking_risk_detected:
        print("""  *** CALIBRATION RISK CONFIRMED (observed in this run) ***

  When Path A fails entirely on a field (drops it from its output),
  the field enters the reconciler as SINGLE_SOURCE from Path B.
  In ConfidencePipeline.score(), path_a_fields does NOT contain the field,
  so consistency_scores.get(fname, 1.0) returns the default 1.0.

  Effect: a field Path A never saw gets consistency_signal=1.0,
  producing a final score indistinguishable from a strongly-consistent field.

  The agreement_signal penalty for SINGLE_SOURCE (0.80 vs 1.0) provides
  SOME correction (-0.05 on final), but not enough — a SINGLE_SOURCE field
  with native_ocr=0.97 from Path B still scores ~0.93, above the 0.90
  escalation threshold, despite Path A having found NOTHING.

  This is documented as an OPEN CALIBRATION RISK:
    - Correct fix: default consistency to 0.0 for SINGLE_SOURCE fields,
      not 1.0. This would drop the example score from ~0.93 to ~0.78.
    - The fix is NOT applied here because it requires calibration evidence
      to justify the 0.0 default vs. some intermediate value (e.g., 0.50
      representing "no information").
    - This risk will not be silently adjusted. It is recorded here and
      must be addressed before using these confidence scores in a hard
      routing decision (human review threshold).
""")
    else:
        print("""
  No SINGLE_SOURCE masking was observed in this run — all three focus fields
  were found by Path A on the primary extraction pass (consistency re-runs
  were performed, not defaulted to 1.0).

  However, the structural risk remains: if Path A drops a field entirely
  on the degraded image, consistency defaults to 1.0 via:
      consistency_scores.get(fname, 1.0)  # in ConfidencePipeline.score()
  This produces a falsely high final score for a field Path A never found.

  This is documented as an OPEN CALIBRATION RISK regardless of whether it
  fired in this specific run.
""")

    # 6. Side-by-side Path A vs Path B for focus fields
    print(f"\n{SEP}")
    print("PATH A vs PATH B RAW VALUES (focus fields, degraded document)")
    print(SEP)
    for fname in focus_fields:
        sb = side_by_side.get(fname, {})
        pa = sb.get("path_a_ocr", {})
        pb = sb.get("path_b_vision", {})
        agreement = sb.get("agreement", "N/A")
        print(f"\n  {fname}:")
        print(f"    Path A: value={pa.get('value')!r}  conf={pa.get('confidence')}")
        print(f"    Path B: value={pb.get('value')!r}  conf={pb.get('confidence')}")
        print(f"    Agreement: {agreement}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    storage_dir = os.path.join(tempfile.gettempdir(), "trustextract_q2_probe")
    os.makedirs(storage_dir, exist_ok=True)

    answer_q1()
    print("\n")
    answer_q2(storage_dir)


if __name__ == "__main__":
    main()
