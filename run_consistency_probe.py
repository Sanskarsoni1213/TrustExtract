"""
Self-Consistency Probe: run_consistency_probe.py
=================================================
Tests the SelfConsistencySampler against a BORDERLINE-QUALITY sample —
a synthetically degraded version of clean_invoice.pdf that still PASSES
the quality gate but is meaningfully harder than the clean original.

Degradation applied:
  - Moderate Gaussian blur (sigma=2.0, kernel=7x7) — simulates a slightly
    soft camera capture. Laplacian variance drops from ~20+ down to ~3-6,
    which is above our gate of 2.5 but clearly degraded.
  - JPEG re-compression at quality=55 — introduces block artefacts around
    character edges, a common real-world source of OCR instability.

For each field the sampler found, this script prints:
  Pass 1 (baseline, unperturbed) : <value>
  Pass 2 (noise sigma=3, +1px shift): <value>
  Pass 3 (noise sigma=4.5, +1px shift): <value>
  Consistency score: X.XXX

If all 3 passes agree on every field, this is reported explicitly as
"self-consistency signal unproven: perturbations did not cause any divergence"
and the limitation is documented in the analysis.
"""

import io
import sys
import cv2
import numpy as np
from PIL import Image

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import fitz  # pymupdf
from trustextract.extraction.path_a_ocr import PathAOCRExtractor
from trustextract.extraction.confidence import SelfConsistencySampler
from trustextract.ingestion.quality import QualityChecker, QualityThresholds

SEP = "=" * 80


def render_pdf_to_pil(pdf_path: str, dpi: int = 150) -> Image.Image:
    """Render first page of a PDF to a PIL image."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def degrade_image(img: Image.Image, blur_sigma: float = 2.0, jpeg_quality: int = 55) -> Image.Image:
    """
    Apply moderate blur + JPEG re-compression to create a borderline-quality image
    that still passes the quality gate (Laplacian variance >= 2.5) but is noticeably
    harder for OCR than the clean original.
    """
    arr = np.array(img.convert("RGB"))

    # Moderate Gaussian blur (kernel must be odd)
    ksize = 7
    blurred = cv2.GaussianBlur(arr, (ksize, ksize), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # JPEG re-compress at moderate quality to introduce block artefacts
    pil_blurred = Image.fromarray(blurred)
    buf = io.BytesIO()
    pil_blurred.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    degraded = Image.open(buf).copy()   # copy() forces load so buf can be GC'd
    return degraded


def measure_laplacian(img: Image.Image) -> float:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    print(SEP)
    print("SELF-CONSISTENCY PROBE: BORDERLINE-QUALITY SAMPLE")
    print(SEP)

    # --- Step 1: Render clean PDF and measure baseline quality ---
    print("\n[STEP 1] Render clean_invoice.pdf and degrade to borderline quality")
    clean_img = render_pdf_to_pil("sample_data/clean_invoice.pdf", dpi=150)
    lap_clean = measure_laplacian(clean_img)
    print(f"  Clean image size      : {clean_img.size}")
    print(f"  Clean Laplacian var   : {lap_clean:.2f}  (well above gate of 2.5)")

    degraded_img = degrade_image(clean_img, blur_sigma=2.0, jpeg_quality=55)
    lap_degraded = measure_laplacian(degraded_img)
    print(f"\n  Degraded image size   : {degraded_img.size}")
    print(f"  Degraded Laplacian var: {lap_degraded:.2f}")

    # --- Step 2: Quality gate check on degraded image ---
    print("\n[STEP 2] Quality gate check on degraded image")
    checker = QualityChecker(QualityThresholds())
    report = checker.assess_page_image(degraded_img)
    print(f"  Quality gate PASSED   : {report.passed}")
    print(f"  Metrics               : {report.metrics}")
    if not report.passed:
        print(f"  FAILURE REASONS       : {report.failure_reasons}")
        print("\n  NOTE: Degraded image failed the quality gate. Cannot run OCR.")
        print("  Reducing degradation and retrying with sigma=1.5, jpeg_quality=70...")
        degraded_img = degrade_image(clean_img, blur_sigma=1.5, jpeg_quality=70)
        lap_degraded = measure_laplacian(degraded_img)
        report = checker.assess_page_image(degraded_img)
        print(f"  Retry Laplacian var   : {lap_degraded:.2f}  |  Gate passed: {report.passed}")
        if not report.passed:
            print("  Cannot find a degradation level that both passes the gate and degrades OCR.")
            print("  Exiting probe — self-consistency limitation documented below.")
            return

    print(f"  Degraded Laplacian is {lap_degraded:.2f} (gate=2.5) — image passes quality gate.")

    # --- Step 3: Baseline OCR on degraded image ---
    print("\n[STEP 3] Baseline Path A OCR on degraded image")
    ocr = PathAOCRExtractor()
    tokens, fields = ocr.extract_page_ocr(degraded_img, page_number=1)
    print(f"  Tokens found          : {len(tokens)}")
    print(f"  Fields extracted      : {list(fields.keys())}")

    if not fields:
        print("\n  No fields extracted from degraded image — OCR failed entirely.")
        print("  Self-consistency signal unproven: image too degraded for baseline extraction.")
        return

    # --- Step 4: Self-consistency verbose run ---
    print("\n[STEP 4] Self-consistency sampling — 3 passes with perturbations")
    sampler = SelfConsistencySampler(n_passes=3, seed=42)
    consistency_scores, per_pass_values = sampler.run_verbose(degraded_img, page_number=1, reference_fields=fields)

    print(f"\n  {'Field':<17} {'Pass1 (baseline)':<30} {'Pass2 (+noise,shift)':<30} {'Pass3 (+noise,shift)':<30}  Score")
    print("  " + "-" * 115)

    all_consistent = True
    for fname in sorted(per_pass_values):
        passes = per_pass_values[fname]
        # Pad to 3 if field was only partially found
        while len(passes) < 3:
            passes.append("<not found>")
        p1, p2, p3 = passes[0], passes[1], passes[2]
        score = consistency_scores[fname]
        diverged = not (p1 == p2 == p3)
        marker = " <<< DIVERGED" if diverged else ""
        if diverged:
            all_consistent = False
        print(
            f"  {fname:<17}"
            f" {str(p1):<30}"
            f" {str(p2):<30}"
            f" {str(p3):<30}"
            f"  {score:.3f}{marker}"
        )

    # --- Step 5: Verdict ---
    print("\n" + SEP)
    print("SELF-CONSISTENCY PROBE VERDICT")
    print(SEP)
    print(f"  Laplacian variance of test image: {lap_degraded:.2f} (gate=2.5, clean={lap_clean:.2f})")
    print(f"  JPEG quality applied: 55")
    print(f"  Perturbation: Gaussian noise sigma=[3.0, 4.5] + 1-2px crop shift")

    if all_consistent:
        print("\n  RESULT: All 3 passes agreed on every field.")
        print()
        print("  *** SELF-CONSISTENCY SIGNAL UNPROVEN ***")
        print("  The perturbations (pixel noise + sub-pixel crop) did NOT cause any")
        print("  divergence in OCR output, even on the degraded sample.")
        print()
        print("  Interpretation: RapidOCR's internal tokenisation is robust enough that")
        print("  our current perturbation magnitude (sigma<=4.5px, 1-2px shift, JPEG@55)")
        print("  does not cross the decision boundary for any extracted token.")
        print()
        print("  What this means for Loop 3:")
        print("  - The consistency_signal=1.0 for all fields is accurate but uninformative.")
        print("  - The signal IS NOT contributing discrimination to the fused score.")
        print("  - Proven discriminating cases would require: heavier blur (sigma>=5),")
        print("    aggressive rotation/skew, low-DPI rendering, or genuinely ambiguous")
        print("    characters (0/O, 1/I, 5/S) in the document text.")
        print()
        print("  Documented limitation: self-consistency is a sound architectural")
        print("  signal that WILL discriminate on real noisy scans, but has NOT been")
        print("  observed to catch anything in the current synthetic test set.")
        print("  This is stated explicitly — it does not pass silently as 'working'.")
    else:
        diverged_fields = [f for f in per_pass_values if len(set(per_pass_values[f])) > 1]
        print(f"\n  RESULT: {len(diverged_fields)} field(s) diverged across passes: {diverged_fields}")
        print("  Self-consistency signal IS discriminating on this harder sample.")
        for fname in diverged_fields:
            passes = per_pass_values[fname]
            print(f"\n  Field '{fname}':")
            for i, v in enumerate(passes, start=1):
                print(f"    Pass {i}: {v!r}")
            print(f"    Consistency score: {consistency_scores[fname]:.3f}")


if __name__ == "__main__":
    main()
