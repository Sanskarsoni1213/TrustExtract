"""
Audit Execution Script for TrustExtract Loop 1 Self-Audit.
Measures and records exact numerical metrics and pipeline behavior across:
1. Borderline samples (slight blur, skew, shadow, JPEG compression)
2. Untested input paths (skewed scan, scanned PDF, corrupt PDF, fake txt.pdf, zero-byte, encrypted PDF)
"""

import os
import sys
import json
from PIL import Image

sys.path.insert(0, ".")
from trustextract.ingestion.quality import QualityChecker, QualityThresholds
from trustextract.ingestion.pipeline import IngestionPipeline


def audit_quality_metrics():
    checker = QualityChecker(QualityThresholds(min_laplacian_variance=80.0))
    borderline_files = [
        ("Clean Original Receipt", "sample_data/clean_receipt.png", True),
        ("Slightly Blurry (Readable)", "sample_data/audit_samples/borderline_slight_blur.png", True),
        ("Mildly Skewed (6 deg)", "sample_data/audit_samples/borderline_skew_6deg.png", True),
        ("Uneven Lighting / Shadow", "sample_data/audit_samples/borderline_uneven_lighting.png", True),
        ("Moderate JPEG (Q=25)", "sample_data/audit_samples/borderline_jpeg_q25.jpg", True),
        ("Heavy JPEG (Q=10)", "sample_data/audit_samples/borderline_jpeg_q10.jpg", True),
        ("Deliberately Heavy Blurry (Unreadable)", "sample_data/deliberately_blurry_sample.jpg", False),
    ]

    print("=== PART 1: BLUR & BORDERLINE QUALITY EVALUATION ===")
    results = []
    for label, path, human_readable in borderline_files:
        img = Image.open(path)
        report = checker.assess_page_image(img)
        res = {
            "sample": label,
            "path": path,
            "human_readable": human_readable,
            "laplacian_var": report.metrics.get("laplacian_variance"),
            "edge_density": report.metrics.get("edge_density"),
            "mean_brightness": report.metrics.get("mean_brightness"),
            "passed": report.passed,
            "failure_reasons": report.failure_reasons
        }
        results.append(res)
        print(f"Sample: {label}")
        print(f"  Human readable: {human_readable}")
        print(f"  Laplacian Var: {res['laplacian_var']}")
        print(f"  Edge Density:  {res['edge_density']}")
        print(f"  Mean Bright:   {res['mean_brightness']}")
        print(f"  Quality Gate:  {'PASS' if res['passed'] else 'FAIL'}")
        if not res['passed']:
            print(f"  Reasons:       {res['failure_reasons']}")
        print()

    return results


def audit_untested_inputs():
    pipeline = IngestionPipeline(storage_dir="tests_encrypted_store")
    test_files = [
        ("Skewed Scan (12 deg)", "sample_data/audit_samples/skewed_12deg.png"),
        ("Scanned PDF (No Text Layer)", "sample_data/audit_samples/scanned_image_only.pdf"),
        ("Truncated / Corrupt PDF", "sample_data/audit_samples/truncated_corrupt.pdf"),
        ("Renamed .txt as .pdf", "sample_data/audit_samples/fake_renamed_txt.pdf"),
        ("Zero-Byte Empty File", "sample_data/audit_samples/empty_zero_byte.pdf"),
        ("Password-Protected PDF", "sample_data/audit_samples/password_protected.pdf"),
    ]

    print("\n=== PART 2: UNTESTED INPUT PATHS EVALUATION ===")
    results = []
    for label, path in test_files:
        try:
            result, norm_doc = pipeline.process_document(
                path,
                document_id=f"audit_{os.path.basename(path).replace('.', '_')}",
                actor_id="audit_operator"
            )
            out_json = result.to_output_json()
            status = out_json["document_status"]
            reason = out_json["unprocessable_reason"]
            has_pages = norm_doc is not None and len(norm_doc.pages) > 0
            has_text_layer = has_pages and (norm_doc.pages[0].text_layer is not None)
            res = {
                "label": label,
                "path": path,
                "status": status,
                "reason": reason,
                "has_pages": has_pages,
                "has_text_layer": has_text_layer
            }
            results.append(res)
            print(f"Case: {label}")
            print(f"  Status:         {status}")
            print(f"  Reason:         {reason}")
            print(f"  Has Pages:      {has_pages}")
            print(f"  Has Text Layer: {has_text_layer}")
        except Exception as e:
            print(f"Case: {label} CRASHED!")
            print(f"  Exception: {type(e).__name__}: {str(e)}")
            results.append({"label": label, "path": path, "crashed": True, "error": str(e)})
        print()

    return results


if __name__ == "__main__":
    audit_quality_metrics()
    audit_untested_inputs()
