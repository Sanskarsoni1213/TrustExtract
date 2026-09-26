"""
Deskew Accuracy & Verification Test
Tests:
1. 6-degree skew sample (sample_data/audit_samples/borderline_skew_6deg.png)
2. 18-degree aggressive skew sample (sample_data/audit_samples/skewed_18deg.png)
3. 0-degree unskewed clean sample (sample_data/clean_receipt.png)
Reports:
- Ground truth angle
- Detected/corrected angle
- Residual angle after correction
"""

import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from trustextract.ingestion.deskew import Deskewer
from trustextract.ingestion.pipeline import IngestionPipeline


def test_deskew_performance():
    # 1. Prepare 18-degree aggressive skew sample
    src_img = Image.open("sample_data/clean_receipt.png").convert("RGB")
    src_np = np.array(src_img)
    (h, w) = src_np.shape[:2]
    center = (w // 2, h // 2)

    M18 = cv2.getRotationMatrix2D(center, 18.0, 1.0)
    skewed_18_np = cv2.warpAffine(src_np, M18, (w, h), borderValue=(255, 255, 255))
    Image.fromarray(skewed_18_np).save("sample_data/audit_samples/skewed_18deg.png")

    cases = [
        ("Mild Skew (6 deg)", "sample_data/audit_samples/borderline_skew_6deg.png", -6.0),
        ("Aggressive Skew (18 deg)", "sample_data/audit_samples/skewed_18deg.png", -18.0),
        ("Clean Unskewed (0 deg)", "sample_data/clean_receipt.png", 0.0),
    ]

    pipeline = IngestionPipeline(storage_dir="tests_encrypted_store")

    print("| Case | Ground Truth Skew | Detected/Applied Correction | Residual Post-Correction | Status |")
    print("|---|---|---|---|---|")

    for name, path, true_angle in cases:
        raw_img = Image.open(path)
        detected_initial = Deskewer.estimate_skew_angle(raw_img)
        deskewed_img, applied_correction = Deskewer.correct_skew(raw_img)
        residual_angle = Deskewer.estimate_skew_angle(deskewed_img)

        # Also process through entire pipeline
        res, norm_doc = pipeline.process_document(path, actor_id="test_deskew")

        print(f"| {name} | {true_angle:+.1f}° | {applied_correction:+.2f}° | {residual_angle:+.2f}° | {res.document_status.value} |")

        # Save deskewed result for inspection
        deskewed_img.save(f"sample_data/audit_samples/deskewed_{name.split()[0].lower()}.png")


if __name__ == "__main__":
    test_deskew_performance()
