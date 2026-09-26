import sys
from PIL import Image

sys.path.insert(0, ".")
from trustextract.ingestion.quality import QualityChecker, QualityThresholds

checker_old = QualityChecker(QualityThresholds(min_laplacian_variance=80.0))
checker_new = QualityChecker(QualityThresholds(min_laplacian_variance=2.0))

files = [
    ("Clean Original Receipt", "sample_data/clean_receipt.png", True),
    ("Slightly Blurry (Readable)", "sample_data/audit_samples/borderline_slight_blur.png", True),
    ("Mildly Skewed (6 deg)", "sample_data/audit_samples/borderline_skew_6deg.png", True),
    ("Uneven Lighting / Shadow", "sample_data/audit_samples/borderline_uneven_lighting.png", True),
    ("Moderate JPEG (Q=25)", "sample_data/audit_samples/borderline_jpeg_q25.jpg", True),
    ("Heavy JPEG (Q=10)", "sample_data/audit_samples/borderline_jpeg_q10.jpg", True),
    ("Deliberately Heavy Blurry (Unreadable)", "sample_data/deliberately_blurry_sample.jpg", False)
]

print("| Sample | Laplacian Var | Edge Density | Old Thresh (80.0) | Proposed Thresh (2.0) | Human Readable? | Correct Gate? |")
print("|---|---|---|---|---|---|---|")
for name, p, readable in files:
    img = Image.open(p)
    r_old = checker_old.assess_page_image(img)
    r_new = checker_new.assess_page_image(img)
    l_var = r_old.metrics["laplacian_variance"]
    e_dens = r_old.metrics["edge_density"]
    old_status = "PASS" if r_old.passed else "FAIL"
    new_status = "PASS" if r_new.passed else "FAIL"
    target = "YES" if readable else "NO"
    correct = "YES" if (r_new.passed == readable) else "NO"
    print(f"| {name} | {l_var} | {e_dens} | {old_status} | {new_status} | {target} | {correct} |")
