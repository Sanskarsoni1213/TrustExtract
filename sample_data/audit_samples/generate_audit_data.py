"""
Self-Audit Test Script for TrustExtract Loop 1.
Generates borderline quality samples and untested input paths to rigorously evaluate:
1. Blur & Borderline Quality:
   - Slightly blurry (readable)
   - Mildly skewed (5-8 deg)
   - Uneven lighting / shadow gradient
   - Moderate JPEG compression (Q=30)
   - Heavy JPEG compression (Q=10)
2. Untested Input Paths:
   - Rotated / skewed scan (10 deg off-axis)
   - Scanned PDF (pure image raster, no digital text layer)
   - Truncated / malformed PDF
   - Renamed .txt file with .pdf extension
   - Zero-byte file
   - Password-protected / encrypted PDF
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pymupdf as fitz
from trustextract.ingestion.quality import QualityChecker, QualityThresholds

os.makedirs("sample_data/audit_samples", exist_ok=True)


def create_borderline_samples():
    """Create borderline quality test samples from clean_receipt.png."""
    src_img = Image.open("sample_data/clean_receipt.png").convert("RGB")
    src_np = np.array(src_img)

    # 1. Slightly blurry but clearly human-readable (small Gaussian blur sigma=1.2)
    slight_blur = cv2.GaussianBlur(src_np, (5, 5), 1.2)
    Image.fromarray(slight_blur).save("sample_data/audit_samples/borderline_slight_blur.png")

    # 2. Mildly skewed (6 degrees rotation with white fill)
    (h, w) = src_np.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 6.0, 1.0)
    skewed_6deg = cv2.warpAffine(src_np, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    Image.fromarray(skewed_6deg).save("sample_data/audit_samples/borderline_skew_6deg.png")

    # 3. Uneven lighting / shadow gradient across the page
    gradient = np.tile(np.linspace(0.45, 1.0, w), (h, 1))
    gradient = np.stack([gradient]*3, axis=2)
    uneven_lighting = np.clip(src_np.astype(np.float64) * gradient, 0, 255).astype(np.uint8)
    Image.fromarray(uneven_lighting).save("sample_data/audit_samples/borderline_uneven_lighting.png")

    # 4. Moderate JPEG compression (quality = 25)
    src_img.save("sample_data/audit_samples/borderline_jpeg_q25.jpg", "JPEG", quality=25)

    # 5. Heavy JPEG compression (quality = 10)
    src_img.save("sample_data/audit_samples/borderline_jpeg_q10.jpg", "JPEG", quality=10)

    print("Created borderline samples.")


def create_untested_input_samples():
    """Create untested input path samples."""
    # 1. Rotated scan: 12 degrees off-axis
    src_img = Image.open("sample_data/clean_receipt.png").convert("RGB")
    src_np = np.array(src_img)
    (h, w) = src_np.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 12.0, 1.0)
    skewed_12deg = cv2.warpAffine(src_np, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    Image.fromarray(skewed_12deg).save("sample_data/audit_samples/skewed_12deg.png")

    # 2. Scanned PDF with NO digital text layer (raster image embedded in PDF)
    doc_scan = fitz.open()
    page_scan = doc_scan.new_page(width=src_img.width, height=src_img.height)
    # Insert raster image only, no font text
    page_scan.insert_image(page_scan.rect, filename="sample_data/clean_receipt.png")
    doc_scan.save("sample_data/audit_samples/scanned_image_only.pdf")
    doc_scan.close()

    # 3. Truncated / malformed PDF
    with open("sample_data/audit_samples/truncated_corrupt.pdf", "wb") as f:
        # Valid header but abruptly cut off in the middle of stream
        f.write(b"%PDF-1.7\n%\x82\x82\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Count 1 /Ki")

    # 4. Renamed .txt file with .pdf extension
    with open("sample_data/audit_samples/fake_renamed_txt.pdf", "wb") as f:
        f.write(b"This is just a text file renamed to fake a PDF document extension.\n")

    # 5. Zero-byte file
    with open("sample_data/audit_samples/empty_zero_byte.pdf", "wb") as f:
        pass

    # 6. Password-protected / encrypted PDF
    doc_enc = fitz.open()
    p = doc_enc.new_page()
    p.insert_text((50, 100), "Confidential Protected Document", fontsize=14)
    # Encrypt with owner and user password
    perm = int(
        fitz.PDF_PERM_ACCESSIBILITY |
        fitz.PDF_PERM_PRINT |
        fitz.PDF_PERM_COPY |
        fitz.PDF_PERM_ANNOTATE
    )
    doc_enc.save(
        "sample_data/audit_samples/password_protected.pdf",
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="AdminSecretPassword99!",
        user_pw="UserSecret123!"
    )
    doc_enc.close()

    print("Created untested input path samples.")


if __name__ == "__main__":
    create_borderline_samples()
    create_untested_input_samples()
