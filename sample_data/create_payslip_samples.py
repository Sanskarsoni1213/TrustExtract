"""
Generate payslip samples for Loop 6 document-type generalization.
Creates:
  1. clean_payslip.png  — crisp, high-resolution payslip
  2. degraded_payslip.jpg — skewed, JPEG-compressed, camera-captured quality
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.makedirs("sample_data", exist_ok=True)


def create_clean_payslip(output_path: str):
    """Create a high-resolution clean payslip image with all standard fields."""
    img = Image.new("RGB", (1600, 1200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Company header
    draw.rectangle([(40, 30), (1560, 110)], fill=(30, 60, 120))
    draw.text((60, 50), "ACME CORPORATION", fill=(255, 255, 255))
    draw.text((60, 80), "Employee Pay Statement", fill=(200, 210, 240))

    # Employee details section
    draw.text((60, 140), "EmployeeName: Sarah Chen", fill=(20, 20, 20))
    draw.text((60, 175), "EmployeeId: EMP-2024-0471", fill=(20, 20, 20))
    draw.text((60, 210), "Department: Engineering", fill=(60, 60, 60))

    # Pay period and date
    draw.text((900, 140), "PayPeriodStart: 2024-03-01", fill=(20, 20, 20))
    draw.text((900, 175), "PayPeriodEnd: 2024-03-31", fill=(20, 20, 20))
    draw.text((900, 210), "PayDate: 2024-04-05", fill=(20, 20, 20))

    # Separator
    draw.line([(50, 250), (1550, 250)], fill=(180, 180, 180), width=2)

    # Earnings table header
    draw.rectangle([(50, 270), (780, 310)], fill=(230, 235, 245))
    draw.text((60, 280), "EARNINGS", fill=(30, 30, 30))
    draw.text((400, 280), "Hours", fill=(30, 30, 30))
    draw.text((520, 280), "Rate", fill=(30, 30, 30))
    draw.text((660, 280), "Amount", fill=(30, 30, 30))

    # Earnings rows
    draw.text((60, 330), "Regular Pay", fill=(40, 40, 40))
    draw.text((400, 330), "160", fill=(40, 40, 40))
    draw.text((520, 330), "$43.75", fill=(40, 40, 40))
    draw.text((660, 330), "$7,000.00", fill=(40, 40, 40))

    draw.text((60, 370), "Overtime", fill=(40, 40, 40))
    draw.text((400, 370), "12", fill=(40, 40, 40))
    draw.text((520, 370), "$65.63", fill=(40, 40, 40))
    draw.text((660, 370), "$787.50", fill=(40, 40, 40))

    # Deductions table header
    draw.rectangle([(820, 270), (1550, 310)], fill=(245, 230, 230))
    draw.text((830, 280), "DEDUCTIONS", fill=(30, 30, 30))
    draw.text((1350, 280), "Amount", fill=(30, 30, 30))

    # Deductions rows
    draw.text((830, 330), "Federal Tax", fill=(40, 40, 40))
    draw.text((1350, 330), "$1,557.50", fill=(40, 40, 40))

    draw.text((830, 370), "State Tax", fill=(40, 40, 40))
    draw.text((1350, 370), "$389.38", fill=(40, 40, 40))

    draw.text((830, 410), "Social Security", fill=(40, 40, 40))
    draw.text((1350, 410), "$482.83", fill=(40, 40, 40))

    draw.text((830, 450), "Medicare", fill=(40, 40, 40))
    draw.text((1350, 450), "$112.92", fill=(40, 40, 40))

    draw.text((830, 490), "Health Insurance", fill=(40, 40, 40))
    draw.text((1350, 490), "$275.00", fill=(40, 40, 40))

    draw.text((830, 530), "401(k) Contribution", fill=(40, 40, 40))
    draw.text((1350, 530), "$389.38", fill=(40, 40, 40))

    # Separator
    draw.line([(50, 580), (1550, 580)], fill=(180, 180, 180), width=2)

    # Summary totals
    draw.rectangle([(50, 600), (1550, 740)], fill=(248, 248, 252))

    draw.text((60, 620), "GrossPay: $7,787.50", fill=(20, 20, 20))
    draw.text((60, 660), "TotalDeductions: $3,207.01", fill=(20, 20, 20))
    draw.text((60, 700), "NetPay: $4,580.49", fill=(10, 10, 100))

    # YTD section
    draw.text((900, 620), "YTD Gross: $23,362.50", fill=(80, 80, 80))
    draw.text((900, 660), "YTD Net: $13,741.47", fill=(80, 80, 80))

    # Footer
    draw.line([(50, 760), (1550, 760)], fill=(180, 180, 180), width=1)
    draw.text((60, 780), "Direct Deposit to Account ending in 4821", fill=(100, 100, 100))
    draw.text((60, 810), "Employer: Acme Corporation | EIN: 91-1234567", fill=(100, 100, 100))

    img.save(output_path, "PNG")
    print(f"Created clean payslip: {output_path}")


def create_degraded_payslip(clean_path: str, output_path: str):
    """
    Create a degraded payslip: skewed, JPEG-compressed, camera-captured quality.
    Reuses Loop 1 degradation techniques.
    """
    img = cv2.imread(clean_path)
    h, w = img.shape[:2]

    # 1. Apply rotation (skew) — simulates camera capture at an angle
    angle = 3.5  # degrees
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    skewed = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Add Gaussian noise (camera sensor noise)
    noise = np.random.normal(0, 12, skewed.shape).astype(np.int16)
    noisy = np.clip(skewed.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 3. Reduce brightness unevenly (simulates office lighting)
    rows, cols = noisy.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols * 0.4)
    kernel_y = cv2.getGaussianKernel(rows, rows * 0.4)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    for c in range(3):
        noisy[:, :, c] = np.clip(noisy[:, :, c] * (0.7 + 0.3 * mask), 0, 255).astype(np.uint8)

    # 4. Moderate blur (slightly out of focus)
    blurred = cv2.GaussianBlur(noisy, (5, 5), 1.5)

    # 5. Save as JPEG with moderate compression (quality 55)
    cv2.imwrite(output_path, blurred, [cv2.IMWRITE_JPEG_QUALITY, 55])
    print(f"Created degraded payslip: {output_path}")


if __name__ == "__main__":
    create_clean_payslip("sample_data/clean_payslip.png")
    create_degraded_payslip("sample_data/clean_payslip.png", "sample_data/degraded_payslip.jpg")
