"""
Generate realistic sample documents for testing TrustExtract.
Includes clean PDF, clean PNG, clean DOCX, deliberately blurred photo, and low-res sample.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf as fitz
import docx

os.makedirs("sample_data", exist_ok=True)


def create_clean_invoice_pdf(output_path: str):
    """Create a realistic clean PDF invoice with text layer and sharp layout."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Standard Letter size

    # Draw header
    page.insert_text((50, 60), "INVOICE", fontsize=24, color=(0.1, 0.1, 0.3))
    page.insert_text((50, 95), "VendorName: Contoso Logistics Inc.", fontsize=12, color=(0.2, 0.2, 0.2))
    page.insert_text((50, 115), "VendorAddress: 100 Main Street, Suite 400, Seattle, WA 98101", fontsize=10, color=(0.3, 0.3, 0.3))

    page.insert_text((400, 60), "InvoiceId: INV-2024-9841", fontsize=12, color=(0.1, 0.1, 0.1))
    page.insert_text((400, 80), "InvoiceDate: 2024-03-15", fontsize=10, color=(0.3, 0.3, 0.3))
    page.insert_text((400, 100), "DueDate: 2024-04-15", fontsize=10, color=(0.3, 0.3, 0.3))

    page.insert_text((50, 160), "CustomerName: Acme Enterprise Corp", fontsize=11, color=(0.2, 0.2, 0.2))
    page.insert_text((50, 180), "CustomerAddress: 500 Industrial Pkwy, Chicago, IL 60601", fontsize=10, color=(0.3, 0.3, 0.3))

    # Draw table border and header line
    page.draw_rect(fitz.Rect(50, 220, 560, 360), color=(0.7, 0.7, 0.7), width=1)
    page.draw_line(fitz.Point(50, 250), fitz.Point(560, 250), color=(0.5, 0.5, 0.5), width=1)

    page.insert_text((60, 240), "Description", fontsize=10, color=(0, 0, 0))
    page.insert_text((300, 240), "Quantity", fontsize=10, color=(0, 0, 0))
    page.insert_text((380, 240), "UnitPrice", fontsize=10, color=(0, 0, 0))
    page.insert_text((480, 240), "LineTotal", fontsize=10, color=(0, 0, 0))

    items = [
        ("Cloud Server Hosting - March", "2", "$350.00", "$700.00"),
        ("Database Backup Storage", "1", "$150.00", "$150.00"),
        ("Dedicated IP Allocation", "5", "$10.00", "$50.00"),
    ]

    y = 280
    for desc, qty, price, total in items:
        page.insert_text((60, y), desc, fontsize=9, color=(0.2, 0.2, 0.2))
        page.insert_text((310, y), qty, fontsize=9, color=(0.2, 0.2, 0.2))
        page.insert_text((390, y), price, fontsize=9, color=(0.2, 0.2, 0.2))
        page.insert_text((490, y), total, fontsize=9, color=(0.2, 0.2, 0.2))
        y += 25

    # Totals
    page.insert_text((400, 400), "Subtotal: $900.00", fontsize=11, color=(0.1, 0.1, 0.1))
    page.insert_text((400, 420), "Tax (8%): $72.00", fontsize=11, color=(0.1, 0.1, 0.1))
    page.insert_text((400, 445), "InvoiceTotal: $972.00", fontsize=13, color=(0.1, 0.1, 0.5))

    doc.save(output_path)
    doc.close()
    print(f"Created {output_path}")


def create_clean_png_receipt(output_path: str):
    """Create a high-resolution clean PNG receipt."""
    img = Image.new("RGB", (1000, 1400), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)

    draw.text((350, 80), "STARBUCKS COFFEE #1042", fill=(20, 20, 20))
    draw.text((320, 120), "MerchantName: Starbucks Coffee", fill=(30, 30, 30))
    draw.text((340, 150), "MerchantAddress: 456 Pine St", fill=(50, 50, 50))
    draw.text((360, 180), "TransactionDate: 2024-04-02", fill=(50, 50, 50))

    draw.line([(100, 230), (900, 230)], fill=(180, 180, 180), width=2)
    draw.text((120, 260), "1x Caffe Latte (Grande)          $5.45", fill=(20, 20, 20))
    draw.text((120, 300), "1x Blueberry Muffin              $3.75", fill=(20, 20, 20))
    draw.text((120, 340), "1x Espresso Shot Extra           $0.80", fill=(20, 20, 20))
    draw.line([(100, 400), (900, 400)], fill=(180, 180, 180), width=2)

    draw.text((550, 430), "Subtotal:        $10.00", fill=(30, 30, 30))
    draw.text((550, 470), "Tax (8.5%):      $0.85", fill=(30, 30, 30))
    draw.text((550, 510), "InvoiceTotal:    $10.85", fill=(10, 10, 80))

    img.save(output_path, "PNG")
    print(f"Created {output_path}")


def create_blurry_sample(clean_png_path: str, output_path: str):
    """Generate a deliberately blurry, out-of-focus photographed document."""
    img = cv2.imread(clean_png_path)
    # Apply severe Gaussian blur mimicking out-of-focus camera capture
    blurred = cv2.GaussianBlur(img, (45, 45), 18.0)
    # Add slight camera lens vignette / uneven shading
    rows, cols = blurred.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / 2)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    blurred = np.clip(blurred * 0.8, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, blurred)
    print(f"Created {output_path}")


def create_low_res_sample(output_path: str):
    """Create a deliberately tiny, low-resolution unreadable thumbnail sample."""
    img = Image.new("RGB", (200, 150), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Tiny Unreadable Doc", fill=(80, 80, 80))
    img.save(output_path, "PNG")
    print(f"Created {output_path}")


def create_clean_docx(output_path: str):
    """Create a clean DOCX sample."""
    doc = docx.Document()
    doc.add_heading("VENDOR SERVICE AGREEMENT", 0)
    p = doc.add_paragraph()
    p.add_run("VendorName: Apex Technologies LLC\n").bold = True
    p.add_run("AgreementDate: 2024-01-10\n")
    p.add_run("DocumentNumber: AGR-2024-5502\n")

    table = doc.add_table(rows=1, cols=3)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Service Code"
    hdr_cells[1].text = "Description"
    hdr_cells[2].text = "Fee"

    services = [("SRV-101", "Systems Integration", "$4,500.00"), ("SRV-102", "Annual Maintenance", "$1,200.00")]
    for code, desc, fee in services:
        row_cells = table.add_row().cells
        row_cells[0].text = code
        row_cells[1].text = desc
        row_cells[2].text = fee

    doc.save(output_path)
    print(f"Created {output_path}")


def create_clean_po_pdf(output_path: str):
    """Create a realistic clean Purchase Order PDF."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text((50, 60), "PURCHASE ORDER", fontsize=24, color=(0.1, 0.3, 0.1))
    page.insert_text((50, 95), "VendorName: Contoso Logistics Inc.", fontsize=12, color=(0.2, 0.2, 0.2))
    page.insert_text((50, 115), "VendorAddress: 100 Main Street, Seattle, WA 98101", fontsize=10, color=(0.3, 0.3, 0.3))

    page.insert_text((400, 60), "DocumentNumber: PO-2024-8821", fontsize=12, color=(0.1, 0.1, 0.1))
    page.insert_text((400, 80), "InvoiceDate: 2024-03-15", fontsize=10, color=(0.3, 0.3, 0.3))
    page.insert_text((400, 100), "DueDate: 2024-04-15", fontsize=10, color=(0.3, 0.3, 0.3))

    page.insert_text((50, 160), "CustomerName: Acme Enterprise Corp", fontsize=11, color=(0.2, 0.2, 0.2))

    page.draw_rect(fitz.Rect(50, 220, 560, 360), color=(0.7, 0.7, 0.7), width=1)
    page.draw_line(fitz.Point(50, 250), fitz.Point(560, 250), color=(0.5, 0.5, 0.5), width=1)
    page.insert_text((60, 240), "Description", fontsize=10, color=(0, 0, 0))
    page.insert_text((490, 240), "LineTotal", fontsize=10, color=(0, 0, 0))

    page.insert_text((60, 280), "Logistics & Freight Services", fontsize=9, color=(0.2, 0.2, 0.2))
    page.insert_text((490, 280), "$900.00", fontsize=9, color=(0.2, 0.2, 0.2))

    page.insert_text((400, 400), "Subtotal: $900.00", fontsize=11, color=(0.1, 0.1, 0.1))
    page.insert_text((400, 420), "Tax (8%): $72.00", fontsize=11, color=(0.1, 0.1, 0.1))
    page.insert_text((400, 445), "InvoiceTotal: $972.00", fontsize=13, color=(0.1, 0.3, 0.1))

    doc.save(output_path)
    doc.close()
    print(f"Created {output_path}")


if __name__ == "__main__":
    create_clean_invoice_pdf("sample_data/clean_invoice.pdf")
    create_clean_png_receipt("sample_data/clean_receipt.png")
    create_clean_po_pdf("sample_data/clean_po.pdf")
    create_blurry_sample("sample_data/clean_receipt.png", "sample_data/deliberately_blurry_sample.jpg")
    create_low_res_sample("sample_data/low_res_sample.png")
    create_clean_docx("sample_data/clean_agreement.docx")
