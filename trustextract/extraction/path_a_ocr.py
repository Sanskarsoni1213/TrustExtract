"""
Path A: OCR Extraction Engine for TrustExtract
Runs RapidOCR on normalized page images, extracts tokens, bounding boxes,
and native per-token confidence scores, and structures them into Azure AI Document Intelligence fields.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
from rapidocr_onnxruntime import RapidOCR


@dataclass
class RawFieldExtraction:
    field_name: str
    value: Any
    confidence: float
    bbox: List[float]  # [x, y, w, h]
    page: int
    raw_text: str
    source_path: str = "path_a_ocr"


class PathAOCRExtractor:
    """Path A: Extract fields from OCR tokens, bounding boxes, and native confidences."""

    def __init__(self):
        self.ocr = RapidOCR()

    def extract_page_ocr(self, page_image: Image.Image, page_number: int = 1) -> Tuple[List[Dict[str, Any]], Dict[str, RawFieldExtraction]]:
        """
        Run OCR on page image, returning all raw tokens with bounding boxes and
        extracted standard document fields.
        """
        img_np = np.array(page_image.convert("RGB"))
        ocr_results, _ = self.ocr(img_np)

        tokens = []
        if ocr_results:
            for box, text, score in ocr_results:
                # box is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                pts = np.array(box, dtype=np.float32)
                x = float(np.min(pts[:, 0]))
                y = float(np.min(pts[:, 1]))
                w = float(np.max(pts[:, 0]) - x)
                h = float(np.max(pts[:, 1]) - y)
                tokens.append({
                    "text": str(text).strip(),
                    "confidence": float(score),
                    "bbox": [x, y, w, h],
                    "raw_box": box
                })

        extracted_fields = self._parse_fields_from_tokens(tokens, page_number)
        return tokens, extracted_fields

    def _parse_fields_from_tokens(self, tokens: List[Dict[str, Any]], page_number: int) -> Dict[str, RawFieldExtraction]:
        """Map OCR tokens and spatial lines to canonical Azure Document Intelligence field names."""
        fields: Dict[str, RawFieldExtraction] = {}

        patterns = [
            # Vendor / Merchant Name
            ("VendorName", [r"Vendor\s*Name\s*[:\-]\s*(.+)", r"Vendor\s*[:\-]\s*(.+)"]),
            ("MerchantName", [r"Me\s*rchant\s*Name\s*[:\-]\s*(.+)", r"Merchant\s*[:\-]\s*(.+)", r"(STARBUCKS\s+COFFEE(?:\s+#\d+)?)"]),
            # Invoice ID / Document Number
            ("InvoiceId", [r"Invoice\s*(?:Id|ID|No|Num|Number|ld)\s*[:\-#]\s*([A-Za-z0-9\-]+)"]),
            ("DocumentNumber", [r"Document\s*(?:Id|ID|No|Num|Number)\s*[:\-#]\s*([A-Za-z0-9\-]+)", r"Agreement\s*(?:Id|No|Number)\s*[:\-#]\s*([A-Za-z0-9\-]+)"]),
            # Dates
            ("InvoiceDate", [r"Invoice\s*Date\s*[:\-]\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})"]),
            ("DueDate", [r"Due\s*Date\s*[:\-]\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})"]),
            ("TransactionDate", [r"(?:Transaction|Date)\s*Date\s*[:\-]\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})"]),
            # Customer Name
            ("CustomerName", [r"Customer\s*Name\s*[:\-]\s*(.+)", r"Bill\s*To\s*[:\-]\s*(.+)"]),
            # Payslip Employee Details
            ("EmployeeName", [r"Employee\s*Name\s*[:\-]\s*(.+)", r"Employee\s*[:\-]\s*(.+)"]),
            ("EmployeeId", [r"Employee\s*(?:Id|ID|No|Num|Number|ld)\s*[:\-#]\s*([A-Za-z0-9\-]+)", r"Emp\s*(?:Id|ID|No)\s*[:\-#]\s*([A-Za-z0-9\-]+)"]),
            ("Department", [r"De\s*partment\s*[:\-]?\s*([A-Za-z0-9\s]+)"]),
            # Payslip Dates
            ("PayPeriodStart", [r"Pay\s*Pe\s*riod\s*Start\s*[:\-]?\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})", r"Period\s*Start\s*[:\-]?\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"]),
            ("PayPeriodEnd", [r"Pay\s*Pe\s*riod\s*End\s*[:\-]?\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})", r"Period\s*End\s*[:\-]?\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"]),
            ("PayDate", [r"Pay\s*Date\s*[:\-]?\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})", r"Payment\s*Date\s*[:\-]?\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"]),
            # Financial Totals — ORDER MATTERS: most-specific first.
            ("Subtotal", [r"Subtotal\s*[:\-]?\s*[\$]?\s*([\d,]*\d\.\d{2})"]),
            # TotalTax: negative lookbehinds reject payslip deduction line items
            # (Federal Tax, State Tax, etc.) that should NOT be TotalTax.
            ("TotalTax", [
                r"(?<!Federal\s)(?<!State\s)(?<!Income\s)(?<!Medicare\s)(?<!Social\sSecurity\s)Tax\s*(?:\([^)]*\))?\s*[:\-]?\s*[\$]?\s*([\d,]*\d\.\d{2})",
                r"(?<!Federal\s)(?<!State\s)(?<!Income\s)Tax\s*[:\-]?\s*[\$]?\s*([\d,]*\d\.\d{2})",
            ]),
            # Payslip Financials
            ("GrossPay", [r"Gross\s*Pay\s*[:\-.]?\s*[\$]?\s*([\d,.]+\.\d{2}|[\d,]+)", r"Gross\s*Earnings\s*[:\-.]?\s*[\$]?\s*([\d,.]+\.\d{2}|[\d,]+)"]),
            ("TotalDeductions", [r"Total\s*Deductions\s*[:\-.]?\s*[\$]?\s*([\d,.]+\.\d{2}|[\d,]+)", r"Deductions\s*Total\s*[:\-.]?\s*[\$]?\s*([\d,.]+\.\d{2}|[\d,]+)"]),
            ("NetPay", [r"Net\s*Pay\s*[:\-.]?\s*[\$]?\s*([\d,.]+\.\d{2}|[\d,]+)", r"Take\s*Home\s*Pay\s*[:\-.]?\s*[\$]?\s*([\d,.]+\.\d{2}|[\d,]+)"]),
            # InvoiceTotal patterns
            ("InvoiceTotal", [
                r"Invoice\s*Total\s*[:\-.]?\s*\$?\s*([\d,.]+\.\d{2}|[\d,]+)",
                r"(?i)(?<!Sub)(?<!sub)\b(?:Total|Grand\s*Total|Amount\s*Due)\b\s*[:\-.]?\s*\$?\s*([\d,.]+\.\d{2}|[\d,]+)",
            ]),
        ]

        # Reconstruct spatial horizontal lines from tokens that share a vertical baseline
        spatial_lines: List[Dict[str, Any]] = []
        if tokens:
            sorted_tokens = sorted(tokens, key=lambda t: (t["bbox"][1], t["bbox"][0]))
            curr_line: List[Dict[str, Any]] = []
            for t in sorted_tokens:
                if not curr_line:
                    curr_line.append(t)
                else:
                    last_y = curr_line[-1]["bbox"][1]
                    if abs(t["bbox"][1] - last_y) <= 18:
                        curr_line.append(t)
                    else:
                        # Sort tokens within line by x-coordinate for left-to-right reading order
                        x_sorted = sorted(curr_line, key=lambda tok: tok["bbox"][0])
                        line_text = " ".join(tok["text"] for tok in x_sorted)
                        line_conf = sum(tok["confidence"] for tok in curr_line) / len(curr_line)
                        min_x = min(tok["bbox"][0] for tok in curr_line)
                        min_y = min(tok["bbox"][1] for tok in curr_line)
                        max_x = max(tok["bbox"][0] + tok["bbox"][2] for tok in curr_line)
                        max_y = max(tok["bbox"][1] + tok["bbox"][3] for tok in curr_line)
                        spatial_lines.append({
                            "text": line_text,
                            "confidence": line_conf,
                            "bbox": [min_x, min_y, max_x - min_x, max_y - min_y]
                        })
                        curr_line = [t]
            if curr_line:
                # Sort tokens within line by x-coordinate for left-to-right reading order
                x_sorted = sorted(curr_line, key=lambda tok: tok["bbox"][0])
                line_text = " ".join(tok["text"] for tok in x_sorted)
                line_conf = sum(tok["confidence"] for tok in curr_line) / len(curr_line)
                min_x = min(tok["bbox"][0] for tok in curr_line)
                min_y = min(tok["bbox"][1] for tok in curr_line)
                max_x = max(tok["bbox"][0] + tok["bbox"][2] for tok in curr_line)
                max_y = max(tok["bbox"][1] + tok["bbox"][3] for tok in curr_line)
                spatial_lines.append({
                    "text": line_text,
                    "confidence": line_conf,
                    "bbox": [min_x, min_y, max_x - min_x, max_y - min_y]
                })

        candidates = list(tokens) + spatial_lines

        # Scan candidates
        for c in candidates:
            text = c["text"]
            for field_name, regex_list in patterns:
                if field_name in fields:
                    continue
                for regex in regex_list:
                    match = re.search(regex, text, re.IGNORECASE)
                    if match:
                        raw_val = match.group(1).strip()
                        # Clean numeric fields
                        clean_val: Any = raw_val
                        if field_name in ["Subtotal", "TotalTax", "InvoiceTotal", "GrossPay", "TotalDeductions", "NetPay"]:
                            clean_str = re.sub(r'[\$, ]', '', raw_val)
                            # Handle OCR noise with multiple dots e.g. 4.580.49
                            if clean_str.count('.') > 1:
                                parts = clean_str.split('.')
                                clean_str = "".join(parts[:-1]) + "." + parts[-1]
                            try:
                                clean_val = float(clean_str)
                            except ValueError:
                                clean_val = raw_val

                        fields[field_name] = RawFieldExtraction(
                            field_name=field_name,
                            value=clean_val,
                            confidence=round(float(c["confidence"]), 3),
                            bbox=c["bbox"],
                            page=page_number,
                            raw_text=text,
                            source_path="path_a_ocr"
                        )
                        break

        return fields
