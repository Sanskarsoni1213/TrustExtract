"""
Generate hand-labeled ground truth reference JSON files for TrustExtract evaluation (Loop 8).
Stores expected field values per document sample.
"""

import json
import os

GROUND_TRUTH_DATA = {
    "clean_invoice.pdf": {
        "document_type": "invoice",
        "description": "Standard clean corporate invoice",
        "fields": {
            "VendorName": "Contoso Logistics Inc.",
            "InvoiceId": "INV-2024-9841",
            "InvoiceDate": "2024-03-15",
            "DueDate": "2024-04-15",
            "CustomerName": "Acme Enterprise Corp",
            "Subtotal": 900.00,
            "TotalTax": 72.00,
            "InvoiceTotal": 972.00,
        }
    },
    "clean_receipt.png": {
        "document_type": "receipt",
        "description": "Clean retail coffee merchant receipt",
        "fields": {
            "MerchantName": "Starbucks Coffee",
            "TransactionDate": "2024-04-02",
            "Subtotal": 10.00,
            "TotalTax": 0.85,
            "InvoiceTotal": 10.85,
        }
    },
    "clean_po.pdf": {
        "document_type": "purchase_order",
        "description": "Clean matching purchase order",
        "fields": {
            "VendorName": "Contoso Logistics Inc.",
            "DocumentNumber": "PO-2024-8821",
            "InvoiceDate": "2024-03-15",
            "DueDate": "2024-04-15",
            "CustomerName": "Acme Enterprise Corp",
            "Subtotal": 900.00,
            "TotalTax": 72.00,
            "InvoiceTotal": 972.00,
        }
    },
    "clean_agreement.docx": {
        "document_type": "agreement",
        "description": "Clean vendor services agreement document",
        "fields": {
            "VendorName": "Apex Technologies LLC",
            "DocumentNumber": "AGR-2024-5502",
        }
    },
    "clean_payslip.png": {
        "document_type": "payslip",
        "description": "High-resolution clean payslip document",
        "fields": {
            "EmployeeName": "Sarah Chen",
            "EmployeeId": "EMP-2024-0471",
            "Department": "Engineering",
            "PayPeriodStart": "2024-03-01",
            "PayPeriodEnd": "2024-03-31",
            "PayDate": "2024-04-05",
            "GrossPay": 7787.50,
            "TotalDeductions": 3207.01,
            "NetPay": 4580.49,
        }
    },
    "degraded_payslip.jpg": {
        "document_type": "payslip",
        "description": "Degraded (skewed, compressed, noisy) payslip sample",
        "fields": {
            "EmployeeName": "Sarah Chen",
            "EmployeeId": "EMP-2024-0471",
            "Department": "Engineering",
            "PayPeriodStart": "2024-03-01",
            "PayPeriodEnd": "2024-03-31",
            "PayDate": "2024-04-05",
            "GrossPay": 7787.50,
            "TotalDeductions": 3207.01,
            "NetPay": 4580.49,
        }
    },
}


def write_ground_truth(gt_dir: str = "sample_data/ground_truth"):
    os.makedirs(gt_dir, exist_ok=True)
    for sample_filename, gt_spec in GROUND_TRUTH_DATA.items():
        base_name = os.path.splitext(sample_filename)[0]
        out_path = os.path.join(gt_dir, f"{base_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(gt_spec, f, indent=2)
        print(f"Wrote ground truth: {out_path} ({len(gt_spec['fields'])} fields)")


if __name__ == "__main__":
    write_ground_truth()
