"""
Loop 8: Precision, Recall, and Confidence Calibration Evaluation Runner.
Evaluates TrustExtract across all document types against hand-labeled ground truth references,
broken down per-document and overall.
"""

import json
import os
from trustextract.evaluation.evaluate import evaluate_dataset, DocumentEvaluator
from trustextract.extraction.pipeline import DualPathExtractor
from trustextract.extraction.path_b_vision import ProductionVisionExtractor, LocalVisionLayoutExtractor


def run_evaluation():
    print("================================================================================")
    print("TRUSTEXTRACT LOOP 8: PRECISION, RECALL & CONFIDENCE CALIBRATION EVALUATION")
    print("================================================================================\n")

    manifest = [
        ("sample_data/clean_invoice.pdf", "sample_data/ground_truth/clean_invoice.json"),
        ("sample_data/clean_receipt.png", "sample_data/ground_truth/clean_receipt.json"),
        ("sample_data/clean_po.pdf", "sample_data/ground_truth/clean_po.json"),
        ("sample_data/clean_agreement.docx", "sample_data/ground_truth/clean_agreement.json"),
        ("sample_data/clean_payslip.png", "sample_data/ground_truth/clean_payslip.json"),
        ("sample_data/degraded_payslip.jpg", "sample_data/ground_truth/degraded_payslip.json"),
    ]

    extractor = DualPathExtractor(
        vision_model=LocalVisionLayoutExtractor(),
        n_consistency_passes=2
    )

    report = evaluate_dataset(manifest, dual_extractor=extractor)

    # 1. Per-Document Evaluation Breakdown
    print("--- [PART 1: PER-DOCUMENT ACCURACY BREAKDOWN] ---")
    for doc in report.per_document_details:
        doc_id = doc["document_id"]
        tp = doc["metrics"]["tp"]
        fp = doc["metrics"]["fp"]
        fn = doc["metrics"]["fn"]
        total_gt = tp + fn
        acc = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / total_gt) if total_gt > 0 else 0.0
        print(f"\nDocument: {doc_id:<25} | GT Fields: {total_gt:<2} | TP: {tp:<2} | FP: {fp:<2} | FN: {fn:<2} | Precision: {acc:.4f} | Recall: {rec:.4f}")
        for fname, fres in sorted(doc["field_results"].items()):
            status = fres["status"]
            ext_val = fres["extracted_value"]
            gt_val = fres["ground_truth_value"]
            conf = fres["confidence"]
            print(f"  Field: {fname:<18} | Status: {status:<3} | Conf: {conf:.3f} | Extracted: {str(ext_val):<22} | GT: {str(gt_val)}")

    # 2. Per-Field Precision & Recall Table
    print("\n--------------------------------------------------------------------------------")
    print("--- [PART 2: AGGREGATED PER-FIELD PRECISION & RECALL (N explicit)] ---")
    print(f"{'Field Name':<20} | {'Type':<12} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<10} | {'Recall':<10} | {'F1':<8} | {'N (GT/Ext)':<12}")
    print("-" * 105)

    sorted_fields = sorted(report.field_metrics.keys())
    for fname in sorted_fields:
        m = report.field_metrics[fname]
        n_str = f"{m.total_ground_truth}/{m.total_extracted}"
        print(f"{fname:<20} | {m.field_type.value:<12} | {m.true_positives:<4} | {m.false_positives:<4} | {m.false_negatives:<4} | {m.precision:<10.4f} | {m.recall:<10.4f} | {m.f1_score:<8.4f} | {n_str:<12}")

    print("-" * 105)
    print(f"{'OVERALL AGGREGATE':<35} | {report.overall_precision:<10.4f} | {report.overall_recall:<10.4f} | {report.overall_f1:<8.4f} | {sum(m.total_ground_truth for m in report.field_metrics.values())} GT total\n")

    # 3. Confidence Calibration by Bucket
    print("--- [PART 3: CONFIDENCE CALIBRATION ANALYSIS BY BUCKET] ---")
    print(f"{'Confidence Bucket':<32} | {'Count (N)':<10} | {'Correct':<8} | {'Empirical Accuracy':<20} | {'Mean Confidence':<16} | {'Calibration Gap':<15}")
    print("-" * 115)

    for bname, b in report.calibration_buckets.items():
        print(f"{bname:<32} | {b.total_count:<10} | {b.correct_count:<8} | {b.accuracy:<20.4f} | {b.mean_confidence:<16.4f} | {b.calibration_gap:<15.4f}")

    print("-" * 115)

    # Save detailed evaluation JSON
    eval_summary = {
        "sample_count": report.sample_count,
        "overall_precision": report.overall_precision,
        "overall_recall": report.overall_recall,
        "overall_f1": report.overall_f1,
        "per_document": report.per_document_details,
        "fields": {
            fname: {
                "field_type": m.field_type.value,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1_score,
                "tp": m.true_positives,
                "fp": m.false_positives,
                "fn": m.false_negatives,
                "n_ground_truth": m.total_ground_truth,
                "n_extracted": m.total_extracted,
            }
            for fname, m in report.field_metrics.items()
        },
        "calibration": {
            bname: {
                "total_count": b.total_count,
                "correct_count": b.correct_count,
                "accuracy": b.accuracy,
                "mean_confidence": b.mean_confidence,
                "gap": b.calibration_gap,
            }
            for bname, b in report.calibration_buckets.items()
        }
    }

    os.makedirs("evaluation_results", exist_ok=True)
    with open("evaluation_results/loop8_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print("\nDetailed evaluation artifact written to evaluation_results/loop8_evaluation_summary.json")
    print("================================================================================")


if __name__ == "__main__":
    run_evaluation()
