"""
Precision, Recall, and Confidence Calibration Evaluator for TrustExtract (Loop 8).

Compares TrustExtract extracted field values against hand-labeled ground truth references:
  1. Semantic type-aware field equivalence checking.
  2. Per-field Precision, Recall, F1, and explicit sample counts N.
  3. Confidence Calibration analysis across confidence buckets ([0.90, 1.00], [0.75, 0.90), [0.00, 0.75)).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from trustextract.schema import (
    FieldType,
    FieldValue,
    TrustExtractResult,
)
from trustextract.extraction.pipeline import classify_field_type, DualPathExtractor
from trustextract.ingestion.pipeline import IngestionPipeline


@dataclass
class FieldEvaluationMetric:
    """Precision and recall evaluation metrics for a single field name."""
    field_name: str
    field_type: FieldType
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_ground_truth: int = 0
    total_extracted: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return round(self.true_positives / denom, 4) if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return round(self.true_positives / denom, 4) if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * (p * r) / (p + r), 4) if (p + r) > 0 else 0.0


@dataclass
class CalibrationBucket:
    """Confidence calibration stats for a specific confidence range."""
    bucket_name: str
    min_conf: float
    max_conf: float
    total_count: int = 0
    correct_count: int = 0
    sum_confidence: float = 0.0

    @property
    def accuracy(self) -> float:
        return round(self.correct_count / self.total_count, 4) if self.total_count > 0 else 0.0

    @property
    def mean_confidence(self) -> float:
        return round(self.sum_confidence / self.total_count, 4) if self.total_count > 0 else 0.0

    @property
    def calibration_gap(self) -> float:
        return round(abs(self.mean_confidence - self.accuracy), 4) if self.total_count > 0 else 0.0


@dataclass
class EvaluationReport:
    """Comprehensive evaluation results over a document test dataset."""
    sample_count: int = 0
    field_metrics: Dict[str, FieldEvaluationMetric] = field(default_factory=dict)
    calibration_buckets: Dict[str, CalibrationBucket] = field(default_factory=dict)
    per_document_details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def overall_precision(self) -> float:
        tp = sum(m.true_positives for m in self.field_metrics.values())
        fp = sum(m.false_positives for m in self.field_metrics.values())
        return round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0

    @property
    def overall_recall(self) -> float:
        tp = sum(m.true_positives for m in self.field_metrics.values())
        fn = sum(m.false_negatives for m in self.field_metrics.values())
        return round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0

    @property
    def overall_f1(self) -> float:
        p, r = self.overall_precision, self.overall_recall
        return round(2 * (p * r) / (p + r), 4) if (p + r) > 0 else 0.0


class DocumentEvaluator:
    """Evaluates TrustExtract extraction results against ground truth reference JSON."""

    def __init__(self, currency_tolerance: float = 0.02):
        self.currency_tolerance = currency_tolerance

    def values_match(self, extracted_val: Any, gt_val: Any, field_name: str) -> bool:
        """Determines semantic equality between extracted and ground truth values."""
        if extracted_val is None or gt_val is None:
            return extracted_val == gt_val

        ftype = classify_field_type(field_name)

        # 1. IDENTIFIER: Strict exact match (whitespace stripped)
        if ftype == FieldType.IDENTIFIER:
            return str(extracted_val).strip() == str(gt_val).strip()

        # 2. NUMERIC: Float parse and tolerance comparison
        if ftype == FieldType.NUMERIC:
            num_ext = self._parse_num(extracted_val)
            num_gt = self._parse_num(gt_val)
            if num_ext is not None and num_gt is not None:
                return abs(num_ext - num_gt) <= self.currency_tolerance
            return False

        # 3. DATE: Canonical date parse
        if ftype == FieldType.DATE:
            date_ext = self._parse_date(extracted_val)
            date_gt = self._parse_date(gt_val)
            if date_ext is not None and date_gt is not None:
                return date_ext == date_gt
            return str(extracted_val).strip() == str(gt_val).strip()

        # 4. FREE_TEXT: Normalized string comparison with GT-in-extracted substring tolerance
        norm_ext = re.sub(r'[^a-zA-Z0-9]', '', str(extracted_val).lower())
        norm_gt = re.sub(r'[^a-zA-Z0-9]', '', str(gt_val).lower())
        if norm_ext == norm_gt:
            return True
        # GT-in-extracted direction only: accept if the normalized GT is a contiguous
        # substring of the normalized extracted value. This handles cases like
        # extracted="STARBUCKS COFFEE #1042" vs GT="Starbucks Coffee".
        # Do NOT accept extracted-in-GT (that direction would over-forgive truncated values).
        if len(norm_gt) >= 3 and norm_gt in norm_ext:
            return True
        return False

    @staticmethod
    def _parse_num(val: Any) -> Optional[float]:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            cleaned = re.sub(r'[\$,€£\s]', '', val)
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_date(val: Any) -> Optional[str]:
        if not isinstance(val, str):
            return None
        formats = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%d-%m-%Y", "%m-%d-%Y"]
        for fmt in formats:
            try:
                dt = datetime.strptime(val.strip(), fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    def evaluate_sample(
        self,
        extracted_fields: Dict[str, FieldValue],
        gt_fields: Dict[str, Any],
        doc_id: str = "doc"
    ) -> Dict[str, Any]:
        """
        Evaluates a single document extraction against ground truth.
        Returns per-field classification results: TP, FP, FN with confidence.
        """
        sample_eval = {
            "document_id": doc_id,
            "field_results": {},
            "metrics": {"tp": 0, "fp": 0, "fn": 0}
        }

        all_keys = set(extracted_fields.keys()).union(gt_fields.keys())

        for fname in all_keys:
            has_gt = fname in gt_fields and gt_fields[fname] is not None
            has_ext = fname in extracted_fields and extracted_fields[fname].value is not None

            gt_val = gt_fields.get(fname)
            ext_obj = extracted_fields.get(fname)
            ext_val = ext_obj.value if ext_obj else None
            conf = ext_obj.confidence if ext_obj else 0.0

            if has_gt and has_ext:
                is_match = self.values_match(ext_val, gt_val, fname)
                if is_match:
                    status = "TP"
                    sample_eval["metrics"]["tp"] += 1
                else:
                    status = "FP"
                    sample_eval["metrics"]["fp"] += 1
            elif has_ext and not has_gt:
                status = "FP"  # Hallucinated / unwanted extra field
                sample_eval["metrics"]["fp"] += 1
            elif has_gt and not has_ext:
                status = "FN"  # Missed field
                sample_eval["metrics"]["fn"] += 1
            else:
                continue

            sample_eval["field_results"][fname] = {
                "status": status,
                "extracted_value": ext_val,
                "ground_truth_value": gt_val,
                "confidence": conf,
            }

        return sample_eval


def evaluate_dataset(
    dataset_manifest: List[Tuple[str, str]],  # List of (sample_file_path, gt_json_path)
    dual_extractor: Optional[DualPathExtractor] = None,
) -> EvaluationReport:
    """
    Executes end-to-end evaluation over an annotated dataset.
    Computes precision/recall per field and calibration by confidence bucket.
    """
    ingest = IngestionPipeline()
    extractor = dual_extractor or DualPathExtractor(n_consistency_passes=2)
    evaluator = DocumentEvaluator()

    report = EvaluationReport(sample_count=len(dataset_manifest))

    # Initialize calibration buckets
    report.calibration_buckets = {
        "High ([0.90, 1.00])": CalibrationBucket("High ([0.90, 1.00])", 0.90, 1.00),
        "Medium ([0.75, 0.90))": CalibrationBucket("Medium ([0.75, 0.90))", 0.75, 0.90),
        "Low / Escalated ([0.00, 0.75))": CalibrationBucket("Low / Escalated ([0.00, 0.75))", 0.00, 0.75),
    }

    for sample_path, gt_path in dataset_manifest:
        if not os.path.exists(sample_path) or not os.path.exists(gt_path):
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gt_fields = gt_data.get("fields", {})

        doc_id = os.path.basename(sample_path)
        _, norm_doc = ingest.process_document(sample_path, document_id=doc_id)
        if not norm_doc:
            continue

        result, _, _ = extractor.process_normalized_document(norm_doc)
        sample_res = evaluator.evaluate_sample(result.fields, gt_fields, doc_id=doc_id)
        report.per_document_details.append(sample_res)

        # Aggregate metrics
        for fname, fres in sample_res["field_results"].items():
            ftype = classify_field_type(fname)
            metric = report.field_metrics.setdefault(
                fname, FieldEvaluationMetric(field_name=fname, field_type=ftype)
            )

            status = fres["status"]
            conf = fres["confidence"]

            if status == "TP":
                metric.true_positives += 1
                metric.total_extracted += 1
                metric.total_ground_truth += 1
            elif status == "FP":
                metric.false_positives += 1
                metric.total_extracted += 1
            elif status == "FN":
                metric.false_negatives += 1
                metric.total_ground_truth += 1

            # Place into calibration bucket if extracted
            if fres["extracted_value"] is not None:
                is_correct = (status == "TP")
                if conf >= 0.90:
                    bucket = report.calibration_buckets["High ([0.90, 1.00])"]
                elif conf >= 0.75:
                    bucket = report.calibration_buckets["Medium ([0.75, 0.90))"]
                else:
                    bucket = report.calibration_buckets["Low / Escalated ([0.00, 0.75))"]

                bucket.total_count += 1
                if is_correct:
                    bucket.correct_count += 1
                bucket.sum_confidence += conf

    return report
