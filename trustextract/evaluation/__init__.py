"""
Evaluation Package for TrustExtract (Loop 8).
Measures Precision, Recall, and Confidence Calibration against Ground Truth Reference Data.
"""

from trustextract.evaluation.evaluate import (
    FieldEvaluationMetric,
    CalibrationBucket,
    EvaluationReport,
    DocumentEvaluator,
    evaluate_dataset,
)

__all__ = [
    "FieldEvaluationMetric",
    "CalibrationBucket",
    "EvaluationReport",
    "DocumentEvaluator",
    "evaluate_dataset",
]
