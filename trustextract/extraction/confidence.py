"""
Loop 3 Confidence Scoring Pipeline for TrustExtract
====================================================
Fuses four independent signals into a single per-field confidence score:

  Signal 1 — native_ocr      : raw confidence from the winning extraction path
  Signal 2 — agreement        : whether Path A and Path B produced the same value
  Signal 3 — consistency      : variance across 3 re-runs of OCR with image perturbations
  Signal 4 — validation       : business-logic check results (arithmetic, dates, format)

Fusion formula (weighted sum, then optional validation cap):
  score = w_native * native_ocr
        + w_agreement * agreement_signal
        + w_consistency * consistency_signal
        + w_validation * validation_signal

  if any validator FAILED:
      score = min(score, cap_confidence_at)   # hard cap from the failing validator

Weights are documented below and can be overridden. They are NOT calibrated
against a labeled dataset; a trained calibration model is the planned follow-up
once ground-truth labels are available.

--------------------------------------------------------------------------
!!! IMPORTANT SIMULATION WARNING — READ BEFORE INTERPRETING RESULTS !!!
--------------------------------------------------------------------------
Path B (IndependentVisionModel) is a HAND-SCRIPTED SIMULATOR, NOT a real
second model.  Every value and confidence it returns is a Python literal
in path_b_vision.py (e.g. "Contoso Logistics Inc.", 0.96).  The only
"disagreement" it produces is a hard-coded string toggle on InvoiceId.

Consequence: the agreement_signal and all downstream fused scores in this
module are currently TUNED AGAINST SIMULATED PATH B BEHAVIOR, NOT
VALIDATED AGAINST A REAL SECOND MODEL.

Before using these confidence scores in a production decision (e.g. routing
to human review), Path B must be replaced with an actual vision API call
(Gemini Vision, Azure AI Document Intelligence, etc.) and the weights below
must be re-calibrated against a labeled ground-truth dataset.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from trustextract.extraction.path_a_ocr import PathAOCRExtractor, RawFieldExtraction
from trustextract.extraction.validators import DocumentValidationReport, FieldValidationResult
from trustextract.schema import ConfidenceBreakdown, ExtractionAgreement


# ---------------------------------------------------------------------------
# Fusion weights  (must sum to 1.0)
# These are manually tuned starting points — NOT empirically calibrated.
# Replace with a logistic regression / isotonic regression once labeled data
# is available.
# ---------------------------------------------------------------------------
WEIGHT_NATIVE_OCR   = 0.35
WEIGHT_AGREEMENT    = 0.25
WEIGHT_CONSISTENCY  = 0.25
WEIGHT_VALIDATION   = 0.15

assert abs(WEIGHT_NATIVE_OCR + WEIGHT_AGREEMENT + WEIGHT_CONSISTENCY + WEIGHT_VALIDATION - 1.0) < 1e-9, \
    "Fusion weights must sum to 1.0"

# Agreement signal mapping
_AGREEMENT_SCORES: Dict[ExtractionAgreement, float] = {
    ExtractionAgreement.OCR_EQ_VISION:  1.00,   # both paths agreed (normalized match)
    ExtractionAgreement.OCR_NEAR_MATCH: 0.85,   # close match (e.g. edit distance <= 2, ratio >= 0.85)
    ExtractionAgreement.SINGLE_SOURCE:  0.80,   # only one path found the field
    ExtractionAgreement.OCR_NEQ_VISION: 0.60,   # paths returned genuinely different values
}

# Number of OCR re-runs for self-consistency sampling
N_CONSISTENCY_PASSES = 3

# Seed for reproducible perturbations during testing (set to None for true randomness)
_CONSISTENCY_SEED = 42


class SelfConsistencySampler:
    """
    Runs Path A OCR N_CONSISTENCY_PASSES times on the same image with small
    perturbations (pixel-level noise, tiny crops) and measures how often the
    extracted value for each field is stable.

    Why perturbations?  RapidOCR is deterministic; re-running on the exact
    same bytes returns the exact same output.  Small realistic perturbations
    (sensor noise, JPEG re-compression, sub-pixel alignment) are what a real
    second scan would introduce.  Instability under these micro-changes
    signals that the extraction is fragile.
    """

    def __init__(self, n_passes: int = N_CONSISTENCY_PASSES, seed: Optional[int] = _CONSISTENCY_SEED) -> None:
        self.n_passes = n_passes
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._ocr = PathAOCRExtractor()

    def run(
        self,
        page_image: Image.Image,
        page_number: int,
        reference_fields: Dict[str, RawFieldExtraction],
    ) -> Dict[str, float]:
        """
        Args:
            page_image:        The normalised page PIL Image.
            page_number:       Page number (passed through to OCR extractor).
            reference_fields:  Fields already extracted on the unperturbed image
                               (Pass 1 baseline — counted as a consistent pass).

        Returns:
            Dict mapping field_name -> consistency_score (0.0 – 1.0).
            1.0 = all N_CONSISTENCY_PASSES returned identical value.
            0.0 = no two passes agreed.
        """
        # Pass 1 is the reference (unperturbed) already supplied by the caller.
        # We run (n_passes - 1) additional perturbed passes.
        results_per_field: Dict[str, List[str]] = {
            fname: [str(ref.value).strip().lower()]
            for fname, ref in reference_fields.items()
        }

        for i in range(self.n_passes - 1):
            perturbed = self._perturb(page_image, pass_index=i)
            _, pass_fields = self._ocr.extract_page_ocr(perturbed, page_number)
            for fname in reference_fields:
                val = str(pass_fields[fname].value).strip().lower() if fname in pass_fields else ""
                results_per_field[fname].append(val)

        # Consistency score = fraction of passes that matched the reference value
        consistency: Dict[str, float] = {}
        for fname, values in results_per_field.items():
            reference_val = values[0]
            matches = sum(1 for v in values if v == reference_val)
            consistency[fname] = round(matches / len(values), 3)

        return consistency

    def run_verbose(
        self,
        page_image: Image.Image,
        page_number: int,
        reference_fields: Dict[str, RawFieldExtraction],
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """
        Same as run() but also returns per-pass raw values for each field.

        Returns:
            Tuple of:
              - consistency scores  Dict[field_name, float]
              - per_pass_values     Dict[field_name, List[str]]  (one entry per pass)
        """
        results_per_field: Dict[str, List[str]] = {
            fname: [str(ref.value).strip().lower()]
            for fname, ref in reference_fields.items()
        }

        for i in range(self.n_passes - 1):
            perturbed = self._perturb(page_image, pass_index=i)
            _, pass_fields = self._ocr.extract_page_ocr(perturbed, page_number)
            for fname in reference_fields:
                val = str(pass_fields[fname].value).strip().lower() if fname in pass_fields else "<not found>"
                results_per_field[fname].append(val)

        consistency: Dict[str, float] = {}
        for fname, values in results_per_field.items():
            reference_val = values[0]
            matches = sum(1 for v in values if v == reference_val)
            consistency[fname] = round(matches / len(values), 3)

        return consistency, results_per_field

    def _perturb(self, img: Image.Image, pass_index: int) -> Image.Image:
        """Apply a small, repeatable perturbation to img."""
        arr = np.array(img.convert("RGB"), dtype=np.int16)

        # (a) Gaussian pixel noise: sigma scales slightly with pass index
        sigma = 3.0 + pass_index * 1.5
        noise = self._np_rng.normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

        # (b) Tiny 1-2 pixel crop shift to simulate sub-pixel alignment variation
        h, w = arr.shape[:2]
        dx = self._rng.randint(0, 2)
        dy = self._rng.randint(0, 2)
        arr = arr[dy:h - dy if dy else h, dx:w - dx if dx else w]

        # (c) Restore to original size via LANCZOS so OCR resolution is unchanged
        perturbed = Image.fromarray(arr).resize(img.size, Image.LANCZOS)
        return perturbed


class ConfidenceFuser:
    """
    Fuses native OCR confidence, path agreement, self-consistency, and validation
    signals into a single final confidence score per field.

    --------------------------------------------------------------------------
    !!! SIMULATION WARNING — SEE MODULE DOCSTRING !!!
    The agreement signal is computed against a hand-scripted Path B simulator.
    These scores are NOT validated against a real second model.
    --------------------------------------------------------------------------
    """

    def fuse(
        self,
        field_name: str,
        native_confidence: float,
        agreement: ExtractionAgreement,
        consistency_score: Optional[float],
        validation_result: Optional[FieldValidationResult],
    ) -> ConfidenceBreakdown:
        """
        Args:
            field_name:         Field being scored (for flag assembly).
            native_confidence:  Raw confidence from the winning extraction path.
            agreement:          Enum from reconciler (OCR_EQ_VISION / NEAR / NEQ / SINGLE).
            consistency_score:  Fraction of consistency passes that agreed (0–1),
                                or None if single_source / not extracted by Path A.
            validation_result:  Result from DocumentValidator for this field (or None).

        Returns:
            ConfidenceBreakdown with all signal values and fused final score.
        """
        agreement_signal = _AGREEMENT_SCORES[agreement]
        validation_signal = 1.0 if (validation_result is None or validation_result.passed) else 0.0
        validation_flags  = validation_result.flags if validation_result else ["NO_VALIDATOR_APPLIED"]

        if consistency_score is not None:
            # Standard 4-signal fusion
            raw_score = (
                WEIGHT_NATIVE_OCR   * native_confidence
              + WEIGHT_AGREEMENT    * agreement_signal
              + WEIGHT_CONSISTENCY  * consistency_score
              + WEIGHT_VALIDATION   * validation_signal
            )
        else:
            # Single-source / No-OCR reweighting: exclude consistency term entirely.
            # Reweight remaining 3 signals proportionally so their relative proportions
            # are preserved and their sum equals 1.0.
            sum_other_weights = WEIGHT_NATIVE_OCR + WEIGHT_AGREEMENT + WEIGHT_VALIDATION  # 0.75
            w_native = WEIGHT_NATIVE_OCR / sum_other_weights  # ~0.467
            w_agree  = WEIGHT_AGREEMENT  / sum_other_weights  # ~0.333
            w_val    = WEIGHT_VALIDATION / sum_other_weights  # ~0.200
            raw_score = (
                w_native * native_confidence
              + w_agree  * agreement_signal
              + w_val    * validation_signal
            )

        # Hard cap from failing validator
        if validation_result and not validation_result.passed:
            cap = validation_result.cap_confidence_at or 0.50
            final = min(round(raw_score, 3), cap)
        else:
            final = round(raw_score, 3)

        return ConfidenceBreakdown(
            native_ocr=round(native_confidence, 3),
            agreement_signal=round(agreement_signal, 3),
            consistency_signal=round(consistency_score, 3) if consistency_score is not None else None,
            validation_signal=round(validation_signal, 3),
            validation_flags=validation_flags,
            final=final,
        )


class ConfidencePipeline:
    """
    Loop 3 orchestrator: runs self-consistency sampling and business-logic
    validation, then fuses all signals into final per-field ConfidenceBreakdown
    objects and mutates the TrustExtractResult fields in-place.

    --------------------------------------------------------------------------
    !!! SIMULATION WARNING — SEE MODULE DOCSTRING !!!
    --------------------------------------------------------------------------
    """

    def __init__(
        self,
        n_consistency_passes: int = N_CONSISTENCY_PASSES,
        sampler_seed: Optional[int] = _CONSISTENCY_SEED,
    ) -> None:
        self._sampler = SelfConsistencySampler(n_passes=n_consistency_passes, seed=sampler_seed)
        self._fuser   = ConfidenceFuser()

    def score(
        self,
        result_fields: Dict[str, Any],       # FieldValue objects from Loop 2
        page_image: Image.Image,
        page_number: int,
        path_a_fields: Dict[str, RawFieldExtraction],
        validation_report: DocumentValidationReport,
    ) -> None:
        """
        Mutates result_fields in-place, setting confidence and confidence_breakdown
        on every FieldValue.

        Args:
            result_fields:      Dict[field_name, FieldValue] from Loop 2 reconciler.
            page_image:         Normalised page image (PIL).
            page_number:        Page number.
            path_a_fields:      Raw Path A extractions for self-consistency re-runs.
            validation_report:  DocumentValidationReport from DocumentValidator.
        """
        # Self-consistency: only re-run on fields Path A actually found
        consistency_scores = self._sampler.run(page_image, page_number, path_a_fields)

        for fname, fv in result_fields.items():
            native_conf     = fv.confidence            # reconciled (pre-fusion) confidence
            agreement       = fv.extraction_agreement
            # Pass None if Path A did not find this field (single-source / vision-only)
            # This triggers proportional reweighting instead of granting an unearned 1.0
            consistency     = consistency_scores.get(fname, None)
            val_result      = validation_report.per_field.get(fname)

            breakdown = self._fuser.fuse(
                field_name=fname,
                native_confidence=native_conf,
                agreement=agreement,
                consistency_score=consistency,
                validation_result=val_result,
            )

            # Update confidence with fused score and attach breakdown
            fv.confidence = breakdown.final
            fv.confidence_breakdown = breakdown

            # Reflect validation outcome in the top-level validation field
            if val_result and not val_result.passed:
                fv.validation = "FAILED"
            elif val_result and val_result.passed:
                fv.validation = "PASSED"


# ---------------------------------------------------------------------------
# Document-level escalation engine (runs AFTER confidence fusion)
# ---------------------------------------------------------------------------

# Threshold below which a field's final fused confidence triggers escalation.
# Chosen as 0.90: fields above this are considered reliably extracted;
# below it, human verification is warranted.
CONFIDENCE_ESCALATION_THRESHOLD: float = 0.90


class DocumentEscalationEngine:
    """
    Post-fusion escalation: inspects all FieldValue objects after ConfidencePipeline
    has set their final confidence and raises document_status to NEEDS_REVIEW
    if any of the following independent rules fire:

    Rule 1 — CONFIDENCE_BELOW_THRESHOLD
        Any field whose final fused confidence < CONFIDENCE_ESCALATION_THRESHOLD.
        Rationale: even if both paths agreed, a low fused score means the signal
        composite was weak enough to warrant human review.

    Rule 2 — PATH_DISAGREEMENT
        Any field whose extraction_agreement == OCR_NEQ_VISION, regardless of
        which path "won" the confidence vote. Two independent sources disagreeing
        is itself a flag, independent of the numeric score.

    Both rules are evaluated independently; both can fire on the same document
    simultaneously. Every firing rule produces a distinct entry in escalation_reasons
    so the reviewer sees every trigger, not just the first one found.

    Note: arithmetic / validator escalation is handled separately in the pipeline
    and adds its own entry to escalation_reasons. This engine does not re-check
    validators — it only looks at the post-fusion confidence and agreement signals.
    """

    def __init__(self, threshold: float = CONFIDENCE_ESCALATION_THRESHOLD) -> None:
        self.threshold = threshold

    def evaluate(
        self,
        result: Any,   # TrustExtractResult — typed as Any to avoid circular import
    ) -> None:
        """
        Mutates result.document_status and result.escalation_reasons in-place.
        Safe to call multiple times (reasons are appended, not overwritten).
        """
        for fname, fv in result.fields.items():
            bd = fv.confidence_breakdown
            final_conf = bd.final if bd else fv.confidence

            # Rule 1: confidence below threshold
            if final_conf < self.threshold:
                reason = (
                    f"Field '{fname}': final confidence {final_conf:.3f} "
                    f"< threshold {self.threshold:.2f}"
                )
                if reason not in result.escalation_reasons:
                    result.escalation_reasons.append(reason)

            # Rule 2: genuine path disagreement
            from trustextract.schema import ExtractionAgreement  # local import avoids circular
            if fv.extraction_agreement == ExtractionAgreement.OCR_NEQ_VISION:
                reason = (
                    f"Field '{fname}': ocr!=vision disagreement "
                    f"(Path A={fv.value!r} selected; paths diverged)"
                )
                if reason not in result.escalation_reasons:
                    result.escalation_reasons.append(reason)

        if result.escalation_reasons:
            from trustextract.schema import DocumentStatus
            result.document_status = DocumentStatus.NEEDS_REVIEW
