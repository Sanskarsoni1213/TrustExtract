"""
Font and Spacing Consistency Analyzer for TrustExtract (Loop 4)
Measures character spacing, bounding box heights, and baseline alignments
across extracted text regions to detect digitally altered/spliced values.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class FieldFontMetrics:
    field_name: str
    token_text: str
    char_width: float
    token_height: float
    aspect_ratio: float
    bbox: List[float]
    is_anomalous: bool
    deviation_score: float
    reason: Optional[str] = None


@dataclass
class FontConsistencyResult:
    has_anomaly: bool
    median_char_width: float
    median_token_height: float
    anomalous_fields: List[FieldFontMetrics] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


class FontConsistencyDetector:
    """
    Analyzes geometric and typographical metrics across OCR tokens.
    Detects digitally pasted or altered fields whose font size, character pitch,
    or vertical baseline diverges from surrounding document text.
    """

    def __init__(self, deviation_sigma_threshold: float = 2.8):
        self.deviation_sigma_threshold = deviation_sigma_threshold

    def analyze_tokens_and_fields(
        self,
        all_tokens: List[Dict[str, Any]],
        extracted_fields: Dict[str, Any]
    ) -> FontConsistencyResult:
        """
        Args:
            all_tokens: List of raw OCR tokens on the page (from PathAOCRExtractor).
                        Each token has {'text': str, 'bbox': [x, y, w, h], 'confidence': float}.
            extracted_fields: Dict of field_name -> FieldValue (or RawFieldExtraction).

        Returns:
            FontConsistencyResult with detected anomalies and flags.
        """
        if not all_tokens:
            return FontConsistencyResult(
                has_anomaly=False,
                median_char_width=0.0,
                median_token_height=0.0,
                anomalous_fields=[],
                flags=[]
            )

        # 1. Compute global baseline typographical metrics from body tokens
        char_widths = []
        token_heights = []

        for tok in all_tokens:
            text = str(tok.get("text", "")).strip()
            bbox = tok.get("bbox") or [0, 0, 0, 0]
            if len(text) >= 3 and bbox[2] > 0 and bbox[3] > 0:
                # Character pitch = width / character count
                char_w = float(bbox[2]) / len(text)
                char_widths.append(char_w)
                token_heights.append(float(bbox[3]))

        if not char_widths:
            return FontConsistencyResult(
                has_anomaly=False,
                median_char_width=0.0,
                median_token_height=0.0,
                anomalous_fields=[],
                flags=[]
            )

        med_char_w = float(np.median(char_widths))
        std_char_w = float(np.std(char_widths)) or 1.0
        med_height = float(np.median(token_heights))
        std_height = float(np.std(token_heights)) or 1.0

        # 2. Inspect each extracted field against local and global typographical metrics
        anomalous_fields = []

        for fname, fval in extracted_fields.items():
            # Get bbox and text
            bbox = None
            val_str = ""
            if hasattr(fval, "source") and fval.source:
                bbox = fval.source.bbox
                val_str = str(fval.value)
            elif hasattr(fval, "bbox"):
                bbox = fval.bbox
                val_str = str(fval.value)
            elif isinstance(fval, dict) and "bbox" in fval:
                bbox = fval["bbox"]
                val_str = str(fval.get("value", ""))

            if not bbox or len(bbox) < 4 or len(val_str.strip()) < 2:
                continue

            bw = float(bbox[2])
            bh = float(bbox[3])
            if bw <= 0 or bh <= 0:
                continue

            field_char_w = bw / len(val_str.strip())
            aspect = bw / bh

            # Calculate z-scores against document baseline
            z_char_w = abs(field_char_w - med_char_w) / std_char_w
            z_height = abs(bh - med_height) / std_height

            is_anom = False
            reasons = []

            # (a) Font size divergence in fields
            if bh > (med_height * 1.5) or bh < (med_height * 0.5):
                is_anom = True
                reasons.append(f"Abnormal font height ({bh:.1f}px vs median {med_height:.1f}px)")

            # (b) Irregular character pitch
            if field_char_w > (med_char_w * 1.7) or field_char_w < (med_char_w * 0.5):
                is_anom = True
                reasons.append(f"Inconsistent character pitch ({field_char_w:.1f}px/char vs median {med_char_w:.1f}px/char)")

            # (c) Check local baseline misalignment if field is in a numeric column
            local_y = float(bbox[1])
            nearby_tokens = [
                t for t in all_tokens
                if abs(float(t.get("bbox", [0, 0, 0, 0])[1]) - local_y) < 15.0 and t.get("text") != val_str
            ]
            if nearby_tokens:
                neighbor_heights = [float(t["bbox"][3]) for t in nearby_tokens if len(t.get("bbox", [])) >= 4]
                if neighbor_heights:
                    local_med_h = float(np.median(neighbor_heights))
                    if abs(bh - local_med_h) > 10.0 and abs(bh - local_med_h) / max(local_med_h, 1.0) > 0.35:
                        is_anom = True
                        reasons.append(f"Local line height divergence from neighbor tokens ({bh:.1f}px vs neighbor {local_med_h:.1f}px)")

            if is_anom:
                score = max(z_char_w, z_height, round(bh / max(med_height, 1.0), 2))
                anomalous_fields.append(
                    FieldFontMetrics(
                        field_name=fname,
                        token_text=val_str,
                        char_width=round(field_char_w, 2),
                        token_height=round(bh, 2),
                        aspect_ratio=round(aspect, 2),
                        bbox=[round(c, 1) for c in bbox],
                        is_anomalous=True,
                        deviation_score=round(score, 2),
                        reason="; ".join(reasons)
                    )
                )

        has_anomaly = len(anomalous_fields) > 0
        flags = ["FONT_INCONSISTENCY"] if has_anomaly else []

        return FontConsistencyResult(
            has_anomaly=has_anomaly,
            median_char_width=round(med_char_w, 2),
            median_token_height=round(med_height, 2),
            anomalous_fields=anomalous_fields,
            flags=flags
        )
