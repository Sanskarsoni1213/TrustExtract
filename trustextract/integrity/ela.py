"""
Error Level Analysis (ELA) for TrustExtract (Loop 4)
Detects digital tampering, localized image splicing, and recompression anomalies
by measuring compression error differences across image regions.
"""

from dataclasses import dataclass, field
import io
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image


@dataclass
class ELARegion:
    bbox: List[int]  # [x, y, width, height]
    anomaly_score: float
    description: str


@dataclass
class ELAResult:
    has_anomaly: bool
    peak_error: float
    mean_error: float
    std_error: float
    anomalous_regions: List[ELARegion] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


class ErrorLevelAnalyzer:
    """
    Performs Error Level Analysis (ELA) by re-compressing images at a known
    quality factor and measuring differential compression artifact rates.
    
    When an image is modified (e.g. an amount or text box pasted in), the pasted
    area has a different compression history than the background, producing a
    measurable spike or drop in error level during recompression.
    """

    def __init__(self, quality: int = 90, scale_factor: float = 15.0, block_size: int = 16, threshold_sigma: float = 2.8):
        self.quality = quality
        self.scale_factor = scale_factor
        self.block_size = block_size
        self.threshold_sigma = threshold_sigma

    def analyze(self, image: Image.Image) -> ELAResult:
        """
        Run ELA on a PIL Image.
        Returns:
            ELAResult containing anomaly metrics, flags, and localized tampered regions.
        """
        img_rgb = image.convert("RGB")
        orig_arr = np.array(img_rgb, dtype=np.float32)

        # 1. Recompress image at standard reference quality
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        recomp_arr = np.array(recompressed, dtype=np.float32)

        # 2. Compute absolute difference per pixel
        diff = np.abs(orig_arr - recomp_arr)
        # Luminance difference across RGB channels
        gray_diff = np.mean(diff, axis=2)

        mean_error = float(np.mean(gray_diff))
        std_error = float(np.std(gray_diff))
        peak_error = float(np.max(gray_diff))

        # 3. Block-level grid analysis for localized tampering
        h, w = gray_diff.shape
        bs = self.block_size
        stride = bs // 2
        block_stats = []

        for y in range(0, h - bs + 1, stride):
            for x in range(0, w - bs + 1, stride):
                block = gray_diff[y:y + bs, x:x + bs]
                b_mean = float(np.mean(block))
                b_max = float(np.max(block))
                block_stats.append((x, y, bs, bs, b_mean, b_max))

        if not block_stats:
            return ELAResult(
                has_anomaly=False,
                peak_error=peak_error,
                mean_error=mean_error,
                std_error=std_error,
                anomalous_regions=[],
                flags=[]
            )

        bm_arr = np.array([s[4] for s in block_stats])
        content_means = bm_arr[bm_arr > 0.5]

        if len(content_means) > 10:
            ref_med = float(np.median(content_means))
            ref_mad = float(np.median(np.abs(content_means - ref_med))) or 1.0
            # A spliced/altered region must be an outlier against normal content/text
            mean_threshold = max(ref_med + self.threshold_sigma * (ref_mad * 1.4826), 8.5)
            max_threshold = max(ref_med + (self.threshold_sigma + 2.5) * (ref_mad * 1.4826), 28.0)
        else:
            ref_med = float(np.median(bm_arr))
            ref_mad = float(np.median(np.abs(bm_arr - ref_med))) or 0.5
            mean_threshold = max(ref_med + self.threshold_sigma * (ref_mad * 1.4826), 8.5)
            max_threshold = max(ref_med + (self.threshold_sigma + 2.0) * (ref_mad * 1.4826), 28.0)

        anomalous_regions = []
        for x, y, bw, bh, b_mean, b_max in block_stats:
            if b_mean > mean_threshold or (b_mean > 5.0 and b_max > 45.0):
                score = round((b_mean - ref_med) / (ref_mad * 1.4826 + 1e-6), 2)
                anomalous_regions.append(
                    ELARegion(
                        bbox=[int(x), int(y), int(bw), int(bh)],
                        anomaly_score=max(score, round(b_max / max(ref_med, 1.0), 2)),
                        description=f"High ELA variance block (mean={b_mean:.1f}, peak={b_max:.1f} vs content_med={ref_med:.1f})"
                    )
                )

        # Merge overlapping/adjacent anomalous blocks
        merged_regions = self._merge_regions(anomalous_regions)
        has_anomaly = len(merged_regions) > 0

        flags = ["ELA_ANOMALY_DETECTED"] if has_anomaly else []

        return ELAResult(
            has_anomaly=has_anomaly,
            peak_error=round(peak_error, 2),
            mean_error=round(mean_error, 2),
            std_error=round(std_error, 2),
            anomalous_regions=merged_regions,
            flags=flags
        )

    def _merge_regions(self, regions: List[ELARegion]) -> List[ELARegion]:
        """Merge nearby/overlapping anomaly blocks into cohesive bounding boxes."""
        if not regions:
            return []
        
        boxes = [r.bbox for r in regions]
        merged = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, w1, h1 = boxes[i]
            rx1, ry1, rx2, ry2 = x1, y1, x1 + w1, y1 + h1
            max_score = regions[i].anomaly_score
            used[i] = True

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                # Check proximity / intersection (within 16px)
                if not (x2 > rx2 + 16 or x2 + w2 < rx1 - 16 or y2 > ry2 + 16 or y2 + h2 < ry1 - 16):
                    rx1 = min(rx1, x2)
                    ry1 = min(ry1, y2)
                    rx2 = max(rx2, x2 + w2)
                    ry2 = max(ry2, y2 + h2)
                    max_score = max(max_score, regions[j].anomaly_score)
                    used[j] = True

            merged.append(
                ELARegion(
                    bbox=[rx1, ry1, rx2 - rx1, ry2 - ry1],
                    anomaly_score=max_score,
                    description=f"Tampered/Spliced region at ({rx1},{ry1}) with peak anomaly {max_score:.1f} sigma"
                )
            )

        return merged
