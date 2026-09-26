"""
Quality Assessment Engine for TrustExtract
Evaluates document resolution, blur, text density, and contrast.
Rejects degraded or unreadable inputs before extraction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image


@dataclass
class QualityThresholds:
    min_width: int = 600
    min_height: int = 600
    min_laplacian_variance: float = 2.5    # Calibrated threshold: below ~1.5 text smears; 2.5 admits readable soft captures while rejecting smudged blur (<1.0)
    min_edge_density: float = 0.002       # Minimum edge density to ensure readable text strokes
    min_mean_brightness: float = 20.0     # Minimum brightness (not pitch black)
    max_mean_brightness: float = 252.0    # Maximum brightness (not pure white washed out)


@dataclass
class QualityReport:
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class QualityChecker:
    """Evaluates image quality for reliable OCR and document processing."""

    def __init__(self, thresholds: QualityThresholds = QualityThresholds()):
        self.thresholds = thresholds

    def assess_page_image(self, pil_image: Image.Image) -> QualityReport:
        """Analyze a PIL page image and determine if it meets quality standards."""
        reasons = []
        metrics = {}

        # 1. Resolution Check
        width, height = pil_image.size
        metrics["width"] = float(width)
        metrics["height"] = float(height)

        if width < self.thresholds.min_width or height < self.thresholds.min_height:
            reasons.append(
                f"Resolution check failed: {width}x{height} is below minimum requirement of "
                f"{self.thresholds.min_width}x{self.thresholds.min_height}."
            )

        # Convert to OpenCV grayscale
        img_np = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # 2. Blur / Sharpness Check (Laplacian Variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        metrics["laplacian_variance"] = round(laplacian_var, 2)

        if laplacian_var < self.thresholds.min_laplacian_variance:
            reasons.append(
                f"Blur check failed: Laplacian variance {laplacian_var:.2f} is below minimum threshold "
                f"{self.thresholds.min_laplacian_variance:.1f} (document is blurry or out of focus)."
            )

        # 3. Brightness / Contrast Check
        mean_brightness = float(np.mean(gray))
        metrics["mean_brightness"] = round(mean_brightness, 2)

        if mean_brightness < self.thresholds.min_mean_brightness:
            reasons.append(
                f"Contrast check failed: image is severely underexposed/black (mean brightness {mean_brightness:.1f})."
            )

        # 4. Text / Stroke Density Check (Canny edge density)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges) / edges.size)
        metrics["edge_density"] = round(edge_density, 5)

        # If image is almost entirely white (>254.8) and has virtually no edges, it's blank/overexposed
        if mean_brightness > 254.8 and edge_density < self.thresholds.min_edge_density:
            reasons.append(
                f"Contrast check failed: image is severely overexposed/blank (mean brightness {mean_brightness:.1f} with no readable content)."
            )
        elif edge_density < self.thresholds.min_edge_density:
            reasons.append(
                f"Text density check failed: edge density {edge_density:.5f} is below minimum threshold "
                f"{self.thresholds.min_edge_density:.5f} (insufficient visible text or content)."
            )

        passed = len(reasons) == 0
        return QualityReport(passed=passed, failure_reasons=reasons, metrics=metrics)
