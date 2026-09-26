"""
Deskew and Orientation Normalizer for TrustExtract
Detects document skew angle using morphological text line contour orientation and Hough analysis,
and rotates the page image back to horizontal alignment (0 degrees) for reliable OCR bounding boxes.
"""

from typing import Tuple
import cv2
import numpy as np
from PIL import Image


class Deskewer:
    """Estimates and corrects rotation skew on document page images."""

    @staticmethod
    def estimate_skew_angle(pil_img: Image.Image) -> float:
        """
        Estimate skew angle in degrees (-45 to 45).
        Returns the angle by which the image is rotated relative to horizontal.
        A negative value indicates counter-clockwise tilt in image coordinates.
        """
        img_np = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Threshold inverted: text becomes 255 (white), background becomes 0 (black)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Morphological dilation: horizontally connect adjacent letters and words into continuous text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours of dilated text lines
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        angles = []

        for c in contours:
            area = cv2.contourArea(c)
            # Filter out tiny noise spots and massive page boundary boxes
            if 250 < area < (img_np.shape[0] * img_np.shape[1] * 0.7):
                rect = cv2.minAreaRect(c)
                (w, h) = rect[1]
                if w > 0 and h > 0:
                    aspect_ratio = max(w, h) / min(w, h)
                    # Text line strips typically have aspect ratio > 2.0
                    if aspect_ratio >= 2.0:
                        angle = rect[2]
                        if w < h:
                            angle = angle - 90 if angle > 0 else angle + 90
                        if -45 < angle < 45:
                            angles.append(angle)

        if not angles:
            # Fallback to HoughLinesP if contour analysis yielded no high-aspect lines
            lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=15)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line.flatten()
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if -45 < angle < 45:
                        angles.append(angle)

        if angles:
            # Median angle provides robust rejection of outlier lines/charts
            return float(np.median(angles))
        return 0.0

    @classmethod
    def correct_skew(cls, pil_img: Image.Image, min_threshold_degrees: float = 0.4) -> Tuple[Image.Image, float]:
        """
        Detect skew and rotate image so that text lines are horizontal.
        Returns:
            Tuple[Image.Image, float]: (deskewed_image, applied_correction_angle_degrees)
        """
        detected_angle = cls.estimate_skew_angle(pil_img)

        # If detected angle is negligible, do not resample the image (preserves maximum sharpness)
        if abs(detected_angle) < min_threshold_degrees:
            return pil_img, 0.0

        img_np = np.array(pil_img.convert("RGB"))
        (h, w) = img_np.shape[:2]
        center = (w // 2, h // 2)

        # In OpenCV:
        # A line sloping upwards has a negative detected angle in image coordinates.
        # Rotating with positive detected_angle rotates it back clockwise to 0 degrees horizontal.
        correction_angle = detected_angle
        M = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
        deskewed_np = cv2.warpAffine(
            img_np,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        deskewed_img = Image.fromarray(deskewed_np)
        return deskewed_img, correction_angle
