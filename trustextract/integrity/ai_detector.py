"""
AI Generation & Synthetic Origin Detection for TrustExtract (Loop 4)
Analyzes metadata, EXIF profiles, generator watermarks/tags, and capture
claims to identify AI-synthesized or origin-inconsistent documents.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from PIL import Image, ExifTags


@dataclass
class AISynthesisResult:
    is_suspicious: bool
    ai_tool_detected: Optional[str]
    missing_camera_metadata: bool
    claimed_capture_method: Optional[str]
    confidence: float
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


KNOWN_AI_GENERATOR_KEYWORDS = [
    "midjourney", "dall-e", "dalle", "stable diffusion", "stablediffusion",
    "comfyui", "automatic1111", "novelai", "deepai", "adobe firefly",
    "bing image creator", "flux.1", "runway", "pika", "sora"
]


class AIGenerationDetector:
    """
    Evaluates metadata, EXIF tags, software strings, and capture claims
    to detect AI-generated documents or synthetic scans posing as real photos.
    """

    def analyze_image_and_metadata(
        self,
        image: Image.Image,
        doc_metadata: Optional[Dict[str, str]] = None,
        claimed_capture_method: Optional[str] = None
    ) -> AISynthesisResult:
        """
        Args:
            image: PIL Image of the document page.
            doc_metadata: Normalized document metadata (creator, producer, author).
            claimed_capture_method: "camera_photo", "flatbed_scanner", or "digital_native".
        """
        flags = []
        details = {}
        ai_tool = None
        doc_meta = doc_metadata or {}

        # 1. Search document metadata strings for AI signatures
        meta_str = " ".join(str(v) for v in doc_meta.values()).lower()
        for kw in KNOWN_AI_GENERATOR_KEYWORDS:
            if kw in meta_str:
                ai_tool = kw
                flags.append("AI_GENERATION_SUSPECTED")
                details["generator_signature"] = f"Found generative AI signature in metadata: {kw!r}"
                break

        # 2. Extract image EXIF metadata
        exif_dict = {}
        try:
            exif_raw = image.getexif()
            if exif_raw:
                for tag_id, val in exif_raw.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    exif_dict[tag_name] = str(val)
        except Exception:
            pass

        exif_str = " ".join(f"{k}:{v}" for k, v in exif_dict.items()).lower()
        if not ai_tool:
            for kw in KNOWN_AI_GENERATOR_KEYWORDS:
                if kw in exif_str:
                    ai_tool = kw
                    if "AI_GENERATION_SUSPECTED" not in flags:
                        flags.append("AI_GENERATION_SUSPECTED")
                    details["exif_generator_signature"] = f"Found generative AI tag in EXIF: {kw!r}"
                    break

        # 3. Capture Origin Consistency Check
        # If user/API claimed the document is a "camera_photo" or "phone_scan"
        # but the image completely lacks camera EXIF data (no Make, Model, DateTimeOriginal)
        missing_camera = False
        if claimed_capture_method in ("camera_photo", "mobile_photo", "phone_photo"):
            has_camera_make = "Make" in exif_dict or "Model" in exif_dict
            has_camera_date = "DateTimeOriginal" in exif_dict or "DateTime" in exif_dict
            if not (has_camera_make or has_camera_date):
                missing_camera = True
                flags.append("METADATA_INCONSISTENT_CAPTURE_ORIGIN")
                details["camera_claim_mismatch"] = (
                    f"Document claimed capture method is {claimed_capture_method!r} "
                    f"but contains zero camera/hardware EXIF metadata (suspiciously clean raster)"
                )

        is_suspicious = len(flags) > 0
        confidence = 0.95 if ai_tool else (0.80 if missing_camera else 0.0)

        return AISynthesisResult(
            is_suspicious=is_suspicious,
            ai_tool_detected=ai_tool,
            missing_camera_metadata=missing_camera,
            claimed_capture_method=claimed_capture_method,
            confidence=confidence,
            flags=flags,
            details=details
        )
