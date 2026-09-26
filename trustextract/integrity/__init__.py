"""
TrustExtract Integrity & Tamper Forensics Package (Loop 4)
"""

from trustextract.integrity.ela import ErrorLevelAnalyzer, ELAResult, ELARegion
from trustextract.integrity.pdf_forensics import PDFMetadataForensicDetector, PDFMetadataResult
from trustextract.integrity.font_analysis import FontConsistencyDetector, FontConsistencyResult
from trustextract.integrity.duplicate_detector import (
    DocumentDeduplicator,
    DuplicateCheckResult,
    compute_sha256,
    compute_image_phash,
    hamming_distance,
)
from trustextract.integrity.ai_detector import AIGenerationDetector, AISynthesisResult
from trustextract.integrity.pipeline import IntegrityPipeline, IntegrityReport

__all__ = [
    "ErrorLevelAnalyzer",
    "ELAResult",
    "ELARegion",
    "PDFMetadataForensicDetector",
    "PDFMetadataResult",
    "FontConsistencyDetector",
    "FontConsistencyResult",
    "DocumentDeduplicator",
    "DuplicateCheckResult",
    "compute_sha256",
    "compute_image_phash",
    "hamming_distance",
    "AIGenerationDetector",
    "AISynthesisResult",
    "IntegrityPipeline",
    "IntegrityReport",
]
