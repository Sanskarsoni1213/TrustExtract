"""Extraction subpackage for TrustExtract."""
from trustextract.extraction.path_a_ocr import PathAOCRExtractor, RawFieldExtraction
from trustextract.extraction.path_b_vision import (
    VisionExtractorInterface,
    ProductionVisionExtractor,
    GeminiVisionExtractor,
    OpenAIVisionExtractor,
    AnthropicVisionExtractor,
    SimulatedOfflineVisionExtractor,
    IndependentVisionModel,
)
from trustextract.extraction.table import TableStructureExtractor
from trustextract.extraction.crop import CropManager
from trustextract.extraction.pipeline import DualPathExtractor
from trustextract.extraction.validators import DocumentValidator, DocumentValidationReport
from trustextract.extraction.confidence import ConfidencePipeline, ConfidenceFuser, SelfConsistencySampler, DocumentEscalationEngine

__all__ = [
    "PathAOCRExtractor",
    "RawFieldExtraction",
    "VisionExtractorInterface",
    "ProductionVisionExtractor",
    "GeminiVisionExtractor",
    "OpenAIVisionExtractor",
    "AnthropicVisionExtractor",
    "SimulatedOfflineVisionExtractor",
    "IndependentVisionModel",
    "TableStructureExtractor",
    "CropManager",
    "DualPathExtractor",
    "DocumentValidator",
    "DocumentValidationReport",
    "ConfidencePipeline",
    "ConfidenceFuser",
    "SelfConsistencySampler",
]
