"""
Format Normalizer for TrustExtract
Ingests PDF, JPG, PNG, TIFF, HEIC, and DOCX documents and normalizes them into
standardized page images with digital text layers where present.
"""

from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional
import os
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont
import docx
import pillow_heif

import json
from trustextract.ingestion.deskew import Deskewer

# Register HEIF opener with Pillow for HEIC support
pillow_heif.register_heif_opener()


@dataclass
class NormalizedPage:
    page_number: int
    image: Image.Image
    text_layer: Optional[str] = None
    deskew_angle: float = 0.0


@dataclass
class NormalizedDocument:
    document_id: str
    original_format: str
    pages: List[NormalizedPage] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class DocumentNormalizer:
    """Normalizes arbitrary document formats to page images and text layers."""

    @staticmethod
    def normalize_pdf(file_path: str, document_id: str) -> NormalizedDocument:
        """Render PDF pages to high-resolution images, deskew raster pages, and extract digital text layers."""
        doc = fitz.open(file_path)
        if doc.is_encrypted or doc.needs_pass:
            doc.close()
            raise PermissionError("Document is encrypted or password-protected and cannot be opened without credentials.")

        pages = []
        doc_meta = doc.metadata or {}
        meta = {
            "title": doc_meta.get("title", "") or "",
            "author": doc_meta.get("author", "") or "",
            "creator": doc_meta.get("creator", "") or "",
            "producer": doc_meta.get("producer", "") or "",
            "creation_date": doc_meta.get("creationDate", "") or "",
            "mod_date": doc_meta.get("modDate", "") or "",
            "page_count": str(len(doc))
        }

        for idx, page in enumerate(doc):
            text = page.get_text("text").strip()
            # Render at 200 DPI for high OCR and vision fidelity
            pix = page.get_pixmap(dpi=200)
            raw_img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
            # Apply deskewing to ensure bounding boxes map to aligned horizontal coordinate space
            deskewed_img, applied_angle = Deskewer.correct_skew(raw_img)
            pages.append(NormalizedPage(
                page_number=idx + 1,
                image=deskewed_img,
                text_layer=text or None,
                deskew_angle=round(applied_angle, 2)
            ))

        doc.close()
        meta["deskew_applied"] = str(any(abs(p.deskew_angle) > 0 for p in pages))
        meta["deskew_angles"] = json.dumps([p.deskew_angle for p in pages])
        return NormalizedDocument(document_id=document_id, original_format="pdf", pages=pages, metadata=meta)

    @staticmethod
    def normalize_image(file_path: str, document_id: str, fmt: str) -> NormalizedDocument:
        """Normalize and deskew JPG, PNG, TIFF, or HEIC images."""
        with Image.open(file_path) as pil_img:
            pages = []
            page_idx = 1
            while True:
                # Copy image into memory and convert to RGB
                page_img = pil_img.copy().convert("RGB")
                deskewed_img, applied_angle = Deskewer.correct_skew(page_img)
                pages.append(NormalizedPage(
                    page_number=page_idx,
                    image=deskewed_img,
                    text_layer=None,
                    deskew_angle=round(applied_angle, 2)
                ))
                page_idx += 1
                try:
                    pil_img.seek(pil_img.tell() + 1)
                except EOFError:
                    break

        meta = {
            "format": fmt,
            "page_count": str(len(pages)),
            "deskew_applied": str(any(abs(p.deskew_angle) > 0 for p in pages)),
            "deskew_angles": json.dumps([p.deskew_angle for p in pages])
        }
        return NormalizedDocument(
            document_id=document_id,
            original_format=fmt,
            pages=pages,
            metadata=meta
        )

    @staticmethod
    def normalize_docx(file_path: str, document_id: str) -> NormalizedDocument:
        """Extract text from DOCX and render formatted page representation."""
        doc = docx.Document(file_path)
        paragraphs_text = [p.text for p in doc.paragraphs if p.text.strip()]

        # Extract table text
        table_lines = []
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                table_lines.append(" | ".join(row_cells))

        full_text = "\n".join(paragraphs_text + (["--- Tables ---"] + table_lines if table_lines else []))

        # Render onto a standard page image (1700x2200, representing ~200 DPI Letter paper)
        page_width, page_height = 1700, 2200
        canvas = Image.new("RGB", (page_width, page_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Attempt to use default font
        try:
            font = ImageFont.load_default(size=24)
        except TypeError:
            font = ImageFont.load_default()

        # Render text lines onto canvas
        margin = 100
        y = margin
        for line in full_text.split("\n"):
            if y > page_height - margin:
                break
            # Wrap long lines if necessary
            draw.text((margin, y), line[:100], fill=(20, 20, 20), font=font)
            y += 36

        pages = [NormalizedPage(page_number=1, image=canvas, text_layer=full_text or None)]
        return NormalizedDocument(
            document_id=document_id,
            original_format="docx",
            pages=pages,
            metadata={"paragraph_count": str(len(paragraphs_text)), "table_count": str(len(doc.tables))}
        )

    @classmethod
    def normalize(cls, file_path: str, detected_format: str, document_id: str) -> NormalizedDocument:
        """Route to appropriate normalization handler based on detected format."""
        if detected_format == "pdf":
            return cls.normalize_pdf(file_path, document_id)
        elif detected_format in ["png", "jpeg", "tiff", "heic"]:
            return cls.normalize_image(file_path, document_id, detected_format)
        elif detected_format == "docx":
            return cls.normalize_docx(file_path, document_id)
        else:
            raise ValueError(f"Unsupported format for normalization: {detected_format}")
