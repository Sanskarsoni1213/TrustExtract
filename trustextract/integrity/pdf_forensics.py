"""
PDF Metadata Forensics for TrustExtract (Loop 4)
Inspects PDF creation vs modification dates, creator/producer chains,
incremental update histories, and editor tool signatures.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from typing import Any, Dict, List, Optional
import pymupdf as fitz


@dataclass
class PDFMetadataResult:
    has_anomaly: bool
    creation_date: Optional[str]
    mod_date: Optional[str]
    creator: str
    producer: str
    time_delta_seconds: Optional[float]
    incremental_updates_count: int
    suspicious_tool_detected: Optional[str]
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


SUSPICIOUS_PRODUCERS = [
    "photoshop", "gimp", "canva", "pdfill", "ilovepdf",
    "sejda", "inkscape", "foxit phantom", "nitro pro",
    "pdf-xchange editor", "smallpdf", "pdfescape"
]


def _parse_pdf_date(date_str: str) -> Optional[datetime]:
    """Parse PDF format date string e.g. 'D:20240315103000Z' or 'D:20240315103000+00'00'."""
    if not date_str:
        return None
    cleaned = date_str.strip()
    if cleaned.startswith("D:"):
        cleaned = cleaned[2:]
    
    # Remove timezone suffix for basic datetime comparison
    cleaned = re.sub(r"[Zz\+\-].*$", "", cleaned)
    # Match YYYYMMDDHHMMSS
    match = re.match(r"^(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", cleaned)
    if not match:
        return None
    
    parts = match.groups()
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    hour = int(parts[3] or 0)
    minute = int(parts[4] or 0)
    second = int(parts[5] or 0)
    
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


class PDFMetadataForensicDetector:
    """
    Forensically audits PDF metadata, revision chains, and software footprints.
    Flags PDFs whose modification timeline or tool history indicates post-generation alteration.
    """

    def __init__(self, suspicious_time_delta_hours: float = 2.0):
        self.suspicious_time_delta_hours = suspicious_time_delta_hours

    def analyze_file(self, file_path: str) -> PDFMetadataResult:
        """Analyze PDF metadata directly from a file path."""
        if not file_path.lower().endswith(".pdf"):
            return PDFMetadataResult(
                has_anomaly=False,
                creation_date=None,
                mod_date=None,
                creator="",
                producer="",
                time_delta_seconds=None,
                incremental_updates_count=0,
                suspicious_tool_detected=None,
                flags=[],
                details={"reason": "Not a PDF document"}
            )

        try:
            doc = fitz.open(file_path)
            meta = doc.metadata or {}
            xref_len = doc.xref_length()
            doc.close()
        except Exception as e:
            return PDFMetadataResult(
                has_anomaly=True,
                creation_date=None,
                mod_date=None,
                creator="",
                producer="",
                time_delta_seconds=None,
                incremental_updates_count=0,
                suspicious_tool_detected=None,
                flags=["METADATA_SUSPICIOUS_EDIT_HISTORY"],
                details={"error": f"Failed to parse PDF metadata: {e}"}
            )

        return self.analyze_metadata(meta, xref_length=xref_len)

    def analyze_metadata(self, meta: Dict[str, str], xref_length: int = 1) -> PDFMetadataResult:
        """
        Analyze extracted metadata dict.
        """
        creation_raw = meta.get("creationDate") or meta.get("creation_date") or ""
        mod_raw = meta.get("modDate") or meta.get("mod_date") or ""
        creator = meta.get("creator") or ""
        producer = meta.get("producer") or ""

        dt_create = _parse_pdf_date(creation_raw)
        dt_mod = _parse_pdf_date(mod_raw)

        flags = []
        details = {}
        suspicious_tool = None
        delta_sec = None

        # 1. Check for suspicious editing software
        combined_tools = f"{creator} {producer}".lower()
        for s_tool in SUSPICIOUS_PRODUCERS:
            if s_tool in combined_tools:
                suspicious_tool = s_tool
                flags.append("METADATA_SUSPICIOUS_EDIT_HISTORY")
                details["suspicious_tool"] = f"Identified document editor tool in metadata: {s_tool}"
                break

        # 2. Check modification date divergence from creation date
        if dt_create and dt_mod:
            delta = dt_mod - dt_create
            delta_sec = delta.total_seconds()
            if delta_sec > (self.suspicious_time_delta_hours * 3600):
                # Document was modified significantly after creation
                if "METADATA_SUSPICIOUS_EDIT_HISTORY" not in flags:
                    flags.append("METADATA_SUSPICIOUS_EDIT_HISTORY")
                details["time_divergence"] = (
                    f"Modification date ({dt_mod}) is {delta_sec / 3600:.1f} hours "
                    f"after creation date ({dt_create})"
                )

        # 3. Discrepancy between automated creator and manual editing producer
        if creator and producer:
            if "report" in creator.lower() and ("acrobat" in producer.lower() or "editor" in producer.lower()):
                if "METADATA_SUSPICIOUS_EDIT_HISTORY" not in flags:
                    flags.append("METADATA_SUSPICIOUS_EDIT_HISTORY")
                details["creator_producer_mismatch"] = f"Created by {creator!r} but modified/produced by {producer!r}"

        has_anomaly = len(flags) > 0

        return PDFMetadataResult(
            has_anomaly=has_anomaly,
            creation_date=str(dt_create) if dt_create else creation_raw or None,
            mod_date=str(dt_mod) if dt_mod else mod_raw or None,
            creator=creator,
            producer=producer,
            time_delta_seconds=delta_sec,
            incremental_updates_count=xref_length,
            suspicious_tool_detected=suspicious_tool,
            flags=flags,
            details=details
        )
