"""
Isolated Untrusted Input Sandbox & Parser Boundary for TrustExtract
Guards against malicious payloads, zip bombs, decompression bombs, and malformed files.
"""

import os
import zipfile
from typing import Tuple, Optional
from PIL import Image

# Enforce PIL decompression bomb limit
Image.MAX_IMAGE_PIXELS = 50_000_000

# Strict maximum file size (50 MB)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

# Maximum allowed zip uncompressed size (200 MB) and max compression ratio (100:1)
MAX_ZIP_EXPANDED_BYTES = 200 * 1024 * 1024
MAX_ZIP_RATIO = 100


class SecurityValidationError(Exception):
    """Raised when an untrusted input fails security inspection."""
    pass


class SafeParserBoundary:
    """Security boundary enforcing strict validation on untrusted input files."""

    @staticmethod
    def validate_file_size(file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file does not exist: {file_path}")
        size = os.path.getsize(file_path)
        if size == 0:
            raise SecurityValidationError("File is empty (0 bytes).")
        if size > MAX_FILE_SIZE_BYTES:
            raise SecurityValidationError(
                f"File size {size} bytes exceeds maximum allowed limit of {MAX_FILE_SIZE_BYTES} bytes."
            )

    @staticmethod
    def detect_format_magic_bytes(header: bytes) -> Optional[str]:
        """Detect file format strictly by magic signature bytes, never relying on untrusted extensions."""
        if header.startswith(b"%PDF-"):
            return "pdf"
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if header.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
            return "tiff"
        if len(header) >= 12 and header[4:8] == b"ftyp" and header[8:12] in [b"heic", b"mif1", b"msf1", b"heix", b"hevc"]:
            return "heic"
        if header.startswith(b"PK\x03\x04"):
            return "docx_or_zip"
        return None

    @staticmethod
    def inspect_untrusted_input(file_path: str) -> str:
        """
        Thoroughly inspect an untrusted document before opening with parsers.
        Returns the validated canonical format name.
        """
        SafeParserBoundary.validate_file_size(file_path)

        with open(file_path, "rb") as f:
            header = f.read(64)

        detected_type = SafeParserBoundary.detect_format_magic_bytes(header)
        if not detected_type:
            raise SecurityValidationError(
                "Unsupported or unrecognized file format: magic bytes did not match any supported document type."
            )

        # Additional inspection for docx / zip archives
        if detected_type == "docx_or_zip":
            SafeParserBoundary._inspect_zip_archive(file_path)
            return "docx"

        return detected_type

    @staticmethod
    def _inspect_zip_archive(file_path: str) -> None:
        """Inspect zip/docx archives against zip-bombs and directory traversal."""
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                total_uncompressed = 0
                compressed_total = os.path.getsize(file_path)

                has_word_doc = False
                for info in zf.infolist():
                    # Guard directory traversal
                    if ".." in info.filename or info.filename.startswith("/"):
                        raise SecurityValidationError("Malicious zip path traversal detected in DOCX archive.")

                    total_uncompressed += info.file_size
                    if total_uncompressed > MAX_ZIP_EXPANDED_BYTES:
                        raise SecurityValidationError(
                            f"Zip bomb detected: uncompressed size exceeds {MAX_ZIP_EXPANDED_BYTES} bytes."
                        )

                    if info.filename.lower() in ["word/document.xml", "[content_types].xml"]:
                        has_word_doc = True

                if compressed_total > 0 and (total_uncompressed / compressed_total) > MAX_ZIP_RATIO:
                    raise SecurityValidationError(
                        f"Zip bomb detected: compression ratio {total_uncompressed / compressed_total:.1f} exceeds safety threshold."
                    )

                if not has_word_doc:
                    raise SecurityValidationError("Archive is a ZIP file but lacks standard DOCX XML structure.")

        except zipfile.BadZipFile as e:
            raise SecurityValidationError(f"Corrupted or invalid DOCX archive: {str(e)}")
