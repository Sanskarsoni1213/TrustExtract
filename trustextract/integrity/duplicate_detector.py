"""
Duplicate & Near-Duplicate Detection for TrustExtract (Loop 4)
Provides cryptographic exact-match detection (SHA-256) and robust
perceptual hash matching (pHash) to catch re-submissions across cropping,
rotation, and re-compression variations.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image


@dataclass
class DuplicateMatch:
    matched_document_id: str
    match_type: str  # "EXACT" or "NEAR_DUPLICATE"
    page_number: int
    hamming_distance: Optional[int]
    similarity_percentage: float
    description: str


@dataclass
class DuplicateCheckResult:
    is_duplicate: bool
    is_exact_match: bool
    is_near_match: bool
    matches: List[DuplicateMatch] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    sha256_hash: str = ""
    page_phashes: List[str] = field(default_factory=list)


def compute_sha256(data: bytes) -> str:
    """Compute standard SHA-256 hex digest for exact matching."""
    return hashlib.sha256(data).hexdigest()


def compute_image_phash(img: Image.Image, hash_size: int = 8) -> str:
    """
    Computes a 64-bit DCT-based Perceptual Hash (pHash) of an image.
    Resistant to JPEG compression, small rotations, brightness shifts, and minor crops.
    """
    # 1. Convert to grayscale and resize to 32x32
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    
    # 2. Compute 2D Discrete Cosine Transform (DCT)
    dct = cv2.dct(np.float32(resized))
    
    # 3. Extract top-left 8x8 low-frequency components (excluding DC term at [0,0])
    dct_low = dct[:hash_size, :hash_size]
    
    # 4. Compare against median of AC coefficients
    med = float(np.median(dct_low[1:, 1:]))
    bit_array = (dct_low > med).flatten()
    
    # 5. Pack 64 boolean bits into 16-hex-character string
    hex_str = "".join(f"{b:x}" for b in np.packbits(bit_array))
    return hex_str


def hamming_distance(hash1_hex: str, hash2_hex: str) -> int:
    """Calculate the Hamming distance (number of bit differences) between two 64-bit hex pHashes."""
    try:
        val1 = int(hash1_hex, 16)
        val2 = int(hash2_hex, 16)
        return bin(val1 ^ val2).count("1")
    except Exception:
        return 64


class DocumentDeduplicator:
    """
    Maintains an index of processed documents and performs multi-tier deduplication:
      - Tier 1: SHA-256 full byte exact-match guarantee.
      - Tier 2: Perceptual pHash near-match guarantee across visual variations.
    """

    def __init__(self, near_match_hamming_threshold: int = 8):
        # 8 bits out of 64 bits = 87.5% visual similarity threshold
        self.near_match_hamming_threshold = near_match_hamming_threshold
        # Registry: document_id -> {'sha256': str, 'phashes': List[str]}
        self._registry: Dict[str, Dict[str, Any]] = {}

    def check_and_register(
        self,
        document_id: str,
        file_bytes: bytes,
        page_images: List[Image.Image],
        auto_index: bool = True
    ) -> DuplicateCheckResult:
        """
        Check if document is an exact or near duplicate of any previously seen document,
        and optionally register it into the deduplication index.
        """
        sha256 = compute_sha256(file_bytes)
        page_phashes = [compute_image_phash(p) for p in page_images]

        matches: List[DuplicateMatch] = []
        is_exact = False
        is_near = False
        flags = []

        # Compare against all indexed documents
        for registered_id, reg_data in self._registry.items():
            if registered_id == document_id:
                continue

            # Check 1: Exact byte match
            if reg_data["sha256"] == sha256:
                is_exact = True
                matches.append(
                    DuplicateMatch(
                        matched_document_id=registered_id,
                        match_type="EXACT",
                        page_number=1,
                        hamming_distance=0,
                        similarity_percentage=100.0,
                        description=f"Exact binary byte match (SHA-256: {sha256[:12]}...)"
                    )
                )

            # Check 2: Perceptual near-duplicate match
            for p_idx, p_hash in enumerate(page_phashes, start=1):
                for reg_p_idx, reg_p_hash in enumerate(reg_data["phashes"], start=1):
                    h_dist = hamming_distance(p_hash, reg_p_hash)
                    sim_pct = round((64 - h_dist) / 64.0 * 100.0, 1)

                    if h_dist <= self.near_match_hamming_threshold:
                        is_near = True
                        if not is_exact:
                            matches.append(
                                DuplicateMatch(
                                    matched_document_id=registered_id,
                                    match_type="NEAR_DUPLICATE",
                                    page_number=p_idx,
                                    hamming_distance=h_dist,
                                    similarity_percentage=sim_pct,
                                    description=(
                                        f"Perceptual near-duplicate (pHash dist={h_dist}/64, "
                                        f"{sim_pct}% visual match with doc '{registered_id}' p.{reg_p_idx})"
                                    )
                                )
                            )

        if is_exact:
            flags.append("DUPLICATE_EXACT_MATCH")
        elif is_near:
            flags.append("DUPLICATE_NEAR_MATCH")

        # Register for future comparisons
        if auto_index:
            self._registry[document_id] = {
                "sha256": sha256,
                "phashes": page_phashes
            }

        return DuplicateCheckResult(
            is_duplicate=(is_exact or is_near),
            is_exact_match=is_exact,
            is_near_match=is_near,
            matches=matches,
            flags=flags,
            sha256_hash=sha256,
            page_phashes=page_phashes
        )

    def clear(self) -> None:
        """Reset index for testing."""
        self._registry.clear()
