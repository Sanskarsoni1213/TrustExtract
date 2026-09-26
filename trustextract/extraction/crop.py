"""
Image Region Crop & Provenance Manager for TrustExtract
Extracts image crops for field values, encrypts them at rest, and provides secure crop references.
"""

import hashlib
import os
from io import BytesIO
from typing import List, Tuple
from PIL import Image
from trustextract.security.crypto import KMSEncryptionInterface, default_kms_service


class CropManager:
    """Safely extracts, pads, and encrypts visual crops for extracted fields."""

    def __init__(self, kms_service: KMSEncryptionInterface = default_kms_service, storage_dir: str = "encrypted_store"):
        self.kms_service = kms_service
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def generate_crop_ref(
        self,
        page_image: Image.Image,
        bbox: List[float],
        page_number: int,
        document_id: str,
        padding: int = 4
    ) -> Tuple[str, List[float]]:
        """
        Crop the specified bbox region from page_image, encrypt the crop at rest,
        and return a secure crop reference identifier.
        """
        w_img, h_img = page_image.size
        x, y, w, h = bbox

        # Apply padding while clamping to image dimensions
        x0 = max(0, int(x - padding))
        y0 = max(0, int(y - padding))
        x1 = min(w_img, int(x + w + padding))
        y1 = min(h_img, int(y + h + padding))

        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, min(10, w_img), min(10, h_img)

        crop_img = page_image.crop((x0, y0, x1, y1))

        # Serialize crop to PNG bytes
        buf = BytesIO()
        crop_img.save(buf, format="PNG")
        crop_bytes = buf.getvalue()

        crop_hash = hashlib.sha256(crop_bytes).hexdigest()[:12]
        crop_id = f"crop_{document_id}_p{page_number}_{int(x)}_{int(y)}_{crop_hash}"

        # Encrypt crop at rest
        encrypted_crop = self.kms_service.encrypt(
            crop_bytes,
            context={"doc_id": document_id, "crop_id": crop_id, "page": str(page_number)}
        )

        enc_path = os.path.join(self.storage_dir, f"{crop_id}.enc")
        with open(enc_path, "wb") as f:
            f.write(encrypted_crop)

        normalized_bbox = [float(x), float(y), float(w), float(h)]
        crop_ref = f"crop://{crop_id}"
        return crop_ref, normalized_bbox
