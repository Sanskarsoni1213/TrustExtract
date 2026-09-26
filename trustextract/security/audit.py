"""
Security Audit Logger for TrustExtract
Ensures strict zero-PII logging, actor tracking, and ISO timestamps on every action.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class RedactingAuditFormatter(logging.Formatter):
    """Logging formatter that strictly prevents any plaintext PII or document text from being written."""

    SENSITIVE_KEYS = {
        "text", "content", "raw_content", "crop", "image", "value", "extracted_text",
        "ssn", "dob", "name", "address", "phone", "account_number", "tax_id"
    }

    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            sanitized = self._sanitize_dict(record.msg)
            record.msg = json.dumps(sanitized)
        return super().format(record)

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(sens in k_lower for sens in self.SENSITIVE_KEYS):
                # Hash or mask
                if isinstance(v, (str, bytes)):
                    val_bytes = v.encode() if isinstance(v, str) else v
                    sanitized[k] = f"[REDACTED:sha256={hashlib.sha256(val_bytes).hexdigest()[:12]}]"
                else:
                    sanitized[k] = "[REDACTED]"
            elif isinstance(v, dict):
                sanitized[k] = self._sanitize_dict(v)
            elif isinstance(v, list):
                sanitized[k] = [
                    self._sanitize_dict(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                sanitized[k] = v
        return sanitized


class AuditLogger:
    """Audit logger tracking system actions, actors, and document hashes with zero PII exposure."""

    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger("trustextract.audit")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        formatter = RedactingAuditFormatter(
            fmt='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "audit_record": %(message)s}',
            datefmt="%Y-%m-%dT%H:%M:%SZ"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_event(
        self,
        action: str,
        actor_id: str,
        document_id: str,
        details: Optional[Dict[str, Any]] = None,
        level: int = logging.INFO
    ) -> None:
        """Log an audited event with UTC timestamp, actor ID, and non-PII details."""
        now = datetime.now(timezone.utc).isoformat()
        formatter = RedactingAuditFormatter()
        clean_details = formatter._sanitize_dict(details or {})
        payload = {
            "timestamp": now,
            "actor_id": actor_id,
            "document_id": document_id,
            "action": action,
            "details": clean_details
        }
        self.logger.log(level, payload)


# Global default audit logger
audit_logger = AuditLogger()
