"""Structured audit logging.

Rejections are logged with exactly the same detail as acceptances. That is the
entire point: a log that records only successful logins tells you nothing about
whether anyone is attacking the system. The interesting security signal is the
shape of the failures -- twenty ``TEXTURE_SPOOF_SUSPECTED`` rejections from one
client in two minutes is somebody holding a phone up to the camera, and it is
invisible in a success-only log.

Records are newline-delimited JSON so the file can be tailed, grepped and fed
to anything that reads JSONL without a parser being written first.

**Privacy.** These records contain identity names and per-attempt biometric
scores, so the log is personal data and is excluded from version control. It
never contains images or embeddings. Client identifiers are hashed before they
are written, so the log can be shared for debugging without exposing the IP
addresses or session tokens it was keyed on.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .schemas import VerificationResult


def hash_client(client_id: str, salt: str = "") -> str:
    """One-way, truncated digest of a client identifier.

    Truncated to 16 hex characters: enough to correlate attempts from the same
    client within a log, far too little to be worth attacking, and it keeps
    lines readable.
    """
    salt = salt or os.environ.get("FACEGUARD_AUDIT_SALT", "faceguard")
    digest = hashlib.sha256(f"{salt}:{client_id}".encode("utf-8")).hexdigest()
    return digest[:16]


class AuditLog:
    """Append-only JSONL audit log, safe for concurrent writers in-process."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _write(self, record: Dict[str, Any]) -> None:
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Opened per write and flushed immediately: an audit record that is
            # still sitting in a buffer when the process is killed is an audit
            # record that does not exist.
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def record_verification(
        self,
        result: VerificationResult,
        client_id: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log one verification attempt, accepted or rejected."""
        signals = {}
        if result.liveness:
            signals = {
                name: {
                    "status": signal.status.value,
                    "score": round(float(signal.score), 4),
                    "reason": signal.reason.value if signal.reason else None,
                }
                for name, signal in result.liveness.signals.items()
            }
        record: Dict[str, Any] = {
            "event": "verification",
            "session_id": result.session_id,
            "client": hash_client(client_id) if client_id else None,
            "accepted": result.accepted,
            "reason": result.reason.value,
            "identity": result.identity,
            "auth_method": result.auth_method,
            "recognition_confidence": round(float(result.recognition_confidence), 4),
            "detection_confidence": round(float(result.detection_confidence), 4),
            "liveness_passed": result.liveness_passed,
            "liveness_score": round(float(result.liveness_score), 4),
            "liveness_signals": signals,
            "degraded": result.degraded,
            "frames_used": result.frames_used,
            "elapsed_ms": round(float(result.elapsed_ms), 1),
        }
        if extra:
            record.update(extra)
        self._write(record)

    def record_event(self, event: str, **fields: Any) -> None:
        """Log a non-verification event: enrolment, lockout, configuration change."""
        record = {"event": event}
        if "client_id" in fields:
            fields["client"] = hash_client(str(fields.pop("client_id")))
        record.update(fields)
        self._write(record)

    def read(self) -> Iterator[Dict[str, Any]]:
        """Iterate parsed records. Malformed lines are skipped, not fatal."""
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
