"""Encrypted storage for face templates.

Raw face images are never written to disk by this system. Only embeddings are
persisted, and only encrypted.

**Why embeddings still count as biometric data.** It is tempting to treat a
512-float vector as anonymous, because it does not look like a face. It is not
anonymous. Template-inversion research has repeatedly shown that a recognisable
face can be reconstructed from an embedding alone, and unlike a password a face
cannot be rotated after a breach. So the store is encrypted at rest with
Fernet (AES-128-CBC with an HMAC-SHA256 authentication tag), and the plaintext
never touches the filesystem.

**What this protects against, stated honestly.** Encryption at rest defends
against the realistic failure modes for a project like this: the directory
being committed to a public repository, ending up in a backup or a cloud sync,
or being read by another user of the same machine. It does **not** defend
against an attacker who already has the key. By default the key sits in
``data/faceguard.key`` next to the ciphertext, which means anyone who can read
the store can read the key. That default exists so the project runs out of the
box; it is not a production posture. Set ``FACEGUARD_KEY`` from a secrets
manager, an OS keyring or an HSM and the key never lands on disk at all.

**Integrity, not just secrecy.** Fernet tokens are authenticated, so a tampered
or truncated store fails to decrypt rather than silently yielding altered
templates. An attacker who can write to the file cannot substitute their own
embedding for yours without the key.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Sequence

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

KEY_ENVIRONMENT_VARIABLE = "FACEGUARD_KEY"
STORE_FORMAT_VERSION = 1


class StoreError(RuntimeError):
    """Raised when the template store cannot be read, written or decrypted."""


@dataclass
class Template:
    """One enrolled biometric template.

    ``source_digest`` is a SHA-256 of the source image, kept purely so a
    duplicate enrolment can be detected. It is a one-way digest: it cannot
    reconstruct the image, and the image itself is never stored.
    """

    identity: str
    embedding: np.ndarray
    quality: float
    created_at: str
    source_digest: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "embedding": [float(v) for v in self.embedding],
            "quality": float(self.quality),
            "created_at": self.created_at,
            "source_digest": self.source_digest,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Template":
        return cls(
            identity=data["identity"],
            embedding=np.asarray(data["embedding"], dtype=np.float32),
            quality=float(data.get("quality", 0.0)),
            created_at=data.get("created_at", ""),
            source_digest=data.get("source_digest", ""),
            note=data.get("note", ""),
        )


def generate_key() -> str:
    """Create a fresh Fernet key as a URL-safe base64 string."""
    return Fernet.generate_key().decode("ascii")


def load_key(key_file: Path, *, create: bool = True) -> bytes:
    """Resolve the encryption key.

    Order of preference: the ``FACEGUARD_KEY`` environment variable, then the
    key file, then -- only if ``create`` is set -- a newly generated key
    written to the key file.

    The environment variable is preferred because a key that never touches the
    filesystem cannot be committed, backed up or synced by accident.
    """
    from_environment = os.environ.get(KEY_ENVIRONMENT_VARIABLE)
    if from_environment:
        try:
            key = from_environment.strip().encode("ascii")
            Fernet(key)  # validates length and encoding
            return key
        except (ValueError, TypeError) as error:
            raise StoreError(
                f"{KEY_ENVIRONMENT_VARIABLE} is set but is not a valid Fernet key: "
                f"{error}. Generate one with: python -c "
                '"from faceguard.store import generate_key; print(generate_key())"'
            ) from error

    key_file = Path(key_file)
    if key_file.is_file():
        key = key_file.read_bytes().strip()
        try:
            Fernet(key)
        except (ValueError, TypeError) as error:
            raise StoreError(f"{key_file} does not contain a valid Fernet key") from error
        return key

    if not create:
        raise StoreError(
            f"No encryption key found. Set {KEY_ENVIRONMENT_VARIABLE} or create "
            f"{key_file}."
        )

    key_file.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    key_file.write_bytes(key)
    _restrict_permissions(key_file)
    warnings.warn(
        f"Generated a new encryption key at {key_file}. It sits alongside the "
        f"template store, so anyone who can read the store can read the key. "
        f"For anything beyond local development, set {KEY_ENVIRONMENT_VARIABLE} "
        f"from a secrets manager instead and delete this file.",
        RuntimeWarning,
        stacklevel=2,
    )
    return key


def _restrict_permissions(path: Path) -> None:
    """Tighten file permissions to owner-only where the OS supports it.

    On POSIX this is chmod 600. On Windows the call is accepted but only
    toggles the read-only bit, so it is best-effort there and the real
    protection is NTFS inheritance from the user profile directory.
    """
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:  # pragma: no cover - platform dependent
        pass


class TemplateStore:
    """Thread-safe, encrypted, atomically-written template store."""

    def __init__(self, store_path: Path, key_file: Path, *, create_key: bool = True):
        self.store_path = Path(store_path)
        self.key_file = Path(key_file)
        self._fernet = Fernet(load_key(self.key_file, create=create_key))
        self._templates: Dict[str, List[Template]] = {}
        self._lock = RLock()
        self.load()

    # -- persistence -------------------------------------------------------- #

    def load(self) -> None:
        """Read and decrypt the store. A missing file is an empty store."""
        with self._lock:
            self._templates = {}
            if not self.store_path.is_file():
                return
            ciphertext = self.store_path.read_bytes()
            if not ciphertext:
                return
            try:
                plaintext = self._fernet.decrypt(ciphertext)
            except InvalidToken as error:
                raise StoreError(
                    f"Could not decrypt {self.store_path}. Either the key is wrong "
                    "or the file has been tampered with -- Fernet tokens are "
                    "authenticated, so this check is meaningful."
                ) from error

            payload = json.loads(plaintext.decode("utf-8"))
            version = payload.get("version")
            if version != STORE_FORMAT_VERSION:
                raise StoreError(
                    f"Unsupported template store version {version!r}; "
                    f"this build reads version {STORE_FORMAT_VERSION}."
                )
            for identity, entries in payload.get("templates", {}).items():
                self._templates[identity] = [Template.from_dict(e) for e in entries]

    def save(self) -> None:
        """Encrypt and write the store atomically.

        Written to a temporary file in the same directory and then replaced, so
        an interrupted write cannot leave a half-written store that fails to
        decrypt and locks every enrolled user out.
        """
        with self._lock:
            payload = {
                "version": STORE_FORMAT_VERSION,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "templates": {
                    identity: [template.to_dict() for template in templates]
                    for identity, templates in self._templates.items()
                },
            }
            ciphertext = self._fernet.encrypt(
                json.dumps(payload).encode("utf-8")
            )
            self.store_path.parent.mkdir(parents=True, exist_ok=True)

            handle, temporary_path = tempfile.mkstemp(
                dir=str(self.store_path.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(handle, "wb") as file:
                    file.write(ciphertext)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary_path, self.store_path)
            except BaseException:
                Path(temporary_path).unlink(missing_ok=True)
                raise
            _restrict_permissions(self.store_path)

    # -- mutation ----------------------------------------------------------- #

    def add(self, template: Template, max_per_identity: int = 10) -> None:
        """Enrol one template, keeping the highest-quality samples per identity.

        When the cap is reached the *lowest quality* template is dropped rather
        than the oldest. Recognition accuracy depends on template quality, and
        silently discarding a good enrolment photo because a poor one arrived
        later would degrade the system over time.
        """
        with self._lock:
            entries = self._templates.setdefault(template.identity, [])
            entries.append(template)
            if len(entries) > max_per_identity:
                entries.sort(key=lambda t: t.quality, reverse=True)
                del entries[max_per_identity:]
            self.save()

    def remove_identity(self, identity: str) -> bool:
        """Delete every template for one person. Returns whether any existed."""
        with self._lock:
            existed = self._templates.pop(identity, None) is not None
            if existed:
                self.save()
            return existed

    # -- queries ------------------------------------------------------------ #

    def identities(self) -> List[str]:
        with self._lock:
            return sorted(self._templates)

    def templates_for(self, identity: str) -> List[Template]:
        with self._lock:
            return list(self._templates.get(identity, []))

    def all_templates(self) -> Dict[str, List[Template]]:
        with self._lock:
            return {k: list(v) for k, v in self._templates.items()}

    def has_digest(self, digest: str) -> bool:
        """Whether an image with this digest has already been enrolled."""
        with self._lock:
            return any(
                template.source_digest == digest
                for templates in self._templates.values()
                for template in templates
            )

    def __len__(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._templates.values())


def image_digest(image: np.ndarray) -> str:
    """SHA-256 of raw image bytes, for duplicate-enrolment detection.

    One-way by construction: this identifies a repeated file without retaining
    anything that could reconstruct the picture.
    """
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
