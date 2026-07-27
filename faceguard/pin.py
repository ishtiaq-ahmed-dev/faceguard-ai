"""PIN fallback authentication.

Biometrics fail for reasons that have nothing to do with security. A user
breaks their glasses, the room is lit from behind, the camera driver dies, the
liveness classifier has a bad day with an unusual skin tone or an unusual
webcam. A system whose only credential is a face locks those people out
permanently, and the usual response is that somebody disables the security
control entirely.

So there is a second factor, and it is deliberately a *different kind* of
secret: something you know rather than something you are. A PIN can be changed
after a compromise, which a face cannot.

**Storage.** PINs are hashed with scrypt -- memory-hard, so a stolen database
cannot be attacked with GPUs anywhere near as effectively as it could with a
fast hash. Each PIN gets its own random salt, so identical PINs across users do
not produce identical hashes and one cracked entry does not reveal others.
Comparison uses ``hmac.compare_digest`` to avoid leaking the answer through
timing.

**Honest limitation.** A six-digit PIN has about 20 bits of entropy, which is
trivially brute-forceable offline if the hash file leaks, and would fall in
seconds online without rate limiting. The scrypt parameters below raise the
offline cost, and the shared lockout counter in :mod:`faceguard.session` -- PIN
failures and biometric failures increment the *same* counter -- raises the
online cost. Neither makes a short PIN strong. It is a fallback, and the README
does not claim otherwise.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Optional

# scrypt cost parameters. n=2**15 with r=8 needs roughly 32 MB per hash, which
# is imperceptible for a single interactive login and expensive to parallelise
# across a stolen database.
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LENGTH = 32
SALT_BYTES = 16


class PinError(ValueError):
    """Raised when a PIN does not meet policy or the store is unreadable."""


def _hash_pin(pin: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        pin.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=KEY_LENGTH,
        maxmem=64 * 1024 * 1024,
    )


class PinStore:
    """Salted, scrypt-hashed PIN store. Plaintext PINs are never persisted."""

    def __init__(self, path: Path, min_length: int = 6) -> None:
        self.path = Path(path)
        self.min_length = min_length
        self._entries: Dict[str, dict] = {}
        self._lock = RLock()
        self.load()

    def load(self) -> None:
        with self._lock:
            self._entries = {}
            if not self.path.is_file():
                return
            try:
                self._entries = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as error:
                raise PinError(f"Could not read PIN store {self.path}: {error}") from error

    def save(self) -> None:
        """Atomic write, so an interrupted save cannot corrupt the store."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as file:
                    json.dump(self._entries, file, indent=2)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, self.path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
            try:
                os.chmod(self.path, 0o600)
            except OSError:  # pragma: no cover - platform dependent
                pass

    def validate_policy(self, pin: str) -> None:
        """Reject PINs that are too short or trivially guessable.

        Sequences and repeats are excluded because they are the first thing any
        attacker tries, and a length check alone happily accepts '123456'.
        """
        if not pin.isdigit():
            raise PinError("PIN must contain digits only.")
        if len(pin) < self.min_length:
            raise PinError(f"PIN must be at least {self.min_length} digits.")
        if len(set(pin)) == 1:
            raise PinError("PIN must not be a single repeated digit.")
        digits = [int(d) for d in pin]
        deltas = {b - a for a, b in zip(digits, digits[1:])}
        if deltas in ({1}, {-1}):
            raise PinError("PIN must not be a simple ascending or descending run.")

    def set_pin(self, identity: str, pin: str) -> None:
        """Create or replace an identity's PIN."""
        self.validate_policy(pin)
        salt = secrets.token_bytes(SALT_BYTES)
        with self._lock:
            self._entries[identity] = {
                "salt": base64.b64encode(salt).decode("ascii"),
                "hash": base64.b64encode(_hash_pin(pin, salt)).decode("ascii"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "scrypt": {"n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P},
            }
            self.save()

    def verify(self, identity: str, pin: str) -> bool:
        """Check a PIN in constant time with respect to the stored value.

        An unknown identity still performs a full scrypt hash against a dummy
        salt before returning False. Skipping the work would make "no such
        user" measurably faster than "wrong PIN" and turn this into an identity
        enumeration oracle.
        """
        with self._lock:
            entry = self._entries.get(identity)
        if entry is None:
            _hash_pin(pin or "0", secrets.token_bytes(SALT_BYTES))
            return False
        try:
            salt = base64.b64decode(entry["salt"])
            expected = base64.b64decode(entry["hash"])
        except (KeyError, ValueError):
            return False
        return hmac.compare_digest(_hash_pin(pin, salt), expected)

    def has_pin(self, identity: str) -> bool:
        with self._lock:
            return identity in self._entries

    def remove(self, identity: str) -> bool:
        with self._lock:
            existed = self._entries.pop(identity, None) is not None
            if existed:
                self.save()
            return existed

    def identities(self) -> list[str]:
        with self._lock:
            return sorted(self._entries)
