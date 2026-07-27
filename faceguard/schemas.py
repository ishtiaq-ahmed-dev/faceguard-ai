"""Structured result types.

Every public entry point in FaceGuard returns one of these dataclasses, and
every rejection carries a machine-readable :class:`ReasonCode`. Nothing returns
a bare boolean or a loose string: a security system that cannot explain *why* it
said no is not auditable, and an operator debugging a demo needs the difference
between "I never saw a blink" and "the surface texture looked like paper".

Detection confidence and recognition confidence are deliberately separate
fields throughout. "Is there a face here" and "is this the right person" are
different questions with different failure modes, and blending them into one
number hides which of the two actually failed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import numpy as np


class ReasonCode(str, Enum):
    """Why the system reached the verdict it did.

    Values are stable strings so they can be logged, aggregated and asserted on
    in tests without depending on human-readable wording.
    """

    ACCEPTED = "ACCEPTED"

    # -- capture and framing -------------------------------------------------
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    FACE_TOO_LARGE = "FACE_TOO_LARGE"
    LOW_QUALITY = "LOW_QUALITY"
    LOW_LIGHT = "LOW_LIGHT"
    FACE_OCCLUDED = "FACE_OCCLUDED"
    INSUFFICIENT_FRAMES = "INSUFFICIENT_FRAMES"
    SESSION_TIMEOUT = "SESSION_TIMEOUT"

    # -- liveness ------------------------------------------------------------
    NO_BLINK_DETECTED = "NO_BLINK_DETECTED"
    TEXTURE_SPOOF_SUSPECTED = "TEXTURE_SPOOF_SUSPECTED"
    NO_MOTION_DETECTED = "NO_MOTION_DETECTED"
    FLAT_GEOMETRY = "FLAT_GEOMETRY"
    CHALLENGE_FAILED = "CHALLENGE_FAILED"
    CHALLENGE_TIMEOUT = "CHALLENGE_TIMEOUT"
    LIVENESS_SCORE_TOO_LOW = "LIVENESS_SCORE_TOO_LOW"
    LIVENESS_UNAVAILABLE = "LIVENESS_UNAVAILABLE"
    DEPTH_UNAVAILABLE = "DEPTH_UNAVAILABLE"

    # -- recognition ---------------------------------------------------------
    NO_MATCH = "NO_MATCH"
    LOW_RECOGNITION_CONFIDENCE = "LOW_RECOGNITION_CONFIDENCE"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    UNSTABLE_IDENTITY = "UNSTABLE_IDENTITY"
    NO_ENROLLED_IDENTITIES = "NO_ENROLLED_IDENTITIES"

    # -- policy --------------------------------------------------------------
    LOCKED_OUT = "LOCKED_OUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    @property
    def message(self) -> str:
        """Plain-language explanation suitable for showing to an end user."""
        return _REASON_MESSAGES[self]


_REASON_MESSAGES: Dict[ReasonCode, str] = {
    ReasonCode.ACCEPTED: "Verified.",
    ReasonCode.NO_FACE_DETECTED: "No face was visible in the camera.",
    ReasonCode.MULTIPLE_FACES: "More than one face was in frame; only one person at a time can be verified.",
    ReasonCode.FACE_TOO_SMALL: "Move closer to the camera.",
    ReasonCode.FACE_TOO_LARGE: "Move further from the camera.",
    ReasonCode.LOW_QUALITY: "The image was too blurry or poorly exposed to judge.",
    ReasonCode.LOW_LIGHT: "There is not enough light to verify safely.",
    ReasonCode.FACE_OCCLUDED: "Part of your face is covered. Remove glasses, a mask or anything blocking it.",
    ReasonCode.INSUFFICIENT_FRAMES: "Not enough usable video was captured to make a decision.",
    ReasonCode.SESSION_TIMEOUT: "The verification attempt took too long.",
    ReasonCode.NO_BLINK_DETECTED: "No natural blink was observed during the attempt.",
    ReasonCode.TEXTURE_SPOOF_SUSPECTED: "The surface texture looks like a printed photo or a screen rather than skin.",
    ReasonCode.NO_MOTION_DETECTED: "The face did not move the way a living face does.",
    ReasonCode.FLAT_GEOMETRY: "The face geometry appears flat, consistent with a photo or screen.",
    ReasonCode.CHALLENGE_FAILED: "The requested head movement was not performed correctly.",
    ReasonCode.CHALLENGE_TIMEOUT: "The requested head movement was not performed in time.",
    ReasonCode.LIVENESS_SCORE_TOO_LOW: "The combined liveness evidence was below the required level.",
    ReasonCode.LIVENESS_UNAVAILABLE: "A required liveness check could not run, so the attempt was refused rather than downgraded.",
    ReasonCode.DEPTH_UNAVAILABLE: "No depth camera is present; depth-based liveness is unavailable on this hardware.",
    ReasonCode.NO_MATCH: "This face does not match any enrolled identity.",
    ReasonCode.LOW_RECOGNITION_CONFIDENCE: "The closest enrolled identity was not a confident enough match.",
    ReasonCode.AMBIGUOUS_MATCH: "Two enrolled identities matched too closely to choose between them.",
    ReasonCode.UNSTABLE_IDENTITY: "The identity was not consistent across enough frames.",
    ReasonCode.NO_ENROLLED_IDENTITIES: "No identities have been enrolled yet.",
    ReasonCode.LOCKED_OUT: "Too many failed attempts. Locked out temporarily.",
    ReasonCode.INTERNAL_ERROR: "An internal error prevented a decision being made.",
}


class SignalStatus(str, Enum):
    """Outcome of one liveness signal.

    ``UNAVAILABLE`` is distinct from ``FAIL`` on purpose. "I checked and this
    looks fake" and "I could not check" must never collapse into the same
    verdict, because the second one is how systems silently degrade.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    UNAVAILABLE = "UNAVAILABLE"
    PENDING = "PENDING"


class BBox(NamedTuple):
    """Face bounding box in pixels, origin top-left."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    def clipped(self, frame_width: int, frame_height: int) -> "BBox":
        """Clamp the box to the frame so crops never index out of bounds."""
        x = max(0, min(self.x, frame_width - 1))
        y = max(0, min(self.y, frame_height - 1))
        w = max(1, min(self.width, frame_width - x))
        h = max(1, min(self.height, frame_height - y))
        return BBox(x, y, w, h)


@dataclass(frozen=True)
class QualityScores:
    """Per-aspect image quality, each normalised to 0..1 (higher is better)."""

    overall: float
    sharpness: float
    brightness: float
    contrast: float
    size: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "overall": _f(self.overall),
            "sharpness": _f(self.sharpness),
            "brightness": _f(self.brightness),
            "contrast": _f(self.contrast),
            "size": _f(self.size),
        }


@dataclass
class FaceObservation:
    """Everything measured about one face in one frame.

    ``landmarks`` is an ``(N, 3)`` array in normalised image coordinates; the z
    component is MediaPipe's *inferred* relative depth, not a measurement (see
    :mod:`faceguard.liveness.depth`).
    """

    bbox: BBox
    detection_confidence: float
    landmarks: Optional[np.ndarray] = None
    blendshapes: Dict[str, float] = field(default_factory=dict)
    pose_matrix: Optional[np.ndarray] = None
    quality: Optional[QualityScores] = None
    timestamp: float = field(default_factory=time.monotonic)
    frame_shape: tuple[int, int] = (0, 0)

    @property
    def has_landmarks(self) -> bool:
        return self.landmarks is not None and len(self.landmarks) > 0


@dataclass
class SignalResult:
    """Verdict from a single liveness signal."""

    name: str
    status: SignalStatus
    score: float = 0.0
    threshold: float = 0.0
    reason: Optional[ReasonCode] = None
    detail: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is SignalStatus.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "score": _f(self.score),
            "threshold": _f(self.threshold),
            "reason": self.reason.value if self.reason else None,
            "detail": self.detail,
            "metrics": {k: _f(v) for k, v in self.metrics.items()},
        }


@dataclass
class LivenessResult:
    """Fused verdict across all liveness signals."""

    passed: bool
    score: float
    signals: Dict[str, SignalResult] = field(default_factory=dict)
    reasons: List[ReasonCode] = field(default_factory=list)
    degraded: bool = False
    degraded_detail: str = ""
    frames_analysed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": _f(self.score),
            "degraded": self.degraded,
            "degraded_detail": self.degraded_detail,
            "frames_analysed": self.frames_analysed,
            "reasons": [r.value for r in self.reasons],
            "signals": {k: v.to_dict() for k, v in self.signals.items()},
        }


@dataclass
class RecognitionResult:
    """Identity decision, entirely independent of liveness."""

    matched: bool
    identity: Optional[str] = None
    confidence: float = 0.0
    runner_up: Optional[str] = None
    runner_up_confidence: float = 0.0
    margin: float = 0.0
    frames_agreeing: int = 0
    frames_compared: int = 0
    reasons: List[ReasonCode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "identity": self.identity,
            "confidence": _f(self.confidence),
            "runner_up": self.runner_up,
            "runner_up_confidence": _f(self.runner_up_confidence),
            "margin": _f(self.margin),
            "frames_agreeing": self.frames_agreeing,
            "frames_compared": self.frames_compared,
            "reasons": [r.value for r in self.reasons],
        }


@dataclass
class VerificationResult:
    """Top-level answer to "should this person be let in".

    ``accepted`` is true only when liveness passed *and* recognition matched.
    The two component verdicts remain readable individually so a reviewer can
    see which half failed.
    """

    accepted: bool
    reason: ReasonCode
    identity: Optional[str] = None
    recognition_confidence: float = 0.0
    detection_confidence: float = 0.0
    liveness_passed: bool = False
    liveness_score: float = 0.0
    liveness: Optional[LivenessResult] = None
    recognition: Optional[RecognitionResult] = None
    frames_used: int = 0
    elapsed_ms: float = 0.0
    degraded: bool = False
    degraded_detail: str = ""
    auth_method: str = "biometric"
    session_id: Optional[str] = None

    @property
    def message(self) -> str:
        return self.reason.message

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form. This is the schema the API and CLI emit."""
        return {
            "accepted": self.accepted,
            "reason": self.reason.value,
            "message": self.message,
            "identity": self.identity,
            "recognition_confidence": _f(self.recognition_confidence),
            "detection_confidence": _f(self.detection_confidence),
            "liveness_passed": self.liveness_passed,
            "liveness_score": _f(self.liveness_score),
            "frames_used": self.frames_used,
            "elapsed_ms": _f(self.elapsed_ms),
            "degraded": self.degraded,
            "degraded_detail": self.degraded_detail,
            "auth_method": self.auth_method,
            "session_id": self.session_id,
            "liveness": self.liveness.to_dict() if self.liveness else None,
            "recognition": self.recognition.to_dict() if self.recognition else None,
        }

    @classmethod
    def rejected(cls, reason: ReasonCode, **kwargs: Any) -> "VerificationResult":
        """Build a rejection. Used wherever an early exit is needed."""
        return cls(accepted=False, reason=reason, **kwargs)


def _f(value: Any) -> float:
    """Coerce numpy scalars and NaNs to plain JSON-safe floats."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if np.isfinite(out) else 0.0


def summarise_reasons(reasons: Sequence[ReasonCode]) -> ReasonCode:
    """Pick the single most informative reason from a list.

    Ordering matters for the user-facing message: telling someone "no blink
    detected" is more actionable than "liveness score too low", even though
    both are true, so specific causes outrank aggregate ones.
    """
    if not reasons:
        return ReasonCode.ACCEPTED
    priority = [
        ReasonCode.LOCKED_OUT,
        ReasonCode.NO_ENROLLED_IDENTITIES,
        ReasonCode.NO_FACE_DETECTED,
        ReasonCode.MULTIPLE_FACES,
        ReasonCode.FACE_TOO_SMALL,
        ReasonCode.FACE_TOO_LARGE,
        ReasonCode.LOW_LIGHT,
        ReasonCode.FACE_OCCLUDED,
        ReasonCode.LOW_QUALITY,
        ReasonCode.INSUFFICIENT_FRAMES,
        ReasonCode.SESSION_TIMEOUT,
        ReasonCode.LIVENESS_UNAVAILABLE,
        ReasonCode.TEXTURE_SPOOF_SUSPECTED,
        ReasonCode.FLAT_GEOMETRY,
        ReasonCode.CHALLENGE_FAILED,
        ReasonCode.CHALLENGE_TIMEOUT,
        ReasonCode.NO_BLINK_DETECTED,
        ReasonCode.NO_MOTION_DETECTED,
        ReasonCode.LIVENESS_SCORE_TOO_LOW,
        ReasonCode.AMBIGUOUS_MATCH,
        ReasonCode.UNSTABLE_IDENTITY,
        ReasonCode.LOW_RECOGNITION_CONFIDENCE,
        ReasonCode.NO_MATCH,
        ReasonCode.INTERNAL_ERROR,
    ]
    present = set(reasons)
    for code in priority:
        if code in present:
            return code
    return reasons[0]
