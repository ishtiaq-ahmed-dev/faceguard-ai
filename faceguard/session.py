"""Verification sessions, rate limiting and the top-level FaceGuard facade.

A *session* is one verification attempt spanning many frames. Sessions exist
because this system refuses to decide from a single frame: the temporal
liveness signals need a window, the identity vote needs several independent
looks, and the challenge needs time to be performed.

The flow:

1. ``FaceGuard.start_session`` issues a session id and a randomly chosen
   head-movement challenge.
2. Frames arrive one at a time through :meth:`VerificationSession.add_frame`,
   which returns immediate framing feedback -- move closer, too dark, only one
   person at a time -- so the user can correct themselves rather than waiting
   out a failure.
3. Once enough usable frames have accumulated, or the clock runs out,
   :meth:`VerificationSession.finalise` runs liveness and recognition and
   produces one :class:`~faceguard.schemas.VerificationResult`.

Rejections are rate limited per client with exponential backoff, and every
outcome is written to the audit log.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audit import AuditLog
from .config import Config, load_config
from .detect import FaceAnalyzer, FrameAnalysis
from .liveness import ChallengeType, LivenessEngine, LivenessEvidence, random_challenge
from .quality import Preprocessor, crop_face, standardise
from .recognition import EmbeddingBackend, Recogniser
from .schemas import (
    FaceObservation,
    ReasonCode,
    VerificationResult,
    summarise_reasons,
)
from .store import TemplateStore


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


@dataclass
class LockoutState:
    """Per-client failure history."""

    failures: Deque[float] = field(default_factory=deque)
    lockout_until: float = 0.0
    consecutive_lockouts: int = 0


class RateLimiter:
    """Sliding-window failure counter with exponential lockout.

    Biometric verification is inherently probabilistic: every threshold has a
    false-acceptance rate above zero, which means enough attempts eventually
    succeed by chance. Somebody cycling through printed photos of different
    people is exactly that attack. Rate limiting converts "keep trying" from a
    viable strategy into an expensive one.

    Successful verifications clear the counter. Lockouts double in length each
    time, up to a configured ceiling, so a legitimate user having a bad day in
    poor lighting is inconvenienced for thirty seconds while a persistent
    attacker is pushed toward an hour.
    """

    def __init__(self, config: Config) -> None:
        self.config = config.security
        self._states: Dict[str, LockoutState] = {}
        self._lock = threading.Lock()

    def _state(self, client_id: str) -> LockoutState:
        return self._states.setdefault(client_id, LockoutState())

    def check(self, client_id: str, now: Optional[float] = None) -> Tuple[bool, float]:
        """Return ``(allowed, seconds_remaining)`` for this client."""
        now = now if now is not None else time.time()
        with self._lock:
            state = self._state(client_id)
            if now < state.lockout_until:
                return False, state.lockout_until - now
            return True, 0.0

    def record_failure(self, client_id: str, now: Optional[float] = None) -> float:
        """Register a failed attempt. Returns seconds of lockout, 0 if none."""
        now = now if now is not None else time.time()
        with self._lock:
            state = self._state(client_id)
            window_start = now - self.config.failure_window_s
            while state.failures and state.failures[0] < window_start:
                state.failures.popleft()
            state.failures.append(now)

            if len(state.failures) < self.config.max_failed_attempts:
                return 0.0

            duration = min(
                self.config.lockout_base_s * (2**state.consecutive_lockouts),
                self.config.lockout_max_s,
            )
            state.consecutive_lockouts += 1
            state.lockout_until = now + duration
            state.failures.clear()
            return duration

    def record_success(self, client_id: str) -> None:
        """Clear failure history after a genuine success."""
        with self._lock:
            self._states.pop(client_id, None)

    def reset(self) -> None:
        with self._lock:
            self._states.clear()


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


@dataclass
class FrameFeedback:
    """Immediate, per-frame guidance returned while a session is collecting.

    This is presentation state, not a verdict. It exists so the UI can tell
    somebody to move closer while it happens, instead of collecting fifteen
    unusable frames and then rejecting them.
    """

    accepted_frame: bool
    frames_collected: int
    frames_required: int
    hint: str
    reason: Optional[ReasonCode] = None
    detection_confidence: float = 0.0
    quality: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    challenge: Optional[str] = None
    ready: bool = False
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "accepted_frame": self.accepted_frame,
            "frames_collected": self.frames_collected,
            "frames_required": self.frames_required,
            "hint": self.hint,
            "reason": self.reason.value if self.reason else None,
            "detection_confidence": round(float(self.detection_confidence), 4),
            "quality": round(float(self.quality), 4),
            "bbox": list(self.bbox) if self.bbox else None,
            "challenge": self.challenge,
            "ready": self.ready,
            "elapsed_s": round(float(self.elapsed_s), 2),
        }


class VerificationSession:
    """One multi-frame verification attempt."""

    def __init__(
        self,
        guard: "FaceGuard",
        client_id: str = "local",
        challenge: Optional[ChallengeType] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.guard = guard
        self.config = guard.config
        self.client_id = client_id
        self.session_id = session_id or uuid.uuid4().hex
        self.challenge = challenge
        if self.challenge is None and self.config.liveness.policy.require_challenge:
            self.challenge = random_challenge()

        self.started_at = time.monotonic()
        self.observations: List[FaceObservation] = []
        self.crops: List[np.ndarray] = []
        self.embeddings: List[np.ndarray] = []
        self._frames_seen = 0
        self._occluded_regions: List[str] = []
        self._finalised: Optional[VerificationResult] = None
        self._lock = threading.Lock()

    # -- collection --------------------------------------------------------- #

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def expired(self) -> bool:
        return self.elapsed_s > self.config.session.max_duration_s

    def add_frame(self, frame_bgr: np.ndarray) -> FrameFeedback:
        """Analyse one frame and add it to the window if it is usable."""
        with self._lock:
            # Normalise resolution first so that quality thresholds mean the
            # same thing whether the frame came from the browser, a video file
            # or a full-resolution photograph.
            frame_bgr = standardise(frame_bgr, self.config.detection.working_width)
            analysis = self.guard.analyzer.analyze(frame_bgr)
            face = analysis.primary
            self._frames_seen += 1

            required = self.config.liveness.min_frames
            bbox = tuple(face.bbox) if face else None
            detection_confidence = face.detection_confidence if face else 0.0
            quality = face.quality.overall if face and face.quality else 0.0

            if analysis.reasons:
                reason = summarise_reasons(analysis.reasons)
                hint = reason.message
                if reason is ReasonCode.LOW_LIGHT and analysis.backlight > 1.6:
                    hint = (
                        "You are lit from behind. Face a light source or move away "
                        "from the window."
                    )
                elif analysis.occluded_regions:
                    covered = ", ".join(
                        r.replace("_", " ") for r in analysis.occluded_regions
                    )
                    hint = f"{hint} Your {covered} may be covered."
                return FrameFeedback(
                    accepted_frame=False,
                    frames_collected=len(self.observations),
                    frames_required=required,
                    hint=hint,
                    reason=reason,
                    detection_confidence=detection_confidence,
                    quality=quality,
                    bbox=bbox,
                    challenge=self.challenge.value if self.challenge else None,
                    elapsed_s=self.elapsed_s,
                )

            assert face is not None  # guaranteed: no reasons means one good face
            if len(self.observations) < self.config.session.max_buffered_frames:
                self.observations.append(face)
                # Liveness must see raw pixels. Enhancement manufactures the
                # micro-texture that separates skin from paper.
                raw_crop, _ = crop_face(frame_bgr, face.bbox, margin=0.08)
                self.crops.append(raw_crop)

                # A frame whose face looks partly covered does not get to vote
                # on identity. Measured reason: with a hand over part of the
                # face, two probes matched the WRONG enrolled person at 0.632
                # and 0.650 -- and no threshold can separate those from a
                # genuine match, whose minimum was 0.651. Occlusion destroys
                # the discriminative detail and the embedding drifts somewhere
                # arbitrary, which is occasionally somewhere dangerous.
                #
                # Excluding the frame rather than refusing the attempt is the
                # right severity: the occlusion detector fires on 17.6% of
                # clean faces, so refusing outright would turn away one honest
                # user in six, whereas dropping a frame costs only a retry.
                # With min_agreeing_frames at 3 and ~87% detection on hand
                # occlusion, an attacker needs three consecutive misses.
                if (
                    self._frames_seen % self.config.session.recognise_every_n_frames == 0
                    and not analysis.occluded_regions
                ):
                    self._embed(frame_bgr, face)

            self._occluded_regions = analysis.occluded_regions
            collected = len(self.observations)
            ready = collected >= required and self._challenge_satisfied()
            return FrameFeedback(
                accepted_frame=True,
                frames_collected=collected,
                frames_required=required,
                hint=self._hint(collected, required),
                detection_confidence=detection_confidence,
                quality=quality,
                bbox=bbox,
                challenge=self.challenge.value if self.challenge else None,
                ready=ready,
                elapsed_s=self.elapsed_s,
            )

    def _embed(self, frame_bgr: np.ndarray, face: FaceObservation) -> None:
        """Compute a recognition embedding from an enhanced crop.

        Enhancement is applied here and only here. Recognition benefits from
        CLAHE and mild sharpening because the embedding network expects
        well-exposed faces; liveness must never see the same pixels.
        """
        crop, _ = crop_face(frame_bgr, face.bbox, margin=0.15)
        if crop.size == 0 or face.quality is None:
            return
        enhanced = Preprocessor.for_recognition(crop, face.quality, self.config.quality)
        embedding = self.guard.recogniser.backend.embed(enhanced)
        if embedding is not None:
            self.embeddings.append(embedding)

    def _challenge_satisfied(self) -> bool:
        if self.challenge is None:
            return True
        from .liveness.motion import evaluate_challenge

        return evaluate_challenge(
            self.observations, self.challenge, self.config.liveness.motion, self.elapsed_s
        ).passed

    def _hint(self, collected: int, required: int) -> str:
        if self.challenge is not None and not self._challenge_satisfied():
            return self.challenge.instruction
        if collected < required:
            return f"Hold still. {required - collected} more frame(s) needed."
        return "Verifying..."

    # -- decision ----------------------------------------------------------- #

    def finalise(self) -> VerificationResult:
        """Run liveness and recognition over the collected window."""
        with self._lock:
            if self._finalised is not None:
                return self._finalised
            result = self.guard._decide(self)
            self._finalised = result
            return result


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #


class FaceGuard:
    """Top-level entry point tying every component together.

    Construct once and reuse: it owns the MediaPipe graphs, the embedding
    model and the template store, all of which are expensive to build.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or load_config()
        self.analyzer = FaceAnalyzer(self.config)
        self.liveness = LivenessEngine.from_config(self.config)
        self.store = TemplateStore(
            self.config.resolve(self.config.security.template_store),
            self.config.resolve(self.config.security.key_file),
        )
        self.recogniser = Recogniser(
            self.config, self.store, EmbeddingBackend(self.config.recognition)
        )
        self.audit = AuditLog(self.config.resolve(self.config.security.audit_log))
        self.rate_limiter = RateLimiter(self.config)

    def warm_up(self) -> None:
        """Pay every one-off model-loading cost up front.

        Without this the first verification of the process pays a ~45 second
        weight download and graph build, which blows the session timeout and
        looks like a hang to anyone trying the demo.
        """
        self.recogniser.backend.warm_up()

    def close(self) -> None:
        self.analyzer.close()

    def __enter__(self) -> "FaceGuard":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- sessions ----------------------------------------------------------- #

    def start_session(
        self, client_id: str = "local", challenge: Optional[ChallengeType] = None
    ) -> VerificationSession:
        """Begin a verification attempt."""
        return VerificationSession(self, client_id=client_id, challenge=challenge)

    def verify_frames(
        self,
        frames: Sequence[np.ndarray],
        client_id: str = "local",
        challenge: Optional[ChallengeType] = None,
    ) -> VerificationResult:
        """Verify a pre-captured sequence of frames.

        Used by the CLI, the evaluation harness and the tests. The live path
        goes through :meth:`start_session` instead, but both converge on the
        same decision logic in :meth:`_decide`.
        """
        session = self.start_session(client_id=client_id, challenge=challenge)
        for frame in frames:
            session.add_frame(frame)
        return session.finalise()

    # -- decision ----------------------------------------------------------- #

    def _decide(self, session: VerificationSession) -> VerificationResult:
        """Combine liveness and recognition into the final verdict.

        Ordering matters. Rate limiting is checked first so a locked-out client
        cannot use verification attempts as an oracle. Liveness is evaluated
        before identity is revealed, so that an attacker holding up a photo
        learns only that liveness failed, never whether the face in the photo
        is enrolled.
        """
        started = time.monotonic()
        allowed, remaining = self.rate_limiter.check(session.client_id)
        if not allowed:
            result = VerificationResult.rejected(
                ReasonCode.LOCKED_OUT,
                session_id=session.session_id,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                degraded_detail=f"Locked out for another {remaining:.0f} seconds.",
            )
            self.audit.record_verification(result, session.client_id)
            return result

        detection_confidence = (
            float(np.median([o.detection_confidence for o in session.observations]))
            if session.observations
            else 0.0
        )

        evidence = LivenessEvidence(
            observations=session.observations,
            crops=session.crops,
            challenge=session.challenge,
            challenge_elapsed_s=session.elapsed_s,
        )
        liveness_result = self.liveness.evaluate(evidence)

        if session.expired and not liveness_result.passed:
            liveness_result.reasons.append(ReasonCode.SESSION_TIMEOUT)

        if not liveness_result.passed:
            result = VerificationResult(
                accepted=False,
                reason=summarise_reasons(liveness_result.reasons),
                detection_confidence=detection_confidence,
                liveness_passed=False,
                liveness_score=liveness_result.score,
                liveness=liveness_result,
                frames_used=len(session.observations),
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                degraded=liveness_result.degraded,
                degraded_detail=liveness_result.degraded_detail,
                session_id=session.session_id,
            )
            self._register_failure(session, result)
            return result

        recognition_result = self.recogniser.identify(session.embeddings)
        accepted = recognition_result.matched
        reason = (
            ReasonCode.ACCEPTED
            if accepted
            else summarise_reasons(recognition_result.reasons)
        )

        result = VerificationResult(
            accepted=accepted,
            reason=reason,
            identity=recognition_result.identity if accepted else None,
            recognition_confidence=recognition_result.confidence,
            detection_confidence=detection_confidence,
            liveness_passed=True,
            liveness_score=liveness_result.score,
            liveness=liveness_result,
            recognition=recognition_result,
            frames_used=len(session.observations),
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            degraded=liveness_result.degraded,
            degraded_detail=liveness_result.degraded_detail,
            session_id=session.session_id,
        )

        if accepted:
            self.rate_limiter.record_success(session.client_id)
            self.audit.record_verification(result, session.client_id)
        else:
            self._register_failure(session, result)
        return result

    def _register_failure(
        self, session: VerificationSession, result: VerificationResult
    ) -> None:
        """Count a failure toward lockout and audit it.

        Framing problems are deliberately excluded from the lockout counter.
        Locking somebody out because their room was dark punishes a usability
        failure as though it were an attack, and it is the fastest way to make
        people disable the security control entirely.
        """
        not_the_users_fault = {
            ReasonCode.INSUFFICIENT_FRAMES,
            ReasonCode.NO_FACE_DETECTED,
            ReasonCode.LOW_LIGHT,
            ReasonCode.LOW_QUALITY,
            ReasonCode.FACE_TOO_SMALL,
            ReasonCode.FACE_TOO_LARGE,
            ReasonCode.MULTIPLE_FACES,
            ReasonCode.NO_ENROLLED_IDENTITIES,
        }
        lockout_seconds = 0.0
        if result.reason not in not_the_users_fault:
            lockout_seconds = self.rate_limiter.record_failure(session.client_id)

        self.audit.record_verification(
            result,
            session.client_id,
            extra={"lockout_seconds": lockout_seconds} if lockout_seconds else None,
        )
        if lockout_seconds:
            self.audit.record_event(
                "lockout",
                client_id=session.client_id,
                seconds=lockout_seconds,
                triggered_by=result.reason.value,
            )
