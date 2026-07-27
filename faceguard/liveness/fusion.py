"""Liveness policy: combining the individual signals into one verdict.

The rules this engine enforces, and why each exists:

* **A minimum number of frames.** No liveness decision is made from a single
  frame, ever. Single-frame liveness is known to be the weakest form, and the
  temporal signals here are meaningless without a window.
* **Required signals must pass.** By default that is texture, which is the only
  signal that inspects the physical surface being presented and the only one an
  uncooperative attacker cannot simply decline to trigger.
* **A required signal that could not run is a refusal, not a pass.** If the
  spoof classifier is missing, the attempt is rejected with
  ``LIVENESS_UNAVAILABLE``. Treating "I could not check" as "the check passed"
  is how systems silently lose the protection they advertise.
* **At least N of the optional signals must pass.** Any single behavioural
  signal is individually defeatable -- a video replay contains real blinks --
  so the policy asks for corroboration rather than trusting one.
* **The challenge, when enabled, must be satisfied.** Passive observation alone
  cannot distinguish a live subject from a recording of one.
* **The weighted score must clear a floor.** Signals that pass marginally on
  every axis should not add up to a confident accept.

Everything about the policy -- which signals are required, how many optional
ones suffice, the weights, the floor -- comes from configuration. The engine
contains no thresholds of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..schemas import (
    FaceObservation,
    LivenessResult,
    ReasonCode,
    SignalResult,
    SignalStatus,
)
from . import blink, depth, motion, texture
from .motion import ChallengeType
from .texture import SpoofClassifier


@dataclass
class LivenessEvidence:
    """Everything gathered during one verification attempt.

    ``crops`` are the raw, unenhanced face regions. Enhanced pixels must never
    reach the texture signal: CLAHE and sharpening manufacture exactly the
    micro-detail that distinguishes skin from paper, so an enhanced photo of a
    photo scores as more live than the original.
    """

    observations: List[FaceObservation] = field(default_factory=list)
    crops: List[np.ndarray] = field(default_factory=list)
    challenge: Optional[ChallengeType] = None
    challenge_elapsed_s: float = 0.0


class LivenessEngine:
    """Runs every liveness signal and applies the configured policy."""

    def __init__(self, config: Config, classifier: Optional[SpoofClassifier] = None):
        self.config = config
        self.classifier = classifier
        # Probed once at startup rather than per attempt: enumerating USB
        # devices takes long enough to be felt in a real-time loop, and a depth
        # camera does not appear halfway through a session.
        self.depth_capability = depth.probe_hardware(config.liveness.depth)

    @classmethod
    def from_config(cls, config: Config) -> "LivenessEngine":
        """Build an engine, loading the spoof classifier if one is present."""
        model_path = config.resolve(config.liveness.texture.model_path)
        return cls(config, SpoofClassifier.load(model_path))

    # -- signal execution --------------------------------------------------- #

    def run_signals(self, evidence: LivenessEvidence) -> Dict[str, SignalResult]:
        """Evaluate every signal independently.

        Signals never see each other's verdicts. Keeping them independent is
        what makes the layering meaningful -- if one signal could suppress
        another, an attacker who defeats the first defeats both.
        """
        signals: Dict[str, SignalResult] = {
            texture.SIGNAL_NAME: texture.evaluate(
                evidence.crops, self.config.liveness.texture, self.classifier
            ),
            blink.SIGNAL_NAME: blink.evaluate(
                evidence.observations, self.config.liveness.blink
            ),
            motion.SIGNAL_NAME: motion.evaluate(
                evidence.observations, self.config.liveness.motion
            ),
            depth.SIGNAL_NAME: depth.evaluate(
                evidence.observations,
                self.config.liveness.depth,
                self.depth_capability,
            ),
        }
        if evidence.challenge is not None:
            signals["challenge"] = motion.evaluate_challenge(
                evidence.observations,
                evidence.challenge,
                self.config.liveness.motion,
                evidence.challenge_elapsed_s,
            )
        return signals

    # -- policy ------------------------------------------------------------- #

    def evaluate(self, evidence: LivenessEvidence) -> LivenessResult:
        """Apply the configured policy to the collected evidence."""
        policy = self.config.liveness.policy
        frames = len(evidence.observations)

        if frames < self.config.liveness.min_frames:
            return LivenessResult(
                passed=False,
                score=0.0,
                reasons=[ReasonCode.INSUFFICIENT_FRAMES],
                frames_analysed=frames,
                degraded=not self.depth_capability.available,
                degraded_detail=self.depth_capability.detail,
            )

        signals = self.run_signals(evidence)
        reasons: List[ReasonCode] = []

        # -- required signals, including their unavailability --------------- #
        for name in policy.required:
            signal = signals.get(name)
            if signal is None or signal.status is SignalStatus.UNAVAILABLE:
                reasons.append(ReasonCode.LIVENESS_UNAVAILABLE)
            elif signal.status is SignalStatus.FAIL:
                reasons.append(signal.reason or ReasonCode.LIVENESS_SCORE_TOO_LOW)

        # -- optional signals: corroboration ------------------------------- #
        optional_results = [signals.get(name) for name in policy.optional]
        passed_optional = sum(
            1 for s in optional_results if s is not None and s.passed
        )
        if passed_optional < policy.min_optional_signals:
            # Only a *required* signal being unavailable raises
            # LIVENESS_UNAVAILABLE. An optional one that could not run is a
            # weaker statement, and letting it dominate produces a worse
            # message: a static photo would report "a check could not run"
            # rather than "no blink was detected", when the second is both
            # true and actionable. The parallax check being unavailable
            # because the subject never turned their head is already explained
            # by the challenge failure.
            for signal in optional_results:
                if signal is None:
                    continue
                if signal.status is SignalStatus.FAIL and signal.reason:
                    reasons.append(signal.reason)
            if not reasons:
                reasons.append(ReasonCode.LIVENESS_SCORE_TOO_LOW)

        # -- active challenge ----------------------------------------------- #
        if policy.require_challenge:
            challenge_result = signals.get("challenge")
            if challenge_result is None:
                reasons.append(ReasonCode.CHALLENGE_FAILED)
            elif challenge_result.status is SignalStatus.UNAVAILABLE:
                reasons.append(ReasonCode.LIVENESS_UNAVAILABLE)
            elif challenge_result.status is SignalStatus.FAIL:
                reasons.append(challenge_result.reason or ReasonCode.CHALLENGE_FAILED)

        # -- fused score ---------------------------------------------------- #
        score = self._fused_score(signals, policy.weights)
        if score < policy.min_fused_score:
            reasons.append(ReasonCode.LIVENESS_SCORE_TOO_LOW)

        # De-duplicate while preserving the order they were discovered in.
        deduped = list(dict.fromkeys(reasons))
        return LivenessResult(
            passed=not deduped,
            score=score,
            signals=signals,
            reasons=deduped,
            degraded=not self.depth_capability.available,
            degraded_detail=self.depth_capability.detail,
            frames_analysed=frames,
        )

    @staticmethod
    def _fused_score(
        signals: Dict[str, SignalResult], weights: Dict[str, float]
    ) -> float:
        """Weighted mean over signals that actually produced a verdict.

        Unavailable signals are excluded from both numerator and denominator
        rather than scored as zero. Scoring them zero would mean a missing
        depth camera silently dragged every honest user's score down, which
        pushes an operator toward loosening the threshold -- weakening the
        system in response to a hardware gap rather than reporting it. The
        gap is reported through ``degraded`` instead.
        """
        total_weight = 0.0
        accumulated = 0.0
        for name, weight in weights.items():
            signal = signals.get(name)
            if signal is None or signal.status is SignalStatus.UNAVAILABLE:
                continue
            if weight <= 0.0:
                continue
            accumulated += weight * float(np.clip(signal.score, 0.0, 1.0))
            total_weight += weight
        if total_weight <= 0.0:
            return 0.0
        return float(accumulated / total_weight)
