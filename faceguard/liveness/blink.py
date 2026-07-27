"""Blink detection.

**What this catches:** a printed photograph, a paper mask, or any other static
target. Eyes that never close are not attached to a living person.

**What this does not catch:** a video replay. A recording of the real user
blinking contains real blinks, and this check will happily count them. Blink is
therefore a cheap, high-value filter against the most common attack, not a
defence against the most capable one. It is one of three layers for that
reason.

**How it works.** Two independent measures of eye closure are computed per
frame:

1. *Eye aspect ratio* (Soukupova & Cech, 2016) -- the ratio of the eye's
   vertical opening to its horizontal width, from six landmarks per eye. Small
   when the lid is down. Simple, inspectable, and completely independent of any
   learned classifier.
2. *MediaPipe blendshape activations* ``eyeBlinkLeft`` / ``eyeBlinkRight`` --
   already calibrated to 0..1 and, unlike raw EAR, robust to head rotation
   (EAR shrinks when you turn your head, purely from foreshortening).

Blendshapes are preferred when present; EAR is the fallback and the
cross-check. A blink is only counted when the closure signal crosses a *closed*
threshold and then returns below a lower *open* threshold -- a deliberate
hysteresis band, so that noise hovering around a single threshold cannot
manufacture blinks -- and when the round trip took between 70 ms and 500 ms,
which is the physiological range of a spontaneous human blink. Anything faster
is landmark jitter. Anything slower is a squint, a wink held for the camera, or
a finger moved across a photograph, none of which are evidence of life.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import BlinkConfig
from ..schemas import FaceObservation, ReasonCode, SignalResult, SignalStatus

# MediaPipe Face Mesh landmark indices, ordered for the EAR formula:
# (outer corner, upper-outer, upper-inner, inner corner, lower-inner, lower-outer)
RIGHT_EYE_EAR_IDX: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR_IDX: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)

SIGNAL_NAME = "blink"


@dataclass(frozen=True)
class BlinkEvent:
    """One completed eye closure that passed the physiological gate."""

    start_s: float
    end_s: float
    duration_ms: float
    peak_closure: float
    source: str


def eye_aspect_ratio(landmarks: np.ndarray, indices: Sequence[int]) -> float:
    """Vertical-to-horizontal opening ratio for one eye.

    ``landmarks`` is the ``(N, 3)`` normalised landmark array. Only x and y are
    used: the z component is inferred rather than measured and adds noise
    without adding information.

    Returns roughly 0.30 for a wide-open eye and under 0.15 for a closed one.
    """
    try:
        p1, p2, p3, p4, p5, p6 = (landmarks[i][:2] for i in indices)
    except (IndexError, ValueError):
        return float("nan")

    horizontal = float(np.linalg.norm(p1 - p4))
    if horizontal <= 1e-9:
        return float("nan")
    vertical = float(np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
    return vertical / (2.0 * horizontal)


def closure_from_ear(landmarks: np.ndarray) -> float:
    """Eye-aspect-ratio of the *more open* of the two eyes.

    The maximum, not the mean. A spontaneous blink is bilateral: both eyes
    close together. Taking the more open eye means the reported value only
    falls when **both** have closed, so one mis-tracked eye -- or a deliberate
    wink -- cannot fabricate proof of life on its own.

    Averaging was tried first and is measurably wrong: with one eye at 30% of
    its open size and the other fully open, the mean lands below the closed
    threshold and a wink registers as a blink.
    """
    left = eye_aspect_ratio(landmarks, LEFT_EYE_EAR_IDX)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR_IDX)
    values = [v for v in (left, right) if np.isfinite(v)]
    return float(max(values)) if values else float("nan")


def closure_from_blendshapes(blendshapes: dict) -> float:
    """Blink activation of the *less closed* eye, 0..1 where 1.0 is shut.

    The minimum, for the same reason :func:`closure_from_ear` takes the
    maximum: the signal must only read "closed" when both eyes are, so that a
    single spuriously high blink activation cannot manufacture a blink.

    The cost is that a genuine blink during which one eye tracks poorly is
    missed. That is the right way to be wrong here -- blink is one of two
    optional signals and a real user blinks repeatedly within the window, so
    the occasional missed blink costs a retry, whereas a fabricated one costs
    the security property.
    """
    left = blendshapes.get("eyeBlinkLeft")
    right = blendshapes.get("eyeBlinkRight")
    values = [v for v in (left, right) if v is not None]
    return float(min(values)) if values else float("nan")


def _extract_series(
    observations: Sequence[FaceObservation],
) -> Tuple[str, List[float], List[float]]:
    """Choose a closure source and extract its time series.

    One source is used for the whole window. Switching mid-window would compare
    values from two differently-scaled signals against one pair of thresholds,
    which silently corrupts the state machine.
    """
    blendshape_values, ear_values, times = [], [], []
    for observation in observations:
        if not observation.has_landmarks:
            continue
        times.append(observation.timestamp)
        blendshape_values.append(closure_from_blendshapes(observation.blendshapes))
        ear_values.append(closure_from_ear(observation.landmarks))

    if not times:
        return "none", [], []

    blendshape_coverage = float(np.mean([np.isfinite(v) for v in blendshape_values]))
    if blendshape_coverage >= 0.8:
        return "blendshape", blendshape_values, times
    return "ear", ear_values, times


def _find_blinks(
    series: Sequence[float],
    times: Sequence[float],
    closed_threshold: float,
    open_threshold: float,
    closed_is_high: bool,
    config: BlinkConfig,
) -> Tuple[List[BlinkEvent], int]:
    """Walk the closure series and extract physiologically plausible blinks.

    Returns the accepted events and the count of closures that were rejected
    for being outside the plausible duration range -- the second number is what
    distinguishes "nobody blinked" from "something occluded the eyes for two
    seconds", and both are worth logging.
    """
    events: List[BlinkEvent] = []
    rejected = 0
    in_closure = False
    start_time = 0.0
    peak = 0.0

    def is_closed(value: float) -> bool:
        return value >= closed_threshold if closed_is_high else value <= closed_threshold

    def is_open(value: float) -> bool:
        return value <= open_threshold if closed_is_high else value >= open_threshold

    for value, timestamp in zip(series, times):
        if not np.isfinite(value):
            continue
        if not in_closure:
            if is_closed(value):
                in_closure = True
                start_time = timestamp
                peak = value
        else:
            peak = max(peak, value) if closed_is_high else min(peak, value)
            if is_open(value):
                duration_ms = (timestamp - start_time) * 1000.0
                if config.min_blink_ms <= duration_ms <= config.max_blink_ms:
                    events.append(
                        BlinkEvent(start_time, timestamp, duration_ms, peak, "")
                    )
                else:
                    rejected += 1
                in_closure = False
    return events, rejected


def evaluate(
    observations: Sequence[FaceObservation], config: BlinkConfig
) -> SignalResult:
    """Decide whether a live blink occurred over this window of frames.

    Pure function of the window: no hidden state, so the same recording always
    yields the same verdict and a failure can be reproduced from a saved clip.
    """
    source, series, times = _extract_series(observations)
    if source == "none" or len(series) < 2:
        return SignalResult(
            name=SIGNAL_NAME,
            status=SignalStatus.UNAVAILABLE,
            reason=ReasonCode.LIVENESS_UNAVAILABLE,
            detail="No facial landmarks were available to measure eye closure.",
        )

    if source == "blendshape":
        # Blendshape activations are already calibrated to 0..1 across
        # individuals, so absolute thresholds are meaningful here.
        closed_threshold = config.blendshape_closed
        open_threshold = config.blendshape_open
        closed_is_high = True
    else:
        # Raw EAR is not comparable between people -- open-eye values span more
        # than 2x across individuals -- so thresholds are taken relative to
        # this subject's own open-eye baseline. The 75th percentile estimates
        # "eyes open" robustly: blinks occupy a small minority of frames, so
        # the upper quartile sits in the open state even when several blinks
        # fall inside the window.
        baseline = float(np.nanpercentile([v for v in series if np.isfinite(v)], 75))
        closed_threshold = baseline * config.ear_closed_ratio
        open_threshold = baseline * config.ear_open_ratio
        closed_is_high = False

    events, rejected = _find_blinks(
        series, times, closed_threshold, open_threshold, closed_is_high, config
    )

    finite = np.array([v for v in series if np.isfinite(v)], dtype=float)
    baseline = float(np.median(finite)) if finite.size else float("nan")

    # A genuine blink is a brief transient in an otherwise-open eye. If the
    # median closure sits in the closed band, the eyes were covered for most of
    # the window -- a finger held over a photograph's eyes, or sunglasses
    # confusing the landmarker -- and the transitions at either end are not
    # blinks.
    eyes_mostly_open = (
        baseline < closed_threshold if closed_is_high else baseline > closed_threshold
    )

    metrics = {
        "blinks_detected": float(len(events)),
        "blinks_rejected_out_of_range": float(rejected),
        "baseline_closure": baseline,
        "frames_with_landmarks": float(len(series)),
        "mean_blink_duration_ms": float(
            np.mean([e.duration_ms for e in events]) if events else 0.0
        ),
    }

    if not eyes_mostly_open:
        return SignalResult(
            name=SIGNAL_NAME,
            status=SignalStatus.FAIL,
            score=0.0,
            threshold=float(config.required_blinks),
            reason=ReasonCode.NO_BLINK_DETECTED,
            detail=(
                "Eyes appear closed or occluded for most of the attempt, so no "
                "blink transition can be trusted."
            ),
            metrics=metrics,
        )

    if len(events) < config.required_blinks:
        detail = f"Saw {len(events)} valid blink(s); {config.required_blinks} required."
        if rejected:
            detail += (
                f" {rejected} eye closure(s) were ignored for falling outside the "
                f"{config.min_blink_ms:.0f}-{config.max_blink_ms:.0f} ms range of a "
                "natural blink."
            )
        return SignalResult(
            name=SIGNAL_NAME,
            status=SignalStatus.FAIL,
            score=len(events) / max(config.required_blinks, 1),
            threshold=1.0,
            reason=ReasonCode.NO_BLINK_DETECTED,
            detail=detail,
            metrics=metrics,
        )

    return SignalResult(
        name=SIGNAL_NAME,
        status=SignalStatus.PASS,
        score=min(1.0, len(events) / max(config.required_blinks, 1)),
        threshold=1.0,
        detail=(
            f"{len(events)} natural blink(s) observed via {source}, "
            f"mean duration {metrics['mean_blink_duration_ms']:.0f} ms."
        ),
        metrics=metrics,
    )
