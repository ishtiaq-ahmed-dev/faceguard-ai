"""Tests for blink detection.

The security-relevant properties here are the *rejections*: a static photograph
must never produce a blink, and neither must landmark jitter or an object held
over the eyes. Detecting a genuine blink is the easy half.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceguard.liveness import blink
from faceguard.schemas import ReasonCode, SignalStatus

from .conftest import blink_sequence, make_observation, synthetic_face_cloud


class TestEyeAspectRatio:
    def test_open_eye_scores_higher_than_closed(self):
        """EAR must fall when the lid closes. Ordering is the whole signal."""
        landmarks = synthetic_face_cloud()
        open_eye = landmarks.copy()
        # Widen the vertical opening of the right eye.
        for index, offset in ((160, -0.03), (158, -0.03), (153, 0.03), (144, 0.03)):
            open_eye[index] = [0.45, 0.46 + offset, 0.0]
        open_eye[33] = [0.40, 0.46, 0.0]
        open_eye[133] = [0.50, 0.46, 0.0]

        closed_eye = open_eye.copy()
        for index in (160, 158, 153, 144):
            closed_eye[index] = [0.45, 0.46, 0.0]

        assert blink.eye_aspect_ratio(open_eye, blink.RIGHT_EYE_EAR_IDX) > (
            blink.eye_aspect_ratio(closed_eye, blink.RIGHT_EYE_EAR_IDX)
        )

    def test_degenerate_eye_returns_nan(self):
        """A collapsed horizontal span must not divide by zero."""
        landmarks = np.zeros((478, 3), dtype=np.float32)
        assert np.isnan(blink.eye_aspect_ratio(landmarks, blink.RIGHT_EYE_EAR_IDX))

    def test_out_of_range_indices_return_nan(self):
        assert np.isnan(blink.eye_aspect_ratio(np.zeros((5, 3)), (0, 1, 2, 3, 4, 99)))


class TestBlendshapeClosure:
    def test_uses_the_less_closed_eye(self):
        """Both eyes must close before the signal reads closed."""
        assert blink.closure_from_blendshapes(
            {"eyeBlinkLeft": 0.2, "eyeBlinkRight": 0.8}
        ) == pytest.approx(0.2)

    def test_missing_blendshapes_return_nan(self):
        assert np.isnan(blink.closure_from_blendshapes({}))


class TestBlinkDetection:
    def test_a_natural_blink_is_counted(self, config):
        # 4 frames at 30fps = 133ms, squarely inside the physiological window.
        result = blink.evaluate(blink_sequence(closed_frames=4), config.liveness.blink)
        assert result.status is SignalStatus.PASS
        assert result.metrics["blinks_detected"] == 1

    def test_a_static_photograph_never_blinks(self, config):
        """The primary claim: eyes that never close fail the signal."""
        frames = [make_observation(blink=0.04, timestamp=i / 30.0) for i in range(40)]
        result = blink.evaluate(frames, config.liveness.blink)
        assert result.status is SignalStatus.FAIL
        assert result.reason is ReasonCode.NO_BLINK_DETECTED

    def test_a_single_frame_flicker_is_rejected_as_too_fast(self, config):
        """One frame of closure at 30fps is 33ms -- landmark noise, not a blink."""
        result = blink.evaluate(blink_sequence(closed_frames=1), config.liveness.blink)
        assert result.status is SignalStatus.FAIL
        assert result.metrics["blinks_rejected_out_of_range"] >= 1

    def test_a_long_closure_is_rejected_as_too_slow(self, config):
        """25 frames at 30fps is 833ms: a squint or an object over the eyes."""
        result = blink.evaluate(
            blink_sequence(closed_frames=25, total_frames=60), config.liveness.blink
        )
        assert result.status is SignalStatus.FAIL
        assert result.metrics["blinks_rejected_out_of_range"] >= 1

    def test_eyes_closed_throughout_cannot_produce_a_blink(self, config):
        """Covering a photo's eyes and uncovering them is not proof of life."""
        frames = [make_observation(blink=0.95, timestamp=i / 30.0) for i in range(30)]
        result = blink.evaluate(frames, config.liveness.blink)
        assert result.status is SignalStatus.FAIL
        assert "occluded" in result.detail.lower() or "closed" in result.detail.lower()

    def test_noise_around_the_threshold_does_not_manufacture_blinks(self, config):
        """Hysteresis test.

        A signal dithering either side of a single threshold would generate a
        blink on every crossing. The open and closed thresholds are separated
        precisely so it cannot.
        """
        rng = np.random.default_rng(5)
        midpoint = (
            config.liveness.blink.blendshape_closed
            + config.liveness.blink.blendshape_open
        ) / 2.0
        frames = [
            make_observation(
                blink=float(np.clip(midpoint + rng.normal(0, 0.02), 0, 1)),
                timestamp=i / 30.0,
            )
            for i in range(60)
        ]
        result = blink.evaluate(frames, config.liveness.blink)
        assert result.metrics["blinks_detected"] == 0

    def test_two_blinks_are_both_counted(self, config):
        frames = []
        for index in range(60):
            closed = (10 <= index < 14) or (34 <= index < 38)
            frames.append(
                make_observation(blink=0.9 if closed else 0.05, timestamp=index / 30.0)
            )
        result = blink.evaluate(frames, config.liveness.blink)
        assert result.metrics["blinks_detected"] == 2

    def test_no_landmarks_reports_unavailable_not_fail(self, config):
        """'I could not look' must not be recorded as 'I looked and saw nothing'."""
        frames = [make_observation() for _ in range(5)]
        for frame in frames:
            frame.landmarks = None
            frame.blendshapes = {}
        result = blink.evaluate(frames, config.liveness.blink)
        assert result.status is SignalStatus.UNAVAILABLE
        assert result.reason is ReasonCode.LIVENESS_UNAVAILABLE

    def test_empty_input_is_unavailable(self, config):
        assert blink.evaluate([], config.liveness.blink).status is (
            SignalStatus.UNAVAILABLE
        )


def _eyes(span: float, right_scale: float = 1.0, left_scale: float = 1.0):
    """Landmark cloud with both eyes opened to a controllable vertical span.

    ``span`` stands in for an individual's natural eye size; the per-eye scales
    let one eye be closed independently of the other.
    """
    landmarks = synthetic_face_cloud()
    eyes = (
        (blink.RIGHT_EYE_EAR_IDX, 0.45, span * right_scale),
        (blink.LEFT_EYE_EAR_IDX, 0.55, span * left_scale),
    )
    for indices, centre_x, opening in eyes:
        outer, upper_a, upper_b, inner, lower_b, lower_a = indices
        landmarks[outer] = [centre_x - 0.05, 0.46, 0.0]
        landmarks[inner] = [centre_x + 0.05, 0.46, 0.0]
        for index in (upper_a, upper_b):
            landmarks[index] = [centre_x, 0.46 - opening * 0.1, 0.0]
        for index in (lower_a, lower_b):
            landmarks[index] = [centre_x, 0.46 + opening * 0.1, 0.0]
    return landmarks


class TestEarFallback:
    def test_ear_thresholds_adapt_to_the_subject(self, config):
        """EAR is not comparable between people, so thresholds are relative.

        Two synthetic subjects whose open-eye EAR differs by 2x -- the range
        measured across this project's real images -- must both be handled.
        """
        for scale in (0.25, 0.52):
            frames = []
            for index in range(40):
                closed = 12 <= index < 16
                span = scale * (0.30 if closed else 1.0)
                landmarks = _eyes(span)
                observation = make_observation(
                    landmarks=landmarks, timestamp=index / 30.0
                )
                observation.blendshapes = {}  # force the EAR path
                frames.append(observation)
            result = blink.evaluate(frames, config.liveness.blink)
            assert result.status is SignalStatus.PASS, (
                f"EAR fallback failed for a subject whose open-eye scale is {scale}"
            )

    def test_a_wink_does_not_count_as_a_blink(self, config):
        """One eye closing is not a blink.

        Closure is averaged across both eyes on purpose: a single mis-tracked
        eye, or a deliberate wink, must not be enough to claim proof of life.
        """
        frames = []
        for index in range(40):
            closed = 12 <= index < 16
            landmarks = _eyes(0.4, right_scale=0.3 if closed else 1.0)
            observation = make_observation(landmarks=landmarks, timestamp=index / 30.0)
            observation.blendshapes = {}
            frames.append(observation)
        result = blink.evaluate(frames, config.liveness.blink)
        assert result.metrics["blinks_detected"] == 0
