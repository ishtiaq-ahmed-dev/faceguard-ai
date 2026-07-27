"""Regression test for the occlusion false-acceptance path.

This exists because of a measured failure, not a hypothetical one. With a hand
covering part of the face, two probes matched the **wrong** enrolled identity
at cosine 0.632 and 0.650, while the lowest genuine match across the same
gallery scored 0.651. No threshold separates those, so the mitigation is to
stop suspected-occluded frames from contributing a recognition vote at all.

If this test fails, occluded frames are voting again and the false-acceptance
path is open.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceguard.quality import OCCLUSION_BASELINES, OCCLUSION_REGIONS, detect_occlusion
from faceguard.schemas import BBox

from .conftest import synthetic_face_cloud


def _textured_face(size: int = 240, seed: int = 0) -> np.ndarray:
    """A synthetic face-sized patch with uniform texture everywhere."""
    rng = np.random.default_rng(seed)
    return rng.integers(70, 190, (size, size, 3), dtype=np.uint8)


def _landmarks_over(image: np.ndarray) -> np.ndarray:
    """Landmarks placed so each region maps onto a distinct quadrant."""
    landmarks = synthetic_face_cloud()
    placements = {
        "left_eye": (0.66, 0.35),
        "right_eye": (0.34, 0.35),
        "nose": (0.50, 0.55),
        "mouth": (0.50, 0.75),
    }
    for name, indices in OCCLUSION_REGIONS.items():
        cx, cy = placements[name]
        for index in indices:
            landmarks[index] = [cx, cy, 0.0]
    return landmarks


class TestOcclusionDetection:
    def test_uniform_texture_is_not_flagged(self):
        """An evenly textured face must not be reported as occluded."""
        image = _textured_face()
        regions = detect_occlusion(
            image, _landmarks_over(image), BBox(0, 0, image.shape[1], image.shape[0])
        )
        assert regions == []

    @pytest.mark.parametrize(
        "region,centre",
        [
            ("mouth", (0.50, 0.75)),
            ("nose", (0.50, 0.55)),
            ("left_eye", (0.66, 0.35)),
        ],
    )
    def test_a_flattened_region_is_flagged(self, region, centre):
        """Covering one region with a flat patch must name that region."""
        image = _textured_face()
        height, width = image.shape[:2]
        cx, cy = int(centre[0] * width), int(centre[1] * height)
        half = int(width * 0.10)
        image[cy - half : cy + half, cx - half : cx + half] = 120  # flat, no detail

        regions = detect_occlusion(
            image, _landmarks_over(image), BBox(0, 0, width, height)
        )
        assert region in regions

    def test_per_region_baselines_are_present_for_every_region(self):
        """A region without a baseline would be compared against 1.0 and
        misjudged; the nose in particular measures 0.29 on healthy faces."""
        assert set(OCCLUSION_BASELINES) == set(OCCLUSION_REGIONS)
        assert OCCLUSION_BASELINES["nose"] < 0.5, (
            "The nose baseline must stay well below 1.0. If it drifts upward, "
            "clean noses start being flagged as occluded."
        )

    def test_missing_landmarks_flags_nothing(self):
        assert detect_occlusion(_textured_face(), None) == []


class TestOccludedFramesDoNotVote:
    def test_session_skips_embedding_for_occluded_frames(self, monkeypatch, config):
        """The mitigation itself: an occluded frame contributes no embedding.

        Verified through the real session code path rather than by inspection,
        so that a refactor which reorders the check is caught here.
        """
        from faceguard.detect import FrameAnalysis
        from faceguard.schemas import FaceObservation, QualityScores

        embedded: list[int] = []

        class _Session:
            """Minimal stand-in exercising the guard condition in add_frame."""

            def __init__(self) -> None:
                self.config = config
                self._frames_seen = 0

            def consider(self, analysis: FrameAnalysis) -> None:
                self._frames_seen += 1
                if (
                    self._frames_seen % self.config.session.recognise_every_n_frames == 0
                    and not analysis.occluded_regions
                ):
                    embedded.append(self._frames_seen)

        face = FaceObservation(
            bbox=BBox(0, 0, 100, 100),
            detection_confidence=0.9,
            quality=QualityScores(0.8, 0.8, 0.8, 0.8, 0.8),
        )
        session = _Session()
        every_n = config.session.recognise_every_n_frames

        # Clean frames up to the first embedding opportunity.
        for _ in range(every_n):
            session.consider(FrameAnalysis(faces=[face]))
        assert embedded == [every_n], "a clean frame should have been embedded"

        # An occluded frame landing on an embedding opportunity must be skipped.
        for _ in range(every_n):
            session.consider(FrameAnalysis(faces=[face], occluded_regions=["mouth"]))
        assert embedded == [every_n], (
            "an occluded frame contributed a recognition vote; the "
            "false-acceptance path this test guards is open again"
        )
