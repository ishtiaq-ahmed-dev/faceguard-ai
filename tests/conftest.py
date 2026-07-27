"""Shared fixtures and synthetic data factories.

Every test in this suite runs on a fresh clone with **no face images present**.
That is deliberate: the dataset is personal data and is not in the repository,
so a suite that needed it would be a suite nobody else could run.

The core geometric claims are testable without photographs anyway. A synthetic
3-D point cloud with facial relief, and the same cloud flattened, is enough to
verify that the parallax check distinguishes a head from a photograph -- and it
tests the claim more directly than a picture would, because the ground truth is
exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceguard.config import load_config
from faceguard.schemas import BBox, FaceObservation, QualityScores

LANDMARK_COUNT = 478
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33


@pytest.fixture(scope="session")
def config():
    """The real default configuration, so tests exercise shipped thresholds."""
    return load_config()


def synthetic_face_cloud(relief: float = 0.35, seed: int = 0) -> np.ndarray:
    """A 478-point cloud with plausible facial depth relief.

    Points are spread over a face-shaped ellipse in x and y. The z channel is a
    radial bump: maximum at the centre (the nose) falling off toward the edges
    (the cheeks and ears), which is the depth profile the parallax check relies
    on. ``relief=0`` produces a perfectly flat plane -- a photograph.

    Coordinates are normalised to roughly 0..1 to match MediaPipe's output.
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, LANDMARK_COUNT)
    radii = np.sqrt(rng.uniform(0, 1, LANDMARK_COUNT))
    x = 0.5 + 0.16 * radii * np.cos(angles)
    y = 0.5 + 0.22 * radii * np.sin(angles)
    z = relief * 0.16 * (1.0 - radii**2)

    cloud = np.stack([x, y, z], axis=1).astype(np.float32)
    # Pin the eye-corner landmarks so interocular normalisation is stable and
    # every distance in the modules under test has a fixed reference.
    cloud[RIGHT_EYE_OUTER] = [0.42, 0.46, relief * 0.05]
    cloud[LEFT_EYE_OUTER] = [0.58, 0.46, relief * 0.05]
    return cloud


def rotate_cloud(cloud: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate a landmark cloud about its vertical axis, through its centroid."""
    theta = np.radians(yaw_deg)
    cos, sin = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    centre = cloud.mean(axis=0)
    return ((cloud - centre) @ rotation.T + centre).astype(np.float32)


def pose_matrix(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """A 4x4 facial transformation matrix for the given orientation."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    rotation_y = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )
    rotation_x = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_y @ rotation_x
    return matrix


def make_observation(
    landmarks: np.ndarray | None = None,
    blink: float = 0.05,
    timestamp: float = 0.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    detection_confidence: float = 0.95,
    quality: float = 0.8,
) -> FaceObservation:
    """Build one FaceObservation without needing a camera or an image."""
    return FaceObservation(
        bbox=BBox(100, 100, 200, 200),
        detection_confidence=detection_confidence,
        landmarks=landmarks if landmarks is not None else synthetic_face_cloud(),
        blendshapes={"eyeBlinkLeft": blink, "eyeBlinkRight": blink},
        pose_matrix=pose_matrix(yaw_deg, pitch_deg),
        quality=QualityScores(quality, quality, quality, quality, quality),
        timestamp=timestamp,
        frame_shape=(480, 640),
    )


def blink_sequence(
    closed_frames: int,
    total_frames: int = 30,
    fps: float = 30.0,
    closed_value: float = 0.9,
    open_value: float = 0.05,
    start_at: int = 10,
) -> list[FaceObservation]:
    """A frame sequence containing exactly one eye closure of a given length."""
    observations = []
    for index in range(total_frames):
        closed = start_at <= index < start_at + closed_frames
        observations.append(
            make_observation(
                blink=closed_value if closed else open_value,
                timestamp=index / fps,
            )
        )
    return observations


def noisy_crop(size: int = 160, seed: int = 0, smooth: bool = False) -> np.ndarray:
    """A synthetic BGR image patch for feature-extraction tests."""
    rng = np.random.default_rng(seed)
    image = rng.integers(60, 200, (size, size, 3), dtype=np.uint8)
    if smooth:
        import cv2

        image = cv2.GaussianBlur(image, (0, 0), sigmaX=4.0)
    return image
