#!/usr/bin/env python3
"""Measure the planar-parallax discriminator that gates motion liveness.

Produces the numbers quoted in ``config/default.yaml`` and the README, so the
threshold can be re-derived rather than taken on trust.

Two populations are measured:

*Flat target, end to end.* A source photograph is warped by the homography that
corresponds to rotating a flat sheet in front of the camera, the real detector
is run on the warped images, and parallax is measured from the landmarks it
actually returns. This is a faithful simulation of an attacker tilting a
printed photo, and every step after the warp is the production code path.

*Real facial geometry, landmark level.* The 3-D landmark cloud recovered from a
real face is rotated and re-projected, and parallax is measured on the result.

The asymmetry is deliberate and worth stating plainly: the flat-target figure
is end-to-end and trustworthy, while the real-face figure is simulated from
MediaPipe's *inferred* z coordinates and is therefore an estimate of what a
genuine head turn would produce, not a measurement of one. Only a live capture
settles that side. Run ``tools/calibrate_motion.py`` on a real camera to
replace it with measured values.

Usage::

    python tools/measure_parallax.py --images "Data/**/*.jpg"
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faceguard.config import load_config
from faceguard.detect import FaceAnalyzer
from faceguard.quality import read_image
from faceguard.liveness.motion import (
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    homography_residual,
)


def plane_rotation_homography(
    width: int, height: int, yaw_deg: float, focal_scale: float = 1.6
) -> np.ndarray:
    """Homography produced by rotating a flat plane about its vertical axis.

    This is the exact transform a printed photograph undergoes when an attacker
    tilts it toward the camera, which is why a flat target can never produce
    parallax: its entire apparent motion is one homography by construction.
    """
    focal = focal_scale * max(width, height)
    centre_x, centre_y = width / 2.0, height / 2.0
    intrinsics = np.array(
        [[focal, 0, centre_x], [0, focal, centre_y], [0, 0, 1]], dtype=np.float64
    )
    theta = np.radians(yaw_deg)
    rotation = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    # Plane at distance `focal` along z, with normal (0, 0, 1).
    normal = np.array([[0.0], [0.0], [1.0]])
    translation = np.zeros((3, 1))
    homography = rotation + (translation @ normal.T) / focal
    return intrinsics @ homography @ np.linalg.inv(intrinsics)


def rotate_y(degrees: float) -> np.ndarray:
    theta = np.radians(degrees)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])


def normalised_landmarks(observation, width: int, height: int) -> np.ndarray | None:
    """Landmarks in pixels, scaled by interocular distance."""
    if observation is None or not observation.has_landmarks:
        return None
    points = observation.landmarks[:, :2].astype(np.float64).copy()
    points[:, 0] *= width
    points[:, 1] *= height
    interocular = np.linalg.norm(points[RIGHT_EYE_OUTER] - points[LEFT_EYE_OUTER])
    if interocular <= 1e-9:
        return None
    return points / interocular


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        default="Data/**/*.jpg",
        help="Glob of source face images (never committed to the repository).",
    )
    parser.add_argument("--angles", default="10,15,20", help="Yaw angles in degrees.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    angles = [float(a) for a in args.angles.split(",")]
    paths = sorted(glob.glob(args.images, recursive=True))
    if not paths:
        print(f"No images matched {args.images!r}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    flat_rows: list[float] = []
    real_rows: list[float] = []

    with FaceAnalyzer(config) as analyzer:
        print(f"{'image':30} {'yaw':>5} {'flat /deg':>11} {'real /deg':>11}")
        print("-" * 60)
        for path in paths:
            image = read_image(path)
            if image is None:
                continue
            scale = 640.0 / image.shape[1]
            image = cv2.resize(image, (640, int(image.shape[0] * scale)))
            height, width = image.shape[:2]

            base = analyzer.analyze(image).primary
            base_points = normalised_landmarks(base, width, height)
            if base_points is None:
                continue

            # Real facial geometry, from the recovered 3-D landmark cloud.
            cloud = base.landmarks.astype(np.float64).copy()
            cloud[:, 0] *= width
            cloud[:, 1] *= height
            cloud[:, 2] *= width
            cloud -= cloud.mean(axis=0)
            interocular = np.linalg.norm(cloud[RIGHT_EYE_OUTER][:2] - cloud[LEFT_EYE_OUTER][:2])

            for yaw in angles:
                # --- flat target: warp the actual image, re-run detection ----
                warped = cv2.warpPerspective(
                    image,
                    plane_rotation_homography(width, height, yaw),
                    (width, height),
                    borderMode=cv2.BORDER_REPLICATE,
                )
                warped_points = normalised_landmarks(
                    analyzer.analyze(warped).primary, width, height
                )
                flat_per_degree = float("nan")
                if warped_points is not None:
                    flat_per_degree = (
                        homography_residual(base_points, warped_points) / yaw
                    )

                # --- real geometry: rotate the 3-D cloud, re-project ---------
                rotated = (rotate_y(yaw) @ cloud.T).T[:, :2] / interocular
                real_per_degree = (
                    homography_residual(cloud[:, :2] / interocular, rotated) / yaw
                )

                if np.isfinite(flat_per_degree):
                    flat_rows.append(flat_per_degree)
                real_rows.append(real_per_degree)
                print(
                    f"{Path(path).name[:30]:30} {yaw:5.0f} "
                    f"{flat_per_degree:11.5f} {real_per_degree:11.5f}"
                )

    flat = np.array(flat_rows)
    real = np.array(real_rows)
    print("\n" + "=" * 60)
    print("FLAT TARGET (end-to-end: warped image -> real detector)")
    print(f"  mean {flat.mean():.5f}  std {flat.std():.5f}  max {flat.max():.5f}")
    print("REAL FACIAL GEOMETRY (simulated from inferred 3-D landmarks)")
    print(f"  mean {real.mean():.5f}  std {real.std():.5f}  min {real.min():.5f}")
    print(f"\nconfigured threshold: {config.liveness.motion.min_parallax_per_degree:.5f}")
    print(f"  margin above flat max : {config.liveness.motion.min_parallax_per_degree / max(flat.max(), 1e-9):.1f}x")
    print(f"  margin below real min : {real.min() / max(config.liveness.motion.min_parallax_per_degree, 1e-9):.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
