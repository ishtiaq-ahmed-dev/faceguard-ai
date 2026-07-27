"""Depth-based liveness, and an honest account of its absence.

With a real depth sensor this is the strongest anti-spoofing signal available
and the reason Face ID is hard to beat with a photograph: a flat target simply
has no facial relief to measure, and no amount of print quality invents any.
This module opens such a sensor when one is present.

On hardware without one -- which includes almost every laptop webcam, including
the machine this project was developed on -- the module reports
:class:`~faceguard.schemas.SignalStatus.UNAVAILABLE` and says so, loudly and in
the user-visible result. It does not quietly substitute something weaker. That
substitution is the single most common way a security demo ends up claiming a
defence it does not have.

**On the software proxy.** There is a tempting shortcut here: MediaPipe returns
a z coordinate per landmark, so fit a plane to the landmark cloud and call a
flat fit a photograph. This module implements that, and then refuses to let it
vote, because it was measured and it does not work. MediaPipe's z is *inferred
from a 2-D image*, never measured -- so a photograph of a face still produces a
perfectly face-shaped landmark cloud, complete with a protruding nose that is
not there. On this project's own images the live and simulated-spoof
distributions were 0.2060 +/- 0.0149 and 0.2115 +/- 0.0207, a Cohen's d of
0.30, which is no separation at all.

The measurement is still computed and reported, because a reviewer should be
able to see the number and check the claim. It carries zero weight in the
policy by default. The genuine depth-free substitute for this check is the
parallax test in :mod:`faceguard.liveness.motion`, which recovers structure
from *motion* rather than trusting a model's guess about depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..config import DepthConfig
from ..schemas import FaceObservation, ReasonCode, SignalResult, SignalStatus

SIGNAL_NAME = "depth"

LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33


@dataclass(frozen=True)
class DepthCapability:
    """What depth hardware, if any, this machine actually has."""

    available: bool
    backend: str
    detail: str


def probe_hardware(config: DepthConfig) -> DepthCapability:
    """Look for a real depth camera without requiring one to be installed.

    Intel RealSense is checked because it is the common desktop option with a
    Python SDK. The import is intentionally inside the function: ``pyrealsense2``
    is not a dependency of this project, and a machine without it must not fail
    to start.
    """
    if not config.enable_hardware:
        return DepthCapability(
            available=False,
            backend="none",
            detail="Hardware depth probing is disabled in configuration.",
        )

    try:
        import pyrealsense2 as rs  # type: ignore

        context = rs.context()
        devices = list(context.query_devices())
        if devices:
            name = devices[0].get_info(rs.camera_info.name)
            return DepthCapability(
                available=True,
                backend="realsense",
                detail=f"Intel RealSense depth camera detected: {name}",
            )
        return DepthCapability(
            available=False,
            backend="realsense",
            detail="pyrealsense2 is installed but no depth camera is connected.",
        )
    except ImportError:
        return DepthCapability(
            available=False,
            backend="none",
            detail=(
                "No depth camera SDK is installed. Depth-based liveness is "
                "unavailable on this hardware; the system is running in "
                "software-only mode, which is weaker."
            ),
        )
    except Exception as error:  # pragma: no cover - driver-specific failures
        return DepthCapability(
            available=False,
            backend="unknown",
            detail=f"Depth camera probe failed: {error}",
        )


def planarity_residual(landmarks: np.ndarray) -> float:
    """RMS distance of the 3-D landmark cloud from its best-fit plane.

    Normalised by interocular distance so it does not depend on how close the
    subject stood. A genuinely flat target would score near zero.

    Read the module docstring before using this number for anything. It is
    computed from *inferred* depth and was measured not to separate live faces
    from spoofs.
    """
    if landmarks is None or len(landmarks) < 4:
        return float("nan")

    points = np.asarray(landmarks, dtype=np.float64).copy()
    interocular = float(
        np.linalg.norm(points[RIGHT_EYE_OUTER][:2] - points[LEFT_EYE_OUTER][:2])
    )
    if interocular <= 1e-9:
        return float("nan")

    centred = points - points.mean(axis=0)
    # Smallest singular value is the spread along the plane normal, i.e. how
    # far the cloud departs from being flat.
    _, singular_values, _ = np.linalg.svd(centred, full_matrices=False)
    rms = singular_values[2] / np.sqrt(len(points))
    return float(rms / interocular)


def evaluate(
    observations: Sequence[FaceObservation],
    config: DepthConfig,
    capability: Optional[DepthCapability] = None,
) -> SignalResult:
    """Depth verdict, or an explicit statement that depth could not be checked."""
    capability = capability or probe_hardware(config)

    residuals = [
        planarity_residual(observation.landmarks)
        for observation in observations
        if observation.has_landmarks
    ]
    finite = [r for r in residuals if np.isfinite(r)]
    metrics = {
        "hardware_depth_available": float(capability.available),
        "frames_measured": float(len(finite)),
    }
    if finite:
        metrics["landmark_planarity_residual"] = float(np.median(finite))

    if not capability.available:
        if not config.allow_software_proxy:
            return SignalResult(
                name=SIGNAL_NAME,
                status=SignalStatus.UNAVAILABLE,
                reason=ReasonCode.DEPTH_UNAVAILABLE,
                detail=(
                    f"{capability.detail} The landmark-planarity proxy was measured "
                    "and does not separate live faces from spoofs (Cohen's d = 0.30), "
                    "so it is reported for transparency but not used as evidence."
                ),
                metrics=metrics,
            )
        # Operator has explicitly opted in to the proxy despite the measurement.
        if not finite:
            return SignalResult(
                name=SIGNAL_NAME,
                status=SignalStatus.UNAVAILABLE,
                reason=ReasonCode.DEPTH_UNAVAILABLE,
                detail="No landmarks were available to estimate depth.",
                metrics=metrics,
            )
        median = float(np.median(finite))
        passed = median >= config.min_planarity_residual
        return SignalResult(
            name=SIGNAL_NAME,
            status=SignalStatus.PASS if passed else SignalStatus.FAIL,
            score=min(1.0, median / max(config.min_planarity_residual, 1e-9)),
            threshold=1.0,
            reason=None if passed else ReasonCode.FLAT_GEOMETRY,
            detail=(
                f"Software depth proxy only (no depth camera): planarity residual "
                f"{median:.4f}. This proxy is known not to discriminate reliably; "
                "treat the verdict as weak evidence."
            ),
            metrics=metrics,
        )

    if not finite:
        return SignalResult(
            name=SIGNAL_NAME,
            status=SignalStatus.UNAVAILABLE,
            reason=ReasonCode.DEPTH_UNAVAILABLE,
            detail="Depth hardware present but no usable frames were captured.",
            metrics=metrics,
        )

    median = float(np.median(finite))
    passed = median >= config.min_planarity_residual
    return SignalResult(
        name=SIGNAL_NAME,
        status=SignalStatus.PASS if passed else SignalStatus.FAIL,
        score=min(1.0, median / max(config.min_planarity_residual, 1e-9)),
        threshold=1.0,
        reason=None if passed else ReasonCode.FLAT_GEOMETRY,
        detail=(
            f"{capability.detail} Measured facial relief {median:.4f} against a "
            f"{config.min_planarity_residual:.4f} minimum."
        ),
        metrics=metrics,
    )
