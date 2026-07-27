"""Image quality assessment and adaptive preprocessing.

Quality gating exists because every liveness signal downstream is measuring
something physical about the image -- texture micro-detail, eyelid position,
sub-pixel landmark motion -- and all of them degrade in the dark or out of
focus. Scoring a bad frame is worse than discarding it: a blurred print attack
and a blurred live face look equally smooth, so a low-quality frame is
evidence-free rather than evidence-against.

All thresholds and reference values come from :mod:`faceguard.config`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import QualityConfig
from .schemas import BBox, QualityScores, ReasonCode


def read_image(path: str | os.PathLike[str]) -> Optional[np.ndarray]:
    """Read an image, tolerating non-ASCII paths.

    ``cv2.imread`` goes through the ANSI file API on Windows and silently
    returns None for any path containing characters outside the system code
    page -- accented names, CJK, or the narrow no-break space macOS puts in
    screenshot filenames. Reading the bytes ourselves and decoding from memory
    sidesteps the platform entirely.

    Returns None for a missing or undecodable file, matching ``cv2.imread``.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str | os.PathLike[str], image: np.ndarray, *params: int) -> bool:
    """Write an image, tolerating non-ASCII paths. Mirrors ``cv2.imwrite``."""
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image, list(params))
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))
    return True


def standardise(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize an image to a fixed working width, preserving aspect ratio.

    Every entry point must call this before analysis, because several quality
    measurements are **not scale-invariant**. Laplacian-variance sharpness is
    the worst offender: the same photograph measured 0.084 at 3120px wide and
    1.000 at 640px, because a 12-megapixel phone image spreads its detail over
    so many pixels that per-pixel high-frequency energy collapses.

    That asymmetry is easy to ship by accident. The browser downscales before
    upload, so the web path was consistent; enrolling the same faces from
    original files rejected every one of them as LOW_QUALITY. Normalising here
    means one threshold means one thing regardless of who is calling.

    Images already at or below the target are returned untouched -- upscaling
    would invent detail the sharpness measure would then reward.
    """
    if image is None or image.size == 0:
        return image
    width = image.shape[1]
    if width <= target_width:
        return image
    scale = target_width / float(width)
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (target_width, height), interpolation=cv2.INTER_AREA)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


class QualityAssessor:
    """Scores a face crop on four independent axes, each normalised to 0..1."""

    def __init__(self, config: QualityConfig) -> None:
        self.config = config

    def sharpness(self, image: np.ndarray) -> float:
        """Laplacian variance, normalised against the configured reference.

        A defocused or motion-blurred face loses exactly the high-frequency
        detail that the texture-based spoof check relies on, so this is a gate
        on that check being meaningful at all.
        """
        variance = cv2.Laplacian(_to_gray(image), cv2.CV_64F).var()
        return float(min(variance / self.config.sharpness_reference, 1.0))

    def brightness(self, image: np.ndarray) -> float:
        """Distance of mean luma from the configured ideal exposure."""
        mean = float(np.mean(_to_gray(image)))
        ideal = self.config.ideal_brightness
        # Normalise by the larger side so both under- and over-exposure reach 0
        # at the ends of the 0..255 range rather than clipping asymmetrically.
        span = max(ideal, 255.0 - ideal)
        return float(max(0.0, 1.0 - abs(mean - ideal) / span))

    def contrast(self, image: np.ndarray) -> float:
        """Pixel standard deviation, normalised against the reference spread."""
        std = float(np.std(_to_gray(image)))
        return float(min(std / self.config.contrast_reference, 1.0))

    def size(self, bbox: BBox) -> float:
        """Face box area against the area that counts as comfortably large."""
        return float(min(bbox.area / self.config.reference_face_area, 1.0))

    def assess(self, face_crop: np.ndarray, bbox: BBox) -> QualityScores:
        """Full quality profile for one face crop."""
        if face_crop.size == 0:
            return QualityScores(0.0, 0.0, 0.0, 0.0, 0.0)

        sharpness = self.sharpness(face_crop)
        brightness = self.brightness(face_crop)
        contrast = self.contrast(face_crop)
        size = self.size(bbox)

        w = self.config.weights
        total_weight = sum(w.values()) or 1.0
        overall = (
            sharpness * w.get("sharpness", 0.0)
            + brightness * w.get("brightness", 0.0)
            + contrast * w.get("contrast", 0.0)
            + size * w.get("size", 0.0)
        ) / total_weight

        return QualityScores(
            overall=float(overall),
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            size=size,
        )

    def gate(self, scores: QualityScores) -> List[ReasonCode]:
        """Reasons this frame is unfit to contribute to a decision.

        An empty list means the frame is usable. Low light is reported
        separately from general low quality because it is the one an end user
        can actually do something about.
        """
        reasons: List[ReasonCode] = []
        if scores.brightness < self.config.min_brightness:
            reasons.append(ReasonCode.LOW_LIGHT)
        if scores.sharpness < self.config.min_sharpness:
            reasons.append(ReasonCode.LOW_QUALITY)
        elif scores.overall < self.config.min_overall:
            reasons.append(ReasonCode.LOW_QUALITY)
        return reasons


class Preprocessor:
    """Adaptive enhancement applied before recognition, never before liveness.

    This distinction matters. CLAHE and sharpening are helpful for the
    recognition embedding -- they restore detail the network expects -- but they
    also *manufacture* the high-frequency micro-texture that the spoof
    classifier uses to tell skin from paper. Enhancing a photo of a photo makes
    it look more like skin. Liveness therefore always sees the raw pixels.
    """

    @staticmethod
    def enhance_lighting(image: np.ndarray) -> np.ndarray:
        """CLAHE on the L channel of LAB, leaving colour untouched."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        merged = cv2.merge([clahe.apply(l_channel), a_channel, b_channel])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """Mild unsharp mask. Gentler than a hard 3x3 kernel, which amplifies
        sensor noise into false texture detail."""
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    @classmethod
    def for_recognition(
        cls, image: np.ndarray, scores: QualityScores, config: QualityConfig
    ) -> np.ndarray:
        """Conditionally enhance a crop before embedding extraction."""
        result = image
        if scores.brightness < config.min_brightness * 2.0:
            result = cls.enhance_lighting(result)
        if scores.sharpness < config.min_sharpness * 2.0:
            result = cls.sharpen(result)
        return result


def backlight_ratio(frame: np.ndarray, bbox: BBox) -> float:
    """Ratio of background luma to face luma.

    Values well above 1.0 mean the subject is lit from behind -- a window at
    their back -- which is the single most common real-world failure mode for
    webcam face recognition. Reported so the UI can tell the user to turn
    around rather than silently failing them.
    """
    gray = _to_gray(frame)
    box = bbox.clipped(frame.shape[1], frame.shape[0])
    face_region = gray[box.y : box.y + box.height, box.x : box.x + box.width]
    if face_region.size == 0:
        return 1.0

    mask = np.ones(gray.shape, dtype=bool)
    mask[box.y : box.y + box.height, box.x : box.x + box.width] = False
    background = gray[mask]
    if background.size == 0:
        return 1.0

    face_mean = float(np.mean(face_region))
    return float(np.mean(background) / max(face_mean, 1.0))


# Landmark indices bounding the face regions an occlusion typically covers.
# Chosen to be small clusters around each feature rather than outlines, so the
# sampled patch sits squarely on the feature.
OCCLUSION_REGIONS: dict[str, Tuple[int, ...]] = {
    "left_eye": (362, 263, 386, 374, 385, 380),
    "right_eye": (33, 133, 159, 145, 160, 144),
    "nose": (1, 4, 6, 195, 197),
    "mouth": (61, 291, 13, 14, 0, 17),
}

# How much local detail each region carries on an UNOCCLUDED face, relative to
# the median across all four regions. Measured over this project's clean images
# (n=16) with tools/measure_occlusion_baseline.py.
#
# These exist because the obvious approach -- flag any region much flatter than
# its neighbours -- does not work. Facial regions are not interchangeable: the
# eyes carry high-contrast lashes and iris detail while the bridge of the nose
# is a smooth, evenly-lit surface that measures four times flatter than the
# face median on every healthy face. Comparing regions directly flagged a nose
# on nearly every clean image. Dividing by these baselines first is what makes
# the comparison meaningful.
OCCLUSION_BASELINES: dict[str, float] = {
    "left_eye": 1.15,
    "right_eye": 1.17,
    "nose": 0.29,
    "mouth": 0.84,
}


def detect_occlusion(
    frame: np.ndarray,
    landmarks: Optional[np.ndarray],
    bbox: Optional[BBox] = None,
    ratio_threshold: float = 0.30,
) -> List[str]:
    """Name the face regions that appear to be covered.

    Sunglasses, a face mask or a hand all do the same thing to the image: they
    replace a textured region of the face with a comparatively flat one. So
    each region's local detail is measured and compared against the median
    across regions. A region far below its neighbours is covered.

    Comparing regions *against each other*, rather than against an absolute
    threshold, is what makes this work across lighting and skin tones -- a dim
    image lowers every region together and the ratios are unchanged.

    Existing behaviour without this check was measured: occluded faces were
    processed normally and then failed to match, so the user was told "no
    match" when the true problem was "take off your sunglasses". Notably it
    produced **zero** false acceptances across 42 occluded probes, so this is
    a usability fix rather than a security one.
    """
    if landmarks is None or len(landmarks) < max(
        max(indices) for indices in OCCLUSION_REGIONS.values()
    ):
        return []

    gray = _to_gray(frame)
    height, width = gray.shape[:2]
    # Sample window scales with the FACE, not the frame. Scaling it to the
    # frame makes the patch enormous when the subject is far away, so it spans
    # several features at once and the comparison between regions stops
    # meaning anything.
    reference_width = bbox.width if bbox is not None else min(height, width)
    patch = max(4, int(reference_width * 0.09))

    detail: dict[str, float] = {}
    for name, indices in OCCLUSION_REGIONS.items():
        xs = [landmarks[i][0] * width for i in indices]
        ys = [landmarks[i][1] * height for i in indices]
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        x0, x1 = max(0, cx - patch), min(width, cx + patch)
        y0, y1 = max(0, cy - patch), min(height, cy + patch)
        region = gray[y0:y1, x0:x1]
        if region.size == 0:
            continue
        detail[name] = float(cv2.Laplacian(region, cv2.CV_64F).var())

    if len(detail) < 3:
        return []
    reference = float(np.median(list(detail.values())))
    if reference <= 1e-6:
        return []

    # Normalise each region against how much detail that region is *expected*
    # to carry, then flag the ones that fall far short. A healthy face scores
    # near 1.0 everywhere after this step.
    occluded = []
    for name, value in detail.items():
        expected = OCCLUSION_BASELINES.get(name, 1.0)
        if (value / reference) / expected < ratio_threshold:
            occluded.append(name)
    return sorted(occluded)


def crop_face(
    frame: np.ndarray, bbox: BBox, margin: float = 0.0
) -> Tuple[np.ndarray, BBox]:
    """Crop a face region with an optional proportional margin.

    Returns the crop together with the box actually used, which may be smaller
    than requested where the margin ran off the edge of the frame.
    """
    height, width = frame.shape[:2]
    pad_x = int(bbox.width * margin)
    pad_y = int(bbox.height * margin)
    expanded = BBox(
        bbox.x - pad_x,
        bbox.y - pad_y,
        bbox.width + 2 * pad_x,
        bbox.height + 2 * pad_y,
    ).clipped(width, height)
    crop = frame[
        expanded.y : expanded.y + expanded.height,
        expanded.x : expanded.x + expanded.width,
    ]
    return crop, expanded
