"""Texture and frequency analysis: is this surface skin, or paper, or a screen?

**What this catches:** printed photographs and screen replays, judged on the
physical properties of the surface rather than on anything the subject does. It
is the only signal here that works on a single frame, and the only one that
does not care whether the attacker cooperates.

**What this does not catch:** a high-quality 3-D mask made of skin-like
material, or any attack rendered at a resolution and print quality high enough
that the micro-texture survives. It is also the signal most sensitive to
camera hardware -- see the calibration warning at the bottom of this docstring.

**Why the physics works.** Three things happen to a face between the real world
and a re-captured copy of it:

1. *Detail is destroyed.* Printing and re-photographing is a lossy round trip.
   Skin has fine, irregular micro-texture -- pores, fine hair, subsurface
   mottling. Paper smooths it into ink dots; a screen replaces it with a pixel
   grid. Local Binary Patterns measure exactly this kind of local micro-
   structure, and have been the standard baseline for print-attack detection
   since Maatta et al. (2011).
2. *New periodic structure appears.* A screen has a physical pixel lattice, and
   photographing one produces moire interference -- energy concentrated at
   specific spatial frequencies that no real face emits. The radial power
   spectrum makes this visible.
3. *Colour and reflectance shift.* Skin is translucent and scatters light
   beneath the surface; paper and glass are not and do not. Re-captured faces
   show a compressed chroma range and characteristic broad specular highlights
   where a glossy print or screen reflects the room.

Rather than hand-tune thresholds for each of these, the module extracts them as
a feature vector and hands the decision to a small logistic-regression model.
Logistic regression is chosen deliberately over something stronger: with a
training set this size a high-capacity model would memorise it, and the linear
coefficients can be read directly to see which features actually carry the
decision (``tools/train_spoof_classifier.py --explain``).

**Calibration warning.** These features are sensitive to the camera that
captured them. A model trained on one webcam's noise and colour response will
not transfer cleanly to another, and a model trained on *simulated* attacks
will not transfer cleanly to real ones. Read the honest metrics in the README
before believing any number this module produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import TextureConfig
from ..schemas import ReasonCode, SignalResult, SignalStatus

SIGNAL_NAME = "texture"

# Number of Local Binary Pattern bins: 58 rotation-variant uniform patterns for
# an 8-neighbour ring, plus one catch-all bin for everything non-uniform.
LBP_BINS = 59


def _uniform_lbp_lut() -> np.ndarray:
    """Map each 8-bit LBP code to a uniform-pattern bin.

    A pattern is "uniform" when the circular bit string contains at most two
    0->1 or 1->0 transitions. Uniform patterns correspond to edges, corners and
    flat spots -- the structures that actually carry texture information --
    while non-uniform codes are dominated by noise and are pooled into a single
    bin, which keeps the histogram compact and far less prone to overfitting.
    """
    lut = np.empty(256, dtype=np.uint8)
    next_bin = 0
    for code in range(256):
        rotated = ((code >> 1) | (code << 7)) & 0xFF
        transitions = bin(code ^ rotated).count("1")
        if transitions <= 2:
            lut[code] = next_bin
            next_bin += 1
        else:
            lut[code] = LBP_BINS - 1
    return lut


_LBP_LUT = _uniform_lbp_lut()

# Clockwise from top-left, so bit order is consistent between scales.
_LBP_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1),
)


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """Normalised uniform-LBP histogram of a greyscale image.

    Implemented with array shifts rather than a Python loop over pixels: this
    runs on every frame of a live session, so a per-pixel loop would dominate
    the latency budget.
    """
    if gray.size == 0 or min(gray.shape[:2]) < 3:
        return np.zeros(LBP_BINS, dtype=np.float64)

    padded = gray.astype(np.int16)
    height, width = padded.shape
    centre = padded[1 : height - 1, 1 : width - 1]
    codes = np.zeros(centre.shape, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(_LBP_OFFSETS):
        neighbour = padded[1 + dy : height - 1 + dy, 1 + dx : width - 1 + dx]
        codes |= ((neighbour >= centre).astype(np.uint8) << bit)

    binned = _LBP_LUT[codes]
    histogram = np.bincount(binned.ravel(), minlength=LBP_BINS).astype(np.float64)
    total = histogram.sum()
    return histogram / total if total > 0 else histogram


def radial_spectrum(gray: np.ndarray, bands: int = 8) -> np.ndarray:
    """Energy of the 2-D power spectrum in concentric frequency bands.

    A printed photo loses high-frequency energy relative to real skin. A screen
    adds it back at specific frequencies, as moire. Both show up as a change in
    the shape of this profile. Bands are log-scaled in energy because spectral
    magnitude spans several orders and the interesting differences live in the
    quiet high-frequency tail.
    """
    if gray.size == 0:
        return np.zeros(bands + 1, dtype=np.float64)

    windowed = gray.astype(np.float64)
    # Hann window in both axes: without it, the discontinuity at the crop edge
    # leaks broadband energy across the whole spectrum and swamps the signal.
    windowed = (
        windowed
        * np.hanning(windowed.shape[0])[:, None]
        * np.hanning(windowed.shape[1])[None, :]
    )
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(windowed)))

    height, width = spectrum.shape
    cy, cx = height / 2.0, width / 2.0
    yy, xx = np.ogrid[:height, :width]
    radius = np.hypot(yy - cy, xx - cx)
    max_radius = radius.max() or 1.0
    normalised = radius / max_radius

    energies = np.empty(bands, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bands + 1)
    for index in range(bands):
        mask = (normalised >= edges[index]) & (normalised < edges[index + 1])
        energies[index] = spectrum[mask].mean() if mask.any() else 0.0

    total = energies.sum() or 1.0
    profile = np.log1p(energies / total)
    # Ratio of the top half of the spectrum to the bottom: one scalar that
    # captures "how much fine detail survived".
    high_ratio = energies[bands // 2 :].sum() / total
    return np.concatenate([profile, [high_ratio]])


def colour_statistics(crop_bgr: np.ndarray) -> np.ndarray:
    """Per-channel mean and standard deviation in HSV and YCrCb.

    Skin is translucent: light enters, scatters and leaves, which produces a
    characteristic spread of chroma. A print or a screen is an opaque emitter
    or reflector with a narrower gamut, so both the means and especially the
    spreads shift.
    """
    if crop_bgr.size == 0:
        return np.zeros(12, dtype=np.float64)
    features: List[float] = []
    for conversion in (cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2YCrCb):
        converted = cv2.cvtColor(crop_bgr, conversion).astype(np.float64) / 255.0
        for channel in cv2.split(converted):
            features.append(float(channel.mean()))
            features.append(float(channel.std()))
    return np.array(features, dtype=np.float64)


def specular_features(crop_bgr: np.ndarray) -> np.ndarray:
    """Extent and intensity of blown-out, desaturated highlights.

    A glossy print or a glass screen reflects the room as a broad, bright,
    low-saturation patch. Skin produces specular highlights too, but smaller
    and tighter, because it is not flat.
    """
    if crop_bgr.size == 0:
        return np.zeros(3, dtype=np.float64)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float64)
    value = hsv[:, :, 2].astype(np.float64)
    specular = (value > 235) & (saturation < 40)
    brightest = np.percentile(value, 99) / 255.0
    return np.array(
        [
            float(specular.mean()),
            float(brightest),
            float(value.std() / 255.0),
        ],
        dtype=np.float64,
    )


def extract_features(crop_bgr: np.ndarray, crop_size: int = 128) -> np.ndarray:
    """Full spoof-detection feature vector for one face crop.

    The crop is resized to a fixed size first so that features describe the
    surface rather than how close the subject was standing -- otherwise
    "stood further away" and "held up a small photo" become the same
    measurement.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(feature_dimension(), dtype=np.float64)

    resized = cv2.resize(
        crop_bgr, (crop_size, crop_size), interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Half resolution approximates a larger LBP radius, capturing coarser
    # texture, at a fraction of the cost of true multi-radius sampling.
    gray_half = cv2.resize(
        gray, (crop_size // 2, crop_size // 2), interpolation=cv2.INTER_AREA
    )

    return np.concatenate(
        [
            lbp_histogram(gray),
            lbp_histogram(gray_half),
            radial_spectrum(gray),
            colour_statistics(resized),
            specular_features(resized),
        ]
    )


def feature_dimension() -> int:
    """Length of the feature vector, for allocating arrays and sanity checks."""
    return LBP_BINS * 2 + 9 + 12 + 3


def feature_names() -> List[str]:
    """Human-readable name per feature, for model explanation output."""
    names = [f"lbp_fine_{i}" for i in range(LBP_BINS)]
    names += [f"lbp_coarse_{i}" for i in range(LBP_BINS)]
    names += [f"spectrum_band_{i}" for i in range(8)] + ["spectrum_high_ratio"]
    for space in ("hsv", "ycrcb"):
        for channel in range(3):
            names += [f"{space}_c{channel}_mean", f"{space}_c{channel}_std"]
    names += ["specular_fraction", "brightest_p99", "value_std"]
    return names


class SpoofClassifier:
    """Thin wrapper over the trained scikit-learn pipeline.

    Kept deliberately small: loading, predicting and reporting availability.
    Training lives in ``tools/train_spoof_classifier.py`` so that nothing in
    the serving path can accidentally fit a model.
    """

    def __init__(self, model: object, metadata: Optional[dict] = None) -> None:
        self.model = model
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: Path) -> Optional["SpoofClassifier"]:
        """Load a trained model, or return None if it is not present.

        Returning None rather than raising is intentional: a missing model is a
        deployment state the system must handle explicitly, by reporting the
        texture signal UNAVAILABLE and refusing the attempt, rather than by
        crashing.
        """
        if not Path(path).is_file():
            return None
        try:
            import joblib

            payload = joblib.load(path)
        except Exception:
            return None
        if isinstance(payload, dict) and "model" in payload:
            return cls(payload["model"], payload.get("metadata"))
        return cls(payload)

    def live_probability(self, features: np.ndarray) -> np.ndarray:
        """P(live) for a batch of feature vectors, shape ``(n_samples,)``."""
        features = np.atleast_2d(features)
        probabilities = self.model.predict_proba(features)
        # Class 1 is "live" by the convention established in the training tool.
        return probabilities[:, 1]


def evaluate(
    crops: Sequence[np.ndarray],
    config: TextureConfig,
    classifier: Optional[SpoofClassifier],
) -> SignalResult:
    """Decide whether the surface shown to the camera looks like real skin.

    Scores every supplied crop independently and requires a configurable
    majority to look live. Per-frame voting rather than averaging means one
    unusually good frame cannot carry an attempt: an attacker who catches a
    single lucky reflection still fails the fraction test.
    """
    if classifier is None:
        return SignalResult(
            name=SIGNAL_NAME,
            status=SignalStatus.UNAVAILABLE,
            reason=ReasonCode.LIVENESS_UNAVAILABLE,
            detail=(
                "No trained spoof classifier is present. Train one with "
                "tools/train_spoof_classifier.py. The attempt is refused rather "
                "than falling back to a weaker check."
            ),
        )

    usable = [crop for crop in crops if crop is not None and crop.size > 0]
    if not usable:
        return SignalResult(
            name=SIGNAL_NAME,
            status=SignalStatus.UNAVAILABLE,
            reason=ReasonCode.LIVENESS_UNAVAILABLE,
            detail="No face crops were available to analyse.",
        )

    features = np.stack(
        [extract_features(crop, config.crop_size) for crop in usable]
    )
    probabilities = classifier.live_probability(features)
    live_flags = probabilities >= config.live_threshold
    live_ratio = float(np.mean(live_flags))
    mean_probability = float(np.mean(probabilities))

    metrics = {
        "mean_live_probability": mean_probability,
        "median_live_probability": float(np.median(probabilities)),
        "min_live_probability": float(np.min(probabilities)),
        "live_frame_ratio": live_ratio,
        "frames_scored": float(len(usable)),
    }

    if live_ratio < config.min_live_frame_ratio:
        return SignalResult(
            name=SIGNAL_NAME,
            status=SignalStatus.FAIL,
            score=mean_probability,
            threshold=config.live_threshold,
            reason=ReasonCode.TEXTURE_SPOOF_SUSPECTED,
            detail=(
                f"Only {live_ratio:.0%} of frames showed skin-like surface texture "
                f"(need {config.min_live_frame_ratio:.0%}); mean live probability "
                f"{mean_probability:.2f}. The surface resembles a printed photo or "
                "a display."
            ),
            metrics=metrics,
        )

    return SignalResult(
        name=SIGNAL_NAME,
        status=SignalStatus.PASS,
        score=mean_probability,
        threshold=config.live_threshold,
        detail=(
            f"{live_ratio:.0%} of frames showed skin-like micro-texture and "
            f"spectral profile (mean live probability {mean_probability:.2f})."
        ),
        metrics=metrics,
    )
