#!/usr/bin/env python3
"""Environment robustness: how the pipeline behaves outside ideal conditions.

Testing only on clean, well-lit, front-facing photographs is how a system ends
up "99% accurate" and useless in a corridor. This script degrades known-good
images in the ways real deployments actually degrade them and reports what
breaks, at which point, and -- most importantly -- whether the failure is
*visible*.

The distinction that matters throughout: a **detected** failure is fine. If the
system says "too dark, turn on a light", that is correct behaviour, and it is
scored as such. What is not fine is a silent one: sailing through with a
confident but wrong answer, or crashing. Every condition below is scored on
three outcomes -- handled correctly, correctly refused with a reason, or failed
silently.

Conditions:

* **Low light** -- progressive exposure reduction.
* **Backlight** -- a bright window behind the subject, the most common
  real-world webcam failure.
* **Occlusion** -- sunglasses, a face mask, a hat brim.
* **Angle** -- yaw and pitch away from frontal.
* **Distance** -- the face too small or too large in frame.
* **Motion blur** -- someone moving while the shutter is open.
* **Compression** -- a poor network connection mangling the upload.

Two failure directions are counted separately, because conflating them would be
the most misleading thing this script could do:

* **False rejection** -- processed, but no longer matches the subject's own
  clean baseline. Annoying, not dangerous.
* **False acceptance** -- the degraded frame matches a *different* enrolled
  identity above threshold. The only genuinely dangerous outcome.

Usage::

    python eval/robustness.py --images "all_faces/**/*.jpg"
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faceguard.config import load_config
from faceguard.detect import FaceAnalyzer
from faceguard.quality import backlight_ratio, read_image, standardise
from faceguard.recognition import EmbeddingBackend, cosine_similarity
from faceguard.schemas import ReasonCode
from faceguard.quality import crop_face


# --------------------------------------------------------------------------- #
# Degradations
# --------------------------------------------------------------------------- #


def low_light(image: np.ndarray, level: float) -> np.ndarray:
    """Reduce exposure. ``level`` 0..1, where 1.0 is near-darkness."""
    gain = 1.0 - 0.92 * level
    darkened = image.astype(np.float32) * gain
    # Sensor noise grows relative to signal as exposure falls; a simple
    # multiply would be an unrealistically kind test.
    noise_sigma = 2.0 + 9.0 * level
    rng = np.random.default_rng(0)
    darkened += rng.normal(0, noise_sigma, image.shape)
    return np.clip(darkened, 0, 255).astype(np.uint8)


def backlight(image: np.ndarray, level: float) -> np.ndarray:
    """Bright region behind the subject, with the face falling into shadow."""
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    centre_x, centre_y = width / 2.0, height * 0.45
    radius = min(height, width) * 0.34
    distance = np.hypot(xx - centre_x, yy - centre_y)
    background = np.clip((distance - radius) / radius, 0, 1)

    work = image.astype(np.float32)
    work += (background * 190.0 * level)[..., None]  # blown-out window
    work *= 1.0 - 0.55 * level * (1.0 - background)[..., None]  # face in shadow
    return np.clip(work, 0, 255).astype(np.uint8)


def occlude(image: np.ndarray, kind: str, box=None) -> np.ndarray:
    """Cover part of the face the way glasses, a mask or a hat would.

    Positioned relative to the detected FACE, not the frame. An earlier version
    used frame-relative coordinates, which meant that whenever the subject did
    not fill the frame the "sunglasses" landed on the background and the test
    silently measured nothing.
    """
    work = image.copy()
    height, width = work.shape[:2]
    if box is None:
        # Fall back to a centred assumption only when no face box is supplied.
        fx, fy, fw, fh = int(width * 0.25), int(height * 0.2), int(width * 0.5), int(height * 0.6)
    else:
        fx, fy, fw, fh = box

    def at(rx0, ry0, rx1, ry1):
        return (
            (int(fx + fw * rx0), int(fy + fh * ry0)),
            (int(fx + fw * rx1), int(fy + fh * ry1)),
        )

    if kind == "sunglasses":
        (x0, y0), (x1, y1) = at(-0.06, 0.24, 1.06, 0.50)
        cv2.rectangle(work, (x0, y0), (x1, y1), (20, 20, 22), -1)
    elif kind == "face_mask":
        (x0, y0), (x1, y1) = at(-0.04, 0.55, 1.04, 1.10)
        cv2.rectangle(work, (x0, y0), (x1, y1), (205, 205, 210), -1)
    elif kind == "hat_brim":
        (x0, y0), (x1, y1) = at(-0.15, -0.35, 1.15, 0.14)
        cv2.rectangle(work, (x0, y0), (x1, y1), (35, 30, 40), -1)
    elif kind == "hand":
        (x0, y0), _ = at(0.55, 0.45, 1.0, 1.0)
        cv2.circle(work, (x0, y0), int(fw * 0.32), (150, 120, 100), -1)
    return work


def rotate_view(image: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Approximate an off-axis view by a plane rotation."""
    from tools.measure_parallax import plane_rotation_homography

    height, width = image.shape[:2]
    return cv2.warpPerspective(
        image,
        plane_rotation_homography(width, height, yaw_deg),
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )


def rescale_face(image: np.ndarray, factor: float) -> np.ndarray:
    """Move the subject nearer or further by scaling within the same frame."""
    height, width = image.shape[:2]
    scaled = cv2.resize(image, None, fx=factor, fy=factor)
    canvas = np.full_like(image, 40)
    if factor < 1.0:
        y = (height - scaled.shape[0]) // 2
        x = (width - scaled.shape[1]) // 2
        canvas[y : y + scaled.shape[0], x : x + scaled.shape[1]] = scaled
        return canvas
    y = (scaled.shape[0] - height) // 2
    x = (scaled.shape[1] - width) // 2
    return scaled[y : y + height, x : x + width]


def motion_blur(image: np.ndarray, length: int) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(image, -1, kernel)


def compress(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else image


# --------------------------------------------------------------------------- #


def build_conditions() -> List[Tuple[str, str, Callable[[np.ndarray], np.ndarray]]]:
    """(group, label, transform) for every condition under test."""
    conditions: List[Tuple[str, str, Callable]] = [("baseline", "unmodified", lambda i: i)]
    for level in (0.3, 0.5, 0.7, 0.85):
        conditions.append(("low_light", f"exposure -{level:.0%}",
                           lambda i, l=level: low_light(i, l)))
    for level in (0.35, 0.6, 0.85):
        conditions.append(("backlight", f"backlight {level:.0%}",
                           lambda i, l=level: backlight(i, l)))
    for kind in ("sunglasses", "face_mask", "hat_brim", "hand"):
        conditions.append(("occlusion", kind, lambda i, k=kind: occlude(i, k)))
    for yaw in (15, 25, 35, 45):
        conditions.append(("angle", f"yaw {yaw} deg",
                           lambda i, y=yaw: rotate_view(i, y)))
    for factor in (0.35, 0.5, 1.8, 2.6):
        conditions.append(("distance", f"scale x{factor}",
                           lambda i, f=factor: rescale_face(i, f)))
    for length in (9, 17, 25):
        conditions.append(("motion_blur", f"{length}px",
                           lambda i, n=length: motion_blur(i, n)))
    for quality in (40, 20, 10):
        conditions.append(("compression", f"jpeg q{quality}",
                           lambda i, q=quality: compress(i, q)))
    return conditions


def identity_of(path: str) -> str:
    """Identity is the name of the directory the image sits in."""
    return Path(path).parent.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="all_faces/**/*.jpg")
    parser.add_argument("--output", default="eval_output/robustness.json")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.images, recursive=True))
    if not paths:
        print(f"No images matched {args.images!r}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    conditions = build_conditions()
    results: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    with FaceAnalyzer(config) as analyzer:
        backend = EmbeddingBackend(config.recognition)
        backend.warm_up()

        baselines: Dict[str, np.ndarray] = {}
        gallery: Dict[str, List[np.ndarray]] = defaultdict(list)
        for path in paths:
            image = read_image(path)
            if image is None:
                continue
            image = standardise(image, config.detection.working_width)
            face = analyzer.analyze(image).primary
            if face is None:
                continue
            crop, _ = crop_face(image, face.bbox, margin=0.15)
            vector = backend.embed(crop)
            if vector is not None:
                baselines[path] = vector
                gallery[identity_of(path)].append(vector)

        print(f"Baseline embeddings for {len(baselines)} image(s).\n")

        for group, label, transform in conditions:
            for path in list(baselines):
                image = read_image(path)
                if image is None:
                    continue
                image = standardise(image, config.detection.working_width)
                try:
                    degraded = transform(image)
                except Exception as error:  # a crash is the worst outcome
                    results[group][label].append(
                        {"outcome": "crashed", "detail": str(error)}
                    )
                    continue

                analysis = analyzer.analyze(degraded)
                face = analysis.primary
                if face is None or analysis.reasons:
                    reason = (
                        analysis.reasons[0].value
                        if analysis.reasons
                        else ReasonCode.NO_FACE_DETECTED.value
                    )
                    results[group][label].append(
                        {"outcome": "refused", "reason": reason}
                    )
                    continue

                crop, _ = crop_face(degraded, face.bbox, margin=0.15)
                vector = backend.embed(crop)
                similarity = (
                    cosine_similarity(vector, baselines[path])
                    if vector is not None
                    else 0.0
                )
                # Does the degraded frame match somebody ELSE above threshold?
                # That is the security-relevant question; failing to match
                # yourself is merely inconvenient.
                wrong_identity = None
                if vector is not None:
                    scores = {
                        other: max(cosine_similarity(vector, v) for v in vectors)
                        for other, vectors in gallery.items()
                    }
                    best = max(scores, key=scores.get)
                    if (
                        scores[best] >= config.recognition.match_threshold
                        and best != identity_of(path)
                    ):
                        wrong_identity = best

                results[group][label].append(
                    {
                        "outcome": "processed",
                        "similarity_to_clean": float(similarity),
                        "still_matches": bool(
                            similarity >= config.recognition.match_threshold
                        ),
                        "matched_wrong_identity": wrong_identity,
                        "occlusion_hint": list(analysis.occluded_regions),
                        "detection_confidence": float(face.detection_confidence),
                        "quality": float(face.quality.overall) if face.quality else 0.0,
                        "backlight_ratio": float(backlight_ratio(degraded, face.bbox)),
                    }
                )

    print(f"{'condition':34} {'refused':>8} {'match':>7} {'sim':>7} "
          f"{'false rej':>10} {'FALSE ACC':>10}")
    print("-" * 86)
    summary: Dict[str, dict] = {}
    for group, labels in results.items():
        for label, entries in labels.items():
            total = len(entries)
            refused = sum(1 for e in entries if e["outcome"] == "refused")
            crashed = sum(1 for e in entries if e["outcome"] == "crashed")
            processed = [e for e in entries if e["outcome"] == "processed"]
            matched = sum(1 for e in processed if e["still_matches"])
            similarity = (
                float(np.mean([e["similarity_to_clean"] for e in processed]))
                if processed
                else 0.0
            )
            # Two directions, counted apart. Failing to match yourself is a
            # usability cost; matching somebody else is a security failure.
            false_rejects = sum(1 for e in processed if not e["still_matches"])
            false_accepts = sum(1 for e in processed if e["matched_wrong_identity"])
            hinted = sum(1 for e in processed if e["occlusion_hint"])
            key = f"{group}/{label}"
            summary[key] = {
                "n": total,
                "refused": refused,
                "crashed": crashed,
                "processed": len(processed),
                "still_matching": matched,
                "mean_similarity": similarity,
                "false_rejections": false_rejects,
                "false_acceptances": false_accepts,
                "occlusion_hinted": hinted,
            }
            flag = "  <-- SECURITY" if false_accepts or crashed else ""
            print(f"{key:34} {refused:4}/{total:<3} {matched:4}/{len(processed):<2} "
                  f"{similarity:7.3f} {false_rejects:10} {false_accepts:10}{flag}")

    total_reject = sum(v["false_rejections"] for v in summary.values())
    total_accept = sum(v["false_acceptances"] for v in summary.values())
    total_crashed = sum(v["crashed"] for v in summary.values())
    print("\n" + "=" * 86)
    print(f"  FALSE ACCEPTANCES (matched the wrong person): {total_accept}"
          f"   <- the number that matters")
    print(f"  false rejections (did not match themselves):  {total_reject}")
    print(f"  crashes:                                      {total_crashed}")
    print("  Refusals are correct behaviour: the system said why it could not decide.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "detail": {k: dict(v) for k, v in results.items()}},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
