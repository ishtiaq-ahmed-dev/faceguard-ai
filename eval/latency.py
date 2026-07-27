#!/usr/bin/env python3
"""Per-stage latency benchmark on the machine it is run on.

Batch accuracy on a static test set says nothing about whether a security
system feels broken in use. A verification that lags visibly reads as failure
even when the model is right, so this measures what actually happens per frame,
stage by stage, and reports the distribution rather than a mean -- the tail is
what a user notices.

Stages measured:

* **detection + landmarks** -- runs on every frame.
* **texture features** -- runs on every frame.
* **liveness fusion** -- runs once per decision, over the whole window.
* **recognition embedding** -- the expensive one, which is why the pipeline
  runs it on every Nth frame rather than all of them.

The summary works out the effective per-frame cost under that policy and the
sustainable frame rate, which is the number that matters for the demo.

Usage::

    python eval/latency.py --images "all_faces/**/*.jpg" --runs 30
"""

from __future__ import annotations

import argparse
import glob
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faceguard.config import load_config
from faceguard.detect import FaceAnalyzer
from faceguard.liveness import LivenessEngine, LivenessEvidence
from faceguard.liveness.motion import ChallengeType
from faceguard.liveness.texture import extract_features
from faceguard.quality import crop_face, read_image, standardise
from faceguard.recognition import EmbeddingBackend


def summarise(samples: List[float]) -> Dict[str, float]:
    """Median and tail percentiles. The mean hides exactly what matters."""
    if not samples:
        return {}
    array = np.array(samples)
    return {
        "n": int(array.size),
        "median_ms": float(np.median(array)),
        "mean_ms": float(array.mean()),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def describe_hardware() -> Dict[str, str]:
    """Record what the numbers were measured on, so they mean something."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
    }
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except ImportError:
        pass
    try:
        import os

        info["cpu_count"] = str(os.cpu_count())
    except Exception:
        pass
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="all_faces/**/*.jpg")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output", default="eval_output/latency.json")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.images, recursive=True))
    if not paths:
        print(f"No images matched {args.images!r}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    working_width = config.detection.working_width
    images = []
    for path in paths:
        image = read_image(path)
        if image is not None:
            images.append(standardise(image, working_width))
    if not images:
        print("No readable images.", file=sys.stderr)
        return 1

    timings: Dict[str, List[float]] = {
        "detect_landmarks": [],
        "texture_features": [],
        "liveness_fusion": [],
        "recognition_embed": [],
    }

    print(f"Benchmarking on {len(images)} image(s), {args.runs} runs, "
          f"working width {working_width}px...\n")

    with FaceAnalyzer(config) as analyzer:
        engine = LivenessEngine.from_config(config)
        backend = EmbeddingBackend(config.recognition)
        backend.warm_up()  # excluded from measurement: one-off startup cost

        # Warm-up pass. The first call through any of these paths pays lazy
        # allocation and cache costs that never recur, and including it would
        # overstate the steady-state figure a user actually experiences.
        analyzer.analyze(images[0])
        extract_features(images[0], config.liveness.texture.crop_size)

        observations, crops = [], []
        for index in range(args.runs):
            image = images[index % len(images)]

            start = time.perf_counter()
            analysis = analyzer.analyze(image)
            timings["detect_landmarks"].append((time.perf_counter() - start) * 1000)

            face = analysis.primary
            if face is None:
                continue
            crop, _ = crop_face(image, face.bbox, margin=0.08)

            start = time.perf_counter()
            extract_features(crop, config.liveness.texture.crop_size)
            timings["texture_features"].append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            backend.embed(crop)
            timings["recognition_embed"].append((time.perf_counter() - start) * 1000)

            face.timestamp = index / 15.0
            observations.append(face)
            crops.append(crop)

        window = observations[: args.window]
        window_crops = crops[: args.window]
        for _ in range(10):
            start = time.perf_counter()
            engine.evaluate(
                LivenessEvidence(
                    observations=window,
                    crops=window_crops,
                    challenge=ChallengeType.TURN_LEFT,
                    challenge_elapsed_s=2.0,
                )
            )
            timings["liveness_fusion"].append((time.perf_counter() - start) * 1000)

    report = {
        "hardware": describe_hardware(),
        "working_width": working_width,
        "recognise_every_n_frames": config.session.recognise_every_n_frames,
        "liveness_window_frames": args.window,
        "stages": {name: summarise(values) for name, values in timings.items()},
    }

    print(f"{'stage':22} {'median':>9} {'p95':>9} {'p99':>9} {'max':>9}")
    print("-" * 62)
    for name, stats in report["stages"].items():
        if stats:
            print(f"{name:22} {stats['median_ms']:8.1f}ms {stats['p95_ms']:8.1f}ms "
                  f"{stats['p99_ms']:8.1f}ms {stats['max_ms']:8.1f}ms")

    per_frame = report["stages"]["detect_landmarks"]["median_ms"]
    per_frame += report["stages"]["texture_features"]["median_ms"]
    every_n = config.session.recognise_every_n_frames
    embed_median = report["stages"]["recognition_embed"]["median_ms"]
    amortised = per_frame + embed_median / max(every_n, 1)

    report["effective"] = {
        "per_frame_always_on_ms": per_frame,
        "amortised_per_frame_ms": amortised,
        "sustainable_fps": 1000.0 / amortised if amortised else 0.0,
        "decision_overhead_ms": report["stages"]["liveness_fusion"]["median_ms"],
        "typical_attempt_s": (
            config.liveness.min_frames * amortised / 1000.0
            + report["stages"]["liveness_fusion"]["median_ms"] / 1000.0
        ),
    }

    effective = report["effective"]
    print("\n" + "=" * 62)
    print("EFFECTIVE THROUGHPUT")
    print("=" * 62)
    print(f"  every frame (detect + texture)   {per_frame:.1f} ms")
    print(f"  recognition, 1 frame in {every_n}        "
          f"{embed_median:.1f} ms -> +{embed_median / every_n:.1f} ms amortised")
    print(f"  amortised per frame              {amortised:.1f} ms")
    print(f"  sustainable                      {effective['sustainable_fps']:.1f} fps")
    print(f"  fusion, once per decision        "
          f"{effective['decision_overhead_ms']:.1f} ms")
    print(f"  compute for a {config.liveness.min_frames}-frame attempt   "
          f"{effective['typical_attempt_s']:.2f} s")
    print("\n  Wall-clock time for an attempt is longer than the compute time:")
    print("  the challenge is paced by how fast a person turns their head, not")
    print("  by the pipeline.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
