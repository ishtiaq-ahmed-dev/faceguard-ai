#!/usr/bin/env python3
"""Evaluation harness. Reports recognition and spoof metrics as separate numbers.

**Why they are separate.** A face recogniser with 99% accuracy and no liveness
check is not 99% secure; it is 0% secure against a printed photograph. Blending
recognition accuracy and spoof detection into one headline figure hides exactly
the failure that matters. This script therefore never produces a single score.

**Two different splitting protocols, on purpose.** The distinction trips people
up, so it is worth stating:

* *Spoof detection* has to generalise to people it has never seen, so it is
  evaluated on **identity-disjoint** splits -- train and test share no person.
* *Recognition* is a gallery task. Verifying someone requires them to be
  enrolled, so an identity-disjoint split would make every genuine trial
  impossible by construction. It is evaluated **leave-one-image-out**: enrol
  all but one image of a person, probe with the held-out one for a genuine
  trial, and probe with every other person's images for impostor trials.

Reported:

* **Recognition accuracy** -- correct identity on genuine trials.
* **FAR** (false acceptance rate) -- impostors accepted as an enrolled identity.
  The security number.
* **FRR** (false rejection rate) -- genuine users rejected. The usability number.
* **Spoof detection** -- APCER, BPCER and per-attack-type breakdown, from the
  trained classifier's own held-out report.
* **End-to-end liveness** -- what the *whole layered policy* does with a
  sequence, which is the number that actually describes the deployed system.

Usage::

    python eval/evaluate.py --manifest splits/manifest.json --output eval_output/
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faceguard.config import load_config
from faceguard.detect import FaceAnalyzer
from faceguard.liveness import LivenessEngine, LivenessEvidence
from faceguard.liveness.motion import ChallengeType
from faceguard.quality import crop_face, read_image, standardise
from faceguard.recognition import EmbeddingBackend, cosine_similarity
from faceguard.schemas import ReasonCode


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #


def embed_all(
    manifest: dict, analyzer: FaceAnalyzer, backend: EmbeddingBackend, working_width: int
) -> Dict[str, List[np.ndarray]]:
    """Embed every usable image, grouped by identity."""
    embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    for identity, paths in manifest["files"].items():
        for path in paths:
            image = read_image(path)
            if image is None:
                continue
            image = standardise(image, working_width)
            face = analyzer.analyze(image).primary
            if face is None:
                continue
            crop, _ = crop_face(image, face.bbox, margin=0.15)
            vector = backend.embed(crop)
            if vector is not None:
                embeddings[identity].append(vector)
    return dict(embeddings)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Reported alongside every rate because the sample sizes here are small
    enough that a point estimate on its own is misleading. Zero false accepts
    out of 29 trials is not "FAR = 0"; it is "FAR is somewhere below about
    12%", and quoting the former would be the single most overstated claim
    this project could make.

    Wilson rather than the textbook normal approximation specifically because
    it stays sensible at 0 and 1, which is exactly where these results land.
    """
    if trials <= 0:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    spread = (
        z
        * np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return (float(max(0.0, centre - spread)), float(min(1.0, centre + spread)))


def evaluate_recognition(
    embeddings: Dict[str, List[np.ndarray]], match_threshold: float, margin: float
) -> dict:
    """Leave-one-image-out verification over the enrolled gallery.

    For every image of every identity with at least two images: enrol the rest,
    probe with the held-out one (a genuine trial), and probe the same gallery
    with every other identity's images (impostor trials).
    """
    usable = {k: v for k, v in embeddings.items() if len(v) >= 2}
    if len(usable) < 2:
        return {
            "error": (
                f"Need at least 2 identities with 2+ images each; have "
                f"{len(usable)}. Recognition metrics cannot be computed."
            )
        }

    genuine_scores: List[float] = []
    impostor_scores: List[float] = []
    correct, genuine_trials = 0, 0
    false_accepts, impostor_trials = 0, 0
    confusions: List[Tuple[str, str, float]] = []

    for identity, vectors in usable.items():
        for held_out_index in range(len(vectors)):
            probe = vectors[held_out_index]
            gallery = {
                name: (
                    [v for i, v in enumerate(values) if i != held_out_index]
                    if name == identity
                    else values
                )
                for name, values in usable.items()
            }
            ranked = sorted(
                (
                    (name, max(cosine_similarity(probe, v) for v in values))
                    for name, values in gallery.items()
                    if values
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            best_name, best_score = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            accepted = best_score >= match_threshold and (best_score - runner_up) >= margin

            genuine_trials += 1
            genuine_scores.append(
                max(cosine_similarity(probe, v) for v in gallery[identity])
            )
            if accepted and best_name == identity:
                correct += 1
            elif accepted and best_name != identity:
                confusions.append((identity, best_name, best_score))

    # Impostor trials: every cross-identity image pair.
    for a, b in itertools.combinations(usable, 2):
        for probe in usable[a]:
            score = max(cosine_similarity(probe, v) for v in usable[b])
            impostor_scores.append(score)
            impostor_trials += 1
            if score >= match_threshold:
                false_accepts += 1

    genuine = np.array(genuine_scores)
    impostor = np.array(impostor_scores)
    return {
        "identities": len(usable),
        "images": sum(len(v) for v in usable.values()),
        "match_threshold": match_threshold,
        "margin_threshold": margin,
        "genuine_trials": genuine_trials,
        "impostor_trials": impostor_trials,
        "recognition_accuracy": correct / max(genuine_trials, 1),
        "frr": 1.0 - (correct / max(genuine_trials, 1)),
        "far": false_accepts / max(impostor_trials, 1),
        "frr_95ci": wilson_interval(genuine_trials - correct, genuine_trials),
        "far_95ci": wilson_interval(false_accepts, impostor_trials),
        "misidentifications": len(confusions),
        "genuine_similarity": {
            "mean": float(genuine.mean()) if genuine.size else 0.0,
            "std": float(genuine.std()) if genuine.size else 0.0,
            "min": float(genuine.min()) if genuine.size else 0.0,
        },
        "impostor_similarity": {
            "mean": float(impostor.mean()) if impostor.size else 0.0,
            "std": float(impostor.std()) if impostor.size else 0.0,
            "max": float(impostor.max()) if impostor.size else 0.0,
        },
        "separation_cohens_d": (
            float(
                abs(genuine.mean() - impostor.mean())
                / np.sqrt((genuine.var() + impostor.var()) / 2)
            )
            if genuine.size and impostor.size
            else 0.0
        ),
        "confusions": [
            {"true": t, "predicted": p, "score": round(s, 4)} for t, p, s in confusions[:10]
        ],
    }


# --------------------------------------------------------------------------- #
# End-to-end liveness
# --------------------------------------------------------------------------- #


def synthetic_sequence(
    image: np.ndarray, analyzer: FaceAnalyzer, frames: int, turn_degrees: float
) -> Tuple[list, list]:
    """Build a frame sequence from one still by simulating a head turn.

    The turn is applied as a **plane rotation homography**, which is exactly
    what happens when a flat object is tilted toward a camera. That makes this
    a faithful simulation of a presentation attack and a deliberately unfair
    one for a genuine face -- a real person turning their head produces
    parallax this cannot reproduce.

    So the numbers this yields are: for attacks, realistic; for live faces, a
    worst case. A live sequence built this way *should* fail the parallax
    check, because geometrically it is a flat rotation. That is reported
    honestly rather than papered over -- it is the reason a real camera test is
    the only way to measure the live side properly.
    """
    from tools.measure_parallax import plane_rotation_homography

    height, width = image.shape[:2]
    observations, crops = [], []
    for index in range(frames):
        progress = index / max(frames - 1, 1)
        angle = turn_degrees * progress
        warped = (
            cv2.warpPerspective(
                image,
                plane_rotation_homography(width, height, angle),
                (width, height),
                borderMode=cv2.BORDER_REPLICATE,
            )
            if abs(angle) > 1e-6
            else image
        )
        face = analyzer.analyze(warped).primary
        if face is None:
            continue
        face.timestamp = index / 15.0
        observations.append(face)
        crop, _ = crop_face(warped, face.bbox, margin=0.08)
        crops.append(crop)
    return observations, crops


def evaluate_liveness_end_to_end(
    manifest: dict,
    attack_index: dict,
    analyzer: FaceAnalyzer,
    engine: LivenessEngine,
    working_width: int,
    frames: int,
) -> dict:
    """Run the full layered policy over sequences built from live and attack stills."""
    results = {"live": [], "spoof": defaultdict(list)}

    test_identities = set(manifest["splits"]["test"])
    for identity in test_identities:
        for path in manifest["files"][identity]:
            image = read_image(path)
            if image is None:
                continue
            observations, crops = synthetic_sequence(
                standardise(image, working_width), analyzer, frames, 20.0
            )
            if len(observations) < engine.config.liveness.min_frames:
                continue
            outcome = engine.evaluate(
                LivenessEvidence(
                    observations=observations,
                    crops=crops,
                    challenge=ChallengeType.TURN_LEFT,
                    challenge_elapsed_s=2.0,
                )
            )
            results["live"].append(outcome)

    for record in attack_index["records"]:
        if record["split"] != "test":
            continue
        image = read_image(record["path"])
        if image is None:
            continue
        observations, crops = synthetic_sequence(
            standardise(image, working_width), analyzer, frames, 20.0
        )
        if len(observations) < engine.config.liveness.min_frames:
            continue
        outcome = engine.evaluate(
            LivenessEvidence(
                observations=observations,
                crops=crops,
                challenge=ChallengeType.TURN_LEFT,
                challenge_elapsed_s=2.0,
            )
        )
        results["spoof"][record["attack_type"]].append(outcome)

    def reason_counts(outcomes) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for outcome in outcomes:
            for reason in outcome.reasons:
                counts[reason.value] += 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    live = results["live"]
    spoof_all = [o for group in results["spoof"].values() for o in group]
    return {
        "frames_per_sequence": frames,
        "live": {
            "n": len(live),
            "passed": sum(1 for o in live if o.passed),
            "pass_rate": (sum(1 for o in live if o.passed) / len(live)) if live else 0.0,
            "reasons": reason_counts(live),
        },
        "spoof_overall": {
            "n": len(spoof_all),
            "passed": sum(1 for o in spoof_all if o.passed),
            "apcer": (
                sum(1 for o in spoof_all if o.passed) / len(spoof_all)
                if spoof_all
                else 0.0
            ),
            "reasons": reason_counts(spoof_all),
        },
        "spoof_by_type": {
            attack_type: {
                "n": len(group),
                "passed": sum(1 for o in group if o.passed),
                "apcer": sum(1 for o in group if o.passed) / len(group) if group else 0.0,
                "reasons": reason_counts(group),
            }
            for attack_type, group in results["spoof"].items()
        },
        "caveat": (
            "Sequences are synthesised from stills by plane-rotation homography, "
            "which is geometrically a flat rotation. That is faithful for attacks "
            "and a WORST CASE for live faces, which is why the live pass rate here "
            "is a floor rather than an estimate. Only a real camera capture "
            "measures the live side."
        ),
    }


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="splits/manifest.json")
    parser.add_argument("--attacks", default="attacks/index.json")
    parser.add_argument("--spoof-report", default="eval_output/spoof_report.json")
    parser.add_argument("--output", default="eval_output/evaluation.json")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-liveness", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    report: Dict[str, object] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    with FaceAnalyzer(config) as analyzer:
        backend = EmbeddingBackend(config.recognition)
        backend.warm_up()

        print("Embedding images for recognition evaluation...")
        embeddings = embed_all(
            manifest, analyzer, backend, config.detection.working_width
        )
        recognition = evaluate_recognition(
            embeddings,
            config.recognition.match_threshold,
            config.recognition.margin_threshold,
        )
        report["recognition"] = recognition

        print("\n" + "=" * 70)
        print("RECOGNITION  (leave-one-image-out over the enrolled gallery)")
        print("=" * 70)
        if "error" in recognition:
            print(f"  {recognition['error']}")
        else:
            print(f"  identities              {recognition['identities']}")
            print(f"  genuine / impostor      {recognition['genuine_trials']} / "
                  f"{recognition['impostor_trials']} trials")
            print(f"  recognition accuracy    {recognition['recognition_accuracy']:.3f}")
            frr_low, frr_high = recognition["frr_95ci"]
            far_low, far_high = recognition["far_95ci"]
            print(f"  FRR (users rejected)    {recognition['frr']:.3f}   "
                  f"95% CI [{frr_low:.3f}, {frr_high:.3f}]")
            print(f"  FAR (impostors let in)  {recognition['far']:.3f}   "
                  f"95% CI [{far_low:.3f}, {far_high:.3f}]")
            print("  NOTE: with this few trials the upper bounds are what matter.")
            print(f"  genuine similarity      "
                  f"{recognition['genuine_similarity']['mean']:.3f} "
                  f"+/- {recognition['genuine_similarity']['std']:.3f}")
            print(f"  impostor similarity     "
                  f"{recognition['impostor_similarity']['mean']:.3f} "
                  f"+/- {recognition['impostor_similarity']['std']:.3f}")
            print(f"  separation (Cohen's d)  {recognition['separation_cohens_d']:.2f}")

        spoof_report_path = Path(args.spoof_report)
        if spoof_report_path.is_file():
            spoof = json.loads(spoof_report_path.read_text(encoding="utf-8"))
            report["spoof_classifier"] = spoof
            metrics = (spoof.get("metrics") or {}).get("test")
            print("\n" + "=" * 70)
            print("SPOOF CLASSIFIER  (identity-disjoint, simulated attacks)")
            print("=" * 70)
            if metrics:
                print(f"  AUC                     {spoof.get('test_auc', float('nan')):.3f}")
                print(f"  accuracy                {metrics['accuracy']:.3f}")
                print(f"  APCER (attacks let in)  {metrics['apcer']:.3f}")
                print(f"  BPCER (users rejected)  {metrics['bpcer']:.3f}")
                print(f"  ACER                    {metrics['acer']:.3f}")
                for attack_type, stats in metrics["per_attack_type"].items():
                    print(f"    {attack_type:10} APCER {stats['apcer']:.3f} (n={stats['n']})")

        if not args.skip_liveness and Path(args.attacks).is_file():
            print("\nRunning the full layered policy over sequences...")
            attack_index = json.loads(Path(args.attacks).read_text(encoding="utf-8"))
            engine = LivenessEngine.from_config(config)
            liveness = evaluate_liveness_end_to_end(
                manifest,
                attack_index,
                analyzer,
                engine,
                config.detection.working_width,
                args.frames,
            )
            report["liveness_end_to_end"] = liveness
            print("\n" + "=" * 70)
            print("END-TO-END LIVENESS POLICY  (all layers, on sequences)")
            print("=" * 70)
            print(f"  live sequences passed   {liveness['live']['passed']}/"
                  f"{liveness['live']['n']}  ({liveness['live']['pass_rate']:.3f})")
            print(f"  attacks passed (APCER)  {liveness['spoof_overall']['passed']}/"
                  f"{liveness['spoof_overall']['n']}  "
                  f"({liveness['spoof_overall']['apcer']:.3f})")
            for attack_type, stats in liveness["spoof_by_type"].items():
                print(f"    {attack_type:10} APCER {stats['apcer']:.3f} (n={stats['n']})")
            print("\n  Why attacks were rejected:")
            for reason, count in list(liveness["spoof_overall"]["reasons"].items())[:6]:
                print(f"    {reason:28} {count}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
