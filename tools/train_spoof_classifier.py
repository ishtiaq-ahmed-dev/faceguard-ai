#!/usr/bin/env python3
"""Train the texture-based spoof classifier.

**The shortcut this script exists to avoid.** Every simulated attack ends with
a camera pass -- sensor noise and JPEG quantisation -- because every real attack
does. If the live images skip that pass, the two classes differ by JPEG
artefacts alone, and a classifier will happily learn "compressed means fake".
It would score beautifully here and collapse the instant it met a real webcam,
because real webcam frames are also compressed. So the live class goes through
the *same* camera model as the attacks, with the same noise and quality
distributions, and the only thing left to separate the classes is the physics
of the surface being photographed.

Splits are identity-disjoint (see ``tools/make_splits.py``). The threshold is
chosen by leave-one-identity-out cross-validation over the pooled train and
validation identities -- with this few people, tuning on a single held-out
identity yields a threshold fitted to one person's skin, lighting and camera.
The test split is touched exactly once, at the end.

Metrics are reported the way presentation-attack work reports them, because a
single blended "accuracy" hides the trade-off that matters:

* **APCER** -- attack presentation classification error rate. The fraction of
  attacks accepted as live. This is the security number.
* **BPCER** -- bona fide presentation classification error rate. The fraction of
  real faces rejected as attacks. This is the usability number.
* **ACER** -- the mean of the two.

Usage::

    python tools/train_spoof_classifier.py --manifest splits/manifest.json \\
        --attacks attacks/index.json --output models/spoof_classifier.joblib
    python tools/train_spoof_classifier.py --explain
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faceguard.config import load_config
from faceguard.detect import FaceAnalyzer
from faceguard.liveness.texture import extract_features, feature_names
from faceguard.quality import crop_face, read_image, standardise


def benign_camera_pass(
    image: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Put a live image through the same camera model the attacks went through.

    Noise sigma and JPEG quality are drawn from the union of the ranges used by
    ``tools/attack_sim.py``, so neither is informative about the label. Anything
    the classifier learns has to come from the surface, not the codec.
    """
    noisy = image.astype(np.float32) + rng.normal(0.0, rng.uniform(2.0, 5.0), image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(
        ".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(70, 95))]
    )
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else noisy


def benign_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric and small geometric jitter that introduces no spoof cue.

    Exposure, white balance and a degree or two of head tilt vary between
    genuine captures. Varying them here stops the model keying on the specific
    lighting of the handful of enrolment photos available.
    """
    work = image.astype(np.float32)
    work *= rng.uniform(0.82, 1.18)  # exposure
    work += rng.uniform(-12.0, 12.0)  # black level
    for channel in range(3):  # white balance
        work[:, :, channel] *= rng.uniform(0.96, 1.04)
    work = np.clip(work, 0, 255).astype(np.uint8)

    # Saturation jitter, spanning the range the print simulator applies.
    # Without this the classifier discovers that "slightly desaturated" means
    # "printed", which is true of this simulator specifically rather than of
    # printing in general, and the model would be learning the generator
    # instead of the physics.
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.78, 1.15)
    work = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    height, width = work.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), rng.uniform(-4.0, 4.0), rng.uniform(0.97, 1.03)
    )
    work = cv2.warpAffine(work, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    return benign_camera_pass(work, rng)


def face_features(
    analyzer: FaceAnalyzer, image: np.ndarray, crop_size: int
) -> Optional[np.ndarray]:
    """Detect the face and extract the spoof feature vector from its crop.

    Uses the same margin the live session uses, so training and serving see the
    same framing. A mismatch here is a classic silent accuracy leak.
    """
    analysis = analyzer.analyze(image)
    face = analysis.primary
    if face is None:
        return None
    crop, _ = crop_face(image, face.bbox, margin=0.08)
    if crop.size == 0:
        return None
    return extract_features(crop, crop_size)


def build_dataset(
    manifest: dict,
    attack_index: dict,
    analyzer: FaceAnalyzer,
    crop_size: int,
    live_variants: int,
    seed: int,
) -> Dict[str, Dict[str, list]]:
    """Extract features for every live and spoof sample, grouped by split."""
    data: Dict[str, Dict[str, list]] = {
        split: {"X": [], "y": [], "identity": [], "attack_type": []}
        for split in manifest["splits"]
    }

    identity_split = {
        identity: split
        for split, identities in manifest["splits"].items()
        for identity in identities
    }

    # --- live samples ------------------------------------------------------ #
    skipped_live = 0
    for identity, paths in manifest["files"].items():
        split = identity_split.get(identity)
        if split is None:
            continue
        for path in paths:
            image = read_image(path)
            if image is None:
                skipped_live += 1
                continue
            image = standardise(image, crop_size * 5)
            for variant in range(live_variants):
                rng = np.random.default_rng(
                    abs(hash((path, "live", variant, seed))) % (2**32)
                )
                features = face_features(
                    analyzer, benign_augment(image, rng), crop_size
                )
                if features is None:
                    skipped_live += 1
                    continue
                data[split]["X"].append(features)
                data[split]["y"].append(1)  # 1 == live
                data[split]["identity"].append(identity)
                data[split]["attack_type"].append("live")

    # --- spoof samples ----------------------------------------------------- #
    skipped_spoof = 0
    for record in attack_index["records"]:
        split = record["split"]
        if split not in data:
            continue
        image = read_image(record["path"])
        if image is None:
            skipped_spoof += 1
            continue
        features = face_features(analyzer, image, crop_size)
        if features is None:
            skipped_spoof += 1
            continue
        data[split]["X"].append(features)
        data[split]["y"].append(0)  # 0 == spoof
        data[split]["identity"].append(record["identity"])
        data[split]["attack_type"].append(record["attack_type"])

    if skipped_live or skipped_spoof:
        print(
            f"  (skipped {skipped_live} live and {skipped_spoof} spoof samples "
            "where no face could be detected)"
        )
    return data


def error_rates(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    attack_types: List[str],
    threshold: float,
) -> dict:
    """APCER, BPCER, ACER and a per-attack-type breakdown."""
    predicted_live = probabilities >= threshold
    live_mask = y_true == 1
    spoof_mask = ~live_mask

    bpcer = float(np.mean(~predicted_live[live_mask])) if live_mask.any() else 0.0
    apcer = float(np.mean(predicted_live[spoof_mask])) if spoof_mask.any() else 0.0

    per_type: Dict[str, dict] = {}
    types = np.array(attack_types)
    for attack_type in sorted(set(types[spoof_mask])) if spoof_mask.any() else []:
        mask = spoof_mask & (types == attack_type)
        per_type[attack_type] = {
            "n": int(mask.sum()),
            "apcer": float(np.mean(predicted_live[mask])),
        }

    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predicted_live == live_mask)),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2.0,
        "n_live": int(live_mask.sum()),
        "n_spoof": int(spoof_mask.sum()),
        "per_attack_type": per_type,
    }


def choose_threshold(
    y_true: np.ndarray, probabilities: np.ndarray
) -> Tuple[float, float]:
    """Pick the threshold minimising ACER.

    Minimising ACER weights a wrongly-accepted attack and a wrongly-rejected
    user equally. For a real deployment that is the wrong trade -- an accepted
    attack is far more costly -- so this is a starting point an operator should
    move, which is why ``liveness.texture.live_threshold`` is configurable.
    """
    best_threshold, best_acer = 0.5, float("inf")
    live_mask = y_true == 1
    if not live_mask.any() or live_mask.all():
        return best_threshold, best_acer
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted_live = probabilities >= threshold
        bpcer = float(np.mean(~predicted_live[live_mask]))
        apcer = float(np.mean(predicted_live[~live_mask]))
        acer = (apcer + bpcer) / 2.0
        if acer < best_acer:
            best_threshold, best_acer = float(threshold), acer
    return best_threshold, best_acer


def leave_one_identity_out_threshold(
    features: np.ndarray,
    labels: np.ndarray,
    identities: List[str],
    make_model,
    seed: int,
) -> Tuple[float, float, float, List[dict]]:
    """Choose a threshold by leave-one-identity-out cross-validation.

    With only a handful of people available, tuning on a single held-out
    identity produces a threshold fitted to that one person's skin, lighting
    and camera. Rotating the held-out identity and pooling the out-of-fold
    predictions gives every identity a turn, which is the most honest estimate
    the data supports.

    Folds are over *identities*, never over samples, for the same reason the
    top-level split is: two augmented copies of one photo on opposite sides of
    a fold would leak.

    Returns ``(threshold, pooled_acer, pooled_auc, per_fold_records)``.
    """
    from sklearn.metrics import roc_auc_score

    unique_identities = sorted(set(identities))
    identity_array = np.array(identities)
    pooled_probabilities = np.zeros(len(labels), dtype=float)
    pooled_mask = np.zeros(len(labels), dtype=bool)
    folds: List[dict] = []

    for held_out in unique_identities:
        test_mask = identity_array == held_out
        train_mask = ~test_mask
        if len(set(labels[train_mask])) < 2 or not test_mask.any():
            continue
        model = make_model()
        model.fit(features[train_mask], labels[train_mask])
        probabilities = model.predict_proba(features[test_mask])[:, 1]
        pooled_probabilities[test_mask] = probabilities
        pooled_mask[test_mask] = True

        fold_record = {"held_out_identity": held_out, "n": int(test_mask.sum())}
        if len(set(labels[test_mask])) > 1:
            fold_record["auc"] = float(
                roc_auc_score(labels[test_mask], probabilities)
            )
        folds.append(fold_record)

    if not pooled_mask.any():
        return 0.5, float("inf"), float("nan"), folds

    threshold, acer = choose_threshold(
        labels[pooled_mask], pooled_probabilities[pooled_mask]
    )
    auc = float("nan")
    if len(set(labels[pooled_mask])) > 1:
        auc = float(
            roc_auc_score(labels[pooled_mask], pooled_probabilities[pooled_mask])
        )
    return threshold, acer, auc, folds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="splits/manifest.json")
    parser.add_argument("--attacks", default="attacks/index.json")
    parser.add_argument("--output", default="models/spoof_classifier.joblib")
    parser.add_argument("--report", default="eval_output/spoof_report.json")
    parser.add_argument("--live-variants", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--explain", action="store_true", help="Print the strongest coefficients."
    )
    args = parser.parse_args()

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    config = load_config(args.config)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    attack_index = json.loads(Path(args.attacks).read_text(encoding="utf-8"))

    print("Extracting features...")
    with FaceAnalyzer(config) as analyzer:
        data = build_dataset(
            manifest,
            attack_index,
            analyzer,
            config.liveness.texture.crop_size,
            args.live_variants,
            args.seed,
        )

    for split, bundle in data.items():
        labels = np.array(bundle["y"])
        print(
            f"  {split:5} {len(labels):4} samples  "
            f"({int((labels == 1).sum())} live, {int((labels == 0).sum())} spoof)  "
            f"identities: {sorted(set(bundle['identity']))}"
        )

    # Train and validation are pooled for fitting and threshold selection. They
    # are already identity-disjoint from test, and with this few people a
    # single held-out validation identity yields a threshold fitted to one
    # person's skin and lighting. Cross-validating over identities uses every
    # available person instead. Test remains untouched until the final report.
    fit_x = np.array(data["train"]["X"] + data["val"]["X"])
    fit_y = np.array(data["train"]["y"] + data["val"]["y"])
    fit_identities = data["train"]["identity"] + data["val"]["identity"]
    if len(set(fit_y)) < 2:
        print("Training data needs both classes.", file=sys.stderr)
        return 1

    # Logistic regression, not a forest: with a few hundred samples and 142
    # features a high-capacity model memorises the training identities, and the
    # linear coefficients are readable, which matters for a component reviewers
    # will want to interrogate.
    def make_model() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.05,  # strong regularisation for the sample size
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=args.seed,
                    ),
                ),
            ]
        )

    threshold, cv_acer, cv_auc, folds = leave_one_identity_out_threshold(
        fit_x, fit_y, fit_identities, make_model, args.seed
    )
    print(
        f"\nLeave-one-identity-out CV over {len(set(fit_identities))} identities: "
        f"AUC {cv_auc:.3f}, best ACER {cv_acer:.3f} at threshold {threshold:.3f}"
    )
    for fold in folds:
        auc_text = f"{fold['auc']:.3f}" if "auc" in fold else "n/a"
        print(f"    held out {fold['held_out_identity']:<18} n={fold['n']:<4} AUC {auc_text}")

    model = make_model()
    model.fit(fit_x, fit_y)

    def evaluate(split: str, threshold: float) -> Optional[dict]:
        bundle = data[split]
        if not bundle["X"]:
            return None
        probabilities = model.predict_proba(np.array(bundle["X"]))[:, 1]
        return error_rates(
            np.array(bundle["y"]), probabilities, bundle["attack_type"], threshold
        )

    # Pooled CV AUC vs mean per-fold AUC is worth reporting separately. When
    # the pooled figure is the lower of the two, the model separates live from
    # spoof well *within* each person but its output probabilities are not
    # calibrated *across* people -- so a single global threshold is the weak
    # link, not the features. That is the same lesson the blink module hit with
    # absolute EAR thresholds, and it is a real limitation of this dataset size.
    fold_aucs = [fold["auc"] for fold in folds if "auc" in fold]
    mean_fold_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    print(
        f"  pooled AUC {cv_auc:.3f} vs mean per-fold AUC {mean_fold_auc:.3f} "
        f"-- a gap here means scores are not comparable between identities."
    )

    report = {
        "trained_on": args.manifest,
        "attacks": args.attacks,
        "threshold": threshold,
        "cv_auc": cv_auc,
        "cv_acer": cv_acer,
        "cv_folds": folds,
        "cv_mean_fold_auc": mean_fold_auc,
        "live_variants": args.live_variants,
        "n_features": int(fit_x.shape[1]),
        "splits": {
            split: {
                "identities": manifest["splits"][split],
                "n_samples": len(data[split]["X"]),
            }
            for split in data
        },
        "metrics": {
            split: evaluate(split, threshold) for split in ("train", "val", "test")
        },
        "caveat": (
            "Trained and evaluated on SIMULATED attacks. These numbers are an "
            "upper bound on real-world performance, not an estimate of it. "
            "Identity-disjoint splits, so no person appears in both train and "
            "test. Threshold chosen by leave-one-identity-out CV over the "
            "pooled train+val identities; test used once. APCER here is for the "
            "texture signal ALONE -- the deployed policy also requires blink "
            "and parallax, which a printed photo fails independently."
        ),
    }

    print("\n" + "=" * 68)
    print("SPOOF DETECTION (identity-disjoint splits, SIMULATED attacks)")
    print("=" * 68)
    for split in ("train", "val", "test"):
        metrics = report["metrics"][split]
        if metrics is None:
            continue
        print(
            f"{split:5}  acc {metrics['accuracy']:.3f}   "
            f"APCER {metrics['apcer']:.3f}   BPCER {metrics['bpcer']:.3f}   "
            f"ACER {metrics['acer']:.3f}   "
            f"(n={metrics['n_live']} live / {metrics['n_spoof']} spoof)"
        )
    test_bundle = data["test"]
    if test_bundle["X"] and len(set(test_bundle["y"])) > 1:
        from sklearn.metrics import roc_auc_score

        test_auc = float(
            roc_auc_score(
                np.array(test_bundle["y"]),
                model.predict_proba(np.array(test_bundle["X"]))[:, 1],
            )
        )
        report["test_auc"] = test_auc
        print(
            f"\ntest AUC {test_auc:.3f}  (threshold-independent: how well the "
            "model separates the two classes at all)"
        )

    test_metrics = report["metrics"]["test"]
    if test_metrics and test_metrics["per_attack_type"]:
        print("\nAttacks accepted as live, by type (test split):")
        for attack_type, stats in test_metrics["per_attack_type"].items():
            print(f"  {attack_type:10} APCER {stats['apcer']:.3f}  (n={stats['n']})")

    if args.explain:
        coefficients = model.named_steps["clf"].coef_[0]
        names = feature_names()
        order = np.argsort(np.abs(coefficients))[::-1][:15]
        print("\nStrongest features (positive = evidence of LIVE):")
        for index in order:
            print(f"  {names[index]:24} {coefficients[index]:+.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": report}, output)
    print(f"\nSaved model -> {output}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report -> {report_path}")

    if abs(threshold - config.liveness.texture.live_threshold) > 1e-6:
        print(
            f"\nNOTE: set liveness.texture.live_threshold to {threshold:.3f} in "
            "config/default.yaml to use the tuned operating point."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
