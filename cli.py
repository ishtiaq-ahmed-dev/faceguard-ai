#!/usr/bin/env python3
"""FaceGuard command-line interface -- for development and testing.

The web UI is the demo. This exists for the things a browser is bad at:
scripted enrolment, running the pipeline over a recorded clip, inspecting the
audit log, and checking what the system thinks its own capabilities are.

Every command prints the same structured result the API returns, so a failure
seen here is the same failure a reviewer would see in the UI.

Commands::

    python cli.py status                          what is loaded, what is degraded
    python cli.py enrol --name "Ada" --images "photos/*.jpg"
    python cli.py verify --video clip.mp4         verify from a recording
    python cli.py verify --camera 0               verify from a live webcam
    python cli.py identities                      list, or --delete NAME
    python cli.py audit --limit 20                recent decisions
    python cli.py set-pin --name "Ada"            set a fallback PIN
"""

from __future__ import annotations

import argparse
import getpass
import glob
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from faceguard.config import load_config
from faceguard.liveness import ChallengeType
from faceguard.pin import PinError, PinStore
from faceguard.quality import crop_face, read_image, standardise
from faceguard.session import FaceGuard


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def command_status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    with FaceGuard(config) as guard:
        capability = guard.liveness.depth_capability
        classifier = guard.liveness.classifier
        _print_json(
            {
                "enrolled_identities": guard.store.identities(),
                "template_count": len(guard.store),
                "depth": {
                    "available": capability.available,
                    "backend": capability.backend,
                    "detail": capability.detail,
                },
                "spoof_classifier_loaded": classifier is not None,
                "spoof_metrics": (
                    (classifier.metadata.get("metrics", {}) or {}).get("test")
                    if classifier
                    else None
                ),
                "policy": {
                    "required": config.liveness.policy.required,
                    "optional": config.liveness.policy.optional,
                    "min_optional": config.liveness.policy.min_optional_signals,
                    "require_challenge": config.liveness.policy.require_challenge,
                    "min_frames": config.liveness.min_frames,
                },
                "template_store": str(config.resolve(config.security.template_store)),
                "audit_log": str(config.resolve(config.security.audit_log)),
            }
        )
    return 0


def command_enrol(args: argparse.Namespace) -> int:
    paths = sorted(glob.glob(args.images, recursive=True))
    if not paths:
        print(f"No images matched {args.images!r}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    with FaceGuard(config) as guard:
        guard.warm_up()
        accepted, rejected = 0, []
        for path in paths:
            image = read_image(path)
            if image is None:
                rejected.append((path, "unreadable"))
                continue
            image = standardise(image, config.detection.working_width)
            analysis = guard.analyzer.analyze(image)
            if analysis.reasons or analysis.primary is None:
                reason = analysis.reasons[0].value if analysis.reasons else "NO_FACE_DETECTED"
                rejected.append((path, reason))
                continue
            face = analysis.primary
            crop, _ = crop_face(image, face.bbox, margin=0.15)
            template = guard.recogniser.enrol(
                args.name, crop, face.quality, note=f"cli:{Path(path).name}"
            )
            if template is None:
                rejected.append((path, "embedding failed"))
            else:
                accepted += 1
                print(f"  + {Path(path).name}  quality {face.quality.overall:.2f}")

        for path, reason in rejected:
            print(f"  - {Path(path).name}  rejected: {reason}")

        guard.audit.record_event(
            "enrolment", identity=args.name, templates_accepted=accepted,
            frames_rejected=len(rejected), source="cli",
        )
        print(
            f"\nEnrolled {accepted} template(s) for {args.name!r}; "
            f"{len(rejected)} rejected. "
            f"Total now {len(guard.store.templates_for(args.name))}."
        )
    return 0 if accepted else 1


def _frames_from_video(path: str, max_frames: int) -> List[np.ndarray]:
    capture = cv2.VideoCapture(path)
    frames: List[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def _frames_from_camera(
    index: int, guard: FaceGuard, challenge: Optional[ChallengeType]
) -> List[np.ndarray]:
    """Capture live frames, showing the challenge instruction in a preview window."""
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    instruction = challenge.instruction if challenge else "Look at the camera"
    frames: List[np.ndarray] = []
    deadline = time.monotonic() + guard.config.session.max_duration_s
    print(f"\n  >> {instruction}  (press q to stop early)\n")
    try:
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame.copy())
            preview = cv2.flip(frame, 1)
            cv2.putText(preview, instruction, (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
            cv2.putText(preview, f"{len(frames)} frames  "
                        f"{deadline - time.monotonic():.0f}s left",
                        (16, preview.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)
            cv2.imshow("FaceGuard capture", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return frames


def command_verify(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    challenge = ChallengeType(args.challenge) if args.challenge else None

    with FaceGuard(config) as guard:
        guard.warm_up()
        session = guard.start_session(client_id=args.client, challenge=challenge)

        if args.camera is not None:
            frames = _frames_from_camera(args.camera, guard, session.challenge)
        elif args.video:
            frames = _frames_from_video(args.video, config.session.max_buffered_frames)
        elif args.images:
            paths = sorted(glob.glob(args.images, recursive=True))
            frames = [img for img in (read_image(p) for p in paths) if img is not None]
        else:
            print("Give one of --camera, --video or --images", file=sys.stderr)
            return 1

        if not frames:
            print("No frames captured.", file=sys.stderr)
            return 1

        started = time.perf_counter()
        for frame in frames:
            session.add_frame(frame)
        result = session.finalise()
        wall_ms = (time.perf_counter() - started) * 1000.0

        _print_json(result.to_dict())
        print(
            f"\n{len(frames)} frames processed in {wall_ms:.0f} ms "
            f"({wall_ms / max(len(frames), 1):.1f} ms/frame end to end)"
        )
        print(f"Verdict: {'ACCEPTED' if result.accepted else 'REJECTED'} "
              f"-- {result.reason.value}: {result.message}")
    return 0 if result.accepted else 2


def command_identities(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    with FaceGuard(config) as guard:
        if args.delete:
            removed = guard.store.remove_identity(args.delete)
            PinStore(
                config.resolve(config.security.pin_store), config.security.pin_min_length
            ).remove(args.delete)
            guard.audit.record_event(
                "identity_deleted", identity=args.delete, removed=removed, source="cli"
            )
            print(f"{'Deleted' if removed else 'No such identity'}: {args.delete}")
            return 0 if removed else 1

        names = guard.store.identities()
        if not names:
            print("No identities enrolled.")
            return 0
        for name in names:
            templates = guard.store.templates_for(name)
            qualities = [t.quality for t in templates]
            print(
                f"  {name:28} {len(templates):2} template(s)  "
                f"quality {min(qualities):.2f}-{max(qualities):.2f}"
            )
    return 0


def command_audit(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    with FaceGuard(config) as guard:
        records = list(guard.audit.read())
        if not records:
            print("Audit log is empty.")
            return 0
        for record in records[-args.limit :]:
            verdict = record.get("accepted")
            mark = "OK " if verdict else ("-- " if verdict is None else "NO ")
            print(
                f"{mark} {record.get('timestamp', '')[:19]}  "
                f"{record.get('event', ''):16} "
                f"{str(record.get('reason') or record.get('triggered_by') or ''):26} "
                f"{record.get('identity') or ''}"
            )
        print(f"\n{len(records)} record(s) total.")
    return 0


def command_set_pin(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    store = PinStore(
        config.resolve(config.security.pin_store), config.security.pin_min_length
    )
    # Read from a prompt rather than argv: anything on the command line lands in
    # shell history and in the process table.
    pin = getpass.getpass("PIN: ")
    confirm = getpass.getpass("Confirm PIN: ")
    if pin != confirm:
        print("PINs do not match.", file=sys.stderr)
        return 1
    try:
        store.set_pin(args.name, pin)
    except PinError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"PIN set for {args.name!r}.")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Path to a config YAML.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show loaded models and capabilities.")

    enrol = subparsers.add_parser("enrol", help="Enrol an identity from images.")
    enrol.add_argument("--name", required=True)
    enrol.add_argument("--images", required=True, help="Glob of face images.")

    verify = subparsers.add_parser("verify", help="Run a verification attempt.")
    source = verify.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, help="Camera index for live capture.")
    source.add_argument("--video", help="Path to a video file.")
    source.add_argument("--images", help="Glob of frames, in order.")
    verify.add_argument(
        "--challenge",
        choices=[c.value for c in ChallengeType],
        help="Force a specific challenge instead of a random one.",
    )
    verify.add_argument("--client", default="cli", help="Client id for rate limiting.")

    identities = subparsers.add_parser("identities", help="List or delete identities.")
    identities.add_argument("--delete", help="Identity to delete.")

    audit = subparsers.add_parser("audit", help="Show recent audit records.")
    audit.add_argument("--limit", type=int, default=20)

    set_pin = subparsers.add_parser("set-pin", help="Set a fallback PIN.")
    set_pin.add_argument("--name", required=True)

    args = parser.parse_args()
    handlers = {
        "status": command_status,
        "enrol": command_enrol,
        "verify": command_verify,
        "identities": command_identities,
        "audit": command_audit,
        "set-pin": command_set_pin,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
