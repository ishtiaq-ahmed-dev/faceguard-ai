#!/usr/bin/env python3
"""Download the MediaPipe model assets, verifying their checksums.

Model files are not committed to the repository -- they are 4 MB of binary that
Google publishes and versions themselves. This fetches them into ``models/``.

Checksums are verified because these files are executed as model graphs. A
silently corrupted or substituted ``.task`` file would change detection
behaviour in ways that are invisible in the output, so a mismatch is fatal
rather than a warning.

Usage::

    python tools/fetch_models.py
    python tools/fetch_models.py --force      # re-download even if present
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# name -> (url, sha256). Hashes recorded from the files this project was built
# and measured against.
ASSETS: Dict[str, tuple[str, str]] = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, url: str, expected: str, force: bool) -> bool:
    destination = MODELS_DIR / name

    if destination.is_file() and not force:
        actual = sha256_of(destination)
        if actual == expected:
            print(f"  {name}: already present and verified")
            return True
        print(f"  {name}: checksum mismatch, re-downloading")

    print(f"  {name}: downloading...")
    try:
        # Download beside the target and rename only after verifying, so an
        # interrupted download cannot leave a truncated model in place.
        temporary = destination.with_suffix(destination.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response:
            temporary.write_bytes(response.read())
    except (urllib.error.URLError, OSError) as error:
        print(f"  {name}: FAILED ({error})", file=sys.stderr)
        return False

    actual = sha256_of(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        print(
            f"  {name}: CHECKSUM MISMATCH\n"
            f"    expected {expected}\n"
            f"    got      {actual}\n"
            "    Refusing to install. This file is loaded as a model graph, so a "
            "substituted one would change detection behaviour invisibly.",
            file=sys.stderr,
        )
        return False

    temporary.replace(destination)
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"  {name}: {size_mb:.1f} MB, checksum verified")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching MediaPipe assets into {MODELS_DIR}")
    ok = all(
        fetch(name, url, digest, args.force) for name, (url, digest) in ASSETS.items()
    )

    if ok:
        print("\nAll model assets present and verified.")
        print("The Facenet512 recognition weights (~95 MB) are downloaded")
        print("automatically by DeepFace on first use.")
        return 0
    print("\nOne or more assets could not be fetched.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
