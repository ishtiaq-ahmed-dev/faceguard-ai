#!/usr/bin/env python3
"""Build train/validation/test splits with zero identity leakage.

The split is over **identities**, never over images. If two photographs of the
same person land on opposite sides of the split, the model can score well by
recognising the person rather than by learning anything about spoofing, and the
reported accuracy becomes meaningless. This is the single most common way face
benchmarks get silently inflated, and it is easy to do by accident because a
random shuffle of image paths looks perfectly reasonable.

The output manifest records which identities went where, so the split can be
audited and reproduced rather than trusted. ``--verify`` re-checks an existing
manifest for leakage without regenerating it.

Source layout expected::

    <root>/
        Person Name/
            image1.jpg
            image2.jpg
        Another Person/
            ...

A flat directory of ``Person Name.jpg`` files is also accepted, one identity
per file.

Usage::

    python tools/make_splits.py --root Data --output splits/manifest.json
    python tools/make_splits.py --verify splits/manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_identities(root: Path) -> Dict[str, List[str]]:
    """Map identity name to its image paths.

    Images directly inside ``root`` are treated as one identity per file
    (named after the file); images inside a subdirectory belong to the
    identity named by that directory.
    """
    identities: Dict[str, List[str]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        identity = relative.parts[-2] if len(relative.parts) > 1 else path.stem
        identities[identity].append(str(path.as_posix()))
    return dict(identities)


def split_identities(
    identities: List[str], ratios: tuple[float, float, float], seed: int
) -> Dict[str, List[str]]:
    """Assign whole identities to splits.

    Every split is guaranteed at least one identity where the count allows it:
    with very small datasets a proportional split can otherwise produce an
    empty test set, and an empty test set silently reports perfect accuracy.
    """
    shuffled = sorted(identities)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total < 3:
        raise SystemExit(
            f"Only {total} identities found. At least 3 are needed for a "
            "leakage-free train/val/test split. Add more people, or accept "
            "that no honest generalisation estimate is possible."
        )

    train_count = max(1, round(total * ratios[0]))
    validation_count = max(1, round(total * ratios[1]))
    # Leave at least one identity for test.
    while train_count + validation_count > total - 1:
        if train_count > validation_count:
            train_count -= 1
        else:
            validation_count -= 1

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count :],
    }


def verify(manifest: dict) -> List[str]:
    """Return a list of leakage problems. Empty means the split is clean."""
    problems: List[str] = []
    splits = manifest["splits"]
    seen: Dict[str, str] = {}
    for split_name, identities in splits.items():
        for identity in identities:
            if identity in seen:
                problems.append(
                    f"IDENTITY LEAKAGE: {identity!r} appears in both "
                    f"{seen[identity]!r} and {split_name!r}"
                )
            seen[identity] = split_name

    files = manifest.get("files", {})
    file_owner: Dict[str, str] = {}
    for split_name, identities in splits.items():
        for identity in identities:
            for path in files.get(identity, []):
                if path in file_owner and file_owner[path] != split_name:
                    problems.append(
                        f"FILE LEAKAGE: {path!r} in both {file_owner[path]!r} "
                        f"and {split_name!r}"
                    )
                file_owner[path] = split_name

    for split_name, identities in splits.items():
        if not identities:
            problems.append(f"EMPTY SPLIT: {split_name!r} has no identities")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Data", help="Directory of face images.")
    parser.add_argument("--output", default="splits/manifest.json")
    parser.add_argument("--ratios", default="0.6,0.2,0.2")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--verify", default=None, help="Verify an existing manifest.")
    args = parser.parse_args()

    if args.verify:
        manifest = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        problems = verify(manifest)
        if problems:
            print("SPLIT IS NOT CLEAN:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        counts = {k: len(v) for k, v in manifest["splits"].items()}
        print(f"Split verified clean. Identities per split: {counts}")
        return 0

    root = Path(args.root)
    if not root.is_dir():
        print(f"No such directory: {root}", file=sys.stderr)
        return 1

    identities = discover_identities(root)
    if not identities:
        print(f"No images found under {root}", file=sys.stderr)
        return 1

    ratios = tuple(float(v) for v in args.ratios.split(","))
    splits = split_identities(list(identities), ratios, args.seed)

    manifest = {
        "root": str(root.as_posix()),
        "seed": args.seed,
        "ratios": list(ratios),
        "splits": splits,
        "files": identities,
        "counts": {
            name: {
                "identities": len(members),
                "images": sum(len(identities[i]) for i in members),
            }
            for name, members in splits.items()
        },
    }

    problems = verify(manifest)
    if problems:
        # Refusing to write a leaky manifest is the whole purpose of this tool.
        print("Refusing to write a leaky split:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {output}")
    for name, stats in manifest["counts"].items():
        members = ", ".join(splits[name])
        print(f"  {name:5} {stats['identities']:2} identities, "
              f"{stats['images']:3} images  [{members}]")
    print("\nNo identity appears in more than one split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
