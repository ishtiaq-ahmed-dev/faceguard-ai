#!/usr/bin/env python3
"""Generate simulated presentation attacks from consented source images.

**Read this before believing any number produced with this data.**

These are *simulations*. Each one models the physical process a real attack
goes through, and each one is a simplification of it. A classifier trained
here learns to detect simulated attacks, and its accuracy on real printed
photographs held in front of a real webcam will be lower -- probably
substantially. This module exists so the pipeline is reproducible without
distributing anyone's face, and so the project has a defensible baseline. It
does not replace a real capture, and the README says so where reviewers will
see it.

The three attack families and the physics each one models:

**Print attack** (``print``). A photo goes through a printer and back through a
camera. The printer quantises to a handful of inks and reproduces continuous
tone with a halftone dot screen, which imposes a periodic pattern and destroys
sub-dot detail. Paper adds its own fibre texture and cannot reproduce black or
high chroma, so the gamut compresses. The re-capture adds an uneven lighting
gradient across the sheet, camera noise and JPEG quantisation.

**Screen replay** (``replay``). A photo is displayed on a phone or monitor and
re-photographed. The panel has a physical pixel lattice, so detail below one
panel pixel is gone and a regular grid appears. Photographing that lattice with
another sampled sensor produces moire -- low-frequency interference beating
between two grids. Backlight bleed lifts the blacks, the panel's gamma differs
from the original capture, and the glass produces a broad specular reflection
of the room.

**Generative artefact proxy** (``synthetic``). A crude stand-in for the
frequency-domain signature that generative models leave behind: suppressed
high-frequency detail from decoding at low resolution, upsampling checkerboard
artefacts, and a blend seam where a generated face is composited into a frame.

This is the weakest of the three and it is important to say why: it is a
hand-built imitation of an artefact class, not a real deepfake. A classifier
that detects it has not been shown to detect anything produced by an actual
generative model. It is included so the pipeline has a third class and so the
feature extractor is exercised against smoothed high frequencies, and it is
labelled ``synthetic`` everywhere so no result can quietly imply otherwise.

Usage::

    python tools/attack_sim.py --manifest splits/manifest.json --output attacks/
    python tools/attack_sim.py --input photo.jpg --output attacks/ --preview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faceguard.quality import read_image, write_image

ATTACK_TYPES = ("print", "replay", "synthetic")


# --------------------------------------------------------------------------- #
# Shared camera model
# --------------------------------------------------------------------------- #


def _recapture(
    image: np.ndarray,
    rng: np.random.Generator,
    noise_sigma: float = 3.0,
    jpeg_quality: int = 82,
) -> np.ndarray:
    """Model photographing a physical surface with a consumer camera.

    Every attack in this module ends here, because every attack ends with a
    second trip through a lens and a sensor. Shot noise and JPEG quantisation
    are what make the re-captured copy measurably different from the original
    even when the display is very good.
    """
    noisy = image.astype(np.float32) + rng.normal(0.0, noise_sigma, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(
        ".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else noisy


def _lighting_gradient(
    image: np.ndarray, rng: np.random.Generator, strength: float = 0.18
) -> np.ndarray:
    """Uneven illumination across a flat surface.

    A sheet of paper or a screen held up to a camera is almost never lit
    evenly: there is a falloff from wherever the room light is. Real faces show
    this far less, because they are not flat.
    """
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(angle) * xx / width) + (np.sin(angle) * yy / height)
    # np.ptp(), not ramp.ptp(): the ndarray method was removed in NumPy 2.
    ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-6)
    gain = 1.0 + strength * (ramp - 0.5) * 2.0
    return np.clip(image.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Print attack
# --------------------------------------------------------------------------- #


def simulate_print(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photograph -> printer -> paper -> camera."""
    work = image.astype(np.float32)

    # Ink gamut compression: paper cannot reach true black or full saturation.
    work = work * 0.86 + 18.0

    # Printer resolution limit: detail below the dot pitch is gone.
    work = cv2.GaussianBlur(work, (0, 0), sigmaX=rng.uniform(0.7, 1.2))

    # Halftone screen: continuous tone reproduced as a periodic dot pattern,
    # conventionally rotated ~15 degrees per channel to avoid colour moire.
    height, width = work.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    for channel, angle_deg in enumerate((15.0, 75.0, 0.0)):
        theta = np.radians(angle_deg)
        pitch = rng.uniform(3.0, 4.5)
        rotated = xx * np.cos(theta) + yy * np.sin(theta)
        other = -xx * np.sin(theta) + yy * np.cos(theta)
        screen = np.sin(2 * np.pi * rotated / pitch) * np.sin(
            2 * np.pi * other / pitch
        )
        work[:, :, channel] += screen * rng.uniform(4.0, 8.0)

    # Paper fibre texture: fine, achromatic, and identical in all channels.
    fibre = rng.normal(0.0, rng.uniform(2.0, 4.0), (height, width, 1))
    work = work + fibre

    # Slight desaturation: pigment on paper is duller than an emissive display.
    printed = np.clip(work, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(printed, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.80, 0.92)
    printed = cv2.cvtColor(
        np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
    )

    printed = _lighting_gradient(printed, rng, strength=rng.uniform(0.12, 0.25))
    return _recapture(printed, rng, noise_sigma=rng.uniform(2.0, 4.0), jpeg_quality=int(rng.integers(75, 90)))


# --------------------------------------------------------------------------- #
# Screen replay attack
# --------------------------------------------------------------------------- #


def simulate_replay(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photograph -> display panel -> camera."""
    height, width = image.shape[:2]

    # Panel resolution limit: the display cannot show detail finer than its
    # own pixel pitch, and nearest-neighbour upsampling recreates the visible
    # lattice rather than smoothing it away.
    factor = rng.uniform(2.2, 3.4)
    small = cv2.resize(
        image,
        (max(8, int(width / factor)), max(8, int(height / factor))),
        interpolation=cv2.INTER_AREA,
    )
    panel = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST).astype(
        np.float32
    )

    # RGB subpixel stripes: each panel pixel is three coloured emitters side by
    # side, so vertical stripes appear at the pixel pitch.
    stripe_period = max(2, int(round(factor)))
    columns = np.arange(width)
    for channel in range(3):
        modulation = 1.0 + 0.05 * np.cos(
            2 * np.pi * (columns + channel * stripe_period / 3.0) / stripe_period
        )
        panel[:, :, channel] *= modulation[None, :]

    # Moire: interference between the panel lattice and the camera's sensor
    # grid, appearing as a low-frequency beat rather than a fine pattern.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    beat_period = rng.uniform(18.0, 46.0)
    beat_angle = rng.uniform(0, np.pi)
    projection = xx * np.cos(beat_angle) + yy * np.sin(beat_angle)
    panel += np.sin(2 * np.pi * projection / beat_period)[..., None] * rng.uniform(3.0, 7.0)

    # Backlight bleed lifts the blacks; an LCD cannot render true black.
    panel = panel * rng.uniform(0.90, 0.97) + rng.uniform(8.0, 18.0)

    # Panel gamma differs from the gamma the original was captured at.
    gamma = rng.uniform(0.85, 1.18)
    panel = 255.0 * np.power(np.clip(panel, 0, 255) / 255.0, gamma)

    # Glass glare: a broad, soft, desaturated reflection of the room.
    glare_x = rng.uniform(0.2, 0.8) * width
    glare_y = rng.uniform(0.1, 0.6) * height
    radius = rng.uniform(0.18, 0.38) * max(height, width)
    falloff = np.exp(-(((xx - glare_x) ** 2 + (yy - glare_y) ** 2) / (2 * radius**2)))
    panel += (falloff * rng.uniform(18.0, 42.0))[..., None]

    # Cool colour cast: displays typically run a higher colour temperature.
    panel[:, :, 0] *= rng.uniform(1.01, 1.07)  # blue
    panel[:, :, 2] *= rng.uniform(0.94, 0.99)  # red

    replayed = np.clip(panel, 0, 255).astype(np.uint8)
    return _recapture(replayed, rng, noise_sigma=rng.uniform(2.5, 5.0), jpeg_quality=int(rng.integers(70, 88)))


# --------------------------------------------------------------------------- #
# Generative artefact proxy
# --------------------------------------------------------------------------- #


def simulate_synthetic(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Crude proxy for generative-model artefacts. NOT a real deepfake."""
    height, width = image.shape[:2]

    # Decode at low resolution then upsample: suppresses genuine high-frequency
    # detail the way a generator's limited latent capacity does.
    factor = rng.uniform(3.0, 5.0)
    latent = cv2.resize(
        image,
        (max(8, int(width / factor)), max(8, int(height / factor))),
        interpolation=cv2.INTER_AREA,
    )
    upsampled = cv2.resize(latent, (width, height), interpolation=cv2.INTER_CUBIC).astype(
        np.float32
    )

    # Checkerboard artefact from transposed-convolution upsampling.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    stride = max(2, int(round(factor)))
    checker = np.sign(np.sin(np.pi * xx / stride) * np.sin(np.pi * yy / stride))
    upsampled += checker[..., None] * rng.uniform(1.5, 3.5)

    # Over-smoothed skin: generators average away pore-level texture.
    smoothed = cv2.bilateralFilter(
        np.clip(upsampled, 0, 255).astype(np.uint8), 9, 60, 60
    ).astype(np.float32)
    upsampled = 0.35 * upsampled + 0.65 * smoothed

    # Blend seam where a generated face is composited into a surrounding frame.
    centre = (width // 2, int(height * 0.46))
    axes = (int(width * 0.33), int(height * 0.42))
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=rng.uniform(6.0, 14.0))
    original = image.astype(np.float32)
    blended = upsampled * mask[..., None] + original * (1.0 - mask[..., None])

    return _recapture(
        np.clip(blended, 0, 255).astype(np.uint8),
        rng,
        noise_sigma=rng.uniform(1.0, 2.5),
        jpeg_quality=int(rng.integers(85, 95)),
    )


SIMULATORS: Dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "print": simulate_print,
    "replay": simulate_replay,
    "synthetic": simulate_synthetic,
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def generate_for_image(
    image_path: Path,
    output_root: Path,
    identity: str,
    split: str,
    variants: int,
    seed: int,
    types: List[str],
) -> List[dict]:
    """Write every attack variant for one source image and return its records."""
    image = read_image(image_path)
    if image is None:
        return []
    if image.shape[1] > 720:
        scale = 720.0 / image.shape[1]
        image = cv2.resize(image, (720, int(image.shape[0] * scale)))

    records: List[dict] = []
    for attack_type in types:
        simulator = SIMULATORS[attack_type]
        for variant in range(variants):
            # Seeded per (image, type, variant) so the corpus regenerates
            # byte-identically and results stay reproducible.
            rng = np.random.default_rng(
                abs(hash((str(image_path), attack_type, variant, seed))) % (2**32)
            )
            attacked = simulator(image, rng)
            destination = (
                output_root / split / attack_type / identity
                / f"{image_path.stem}_v{variant}.jpg"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_image(destination, attacked, int(cv2.IMWRITE_JPEG_QUALITY), 95)
            records.append(
                {
                    "path": str(destination.as_posix()),
                    "source": str(image_path.as_posix()),
                    "identity": identity,
                    "split": split,
                    "label": "spoof",
                    "attack_type": attack_type,
                    "variant": variant,
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="splits/manifest.json")
    parser.add_argument("--output", default="attacks")
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--types", default=",".join(ATTACK_TYPES), help="Comma-separated attack types."
    )
    parser.add_argument("--input", default=None, help="Single image, for a preview.")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    unknown = set(types) - set(ATTACK_TYPES)
    if unknown:
        print(f"Unknown attack type(s): {sorted(unknown)}", file=sys.stderr)
        return 1

    output_root = Path(args.output)

    if args.input:
        image = read_image(args.input)
        if image is None:
            print(f"Could not read {args.input}", file=sys.stderr)
            return 1
        output_root.mkdir(parents=True, exist_ok=True)
        panels = [image]
        for attack_type in types:
            rng = np.random.default_rng(args.seed)
            attacked = SIMULATORS[attack_type](image, rng)
            write_image(output_root / f"preview_{attack_type}.jpg", attacked)
            panels.append(attacked)
            print(f"Wrote {output_root / f'preview_{attack_type}.jpg'}")
        if args.preview:
            height = min(p.shape[0] for p in panels)
            strip = np.hstack(
                [cv2.resize(p, (int(p.shape[1] * height / p.shape[0]), height)) for p in panels]
            )
            write_image(output_root / "preview_strip.jpg", strip)
            print(f"Wrote {output_root / 'preview_strip.jpg'} (original | " + " | ".join(types) + ")")
        return 0

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(
            f"No manifest at {manifest_path}. Run tools/make_splits.py first.",
            file=sys.stderr,
        )
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    records: List[dict] = []
    for split, identities in manifest["splits"].items():
        for identity in identities:
            for image_path in manifest["files"][identity]:
                records.extend(
                    generate_for_image(
                        Path(image_path), output_root, identity, split,
                        args.variants, args.seed, types,
                    )
                )

    index_path = output_root / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "generated_from": str(manifest_path.as_posix()),
                "variants_per_image": args.variants,
                "attack_types": types,
                "seed": args.seed,
                "warning": (
                    "SIMULATED attacks. Accuracy measured on this corpus is an "
                    "upper bound on real-world performance, not an estimate of it."
                ),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    by_type: Dict[str, int] = {}
    by_split: Dict[str, int] = {}
    for record in records:
        by_type[record["attack_type"]] = by_type.get(record["attack_type"], 0) + 1
        by_split[record["split"]] = by_split.get(record["split"], 0) + 1
    print(f"Generated {len(records)} attack images -> {output_root}")
    print(f"  by type : {by_type}")
    print(f"  by split: {by_split}")
    print(f"  index   : {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
