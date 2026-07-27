"""Typed, file-driven configuration.

Every accept/reject threshold in FaceGuard lives in ``config/default.yaml`` and
is loaded into the frozen dataclasses below. Nothing else in the codebase is
permitted to hard-code a decision boundary.

The loader is deliberately strict: an unknown key or a missing key raises
:class:`ConfigError` rather than being ignored. In a security system a typo in a
threshold name must not silently fall back to a permissive default -- that is
exactly the failure mode that turns a tightened config into a no-op.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, get_args, get_origin, get_type_hints

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file is malformed, incomplete or unknown."""


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DetectionConfig:
    detector_model: str
    landmarker_model: str
    working_width: int
    min_detection_confidence: float
    max_faces_in_frame: int
    min_face_ratio: float
    max_face_ratio: float


@dataclass(frozen=True)
class QualityConfig:
    min_overall: float
    min_brightness: float
    min_sharpness: float
    sharpness_reference: float
    contrast_reference: float
    ideal_brightness: float
    reference_face_area: float
    weights: Dict[str, float]


@dataclass(frozen=True)
class RecognitionConfig:
    model_name: str
    match_threshold: float
    margin_threshold: float
    min_agreeing_frames: int
    aggregation: str


@dataclass(frozen=True)
class BlinkConfig:
    ear_closed_ratio: float
    ear_open_ratio: float
    blendshape_closed: float
    blendshape_open: float
    min_blink_ms: float
    max_blink_ms: float
    required_blinks: int


@dataclass(frozen=True)
class TextureConfig:
    model_path: str
    live_threshold: float
    crop_size: int
    min_live_frame_ratio: float


@dataclass(frozen=True)
class MotionConfig:
    min_parallax_per_degree: float
    min_parallax_depth_correlation: float
    min_parallax_rotation_deg: float
    min_nonrigid_residual: float
    min_nonrigid_ratio: float
    max_rigid_translation: float
    min_pose_variation_deg: float
    challenge_yaw_deg: float
    challenge_pitch_deg: float
    challenge_timeout_s: float
    max_pose_jump_deg: float


@dataclass(frozen=True)
class DepthConfig:
    enable_hardware: bool
    min_planarity_residual: float
    allow_software_proxy: bool


@dataclass(frozen=True)
class PolicyConfig:
    required: List[str]
    optional: List[str]
    min_optional_signals: int
    require_challenge: bool
    min_fused_score: float
    weights: Dict[str, float]


@dataclass(frozen=True)
class LivenessConfig:
    window_frames: int
    min_frames: int
    blink: BlinkConfig
    texture: TextureConfig
    motion: MotionConfig
    depth: DepthConfig
    policy: PolicyConfig


@dataclass(frozen=True)
class SessionConfig:
    max_duration_s: float
    max_buffered_frames: int
    recognise_every_n_frames: int


@dataclass(frozen=True)
class SecurityConfig:
    max_failed_attempts: int
    failure_window_s: float
    lockout_base_s: float
    lockout_max_s: float
    template_store: str
    key_file: str
    audit_log: str
    pin_store: str
    pin_min_length: int


@dataclass(frozen=True)
class Config:
    detection: DetectionConfig
    quality: QualityConfig
    recognition: RecognitionConfig
    liveness: LivenessConfig
    session: SessionConfig
    security: SecurityConfig

    def resolve(self, relative_path: str) -> Path:
        """Resolve a config path value against the project root.

        Paths in the YAML are written relative to the repository so that a
        checkout works without editing the file, but an absolute path supplied
        by an operator is respected as-is.
        """
        path = Path(relative_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Convert one YAML value to its annotated type, raising on mismatch."""
    if is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return _build(annotation, value, path)

    origin = get_origin(annotation)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (item_type,) = get_args(annotation)
        return [_coerce(v, item_type, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        _, val_type = get_args(annotation)
        return {k: _coerce(v, val_type, f"{path}.{k}") for k, v in value.items()}

    # bool must be checked before int: bool is a subclass of int in Python.
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported config type {annotation!r}")


def _build(cls: type, data: Dict[str, Any], path: str = "") -> Any:
    """Instantiate a config dataclass, rejecting unknown and missing keys."""
    # ``from __future__ import annotations`` turns field types into strings, so
    # resolve them properly rather than reading dataclass field.type directly.
    hints = get_type_hints(cls)
    expected = {f.name: hints[f.name] for f in fields(cls)}

    unknown = set(data) - set(expected)
    if unknown:
        where = path or "<root>"
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}. "
            f"Valid keys are {sorted(expected)}."
        )
    missing = set(expected) - set(data)
    if missing:
        where = path or "<root>"
        raise ConfigError(f"{where}: missing required key(s) {sorted(missing)}")

    kwargs = {
        name: _coerce(data[name], annotation, f"{path}.{name}".lstrip("."))
        for name, annotation in expected.items()
    }
    return cls(**kwargs)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate configuration.

    Resolution order: explicit ``path`` argument, then the ``FACEGUARD_CONFIG``
    environment variable, then ``config/default.yaml``.
    """
    chosen = Path(path or os.environ.get("FACEGUARD_CONFIG") or DEFAULT_CONFIG_PATH)
    if not chosen.is_file():
        raise ConfigError(f"Configuration file not found: {chosen}")

    with chosen.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{chosen}: top level must be a mapping")

    config = _build(Config, raw)
    _validate_semantics(config)
    return config


def _validate_semantics(config: Config) -> None:
    """Check cross-field invariants that the type system cannot express.

    These are the mistakes that would quietly weaken the system: hysteresis
    bands inverted so blinks can never be counted, a decision window shorter
    than the minimum frame count, or a policy naming a signal that does not
    exist.
    """
    blink = config.liveness.blink
    if blink.ear_closed_ratio >= blink.ear_open_ratio:
        raise ConfigError(
            "liveness.blink: ear_closed_ratio must be below ear_open_ratio, "
            "otherwise the hysteresis band is inverted and no blink can ever "
            "complete"
        )
    if not 0.0 < blink.ear_closed_ratio < 1.0 or not 0.0 < blink.ear_open_ratio < 1.0:
        raise ConfigError(
            "liveness.blink: EAR ratios are fractions of the subject's own "
            "open-eye baseline and must lie strictly between 0 and 1"
        )
    if blink.blendshape_open >= blink.blendshape_closed:
        raise ConfigError(
            "liveness.blink: blendshape_open must be below blendshape_closed"
        )
    if blink.min_blink_ms >= blink.max_blink_ms:
        raise ConfigError("liveness.blink: min_blink_ms must be below max_blink_ms")

    if config.liveness.min_frames > config.liveness.window_frames:
        raise ConfigError(
            "liveness: min_frames cannot exceed window_frames; the decision "
            "window would never fill"
        )
    if config.liveness.min_frames < 2:
        raise ConfigError(
            "liveness.min_frames must be at least 2. Single-frame liveness "
            "decisions are the weakness this system exists to avoid."
        )

    known_signals = {"blink", "texture", "motion", "depth"}
    policy = config.liveness.policy
    for name in [*policy.required, *policy.optional, *policy.weights]:
        if name not in known_signals:
            raise ConfigError(
                f"liveness.policy references unknown signal {name!r}; "
                f"known signals are {sorted(known_signals)}"
            )
    overlap = set(policy.required) & set(policy.optional)
    if overlap:
        raise ConfigError(
            f"liveness.policy: {sorted(overlap)} listed as both required and optional"
        )
    if policy.min_optional_signals > len(policy.optional):
        raise ConfigError(
            "liveness.policy: min_optional_signals exceeds the number of "
            "optional signals, so no attempt could ever pass"
        )

    if config.detection.min_face_ratio >= config.detection.max_face_ratio:
        raise ConfigError("detection: min_face_ratio must be below max_face_ratio")
    if config.recognition.min_agreeing_frames < 2:
        raise ConfigError(
            "recognition.min_agreeing_frames must be at least 2. A match "
            "confirmed by a single frame is a single-frame decision."
        )
    if config.recognition.aggregation not in {"max", "mean"}:
        raise ConfigError("recognition.aggregation must be 'max' or 'mean'")
    if config.security.lockout_base_s > config.security.lockout_max_s:
        raise ConfigError("security: lockout_base_s must not exceed lockout_max_s")
