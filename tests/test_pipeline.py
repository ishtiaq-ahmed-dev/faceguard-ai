"""Tests for configuration, quality, texture features, storage, fusion and
rate limiting.

Grouped in one module because each area needs only a handful of assertions and
they share the same fixtures. The security-relevant behaviours are marked in
the docstrings: a failure in any of them is a weakened system, not a cosmetic
regression.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from faceguard.audit import AuditLog, hash_client
from faceguard.config import ConfigError, load_config
from faceguard.liveness import texture
from faceguard.liveness.fusion import LivenessEngine, LivenessEvidence
from faceguard.liveness.texture import SpoofClassifier
from faceguard.pin import PinError, PinStore
from faceguard.quality import (
    Preprocessor,
    QualityAssessor,
    crop_face,
    standardise,
)
from faceguard.schemas import (
    BBox,
    ReasonCode,
    SignalResult,
    SignalStatus,
    VerificationResult,
    summarise_reasons,
)
from faceguard.session import RateLimiter
from faceguard.store import StoreError, Template, TemplateStore, generate_key

from .conftest import make_observation, noisy_crop

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _config_with(**overrides):
    """Write a temporary config with a nested override applied."""
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        node = raw
        *path, leaf = dotted.split(".")
        for key in path:
            node = node[key]
        node[leaf] = value
    path = Path(tempfile.mkdtemp()) / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #


class TestConfig:
    def test_default_config_loads(self, config):
        assert config.recognition.model_name
        assert config.liveness.min_frames >= 2

    def test_unknown_key_is_rejected(self):
        """A typo in a threshold name must not silently use a permissive default."""
        with pytest.raises(ConfigError, match="unknown key"):
            load_config(_config_with(**{"recognition.model_name": "x"}) and
                        _write_with_extra_key())

    def test_missing_key_is_rejected(self):
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        del raw["recognition"]["match_threshold"]
        path = Path(tempfile.mkdtemp()) / "c.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="missing required key"):
            load_config(path)

    def test_wrong_type_is_rejected(self):
        with pytest.raises(ConfigError):
            load_config(_config_with(**{"liveness.min_frames": "fifteen"}))

    def test_single_frame_liveness_is_refused(self):
        """min_frames of 1 is the exact weakness this system exists to avoid."""
        with pytest.raises(ConfigError, match="Single-frame"):
            load_config(_config_with(**{"liveness.min_frames": 1}))

    def test_inverted_blink_hysteresis_is_refused(self):
        with pytest.raises(ConfigError, match="hysteresis"):
            load_config(_config_with(**{"liveness.blink.ear_closed_ratio": 0.95}))

    def test_unknown_policy_signal_is_refused(self):
        with pytest.raises(ConfigError, match="unknown signal"):
            load_config(_config_with(**{"liveness.policy.required": ["telepathy"]}))

    def test_unsatisfiable_policy_is_refused(self):
        """A policy no attempt could ever pass is a configuration error."""
        with pytest.raises(ConfigError, match="min_optional_signals"):
            load_config(_config_with(**{"liveness.policy.min_optional_signals": 9}))

    def test_single_frame_recognition_is_refused(self):
        with pytest.raises(ConfigError, match="single-frame"):
            load_config(_config_with(**{"recognition.min_agreeing_frames": 1}))


def _write_with_extra_key():
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["recognition"]["mach_threshold"] = 0.6  # deliberate typo
    path = Path(tempfile.mkdtemp()) / "typo.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #


class TestQuality:
    def test_standardise_downscales_but_never_upscales(self):
        """Upscaling would invent detail the sharpness measure then rewards."""
        big = np.zeros((1200, 1600, 3), dtype=np.uint8)
        assert standardise(big, 640).shape[1] == 640
        small = np.zeros((60, 80, 3), dtype=np.uint8)
        assert standardise(small, 640).shape == small.shape

    def test_standardise_preserves_aspect_ratio(self):
        image = np.zeros((900, 1600, 3), dtype=np.uint8)
        result = standardise(image, 640)
        assert result.shape[1] / result.shape[0] == pytest.approx(1600 / 900, rel=0.01)

    def test_sharpness_ordering(self, config):
        """A blurred image must score lower than a detailed one."""
        assessor = QualityAssessor(config.quality)
        assert assessor.sharpness(noisy_crop(seed=1)) > assessor.sharpness(
            noisy_crop(seed=1, smooth=True)
        )

    def test_brightness_penalises_both_extremes(self, config):
        assessor = QualityAssessor(config.quality)
        mid = np.full((64, 64, 3), 127, dtype=np.uint8)
        dark = np.zeros((64, 64, 3), dtype=np.uint8)
        bright = np.full((64, 64, 3), 255, dtype=np.uint8)
        assert assessor.brightness(mid) > assessor.brightness(dark)
        assert assessor.brightness(mid) > assessor.brightness(bright)

    def test_low_light_is_reported_separately_from_low_quality(self, config):
        """Low light is the one a user can actually fix, so it gets its own code."""
        assessor = QualityAssessor(config.quality)
        dark = np.full((120, 120, 3), 8, dtype=np.uint8)
        reasons = assessor.gate(assessor.assess(dark, BBox(0, 0, 120, 120)))
        assert ReasonCode.LOW_LIGHT in reasons

    def test_crop_is_clamped_to_the_frame(self):
        """A margin running off the edge must not produce an empty crop."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop, box = crop_face(frame, BBox(90, 90, 40, 40), margin=0.5)
        assert crop.size > 0
        assert box.x + box.width <= 100 and box.y + box.height <= 100

    def test_empty_crop_scores_zero_rather_than_crashing(self, config):
        assessor = QualityAssessor(config.quality)
        scores = assessor.assess(np.zeros((0, 0, 3), dtype=np.uint8), BBox(0, 0, 1, 1))
        assert scores.overall == 0.0

    def test_enhancement_is_available_but_not_applied_to_liveness(self, config):
        """Sharpening manufactures the micro-texture the spoof check measures.

        This test documents the contract: the preprocessor exists for
        recognition only. The session never routes crops through it before
        liveness, and this asserts the enhancement genuinely does change the
        pixels, so that routing mistake would be a real one.
        """
        image = noisy_crop(seed=2, smooth=True)
        enhanced = Preprocessor.sharpen(image)
        assert not np.array_equal(image, enhanced)


# --------------------------------------------------------------------------- #


class TestTextureFeatures:
    def test_feature_vector_length_matches_names(self):
        features = texture.extract_features(noisy_crop())
        assert features.shape == (texture.feature_dimension(),)
        assert len(texture.feature_names()) == texture.feature_dimension()

    def test_features_are_finite(self):
        """A NaN reaching the classifier makes its output meaningless."""
        for seed in range(5):
            assert np.isfinite(texture.extract_features(noisy_crop(seed=seed))).all()

    def test_empty_crop_yields_a_zero_vector(self):
        features = texture.extract_features(np.zeros((0, 0, 3), dtype=np.uint8))
        assert features.shape == (texture.feature_dimension(),)

    def test_lbp_histogram_is_normalised(self):
        histogram = texture.lbp_histogram(noisy_crop()[:, :, 0])
        assert histogram.shape == (texture.LBP_BINS,)
        assert histogram.sum() == pytest.approx(1.0)

    def test_lbp_distinguishes_flat_from_textured(self):
        """Flat regions produce one dominant uniform pattern; noise does not."""
        flat = np.full((64, 64), 128, dtype=np.uint8)
        assert texture.lbp_histogram(flat).max() > 0.9
        assert texture.lbp_histogram(noisy_crop()[:, :, 0]).max() < 0.6

    def test_smoothing_reduces_high_frequency_energy(self):
        """The spectral feature must track the detail loss printing causes."""
        sharp = texture.radial_spectrum(noisy_crop(seed=4)[:, :, 0])
        smooth = texture.radial_spectrum(noisy_crop(seed=4, smooth=True)[:, :, 0])
        assert smooth[-1] < sharp[-1]

    def test_features_are_size_invariant(self):
        """Standing further away must not look like holding up a small photo."""
        import cv2

        crop = noisy_crop(size=200, seed=7)
        a = texture.extract_features(crop)
        b = texture.extract_features(cv2.resize(crop, (320, 320)))
        assert np.corrcoef(a, b)[0, 1] > 0.9

    def test_missing_classifier_is_unavailable_not_pass(self, config):
        """A missing model must refuse, never wave attempts through."""
        result = texture.evaluate([noisy_crop()], config.liveness.texture, None)
        assert result.status is SignalStatus.UNAVAILABLE
        assert result.reason is ReasonCode.LIVENESS_UNAVAILABLE

    def test_classifier_load_of_missing_file_returns_none(self):
        assert SpoofClassifier.load(Path("does/not/exist.joblib")) is None


# --------------------------------------------------------------------------- #


class _StubClassifier:
    """Returns a fixed live-probability for every frame."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def live_probability(self, features):
        return np.full(len(np.atleast_2d(features)), self.probability)


class TestTexturePolicy:
    def test_confident_live_passes(self, config):
        result = texture.evaluate(
            [noisy_crop()] * 5, config.liveness.texture, _StubClassifier(0.95)
        )
        assert result.status is SignalStatus.PASS

    def test_confident_spoof_fails_with_the_right_reason(self, config):
        result = texture.evaluate(
            [noisy_crop()] * 5, config.liveness.texture, _StubClassifier(0.05)
        )
        assert result.status is SignalStatus.FAIL
        assert result.reason is ReasonCode.TEXTURE_SPOOF_SUSPECTED


# --------------------------------------------------------------------------- #


def _signal(name, status, score=1.0):
    return SignalResult(name=name, status=status, score=score)


class TestFusionPolicy:
    def _engine(self, config):
        return LivenessEngine(config, classifier=_StubClassifier(0.95))

    def test_too_few_frames_refuses(self, config):
        engine = self._engine(config)
        evidence = LivenessEvidence(observations=[make_observation()], crops=[])
        result = engine.evaluate(evidence)
        assert not result.passed
        assert ReasonCode.INSUFFICIENT_FRAMES in result.reasons

    def test_unavailable_required_signal_refuses(self, config):
        """A required check that could not run must never read as passed."""
        engine = LivenessEngine(config, classifier=None)  # no texture model
        frames = [make_observation(timestamp=i / 30.0) for i in range(30)]
        result = engine.evaluate(
            LivenessEvidence(observations=frames, crops=[noisy_crop()] * 30)
        )
        assert not result.passed
        assert ReasonCode.LIVENESS_UNAVAILABLE in result.reasons

    def test_missing_depth_marks_the_result_degraded(self, config):
        engine = self._engine(config)
        frames = [make_observation(timestamp=i / 30.0) for i in range(30)]
        result = engine.evaluate(
            LivenessEvidence(observations=frames, crops=[noisy_crop()] * 30)
        )
        assert result.degraded
        assert result.degraded_detail

    def test_unavailable_signals_do_not_drag_the_score_down(self, config):
        """Excluded from the weighted mean, not scored zero.

        Scoring an absent depth camera as zero would penalise every honest
        user and push an operator toward lowering the threshold -- weakening
        the system in response to a hardware gap instead of reporting it.
        """
        weights = {"texture": 0.5, "depth": 0.5}
        both = LivenessEngine._fused_score(
            {
                "texture": _signal("texture", SignalStatus.PASS, 0.9),
                "depth": _signal("depth", SignalStatus.UNAVAILABLE, 0.0),
            },
            weights,
        )
        assert both == pytest.approx(0.9)

    def test_zero_weight_signals_are_ignored(self, config):
        score = LivenessEngine._fused_score(
            {
                "texture": _signal("texture", SignalStatus.PASS, 0.8),
                "depth": _signal("depth", SignalStatus.PASS, 0.0),
            },
            {"texture": 1.0, "depth": 0.0},
        )
        assert score == pytest.approx(0.8)

    def test_no_available_signals_scores_zero(self):
        assert LivenessEngine._fused_score({}, {"texture": 1.0}) == 0.0


# --------------------------------------------------------------------------- #


class TestTemplateStore:
    def _store(self):
        directory = Path(tempfile.mkdtemp())
        return TemplateStore(directory / "t.enc", directory / "k.key"), directory

    def _template(self, identity="alice", quality=0.9):
        return Template(
            identity=identity,
            embedding=np.random.rand(64).astype(np.float32),
            quality=quality,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def test_round_trip(self):
        store, directory = self._store()
        template = self._template()
        store.add(template)
        reopened = TemplateStore(directory / "t.enc", directory / "k.key")
        assert np.allclose(reopened.templates_for("alice")[0].embedding, template.embedding)

    def test_identity_names_are_not_in_plaintext(self):
        """The most basic property: the file must not be readable."""
        store, directory = self._store()
        store.add(self._template(identity="SecretPerson"))
        assert b"SecretPerson" not in (directory / "t.enc").read_bytes()

    def test_tampering_is_detected(self):
        """Fernet tokens are authenticated, so this check is meaningful."""
        store, directory = self._store()
        store.add(self._template())
        raw = (directory / "t.enc").read_bytes()
        (directory / "t.enc").write_bytes(raw[:-4] + b"XXXX")
        with pytest.raises(StoreError):
            TemplateStore(directory / "t.enc", directory / "k.key")

    def test_wrong_key_cannot_decrypt(self):
        store, directory = self._store()
        store.add(self._template())
        (directory / "other.key").write_bytes(generate_key().encode())
        with pytest.raises(StoreError):
            TemplateStore(directory / "t.enc", directory / "other.key")

    def test_lowest_quality_template_is_evicted_first(self):
        """Dropping the best enrolment photo would degrade matching over time."""
        store, _ = self._store()
        for index in range(14):
            store.add(self._template(quality=index / 20.0), max_per_identity=10)
        qualities = sorted(t.quality for t in store.templates_for("alice"))
        assert len(qualities) == 10
        assert qualities[0] == pytest.approx(4 / 20.0)

    def test_delete_removes_everything_for_an_identity(self):
        store, _ = self._store()
        store.add(self._template())
        assert store.remove_identity("alice")
        assert store.identities() == []
        assert not store.remove_identity("alice")


# --------------------------------------------------------------------------- #


class TestRateLimiter:
    def test_lockout_after_the_configured_failures(self, config):
        limiter = RateLimiter(config)
        now = 1000.0
        for index in range(config.security.max_failed_attempts - 1):
            assert limiter.record_failure("a", now + index) == 0.0
        assert limiter.record_failure("a", now + 10) > 0
        assert not limiter.check("a", now + 11)[0]

    def test_lockouts_grow(self, config):
        limiter = RateLimiter(config)
        now = 1000.0
        first = second = 0.0
        for _ in range(config.security.max_failed_attempts):
            first = limiter.record_failure("a", now)
        now += config.security.lockout_base_s + 1
        for _ in range(config.security.max_failed_attempts):
            second = limiter.record_failure("a", now)
        assert second > first

    def test_clients_are_independent(self, config):
        limiter = RateLimiter(config)
        for _ in range(config.security.max_failed_attempts):
            limiter.record_failure("attacker", 1000.0)
        assert not limiter.check("attacker", 1000.0)[0]
        assert limiter.check("bystander", 1000.0)[0]

    def test_success_clears_history(self, config):
        limiter = RateLimiter(config)
        for _ in range(config.security.max_failed_attempts - 1):
            limiter.record_failure("a", 1000.0)
        limiter.record_success("a")
        assert limiter.record_failure("a", 1000.0) == 0.0

    def test_old_failures_fall_out_of_the_window(self, config):
        """Failures spread over hours are not an attack."""
        limiter = RateLimiter(config)
        for index in range(config.security.max_failed_attempts * 2):
            elapsed = index * (config.security.failure_window_s + 1)
            assert limiter.record_failure("a", 1000.0 + elapsed) == 0.0


# --------------------------------------------------------------------------- #


class TestPinStore:
    def _store(self):
        return PinStore(Path(tempfile.mkdtemp()) / "pins.json", min_length=6)

    def test_set_and_verify(self):
        store = self._store()
        store.set_pin("alice", "914273")
        assert store.verify("alice", "914273")
        assert not store.verify("alice", "914274")

    def test_pin_is_never_stored_in_plaintext(self):
        store = self._store()
        store.set_pin("alice", "914273")
        assert "914273" not in store.path.read_text(encoding="utf-8")

    def test_identical_pins_hash_differently(self):
        """Per-entry salts: one cracked entry must not reveal others."""
        store = self._store()
        store.set_pin("alice", "914273")
        store.set_pin("bob", "914273")
        entries = json.loads(store.path.read_text(encoding="utf-8"))
        assert entries["alice"]["hash"] != entries["bob"]["hash"]

    def test_unknown_identity_returns_false(self):
        assert not self._store().verify("nobody", "914273")

    @pytest.mark.parametrize("bad", ["12345", "111111", "123456", "987654", "abcdef"])
    def test_weak_pins_are_rejected(self, bad):
        with pytest.raises(PinError):
            self._store().set_pin("alice", bad)


# --------------------------------------------------------------------------- #


class TestSchemas:
    def test_specific_reasons_outrank_aggregate_ones(self):
        """'No blink detected' is more useful than 'score too low'."""
        assert summarise_reasons(
            [ReasonCode.LIVENESS_SCORE_TOO_LOW, ReasonCode.NO_BLINK_DETECTED]
        ) is ReasonCode.NO_BLINK_DETECTED

    def test_lockout_outranks_everything(self):
        assert summarise_reasons(
            [ReasonCode.NO_MATCH, ReasonCode.LOCKED_OUT]
        ) is ReasonCode.LOCKED_OUT

    def test_empty_reasons_means_accepted(self):
        assert summarise_reasons([]) is ReasonCode.ACCEPTED

    def test_result_serialises_to_json(self):
        result = VerificationResult.rejected(ReasonCode.NO_BLINK_DETECTED)
        payload = json.dumps(result.to_dict())
        restored = json.loads(payload)
        assert restored["accepted"] is False
        assert restored["reason"] == "NO_BLINK_DETECTED"
        assert restored["message"]

    def test_every_reason_code_has_a_message(self):
        """A reason with no explanation is not auditable."""
        for code in ReasonCode:
            assert code.message

    def test_bbox_clipping_never_produces_empty_boxes(self):
        assert BBox(-50, -50, 10, 10).clipped(100, 100).width >= 1
        assert BBox(200, 200, 40, 40).clipped(100, 100).height >= 1


# --------------------------------------------------------------------------- #


class TestAuditLog:
    def test_rejections_are_recorded_with_full_detail(self):
        """A success-only log cannot show an attack in progress."""
        path = Path(tempfile.mkdtemp()) / "audit.jsonl"
        log = AuditLog(path)
        log.record_verification(
            VerificationResult.rejected(ReasonCode.TEXTURE_SPOOF_SUSPECTED),
            client_id="10.0.0.5",
        )
        records = list(log.read())
        assert len(records) == 1
        assert records[0]["accepted"] is False
        assert records[0]["reason"] == "TEXTURE_SPOOF_SUSPECTED"

    def test_client_identifiers_are_hashed(self):
        path = Path(tempfile.mkdtemp()) / "audit.jsonl"
        log = AuditLog(path)
        log.record_verification(
            VerificationResult.rejected(ReasonCode.NO_MATCH), client_id="10.0.0.5"
        )
        assert "10.0.0.5" not in path.read_text(encoding="utf-8")

    def test_hashing_is_stable_and_one_way(self):
        assert hash_client("abc") == hash_client("abc")
        assert hash_client("abc") != hash_client("abd")
        assert "abc" not in hash_client("abc")

    def test_malformed_lines_are_skipped_not_fatal(self):
        path = Path(tempfile.mkdtemp()) / "audit.jsonl"
        path.write_text('{"event":"ok"}\nnot json\n{"event":"also ok"}\n', encoding="utf-8")
        assert len(list(AuditLog(path).read())) == 2


class TestReasonQuality:
    """The reason surfaced to a user must be the most actionable true one."""

    def test_optional_unavailable_does_not_mask_a_specific_failure(self, config):
        """A static photo must report NO_BLINK_DETECTED, not LIVENESS_UNAVAILABLE.

        Both are true -- the parallax check genuinely could not run, because the
        subject never turned their head -- but "you did not blink" tells the
        user something they can act on, while "a check could not run" does not.
        Only a *required* signal being unavailable is loud enough to outrank it.
        """
        engine = LivenessEngine(config, classifier=_StubClassifier(0.95))
        frames = [
            make_observation(blink=0.03, timestamp=index / 30.0, yaw_deg=0.0)
            for index in range(30)
        ]
        result = engine.evaluate(
            LivenessEvidence(observations=frames, crops=[noisy_crop()] * 30)
        )
        assert not result.passed
        assert ReasonCode.NO_BLINK_DETECTED in result.reasons
        assert summarise_reasons(result.reasons) is not ReasonCode.LIVENESS_UNAVAILABLE

    def test_required_unavailable_is_still_loud(self, config):
        """A missing spoof classifier must still surface as LIVENESS_UNAVAILABLE."""
        engine = LivenessEngine(config, classifier=None)
        frames = [make_observation(timestamp=index / 30.0) for index in range(30)]
        result = engine.evaluate(
            LivenessEvidence(observations=frames, crops=[noisy_crop()] * 30)
        )
        assert ReasonCode.LIVENESS_UNAVAILABLE in result.reasons
