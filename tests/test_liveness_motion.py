"""Tests for the parallax check -- the core anti-spoofing claim.

The claim under test is geometric and exact: rotating a flat object in front of
a camera is a homography, so its landmark motion is fully explained by one, and
a real head's is not. These tests assert that directly on synthetic clouds
where the ground truth is known, rather than on photographs where it is not.

If any test in this file fails, a printed photograph turned toward the camera
may be accepted as a live head.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceguard.liveness import motion
from faceguard.liveness.motion import ChallengeType
from faceguard.schemas import ReasonCode, SignalStatus

from .conftest import make_observation, rotate_cloud, synthetic_face_cloud


def _turning_sequence(relief: float, degrees=(0, 6, 12, 18), seed: int = 0):
    """Frames of a cloud rotating through the given yaw angles."""
    base = synthetic_face_cloud(relief=relief, seed=seed)
    return [
        make_observation(
            landmarks=rotate_cloud(base, yaw),
            timestamp=index / 15.0,
            yaw_deg=yaw,
        )
        for index, yaw in enumerate(degrees)
    ]


class TestPlanarParallax:
    def test_real_relief_is_not_explained_by_a_homography(self, config):
        """A cloud with depth must produce residual no homography can absorb."""
        measurement = motion.planar_parallax(
            _turning_sequence(relief=1.0), config.liveness.motion
        )
        assert measurement is not None
        assert measurement.residual_per_degree > 0.0
        assert measurement.rotation_deg >= 18.0

    def test_flat_target_is_explained_by_a_homography(self, config):
        """A flat cloud rotated in 3-D must fit a homography almost exactly.

        This is the property the whole check rests on. A photograph tilted
        toward the camera undergoes exactly this transform.
        """
        measurement = motion.planar_parallax(
            _turning_sequence(relief=0.0), config.liveness.motion
        )
        assert measurement is not None
        assert measurement.residual_per_degree < 1e-4, (
            "A flat rotating target produced parallax residual. The homography "
            "test is broken and photographs will pass."
        )

    def test_flat_target_fails_the_signal(self, config):
        result = motion.evaluate_parallax(
            _turning_sequence(relief=0.0), config.liveness.motion
        )
        assert result.status is SignalStatus.FAIL
        assert result.reason is ReasonCode.FLAT_GEOMETRY

    def test_real_relief_passes_the_signal(self, config):
        result = motion.evaluate_parallax(
            _turning_sequence(relief=1.0), config.liveness.motion
        )
        assert result.status is SignalStatus.PASS
        assert result.metrics["depth_correlation"] >= (
            config.liveness.motion.min_parallax_depth_correlation
        )

    def test_relief_separates_from_flat_by_a_wide_margin(self, config):
        """The two populations must not merely differ, but differ decisively."""
        flat = motion.planar_parallax(
            _turning_sequence(relief=0.0), config.liveness.motion
        )
        real = motion.planar_parallax(
            _turning_sequence(relief=1.0), config.liveness.motion
        )
        assert real.residual_per_degree > flat.residual_per_degree * 50

    def test_no_rotation_reports_unavailable_not_pass(self, config):
        """Without rotation there is no parallax to observe.

        The dangerous failure here would be returning PASS: a subject who never
        turns would sail through the check that is supposed to be hardest.
        """
        still = [
            make_observation(landmarks=synthetic_face_cloud(relief=1.0), yaw_deg=0.0)
            for _ in range(10)
        ]
        result = motion.evaluate_parallax(still, config.liveness.motion)
        assert result.status is SignalStatus.UNAVAILABLE
        assert result.status is not SignalStatus.PASS

    def test_too_few_frames_is_unavailable(self, config):
        assert motion.planar_parallax([], config.liveness.motion) is None
        assert (
            motion.planar_parallax(
                [make_observation()], config.liveness.motion
            )
            is None
        )


class TestDepthCorrelation:
    def test_depth_correlated_residual_scores_high(self):
        depths = np.linspace(0.0, 1.0, 60)
        residuals = np.abs(depths - np.median(depths)) * 3.0
        assert motion.depth_correlation(residuals, depths) > 0.95

    def test_uncorrelated_residual_scores_low(self):
        rng = np.random.default_rng(3)
        depths = np.linspace(0.0, 1.0, 200)
        residuals = rng.normal(1.0, 0.3, 200)
        assert abs(motion.depth_correlation(residuals, depths)) < 0.3

    def test_degenerate_input_returns_nan_not_a_number(self):
        """Constant residuals have no correlation; returning 0.0 would read as
        'measured and found uncorrelated' rather than 'not measurable'."""
        depths = np.linspace(0, 1, 20)
        assert np.isnan(motion.depth_correlation(np.ones(20), depths))


class TestHeadPose:
    def test_identity_matrix_is_neutral(self):
        pose = motion.head_pose(np.eye(4))
        assert pose is not None
        assert abs(pose.yaw) < 1e-6 and abs(pose.pitch) < 1e-6

    @pytest.mark.parametrize("yaw", [-30.0, -12.0, 12.0, 30.0])
    def test_yaw_round_trips(self, yaw):
        from .conftest import pose_matrix

        pose = motion.head_pose(pose_matrix(yaw_deg=yaw))
        assert pose == pytest.approx(pose, abs=1e-6)
        assert pose.yaw == pytest.approx(yaw, abs=0.5)

    @pytest.mark.parametrize("pitch", [-20.0, -8.0, 8.0, 20.0])
    def test_pitch_round_trips(self, pitch):
        from .conftest import pose_matrix

        pose = motion.head_pose(pose_matrix(pitch_deg=pitch))
        assert pose.pitch == pytest.approx(pitch, abs=0.5)

    def test_non_orthogonal_matrix_does_not_produce_nan(self):
        """Fitted matrices drift from pure rotations; a naive arcsin yields NaN."""
        matrix = np.eye(4)
        matrix[:3, :3] = np.array(
            [[1.02, 0.01, 0.0], [0.0, 0.99, 0.02], [0.01, 0.0, 1.03]]
        )
        pose = motion.head_pose(matrix)
        assert pose is not None
        assert np.isfinite([pose.yaw, pose.pitch, pose.roll]).all()

    def test_missing_matrix_returns_none(self):
        assert motion.head_pose(None) is None
        assert motion.head_pose(np.eye(3)) is None


class TestChallenge:
    def test_satisfied_turn_passes(self, config):
        frames = _turning_sequence(relief=1.0, degrees=(0, 8, 16, 24))
        result = motion.evaluate_challenge(
            frames, ChallengeType.TURN_LEFT, config.liveness.motion, 3.0
        )
        assert result.status is SignalStatus.PASS

    def test_insufficient_turn_fails(self, config):
        frames = _turning_sequence(relief=1.0, degrees=(0, 2, 4, 5))
        result = motion.evaluate_challenge(
            frames, ChallengeType.TURN_LEFT, config.liveness.motion, 3.0
        )
        assert result.status is SignalStatus.FAIL
        assert result.reason is ReasonCode.CHALLENGE_FAILED

    def test_turning_the_wrong_way_fails(self, config):
        """Turning right must not satisfy a request to turn left."""
        frames = _turning_sequence(relief=1.0, degrees=(0, -8, -16, -24))
        result = motion.evaluate_challenge(
            frames, ChallengeType.TURN_LEFT, config.liveness.motion, 3.0
        )
        assert result.status is SignalStatus.FAIL

    def test_discontinuous_pose_is_rejected_as_a_swap(self, config):
        """A jump larger than a head can travel means the target was swapped."""
        frames = _turning_sequence(relief=1.0, degrees=(0, 3, 45, 48))
        result = motion.evaluate_challenge(
            frames, ChallengeType.TURN_LEFT, config.liveness.motion, 3.0
        )
        assert result.status is SignalStatus.FAIL
        assert result.reason is ReasonCode.CHALLENGE_FAILED
        assert "jump" in result.detail.lower()

    def test_timeout_is_distinguished_from_failure(self, config):
        """'You ran out of time' and 'you did the wrong thing' are different."""
        frames = _turning_sequence(relief=1.0, degrees=(0, 2, 4, 5))
        result = motion.evaluate_challenge(
            frames,
            ChallengeType.TURN_LEFT,
            config.liveness.motion,
            config.liveness.motion.challenge_timeout_s + 1.0,
        )
        assert result.reason is ReasonCode.CHALLENGE_TIMEOUT

    def test_challenges_are_randomised(self):
        """A fixed challenge could be pre-recorded once and replayed forever."""
        seen = {motion.random_challenge() for _ in range(200)}
        assert len(seen) == len(ChallengeType)


class TestNonRigidMotion:
    def test_identical_frames_produce_no_motion(self):
        cloud = synthetic_face_cloud(relief=1.0)
        frames = [make_observation(landmarks=cloud) for _ in range(8)]
        deformable, rigid, _ = motion.nonrigid_motion(frames)
        assert deformable == pytest.approx(0.0, abs=1e-9)
        assert rigid == pytest.approx(0.0, abs=1e-9)

    def test_rigid_motion_alone_is_removed_by_alignment(self):
        """Waving a photograph around must not register as facial deformation.

        The frames below differ only by a rigid transform, which is exactly
        what a photo on a stick produces.
        """
        cloud = synthetic_face_cloud(relief=1.0)
        frames = [
            make_observation(landmarks=(cloud + shift).astype(np.float32))
            for shift in (0.0, 0.01, 0.02, -0.015, 0.005)
        ]
        deformable, _, _ = motion.nonrigid_motion(frames)
        assert deformable < 1e-6

    def test_too_few_frames_returns_none(self):
        assert motion.nonrigid_motion([make_observation()]) is None


class TestSimilarityAlignment:
    def test_alignment_recovers_a_known_transform(self):
        rng = np.random.default_rng(11)
        source = rng.normal(size=(40, 3))
        theta = np.radians(23.0)
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        target = (source @ rotation.T) * 1.7 + np.array([3.0, -2.0, 0.5])
        transform = motion.align_similarity(source, target)
        assert np.allclose(transform(source), target, atol=1e-9)
