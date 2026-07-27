"""Motion-based liveness: passive micro-movement and active challenge-response.

**What this catches:** a printed photo, a phone or tablet held up to the camera,
and a static mask. All of them can be *moved*, but none of them can *deform*.

**What this does not catch on its own:** a video replay that already contains
the requested movement. The active challenge raises the cost of that attack --
the instruction is chosen at random per attempt, so an attacker needs footage of
the victim performing that specific movement -- but it does not eliminate it.

**How it works.** Three measurements, in descending order of how much weight
they are given.

*Planar parallax (primary).* This is the one that does the real work. When a
head rotates, near features -- the nose, the brow ridge -- sweep across the
image faster than far ones. No flat object can imitate that, for a precise
geometric reason: rotating a plane in front of a camera is **exactly** a
homography. So fit the best homography mapping the landmarks in one frame onto
the landmarks in a later, rotated frame. If it fits, the thing that turned was
flat. If there is stubborn residual error, the thing that turned had depth.

This is what closes the obvious hole in naive challenge-response. Asking "did
you turn your head" is trivially defeated by tilting a printed photo, and the
head-pose estimator is happy to report a turn either way. Asking "did turning
*reveal three-dimensional structure*" is not defeated by that, because the
attacker would have to make a flat sheet behave non-planarly.

Measured on this project's images: residual per degree of rotation is 0.0022
for real face geometry and 0.0000 for a flat target. It only works while the
head is actually rotating -- at a fixed pose there is no parallax to observe --
which is why the challenge and this check are inseparable.

*Non-rigid deformation (corroborating).* Every frame's landmarks are aligned
onto a reference shape using only rigid parts of the head -- nose bridge,
temples, forehead -- which removes all rigid motion, so waving a photograph
contributes nothing after this step. What remains is measured on the parts that
deform: lips, eyebrows, eyelids, reported as a ratio against the rigid-region
motion so that per-frame landmark jitter cancels.

The honest caveat, from measurement rather than theory: a jittered still image
already scores 1.33 +/- 0.12 on that ratio, up to 1.55, because soft-tissue
landmarks sit on low-contrast edges and are genuinely noisier to localise than
the nose bridge. The separation from live faces is not wide enough to gate on,
so this signal can add confidence but never grant a pass alone.

*Head pose trajectory.* Pose comes from MediaPipe's 4x4 facial transformation
matrix. A challenge requires reaching the requested angle within the time
limit along a *continuous* path: a series that jumps more than
``max_pose_jump_deg`` between consecutive frames means the target in front of
the lens was swapped, not turned.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import MotionConfig
from ..schemas import FaceObservation, ReasonCode, SignalResult, SignalStatus

SIGNAL_NAME = "motion"

# Landmarks on rigid structure. The skull does not change shape with
# expression, so motion measured here is alignment error plus sensor noise --
# in other words, the noise floor.
# Chin (152) and jawline are deliberately excluded: the jaw opens.
RIGID_LANDMARKS: Tuple[int, ...] = (1, 4, 6, 9, 10, 33, 151, 168, 195, 197, 234, 263, 454)

# Landmarks on soft tissue that a living face moves constantly and
# involuntarily: mouth, eyebrows, eyelids.
DEFORMABLE_LANDMARKS: Tuple[int, ...] = (
    # mouth
    0, 13, 14, 17, 61, 78, 181, 291, 308, 405,
    # eyebrows
    63, 66, 70, 105, 107, 293, 296, 300, 334, 336,
    # eyelids
    144, 145, 159, 160, 374, 380, 385, 386,
)

# Interocular reference: outer corners of both eyes. Every distance in this
# module is normalised by this so results do not depend on how close the
# subject stood to the camera.
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33


class ChallengeType(str, Enum):
    """An instruction the subject is asked to perform."""

    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    LOOK_UP = "LOOK_UP"
    LOOK_DOWN = "LOOK_DOWN"

    @property
    def instruction(self) -> str:
        return {
            ChallengeType.TURN_LEFT: "Slowly turn your head to your left",
            ChallengeType.TURN_RIGHT: "Slowly turn your head to your right",
            ChallengeType.LOOK_UP: "Slowly tilt your head up",
            ChallengeType.LOOK_DOWN: "Slowly tilt your head down",
        }[self]


def random_challenge(rng: Optional[random.Random] = None) -> ChallengeType:
    """Pick a challenge uniformly at random.

    Randomness is the entire point. A fixed instruction can be pre-recorded
    once and replayed forever; a random one means an attacker needs footage of
    the victim performing whichever of the four movements is asked for on the
    day. Uses :class:`random.SystemRandom` by default so the choice is not
    predictable from previous attempts.
    """
    source = rng or random.SystemRandom()
    return source.choice(list(ChallengeType))


@dataclass(frozen=True)
class HeadPose:
    """Head orientation in degrees.

    Sign conventions, from the subject's point of view:
    ``yaw`` positive = turning to their own left,
    ``pitch`` positive = tilting up,
    ``roll`` positive = tilting their head toward their own right shoulder.
    """

    yaw: float
    pitch: float
    roll: float


def head_pose(pose_matrix: Optional[np.ndarray]) -> Optional[HeadPose]:
    """Decompose MediaPipe's 4x4 facial transformation matrix into Euler angles.

    The rotation block is re-orthonormalised via SVD first: the matrix comes
    from a fitted model and can drift slightly from a pure rotation, which
    makes a naive ``arcsin`` decomposition produce NaNs near the poles.
    """
    if pose_matrix is None or pose_matrix.shape != (4, 4):
        return None

    rotation = np.asarray(pose_matrix[:3, :3], dtype=np.float64)
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:  # reflection, not a rotation
        u[:, -1] *= -1
        rotation = u @ vt

    # Decomposition for R = Ry(yaw) @ Rx(pitch) @ Rz(roll), which matches how a
    # head actually moves: turn, then nod, then tilt. Writing that product out
    # gives R[1,2] = -sin(pitch), R[0,2] = sin(yaw)cos(pitch),
    # R[2,2] = cos(yaw)cos(pitch), R[1,0] = cos(pitch)sin(roll) and
    # R[1,1] = cos(pitch)cos(roll), which is where each line below comes from.
    sin_pitch = float(np.clip(-rotation[1, 2], -1.0, 1.0))
    pitch = np.arcsin(sin_pitch)
    if abs(sin_pitch) < 0.9999:
        yaw = np.arctan2(rotation[0, 2], rotation[2, 2])
        roll = np.arctan2(rotation[1, 0], rotation[1, 1])
    else:  # gimbal lock: yaw and roll are degenerate, attribute it all to yaw
        yaw = np.arctan2(-rotation[2, 0], rotation[0, 0])
        roll = 0.0
    return HeadPose(
        yaw=float(np.degrees(yaw)),
        pitch=float(np.degrees(pitch)),
        roll=float(np.degrees(roll)),
    )


def _to_metric(landmarks: np.ndarray) -> np.ndarray:
    """Scale normalised landmarks so distances are comparable across frames.

    MediaPipe returns x and y in 0..1 of the image and z on roughly the same
    scale as x. Dividing by interocular distance makes every measurement a
    fraction of face width, so it no longer depends on camera distance.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    interocular = np.linalg.norm(
        points[LEFT_EYE_OUTER][:2] - points[RIGHT_EYE_OUTER][:2]
    )
    if interocular <= 1e-9:
        return points
    return points / interocular


def align_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve the least-squares similarity transform mapping source onto target.

    Umeyama's closed form: rotation from the SVD of the cross-covariance, plus
    an isotropic scale. Returns the rotation, scale and translation as a
    callable applied to any point set, so the transform can be fitted on the
    rigid landmarks and then applied to the deformable ones -- which is the
    whole trick. Fitting on points that move would absorb the very motion we
    are trying to measure.
    """
    src_centre = source.mean(axis=0)
    dst_centre = target.mean(axis=0)
    src_centred = source - src_centre
    dst_centred = target - dst_centre

    covariance = src_centred.T @ dst_centred
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(source.shape[1])
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T

    variance = float((src_centred**2).sum())
    scale = (
        float((singular_values * np.diag(correction)).sum()) / variance
        if variance > 1e-12
        else 1.0
    )

    def transform(points: np.ndarray) -> np.ndarray:
        return (scale * (rotation @ (points - src_centre).T).T) + dst_centre

    return transform  # type: ignore[return-value]


def nonrigid_motion(
    observations: Sequence[FaceObservation],
) -> Optional[Tuple[float, float, float]]:
    """Measure deformable motion against the rigid-landmark noise floor.

    Returns ``(deformable_motion, rigid_motion, ratio)``, all normalised by
    interocular distance, or ``None`` if there were too few landmarked frames.

    Every frame is aligned onto the median face shape using rigid landmarks
    only, which removes translation, rotation and scale -- i.e. all the motion
    a photograph on a stick can produce. The spread of what remains is the
    measurement.
    """
    shapes = [
        _to_metric(observation.landmarks)
        for observation in observations
        if observation.has_landmarks
    ]
    if len(shapes) < 3:
        return None

    stacked = np.stack(shapes)
    # Median rather than mean: robust to the occasional badly-fitted frame,
    # which would otherwise drag the reference shape and inflate every residual.
    reference = np.median(stacked, axis=0)

    aligned = []
    for shape in shapes:
        transform = align_similarity(
            shape[list(RIGID_LANDMARKS)], reference[list(RIGID_LANDMARKS)]
        )
        aligned.append(transform(shape))
    aligned_stack = np.stack(aligned)

    def spread(indices: Sequence[int]) -> float:
        # Per-landmark standard deviation over time, then averaged. This is
        # motion *energy*, not displacement, so it does not matter whether the
        # subject moved and came back.
        subset = aligned_stack[:, list(indices), :]
        return float(np.mean(np.std(subset, axis=0)))

    deformable = spread(DEFORMABLE_LANDMARKS)
    rigid = spread(RIGID_LANDMARKS)
    # The floor stops a perfectly static synthetic input (rigid == 0) from
    # producing an infinite ratio.
    ratio = deformable / max(rigid, 1e-6)
    return deformable, rigid, ratio


def pose_series(observations: Sequence[FaceObservation]) -> List[HeadPose]:
    """Head pose for every frame that carries a transformation matrix."""
    poses = [head_pose(observation.pose_matrix) for observation in observations]
    return [pose for pose in poses if pose is not None]


def _pixel_landmarks(observation: FaceObservation) -> Optional[np.ndarray]:
    """2-D landmarks in pixels, normalised by interocular distance.

    Parallax is measured in image space, because that is where a homography
    acts. The z channel is deliberately discarded here: it is inferred by the
    model rather than measured, so including it would let the model's own
    guess about depth answer a question we are asking of the geometry.
    """
    if not observation.has_landmarks:
        return None
    height, width = observation.frame_shape
    if height <= 0 or width <= 0:
        height, width = 1, 1
    points = observation.landmarks[:, :2].astype(np.float64).copy()
    points[:, 0] *= width
    points[:, 1] *= height
    interocular = float(
        np.linalg.norm(points[RIGHT_EYE_OUTER] - points[LEFT_EYE_OUTER])
    )
    if interocular <= 1e-9:
        return None
    return points / interocular


def homography_residuals(
    source: np.ndarray, target: np.ndarray
) -> Optional[np.ndarray]:
    """Per-landmark reprojection error of the best homography source -> target.

    Near zero everywhere means one flat plane moved. Structured error means the
    point cloud has depth relief no single homography can account for.
    """
    if source is None or target is None or len(source) != len(target) or len(source) < 4:
        return None
    matrix, _ = cv2.findHomography(
        source.astype(np.float32), target.astype(np.float32), cv2.RANSAC, 5.0
    )
    if matrix is None:
        return None
    projected = cv2.perspectiveTransform(
        source.reshape(-1, 1, 2).astype(np.float32), matrix
    ).reshape(-1, 2)
    return np.linalg.norm(projected - target, axis=1)


def homography_residual(source: np.ndarray, target: np.ndarray) -> float:
    """Mean reprojection error. Convenience wrapper for measurement tooling."""
    residuals = homography_residuals(source, target)
    return float(np.mean(residuals)) if residuals is not None else float("nan")


def depth_correlation(residuals: np.ndarray, depths: np.ndarray) -> float:
    """Correlate per-landmark residual against per-landmark depth.

    This is the hard-to-fake half of the parallax test. Real parallax displaces
    a landmark in proportion to how far forward it sits: the nose tip moves
    most, the flat of the cheek least. Landmark refit noise -- which is what a
    flat target actually produces, rather than the theoretical zero -- has no
    such relationship with depth.

    Depth is taken relative to the median so that the correlation measures
    relief rather than absolute distance from the camera.
    """
    relief = np.abs(depths - np.median(depths))
    if residuals.std() < 1e-12 or relief.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(residuals, relief)[0, 1])


@dataclass(frozen=True)
class ParallaxMeasurement:
    """Both halves of the parallax test, plus the rotation they were measured over."""

    residual_per_degree: float
    depth_correlation: float
    rotation_deg: float
    mean_residual: float


def planar_parallax(
    observations: Sequence[FaceObservation], config: MotionConfig
) -> Optional[ParallaxMeasurement]:
    """Test whether observed motion required 3-D structure to produce it.

    Picks the two frames furthest apart in head orientation -- parallax is
    strongest there and entirely absent at a fixed pose -- fits a homography
    between their landmarks, and measures both how badly it fits and whether
    the misfit tracks landmark depth.

    Returns ``None`` when the head never rotated enough for the question to be
    answerable. That is reported as UNAVAILABLE, never as a pass.
    """
    usable = [
        (observation, head_pose(observation.pose_matrix))
        for observation in observations
        if observation.has_landmarks and observation.pose_matrix is not None
    ]
    usable = [(o, p) for o, p in usable if p is not None]
    if len(usable) < 2:
        return None

    # Find the pair with the largest angular separation.
    angles = np.array([[p.yaw, p.pitch] for _, p in usable])
    distances = np.linalg.norm(angles[:, None, :] - angles[None, :, :], axis=-1)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    rotation_deg = float(distances[first, second])
    if rotation_deg < config.min_parallax_rotation_deg:
        return None

    source = _pixel_landmarks(usable[first][0])
    target = _pixel_landmarks(usable[second][0])
    residuals = homography_residuals(source, target)
    if residuals is None:
        return None

    depths = usable[first][0].landmarks[:, 2].astype(np.float64)
    mean_residual = float(np.mean(residuals))
    return ParallaxMeasurement(
        residual_per_degree=mean_residual / max(rotation_deg, 1e-6),
        depth_correlation=depth_correlation(residuals, depths),
        rotation_deg=rotation_deg,
        mean_residual=mean_residual,
    )


def evaluate_parallax(
    observations: Sequence[FaceObservation], config: MotionConfig
) -> SignalResult:
    """Verdict on whether the moving target had real depth.

    Both conditions must hold. Requiring only the residual magnitude lets a
    warped photograph through on refit noise; requiring only the depth
    correlation lets through a target whose landmarks happen to jitter in a
    depth-correlated way. Together they are considerably harder to satisfy by
    accident.
    """
    measurement = planar_parallax(observations, config)
    if measurement is None:
        return SignalResult(
            name="parallax",
            status=SignalStatus.UNAVAILABLE,
            reason=ReasonCode.LIVENESS_UNAVAILABLE,
            detail=(
                "The head did not rotate enough to measure parallax "
                f"(need {config.min_parallax_rotation_deg:.0f} deg between frames)."
            ),
        )

    metrics = {
        "residual_per_degree": measurement.residual_per_degree,
        "depth_correlation": measurement.depth_correlation,
        "rotation_deg": measurement.rotation_deg,
        "mean_residual": measurement.mean_residual,
    }

    correlation = measurement.depth_correlation
    relief_ratio = measurement.residual_per_degree / max(
        config.min_parallax_per_degree, 1e-9
    )

    # Residual is checked first, and a failure here is conclusive on its own.
    # Ordering matters: if the correlation check ran first, a target so flat
    # that its landmark depths have no variance at all would produce an
    # unmeasurable correlation and be reported UNAVAILABLE -- letting a
    # secondary check that could not run overturn a primary check that
    # answered decisively. A perfectly flat surface is the strongest possible
    # evidence of a photograph, not an absence of evidence.
    if relief_ratio < 1.0:
        return SignalResult(
            name="parallax",
            status=SignalStatus.FAIL,
            score=float(np.clip(relief_ratio, 0.0, 1.0)),
            threshold=1.0,
            reason=ReasonCode.FLAT_GEOMETRY,
            detail=(
                f"Over {measurement.rotation_deg:.0f} deg of rotation this face moved "
                f"the way a flat surface moves: {measurement.residual_per_degree:.5f} "
                f"depth relief per degree against the {config.min_parallax_per_degree:.5f} "
                "a real head produces. Consistent with a photo or a screen being "
                "turned toward the camera."
            ),
            metrics=metrics,
        )

    if not np.isfinite(correlation):
        return SignalResult(
            name="parallax",
            status=SignalStatus.UNAVAILABLE,
            reason=ReasonCode.LIVENESS_UNAVAILABLE,
            detail=(
                "Landmark depths had no variation, so the residual could not be "
                "correlated against them. Parallax is not measurable here."
            ),
            metrics=metrics,
        )

    if correlation < config.min_parallax_depth_correlation:
        return SignalResult(
            name="parallax",
            status=SignalStatus.FAIL,
            score=float(
                np.clip(
                    correlation / max(config.min_parallax_depth_correlation, 1e-9),
                    0.0,
                    1.0,
                )
            ),
            threshold=1.0,
            reason=ReasonCode.FLAT_GEOMETRY,
            detail=(
                f"Over {measurement.rotation_deg:.0f} deg of rotation the landmarks "
                f"moved, but not in proportion to their depth (correlation "
                f"{correlation:.2f} against the {config.min_parallax_depth_correlation:.2f} "
                "a real head produces, where the nose displaces far more than the "
                "cheek). Consistent with a flat surface plus tracking noise."
            ),
            metrics=metrics,
        )

    return SignalResult(
        name="parallax",
        status=SignalStatus.PASS,
        score=float(np.clip(min(relief_ratio / 2.0, correlation), 0.0, 1.0)),
        threshold=0.5,
        detail=(
            f"Rotating {measurement.rotation_deg:.0f} deg revealed depth relief no "
            f"flat surface can produce: residual {measurement.residual_per_degree:.5f} "
            f"per degree, correlating {correlation:.2f} with landmark depth."
        ),
        metrics=metrics,
    )


def evaluate_challenge(
    observations: Sequence[FaceObservation],
    challenge: ChallengeType,
    config: MotionConfig,
    elapsed_s: float,
) -> SignalResult:
    """Check whether the requested head movement was actually performed."""
    poses = pose_series(observations)
    if len(poses) < 3:
        return SignalResult(
            name="challenge",
            status=SignalStatus.UNAVAILABLE,
            reason=ReasonCode.LIVENESS_UNAVAILABLE,
            detail="Head pose could not be measured for enough frames.",
        )

    yaws = np.array([p.yaw for p in poses])
    pitches = np.array([p.pitch for p in poses])

    if challenge is ChallengeType.TURN_LEFT:
        achieved, required, axis = float(yaws.max()), config.challenge_yaw_deg, yaws
    elif challenge is ChallengeType.TURN_RIGHT:
        achieved, required, axis = float(-yaws.min()), config.challenge_yaw_deg, yaws
    elif challenge is ChallengeType.LOOK_UP:
        achieved, required, axis = (
            float(pitches.max()),
            config.challenge_pitch_deg,
            pitches,
        )
    else:
        achieved, required, axis = (
            float(-pitches.min()),
            config.challenge_pitch_deg,
            pitches,
        )

    # A real head turn is continuous. A cut between two stills, or one photo
    # being swapped for another at a different angle, shows up as a step change
    # far larger than a head can travel between consecutive frames.
    jumps = np.abs(np.diff(axis)) if len(axis) > 1 else np.array([0.0])
    largest_jump = float(jumps.max())

    metrics = {
        "achieved_deg": achieved,
        "required_deg": required,
        "largest_frame_to_frame_jump_deg": largest_jump,
        "elapsed_s": elapsed_s,
        "frames": float(len(poses)),
    }

    if largest_jump > config.max_pose_jump_deg:
        return SignalResult(
            name="challenge",
            status=SignalStatus.FAIL,
            score=0.0,
            threshold=1.0,
            reason=ReasonCode.CHALLENGE_FAILED,
            detail=(
                f"Head orientation jumped {largest_jump:.0f} deg between consecutive "
                f"frames, beyond the {config.max_pose_jump_deg:.0f} deg a head can "
                "physically travel at this frame rate. This is what swapping the "
                "target in front of the lens looks like."
            ),
            metrics=metrics,
        )

    if achieved < required:
        timed_out = elapsed_s >= config.challenge_timeout_s
        return SignalResult(
            name="challenge",
            status=SignalStatus.FAIL,
            score=max(0.0, achieved / required) if required else 0.0,
            threshold=1.0,
            reason=(
                ReasonCode.CHALLENGE_TIMEOUT if timed_out else ReasonCode.CHALLENGE_FAILED
            ),
            detail=(
                f"{challenge.instruction}: reached {achieved:.0f} deg of the "
                f"{required:.0f} deg required."
            ),
            metrics=metrics,
        )

    return SignalResult(
        name="challenge",
        status=SignalStatus.PASS,
        score=min(1.0, achieved / required) if required else 1.0,
        threshold=1.0,
        detail=f"Challenge satisfied: reached {achieved:.0f} deg in {elapsed_s:.1f} s.",
        metrics=metrics,
    )


def evaluate(
    observations: Sequence[FaceObservation], config: MotionConfig
) -> SignalResult:
    """Motion verdict: did the thing that moved have three-dimensional structure?

    Parallax decides. The non-rigid deformation ratio is measured and attached
    to the metrics for transparency, but it is not allowed to overturn or stand
    in for the parallax verdict, because its measured separation between live
    and static targets is too narrow to be trusted (see module docstring).

    When the head never rotated far enough to observe parallax, the honest
    answer is UNAVAILABLE. It is tempting to fall back to the deformation ratio
    here, but a fallback that is known not to discriminate is exactly the
    silent downgrade this system is built to avoid.
    """
    poses = pose_series(observations)
    pose_variation = (
        float(np.std([p.yaw for p in poses]) + np.std([p.pitch for p in poses]))
        if len(poses) >= 2
        else 0.0
    )

    metrics = {
        "pose_variation_deg": pose_variation,
        "frames_analysed": float(len(observations)),
    }
    deformation = nonrigid_motion(observations)
    if deformation is not None:
        deformable, rigid, ratio = deformation
        metrics.update(
            {
                "deformable_motion": deformable,
                "rigid_motion_noise_floor": rigid,
                "deformable_to_rigid_ratio": ratio,
                "nonrigid_corroborates": float(
                    ratio >= config.min_nonrigid_ratio
                    and deformable >= config.min_nonrigid_residual
                ),
            }
        )

    result = evaluate_parallax(observations, config)
    result.name = SIGNAL_NAME
    result.metrics.update(metrics)
    return result
