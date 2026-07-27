"""Face recognition: which enrolled identity is this, and how sure are we?

This module answers **only** the identity question. Whether the face in front
of the camera is attached to a living person is decided entirely separately, in
:mod:`faceguard.liveness`, and the two confidences are never blended. A system
that reports one number cannot tell you whether it failed because it did not
recognise you or because it thought you were a photograph, and those call for
opposite responses from the user.

Three rules make the identity decision harder to fool than a nearest-neighbour
lookup:

1. **A margin, not just a threshold.** The best match must beat the runner-up
   by a configured gap. Two enrolled people who both score 0.63 against a
   0.62 threshold is not a match, it is an ambiguity, and it is reported as
   ``AMBIGUOUS_MATCH`` rather than resolved by picking the larger float.
2. **Agreement across frames.** An identity must win on several independent
   frames. One frame catching a favourable angle is not evidence.
3. **No stickiness.** Once identified, an identity is never latched. The
   prototype this replaced had a "VIP mode" that pinned a name to a tracked
   face and defended it against contrary evidence, and a re-acquisition path
   that assigned a recently-seen name to any new face without comparing it to
   anything. Both are absent here by design.
"""

from __future__ import annotations

import sys
import threading
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, RecognitionConfig
from .schemas import QualityScores, ReasonCode, RecognitionResult
from .store import Template, TemplateStore


def _ensure_utf8_stdout() -> None:
    """Make stdout UTF-8 capable before importing DeepFace.

    DeepFace logs emoji. On a Windows console defaulting to cp1252 that raises
    UnicodeEncodeError from inside its weight downloader and surfaces as a
    bogus "could not download model" failure. Reconfiguring the stream is
    cheaper than fighting the dependency.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and getattr(stream, "encoding", "") and (
                stream.encoding.lower() not in {"utf-8", "utf8"}
            ):
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):  # pragma: no cover - exotic streams
            pass


class EmbeddingBackend:
    """Produces face embeddings, loading the model once and reusing it.

    DeepFace resolves and caches models globally, but the first call also
    downloads weights and builds the graph, which takes tens of seconds. Doing
    that lazily inside a live verification would blow the session timeout, so
    :meth:`warm_up` exists to pay the cost at startup instead.
    """

    def __init__(self, config: RecognitionConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._ready = False

    def warm_up(self) -> None:
        """Load model weights and run one throwaway inference."""
        with self._lock:
            if self._ready:
                return
            _ensure_utf8_stdout()
            from deepface import DeepFace  # imported lazily: heavy TensorFlow import

            DeepFace.represent(
                img_path=np.zeros((160, 160, 3), dtype=np.uint8),
                model_name=self.config.model_name,
                enforce_detection=False,
                detector_backend="skip",
            )
            self._ready = True

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Embed one already-cropped face.

        ``detector_backend="skip"`` is deliberate: detection has already
        happened upstream with MediaPipe, and letting DeepFace re-detect would
        both double the cost and risk it picking a different region than the
        one every liveness signal was measured on.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None
        self.warm_up()
        from deepface import DeepFace

        try:
            with self._lock:
                results = DeepFace.represent(
                    img_path=face_bgr,
                    model_name=self.config.model_name,
                    enforce_detection=False,
                    detector_backend="skip",
                )
        except Exception:
            return None
        if not results:
            return None
        return np.asarray(results[0]["embedding"], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, guarded against zero-norm vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


class Recogniser:
    """Matches probe embeddings against the enrolled template store."""

    def __init__(
        self,
        config: Config,
        store: TemplateStore,
        backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self.config = config
        self.store = store
        self.backend = backend or EmbeddingBackend(config.recognition)

    # -- single frame ------------------------------------------------------- #

    def score_identities(self, embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Similarity of one probe embedding to every enrolled identity.

        Sorted best first. Per-identity aggregation across that person's
        templates is configurable: ``max`` is the standard choice, ``mean`` is
        stricter but needs several clean enrolment images to behave.
        """
        aggregation = self.config.recognition.aggregation
        scores: List[Tuple[str, float]] = []
        for identity, templates in self.store.all_templates().items():
            if not templates:
                continue
            similarities = [
                cosine_similarity(embedding, template.embedding)
                for template in templates
            ]
            value = max(similarities) if aggregation == "max" else float(np.mean(similarities))
            scores.append((identity, value))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    # -- multi frame -------------------------------------------------------- #

    def identify(self, embeddings: Sequence[np.ndarray]) -> RecognitionResult:
        """Decide identity from several frames, requiring them to agree.

        Each frame votes independently for its own best-scoring identity, and
        only frames that clear both the similarity threshold and the margin
        over their own runner-up are allowed to vote at all. The winner must
        then collect ``min_agreeing_frames`` votes.

        Voting rather than averaging embeddings is deliberate: averaging lets
        one badly-lit or half-occluded frame drag the mean across a threshold,
        whereas that frame simply abstains here.
        """
        settings = self.config.recognition
        if not self.store.identities():
            return RecognitionResult(
                matched=False, reasons=[ReasonCode.NO_ENROLLED_IDENTITIES]
            )

        usable = [e for e in embeddings if e is not None and np.size(e)]
        if not usable:
            return RecognitionResult(matched=False, reasons=[ReasonCode.NO_MATCH])

        votes: Counter[str] = Counter()
        confidences: Dict[str, List[float]] = defaultdict(list)
        runner_up_totals: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        ambiguous_frames = 0
        below_threshold_frames = 0

        for embedding in usable:
            ranked = self.score_identities(embedding)
            if not ranked:
                continue
            best_identity, best_score = ranked[0]
            second_identity, second_score = (
                ranked[1] if len(ranked) > 1 else (None, 0.0)
            )

            if best_score < settings.match_threshold:
                below_threshold_frames += 1
                continue
            if best_score - second_score < settings.margin_threshold:
                ambiguous_frames += 1
                continue

            votes[best_identity] += 1
            confidences[best_identity].append(best_score)
            if second_identity is not None:
                runner_up_totals[best_identity].append((second_identity, second_score))

        frames_compared = len(usable)
        if not votes:
            # Distinguish the three ways this can happen. They mean different
            # things to a user: stand still, enrol first, or you are not this
            # person.
            if ambiguous_frames > below_threshold_frames:
                reason = ReasonCode.AMBIGUOUS_MATCH
            elif below_threshold_frames:
                reason = ReasonCode.LOW_RECOGNITION_CONFIDENCE
            else:
                reason = ReasonCode.NO_MATCH
            return RecognitionResult(
                matched=False,
                frames_compared=frames_compared,
                reasons=[reason],
            )

        identity, agreeing = votes.most_common(1)[0]
        scores = confidences[identity]
        confidence = float(np.median(scores))
        runner_up_name, runner_up_score = ("", 0.0)
        if runner_up_totals[identity]:
            runner_up_name, runner_up_score = max(
                runner_up_totals[identity], key=lambda item: item[1]
            )

        if agreeing < settings.min_agreeing_frames:
            return RecognitionResult(
                matched=False,
                identity=None,
                confidence=confidence,
                frames_agreeing=agreeing,
                frames_compared=frames_compared,
                reasons=[ReasonCode.UNSTABLE_IDENTITY],
            )

        return RecognitionResult(
            matched=True,
            identity=identity,
            confidence=confidence,
            runner_up=runner_up_name or None,
            runner_up_confidence=runner_up_score,
            margin=confidence - runner_up_score,
            frames_agreeing=agreeing,
            frames_compared=frames_compared,
        )

    # -- enrolment ---------------------------------------------------------- #

    def enrol(
        self,
        identity: str,
        face_bgr: np.ndarray,
        quality: QualityScores,
        source_digest: str = "",
        note: str = "",
    ) -> Optional[Template]:
        """Create and persist a template. Returns None if embedding failed."""
        from datetime import datetime, timezone

        embedding = self.backend.embed(face_bgr)
        if embedding is None:
            return None
        template = Template(
            identity=identity,
            embedding=embedding,
            quality=float(quality.overall),
            created_at=datetime.now(timezone.utc).isoformat(),
            source_digest=source_digest,
            note=note,
        )
        self.store.add(template)
        return template
