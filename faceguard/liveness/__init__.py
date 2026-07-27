"""Layered liveness detection.

Four independent signals, deliberately chosen so that defeating one does not
help an attacker defeat the next:

===========  ============================  =====================================
Signal       Question it answers           Defeated by
===========  ============================  =====================================
``texture``  Is this surface skin?         A very high quality print or mask
``blink``    Do these eyes behave?         A video replay
``motion``   Did the target have depth?    A cooperative victim, a 3-D mask
``depth``    (hardware only) Is it flat?   Nothing cheap, when a sensor exists
===========  ============================  =====================================

The rightmost column is the point. A printed photo fails texture, blink and
motion. A video replay passes blink but fails texture and motion. A tilted
photo passes a naive head-turn challenge but fails the parallax test inside
``motion``. No single check carries the system.

:mod:`faceguard.liveness.fusion` combines them under a configurable policy and
is the only part of this package that decides anything.
"""

from .fusion import LivenessEngine, LivenessEvidence
from .motion import ChallengeType, random_challenge
from .texture import SpoofClassifier

__all__ = [
    "LivenessEngine",
    "LivenessEvidence",
    "ChallengeType",
    "random_challenge",
    "SpoofClassifier",
]
