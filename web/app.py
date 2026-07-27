#!/usr/bin/env python3
"""FastAPI backend for the FaceGuard demo.

The browser captures webcam frames and streams them over a WebSocket; the
server runs the full pipeline and streams back per-frame feedback, then one
final structured verdict.

**Why a WebSocket rather than posting an image.** The whole design refuses to
decide from a single frame, so the demo has to be a conversation: the server
issues a random challenge, watches a sequence, and tells the user what to fix
while they are still in front of the camera. A request/response endpoint that
takes one photo would be a different, weaker system wearing this one's UI.

**What crosses the wire.** Frames go up as JPEG and are held only for the life
of the session. Nothing is written to disk except the encrypted template on
enrolment and the audit log. There is no frame storage, no image cache, and no
upload directory.

**Deployment note.** ``getUserMedia`` requires a secure context, so browsers
allow camera access on ``localhost`` over plain HTTP but will refuse it on a
LAN address without TLS. Run behind HTTPS to demo from another machine.

Usage::

    python -m uvicorn web.app:app --reload --port 8000
    python web/app.py            # equivalent, with warm-up
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faceguard.config import load_config
from faceguard.liveness import ChallengeType
from faceguard.pin import PinError, PinStore
from faceguard.quality import crop_face, standardise
from faceguard.schemas import ReasonCode
from faceguard.session import FaceGuard

STATIC_DIR = Path(__file__).resolve().parent / "static"

# One frame at 640px wide as JPEG is well under this. The cap stops a client
# from sending an arbitrarily large payload and exhausting server memory.
MAX_FRAME_BYTES = 4 * 1024 * 1024

state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline once at startup and warm the models.

    Model loading takes tens of seconds on a cold cache. Doing it here means
    the first user to try the demo gets a fast response instead of a timeout
    that looks like a crash.
    """
    config = load_config()
    guard = FaceGuard(config)
    state["config"] = config
    state["guard"] = guard
    state["pins"] = PinStore(
        config.resolve(config.security.pin_store), config.security.pin_min_length
    )
    state["ready"] = False

    loop = asyncio.get_running_loop()

    async def warm() -> None:
        await loop.run_in_executor(None, guard.warm_up)
        state["ready"] = True

    warm_task = asyncio.create_task(warm())
    try:
        yield
    finally:
        warm_task.cancel()
        guard.close()


app = FastAPI(title="FaceGuard AI", version="1.0.0", lifespan=lifespan)


def guard() -> FaceGuard:
    instance = state.get("guard")
    if instance is None:  # pragma: no cover - only before startup completes
        raise HTTPException(503, "System still starting up.")
    return instance


def client_id(request: Request) -> str:
    """Identify a client for rate limiting.

    Uses the socket peer address. This is honest about what it is: behind a
    proxy every client shares one address, and a determined attacker can move
    between addresses. A real deployment should key on an authenticated
    session or a device token instead. Deliberately does NOT trust
    X-Forwarded-For, which a client can set to anything and would turn the
    lockout into something an attacker resets at will.
    """
    return request.client.host if request.client else "unknown"


def decode_frame(payload: str) -> Optional[np.ndarray]:
    """Decode a base64 data URL or raw base64 JPEG into a BGR image."""
    if not payload:
        return None
    if "," in payload[:64]:
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not raw or len(raw) > MAX_FRAME_BYTES:
        return None
    buffer = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- #
# Pages and status
# --------------------------------------------------------------------------- #


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status() -> JSONResponse:
    """System capability report, including everything that is degraded.

    The UI shows this verbatim. A demo that hides its own limitations is the
    thing this project is arguing against.
    """
    instance = guard()
    config = state["config"]
    capability = instance.liveness.depth_capability
    classifier = instance.liveness.classifier
    metadata = classifier.metadata if classifier else {}

    return JSONResponse(
        {
            "ready": bool(state.get("ready")),
            "enrolled_identities": instance.store.identities(),
            "template_count": len(instance.store),
            "depth": {
                "available": capability.available,
                "backend": capability.backend,
                "detail": capability.detail,
            },
            "spoof_classifier": {
                "loaded": classifier is not None,
                "detail": (
                    "Trained on simulated attacks; see README for measured limits."
                    if classifier
                    else "NOT TRAINED. Verification will refuse rather than "
                    "fall back to a weaker check."
                ),
                "test_metrics": (metadata.get("metrics", {}) or {}).get("test"),
                "test_auc": metadata.get("test_auc"),
            },
            "policy": {
                "required_signals": config.liveness.policy.required,
                "optional_signals": config.liveness.policy.optional,
                "min_optional": config.liveness.policy.min_optional_signals,
                "challenge_required": config.liveness.policy.require_challenge,
                "min_frames": config.liveness.min_frames,
                "min_fused_score": config.liveness.policy.min_fused_score,
            },
            "recognition": {
                "model": config.recognition.model_name,
                "match_threshold": config.recognition.match_threshold,
                "margin_threshold": config.recognition.margin_threshold,
                "min_agreeing_frames": config.recognition.min_agreeing_frames,
            },
        }
    )


@app.get("/api/identities")
async def identities() -> JSONResponse:
    instance = guard()
    pins: PinStore = state["pins"]
    return JSONResponse(
        {
            "identities": [
                {
                    "name": name,
                    "templates": len(instance.store.templates_for(name)),
                    "has_pin": pins.has_pin(name),
                }
                for name in instance.store.identities()
            ]
        }
    )


@app.delete("/api/identities/{name}")
async def delete_identity(name: str, request: Request) -> JSONResponse:
    instance = guard()
    removed = instance.store.remove_identity(name)
    state["pins"].remove(name)
    instance.audit.record_event(
        "identity_deleted", identity=name, client_id=client_id(request), removed=removed
    )
    if not removed:
        raise HTTPException(404, f"No enrolled identity named {name!r}")
    return JSONResponse({"deleted": name})


@app.get("/api/audit")
async def audit(limit: int = 40) -> JSONResponse:
    """Recent audit records, newest first.

    Exposed because the rejections are the interesting part of the demo: a
    reviewer should be able to hold up a photo and watch the reason code appear.
    """
    records = list(guard().audit.read())
    return JSONResponse({"records": records[-limit:][::-1], "total": len(records)})


# --------------------------------------------------------------------------- #
# Enrolment
# --------------------------------------------------------------------------- #


class EnrolRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    frames: List[str] = Field(min_length=1, max_length=20)
    pin: Optional[str] = Field(default=None, max_length=32)


@app.post("/api/enrol")
async def enrol(payload: EnrolRequest, request: Request) -> JSONResponse:
    """Enrol an identity from several frames.

    Only frames that pass the same framing and quality gates as verification
    are accepted. Enrolling a blurry or badly-lit template quietly degrades
    every future match against that person, and it is invisible afterwards.
    """
    instance = guard()
    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "Name must not be empty.")

    accepted, rejected = 0, []
    for encoded in payload.frames:
        frame = decode_frame(encoded)
        if frame is None:
            rejected.append("undecodable frame")
            continue
        frame = standardise(frame, instance.config.detection.working_width)
        analysis = instance.analyzer.analyze(frame)
        if analysis.reasons or analysis.primary is None:
            reason = analysis.reasons[0] if analysis.reasons else ReasonCode.NO_FACE_DETECTED
            rejected.append(reason.message)
            continue
        face = analysis.primary
        crop, _ = crop_face(frame, face.bbox, margin=0.15)
        template = instance.recogniser.enrol(
            name, crop, face.quality, note="web-enrolment"
        )
        if template is not None:
            accepted += 1
        else:
            rejected.append("embedding failed")

    if payload.pin:
        try:
            state["pins"].set_pin(name, payload.pin)
        except PinError as error:
            raise HTTPException(400, str(error)) from error

    instance.audit.record_event(
        "enrolment",
        identity=name,
        client_id=client_id(request),
        templates_accepted=accepted,
        frames_rejected=len(rejected),
        pin_set=bool(payload.pin),
    )

    if accepted == 0:
        raise HTTPException(
            400,
            {
                "message": "No usable frames. Nothing was enrolled.",
                "rejected": rejected[:5],
            },
        )
    return JSONResponse(
        {
            "identity": name,
            "templates_added": accepted,
            "frames_rejected": rejected[:5],
            "total_templates": len(instance.store.templates_for(name)),
            "pin_set": bool(payload.pin),
        }
    )


# --------------------------------------------------------------------------- #
# PIN fallback
# --------------------------------------------------------------------------- #


class PinRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    pin: str = Field(min_length=1, max_length=32)


@app.post("/api/pin/verify")
async def verify_pin(payload: PinRequest, request: Request) -> JSONResponse:
    """Second-factor fallback when biometrics legitimately cannot succeed.

    Shares the biometric lockout counter on purpose: an attacker must not be
    able to sidestep face rate limiting by switching to PIN guessing.
    """
    instance = guard()
    caller = client_id(request)
    allowed, remaining = instance.rate_limiter.check(caller)
    if not allowed:
        return JSONResponse(
            {
                "accepted": False,
                "reason": ReasonCode.LOCKED_OUT.value,
                "message": f"Locked out for another {remaining:.0f} seconds.",
            },
            status_code=429,
        )

    ok = state["pins"].verify(payload.name.strip(), payload.pin)
    if ok:
        instance.rate_limiter.record_success(caller)
    else:
        instance.rate_limiter.record_failure(caller)

    instance.audit.record_event(
        "pin_verification",
        identity=payload.name.strip(),
        client_id=caller,
        accepted=ok,
        auth_method="pin",
    )
    return JSONResponse(
        {
            "accepted": ok,
            "identity": payload.name.strip() if ok else None,
            "auth_method": "pin",
            # Same message either way: distinguishing "no such user" from
            # "wrong PIN" hands an attacker a list of valid identities.
            "message": "Verified with PIN." if ok else "Incorrect name or PIN.",
        }
    )


# --------------------------------------------------------------------------- #
# Verification WebSocket
# --------------------------------------------------------------------------- #


@app.websocket("/ws/verify")
async def verify_socket(websocket: WebSocket) -> None:
    """Streaming verification.

    Protocol, client to server:
        ``{"type": "start", "challenge": "TURN_LEFT"|null}``
        ``{"type": "frame", "data": "<base64 jpeg>"}``
        ``{"type": "finish"}``

    Server to client:
        ``{"type": "session", ...}`` once, with the challenge to perform
        ``{"type": "feedback", ...}`` per frame
        ``{"type": "result", ...}`` once, the full VerificationResult
    """
    await websocket.accept()
    instance = guard()
    caller = websocket.client.host if websocket.client else "unknown"
    session = None
    loop = asyncio.get_running_loop()

    try:
        while True:
            message = json.loads(await websocket.receive_text())
            kind = message.get("type")

            if kind == "start":
                allowed, remaining = instance.rate_limiter.check(caller)
                if not allowed:
                    await websocket.send_json(
                        {
                            "type": "result",
                            "result": {
                                "accepted": False,
                                "reason": ReasonCode.LOCKED_OUT.value,
                                "message": (
                                    f"Too many failed attempts. Locked out for "
                                    f"another {remaining:.0f} seconds."
                                ),
                            },
                        }
                    )
                    continue

                requested = message.get("challenge")
                challenge = None
                if requested:
                    try:
                        challenge = ChallengeType(requested)
                    except ValueError:
                        challenge = None
                session = instance.start_session(client_id=caller, challenge=challenge)
                await websocket.send_json(
                    {
                        "type": "session",
                        "session_id": session.session_id,
                        "challenge": session.challenge.value if session.challenge else None,
                        "instruction": (
                            session.challenge.instruction if session.challenge else
                            "Look at the camera"
                        ),
                        "frames_required": instance.config.liveness.min_frames,
                        "timeout_s": instance.config.session.max_duration_s,
                    }
                )

            elif kind == "frame":
                if session is None:
                    await websocket.send_json(
                        {"type": "error", "message": "Send a start message first."}
                    )
                    continue
                frame = decode_frame(message.get("data", ""))
                if frame is None:
                    await websocket.send_json(
                        {"type": "error", "message": "Could not decode frame."}
                    )
                    continue
                # The pipeline is CPU-bound and synchronous. Running it on the
                # event loop would stall every other connection for the
                # duration of the inference.
                feedback = await loop.run_in_executor(None, session.add_frame, frame)
                await websocket.send_json(
                    {"type": "feedback", "feedback": feedback.to_dict()}
                )
                if feedback.ready or session.expired:
                    result = await loop.run_in_executor(None, session.finalise)
                    await websocket.send_json(
                        {"type": "result", "result": result.to_dict()}
                    )
                    session = None

            elif kind == "finish":
                if session is not None:
                    result = await loop.run_in_executor(None, session.finalise)
                    await websocket.send_json(
                        {"type": "result", "result": result.to_dict()}
                    )
                    session = None

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type {kind!r}"}
                )

    except WebSocketDisconnect:
        return
    except Exception as error:  # pragma: no cover - defensive
        try:
            await websocket.send_json({"type": "error", "message": str(error)})
        except Exception:
            pass


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False)
