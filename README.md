<div align="center">

# FaceGuard AI

**Face verification that checks you are actually there, not just that you look right.**

[![Python](https://img.shields.io/badge/python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-0097A7?style=flat-square&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![DeepFace](https://img.shields.io/badge/DeepFace-Facenet512-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://github.com/serengil/deepface)
[![Tests](https://img.shields.io/badge/tests-120%20passing-brightgreen?style=flat-square&logo=pytest&logoColor=white)](#testing-and-evaluation)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

[How it works](#how-it-works) · [Results](#measured-results) · [Threat model](#what-is-defended-and-what-is-not) · [Quick start](#quick-start) · [Configuration](#configuration) · [Testing](#testing-and-evaluation) · [Limitations](#known-limitations)

</div>

---

A printed photo of you is recognisable. So is a video of you playing on a phone. A face recogniser on its own cannot tell either of them from the real you, so a recogniser that is 99% accurate with no liveness check is 0% secure against a sheet of paper.

FaceGuard puts three independent liveness signals in front of a face recogniser. It decides from a sequence of frames, never a single photo, and when a check cannot run it refuses instead of quietly doing less. Every number in this README was measured by the scripts in this repo, and the ideas that failed are written up next to the ones that worked.

## Overview

| | |
|---|---|
| **Liveness** | Surface texture, blink, and motion parallax. Each is beaten by a different attack, so beating one does not help with the next |
| **No single-frame decisions** | At least 15 usable frames, and 3 frames agreeing on one identity, before anything is accepted |
| **Active challenge** | A random head movement per attempt (turn left, turn right, look up, look down), checked for both completion and continuity |
| **Explicit refusals** | 26 machine-readable reason codes. Every rejection says why |
| **Separate confidences** | Detection confidence and recognition confidence are never blended into one number |
| **Encrypted templates** | Fernet at rest. Raw face images are never written to disk |
| **Rate limiting** | Sliding-window lockout with exponential backoff, shared between face and PIN |
| **Audit log** | Rejections are logged in the same detail as acceptances |
| **PIN fallback** | scrypt-hashed second factor, so a broken camera cannot lock someone out for good |
| **Web demo** | Browser webcam capture over a WebSocket, plus a CLI for development |

## How it works

Four signals. The last column is the point: each one is defeated by something different, which is the whole argument for layering them.

| Signal | Question it answers | Defeated by |
|---|---|---|
| **Texture** | Is this surface skin? | A very high quality print, or a mask made of skin-like material |
| **Blink** | Do these eyes behave like eyes? | A video replay, which contains real blinks |
| **Motion** | Did the thing that moved have depth? | A cooperative victim, or a real 3-D mask |
| **Depth** | Is it flat? | Nothing cheap, but it needs hardware this machine does not have |

The default policy in [`config/default.yaml`](config/default.yaml) makes texture mandatory, requires at least one of blink or motion to pass, requires the active challenge, and needs a fused score of 0.60 or more.

### 1. Surface texture

Three things happen to a face when it is re-captured from paper or a screen, and all three can be measured:

- **Fine detail is lost.** Skin has pores, fine hair and subsurface mottling. Print replaces that with ink dots and a screen replaces it with a pixel grid. Local Binary Patterns at two scales pick this up.
- **New periodic structure appears.** Photographing a display produces moiré, energy at spatial frequencies no real face emits. A radial FFT power spectrum shows it.
- **Colour and reflectance shift.** Skin scatters light below the surface. Paper and glass do not, so chroma compresses and broad specular highlights appear.

142 features feed a logistic regression. It is linear on purpose: the training set is small enough that a stronger model would memorise it, and the coefficients can be read directly with `python tools/train_spoof_classifier.py --explain`.

### 2. Blink

Two independent measures of eye closure: the classic eye aspect ratio (EAR) from six landmarks per eye, and MediaPipe's blink blendshapes, which hold up under head rotation where raw EAR does not.

A blink only counts if the eye closes and reopens within **70 to 500 ms**, the range of a spontaneous human blink. Faster is landmark noise. Slower is a squint, or something being slid across a photo's eyes. A hysteresis band between the "closed" and "open" thresholds stops a signal hovering around one threshold from producing fake blinks.

Two details came from measurement, not theory:

- EAR thresholds are a **fraction of each person's own open-eye baseline**. Across this project's images, a wide-open eye ranged from 0.245 to 0.523 between people, a 2x spread. Any fixed threshold is out of reach for one person and always tripped for another.
- Closure is read from the **more open** of the two eyes, so both must close. Averaging them, the obvious approach, lets a wink count as a blink.

### 3. Motion parallax

This is the check that does the real work.

Asking "did you turn your head?" is easy to fool: tilt a printed photo and the pose estimator reports a turn. So the system asks a harder question: **did turning reveal three-dimensional structure?**

Rotating a flat plane in front of a camera is exactly a homography. So the code fits the best homography from the landmarks in one frame to a later, rotated frame. If it fits, the thing that turned was flat. If stubborn residual error remains, it had depth.

That alone is not enough, because re-detecting a warped image makes the landmarker refit slightly differently, and that noise is not explained by a homography either. So a second condition is required: the per-landmark residual must **correlate with landmark depth**. Real parallax moves the nose far more than the cheek. Refit noise does not care where a landmark sits. Measured separation on this condition is Cohen's *d* = 2.17.

The challenge is picked at random from four per attempt, so a recorded clip of the victim turning one way does not satisfy the other three. The pose trajectory is also checked for continuity: a jump larger than a head can physically move between frames means the target was swapped, not turned.

### 4. Depth

With a real depth sensor this is the strongest signal there is, and the reason Face ID is hard to beat with a photo. The code probes for an Intel RealSense camera (`pyrealsense2`).

This machine has none. The system says so in the API, in the UI status pill, and on every result as `degraded: true`, and it **refuses to substitute something weaker**. The shortcut that was tried and rejected is described under [What did not work](#what-did-not-work).

## Measured results

All numbers come from the scripts in `eval/` on the hardware listed under Latency. None are quoted from a paper.

<details open>
<summary><strong>Recognition</strong>: leave-one-image-out over the enrolled gallery</summary>

| Metric | Value |
|---|---|
| Recognition accuracy | **1.000** (15 genuine trials) |
| FAR, impostors accepted | **0.000**, 95% CI **[0.000, 0.117]** |
| FRR, genuine users rejected | **0.000**, 95% CI [0.000, 0.204] |
| Genuine similarity | 0.821 ± 0.084 (min 0.651) |
| Impostor similarity | 0.322 ± 0.135 (max 0.602) |
| Separation | Cohen's *d* = **4.45** |

Read the confidence interval, not the point estimate. Zero false accepts in 29 impostor trials does not mean FAR is zero. It means FAR is somewhere below about 12%, which is all 7 identities and 17 images can support. The Cohen's *d* of 4.45 is the more useful number: genuine and impostor scores are cleanly separated.
</details>

<details open>
<summary><strong>Spoof detection</strong>: identity-disjoint splits, simulated attacks</summary>

| Metric | Value |
|---|---|
| AUC | **0.977** |
| Accuracy | 0.935 |
| **APCER**, attacks accepted as live | **0.071** |
| **BPCER**, real faces rejected | 0.060 |
| ACER | 0.065 |

| Attack | APCER | n |
|---|---|---|
| Print | 0.071 | 28 |
| Screen replay | 0.143 | 28 |
| Generative-artefact proxy | 0.000 | 28 |

Train, validation and test share **no identity**. The threshold was picked by leave-one-identity-out cross-validation over the pooled train and validation identities, and the test split was touched once.

Pooled CV AUC was 0.845 against a mean per-fold AUC of 0.954. The model separates live from spoof well *within* each person, but its probabilities are not calibrated *across* people, so the single global threshold is the weak link, not the features.

These are texture-only numbers. The full policy also requires the challenge and at least one of blink or parallax, and a printed photo fails those on its own.
</details>

<details open>
<summary><strong>End-to-end</strong>: the full layered policy on frame sequences</summary>

| | Result |
|---|---|
| Attacks passing all layers | **0 / 51** (APCER 0.000) |
| Print | 0 / 19 |
| Screen replay | 0 / 16 |
| Generative proxy | 0 / 16 |

Rejection reasons (most attacks trip several layers at once):

| Reason | Count |
|---|---|
| `NO_BLINK_DETECTED` | 51 |
| `LIVENESS_SCORE_TOO_LOW` | 49 |
| `TEXTURE_SPOOF_SUSPECTED` | 48 |
| `FLAT_GEOMETRY` | 45 |
| `CHALLENGE_FAILED` | 39 |

**The live pass rate in this harness is 0/4, and that is expected.** The sequences are built from still photos by plane-rotation homography, which is a flat rotation by construction. That is faithful for attacks and a deliberately unfair worst case for a live face, which *should* fail a parallax check when it is literally flat. Measuring the live side needs a real camera. See [Known limitations](#known-limitations).
</details>

<details open>
<summary><strong>Latency</strong>: Windows 11, AMD64 CPU, no GPU</summary>

| Stage | Median | p95 | p99 |
|---|---|---|---|
| Detection + 478 landmarks | 12.4 ms | 12.9 ms | 13.2 ms |
| Texture features | 2.1 ms | 2.4 ms | 2.5 ms |
| Liveness fusion (per decision) | 36.5 ms | 41.5 ms | 43.9 ms |
| Recognition embedding | 208.2 ms | 214.5 ms | 218.4 ms |

Detection and texture run on every frame (14.6 ms). Recognition runs on one frame in five, adding 41.6 ms amortised.

**56.2 ms per frame, or 17.8 fps sustained.** Compute for a 15-frame attempt is 0.88 s. Wall-clock time is longer because the challenge moves at the speed of a person turning their head. Recognition is the obvious thing to optimise next.
</details>

<details>
<summary><strong>Degraded conditions</strong>: 26 degraded conditions</summary>

Two failure directions are counted separately. Failing to match *yourself* is annoying. Matching *someone else* is dangerous.

| Condition | Refused | Still matches | Similarity | False accepts |
|---|---|---|---|---|
| Baseline | 0/17 | 17/17 | 1.000 | 0 |
| Low light −30% | 0/17 | 17/17 | 0.979 | 0 |
| Low light −50% | 0/17 | 17/17 | 0.926 | 0 |
| Low light −70% | 2/17 | 13/15 | 0.789 | 0 |
| Low light −85% | 16/17 | 0/1 | 0.497 | 0 |
| Backlight 35% | 2/17 | 15/15 | 0.988 | 0 |
| Backlight 60% | 3/17 | 14/14 | 0.971 | 0 |
| Backlight 85% | 8/17 | 9/9 | 0.926 | 0 |
| Sunglasses | 2/17 | 12/15 | 0.695 | 0 |
| Face mask | 7/17 | 5/10 | 0.681 | 0 |
| Hat brim | 0/17 | 17/17 | 0.987 | 0 |
| **Hand over face** | 5/17 | 4/12 | 0.507 | **2** |
| Yaw 15° | 14/17 | 3/3 | 0.882 | 0 |
| Yaw ≥ 25° | 17/17 | n/a | n/a | 0 |
| Too far (×0.35) | 11/17 | 6/6 | 0.941 | 0 |
| Too close (×2.6) | 17/17 | n/a | n/a | 0 |
| Motion blur (all) | 17/17 | n/a | n/a | 0 |
| JPEG q10 | 0/17 | 17/17 | 0.955 | 0 |

**Zero crashes.** A refusal is correct behaviour: the system said why it could not decide.

The hand-over-face result was a real security hole. How it was closed is under [What did not work](#what-did-not-work).
</details>

## What is defended, and what is not

This is the section a technical reviewer reads first, so it is stated precisely.

### Defended, with evidence

| Attack | Status | Evidence |
|---|---|---|
| **Printed photo, held still** | Defended | Fails blink, parallax and challenge independently. 0/19 print attacks passed end-to-end |
| **Printed photo, tilted to fake a head turn** | Defended | Closed by the parallax check. Rotating a plane is exactly a homography; measured separation *d* = 2.17 |
| **Screen replay, static image** | Defended | 0/16 passed. Fails blink and texture |
| **Second face in frame** | Defended | Explicit `MULTIPLE_FACES` refusal, never a guess at the biggest face |
| **Brute-force retries** | Defended | Sliding-window lockout with exponential backoff, shared with the PIN |
| **Template theft from disk** | Mitigated | Fernet-encrypted and authenticated. Raw images are never stored |
| **Store tampering** | Defended | A modified store fails Fernet's authentication check and refuses to decrypt |

### Open gaps

| Attack | Status |
|---|---|
| **Video replay of the victim doing the exact challenge** | **Open.** Blink and challenge both pass. Only texture and parallax stand in the way, and a good screen at the right distance may beat texture. This is the most realistic attack on the system |
| **Real printed photo held up to a real webcam** | **Untested.** All print results come from simulation |
| **Real screen replay on a second screen** | **Untested.** Same |
| **3-D mask (resin, silicone)** | **Open.** It has real depth, so it passes parallax. Texture is the only defence and was never trained on mask materials |
| **Real deepfake video** | **Untested.** The `synthetic` class is a hand-built imitation of generative artefacts, not output from a generative model |
| **Adversarial perturbation** | **Untested** |
| **Identical twins** | **Untested.** A recognition problem, not a liveness one |

Every anti-spoofing number here comes from **simulated attacks**. The simulations model the physics (halftone screening and gamut compression for print; pixel lattice, moiré and backlight bleed for replay), and live images go through the same camera noise and JPEG pass as attacks, so the classifier cannot learn "compressed means fake". They are still simulations, and real-world APCER will be higher.

The physical test takes ten minutes, a printed photo and a phone:

```bash
python cli.py verify --camera 0        # you
python cli.py verify --camera 0        # then hold up a printed photo
python cli.py verify --camera 0        # then a video of yourself on a phone
python cli.py audit --limit 10         # compare the reason codes
```

## What did not work

Written down because a project that only lists its wins is hard to trust.

**Landmark-planarity depth proxy: abandoned.** MediaPipe returns a z value per landmark, so the tempting shortcut is to fit a plane to the landmark cloud and call a flat fit a photo. Measured residuals were 0.2060 ± 0.0149 for live faces and 0.2115 ± 0.0207 for simulated spoofs, Cohen's *d* = 0.30: no separation. MediaPipe's z is *inferred* from a 2-D image, so a photo of a face still yields a face-shaped cloud with a nose that is not there. It is still computed and reported so the claim can be checked, but it carries zero weight. The parallax check is the real depth-free substitute, because it recovers structure from motion instead of trusting a model's guess.

**Non-rigid deformation ratio: demoted.** Align frames on rigid landmarks, then measure how much lips and brows moved; a photo should score about 1.0. A jittered still scored 1.33 ± 0.12 (up to 1.55), because soft-tissue landmarks sit on low-contrast edges and are noisier to locate. The original threshold of 1.35 sat in the middle of the spoof distribution and would have let through roughly half of all print attacks. It is now corroborating evidence only.

**Region-contrast occlusion detection: advisory only.** Flagging any face region much flatter than its neighbours fired on 59% of clean faces, because the bridge of the nose is four times flatter than the face median on every healthy face. Per-region baselines brought false positives down to 17.6% with about 86% detection. That is still too noisy to reject on (it would turn away one honest user in six), so it advises and drops frames from the recognition vote instead.

**Hand-occlusion false accepts: found and closed.** The degraded-conditions suite (`eval/robustness.py`) found two probes, with a hand over the face, matching the **wrong** enrolled identity at cosine 0.632 and 0.650. Threshold tuning could not fix it: the lowest *genuine* match scored 0.651. The fix was to exclude suspected-occluded frames from the recognition vote. Dropping a frame only costs a retry, and with 3 agreeing frames required and about 87% detection on hand occlusion, an attacker needs three misses in a row. Regression test: [`tests/test_occlusion_defence.py`](tests/test_occlusion_defence.py).

**Bugs the tests caught.**

| Bug | Consequence |
|---|---|
| Euler decomposition used the wrong rotation order | "Turn left" was satisfied by turning **right** |
| Parallax checked depth correlation before residual | A perfectly flat target reported `UNAVAILABLE` instead of `FLAT_GEOMETRY` |
| Blink averaged both eyes | A single **wink** counted as a blink |
| Laplacian sharpness is not scale-invariant | The same photo scored 0.084 at 3120 px wide and 1.000 at 640 px, so enrolling from original files failed with `LOW_QUALITY` |

The last one was fixed by normalising resolution at every entry point (`working_width: 640`) instead of loosening the threshold. That also made texture features comparable across sources: test AUC went from 0.967 to **0.977** and print APCER from 0.286 to **0.071**.

## Security design

- **No single-frame decisions.** 15 usable frames minimum and 3 agreeing frames. The config loader refuses to start if `min_frames` or `min_agreeing_frames` is below 2.
- **Unavailable is not passed.** `SignalStatus` keeps `FAIL` and `UNAVAILABLE` apart. With no spoof classifier, an attempt is rejected with `LIVENESS_UNAVAILABLE`, never waved through. Unavailable signals are left out of the weighted score instead of scoring zero, so a missing depth camera does not push an operator into lowering the threshold.
- **Liveness before identity.** An attacker holding up a photo learns that liveness failed, never whether the face is enrolled.
- **Template storage.** Fernet (AES-128-CBC + HMAC-SHA256). Embeddings are treated as biometric data: template inversion can rebuild a recognisable face, and a face cannot be rotated after a breach like a password. Writes are atomic (temp file, fsync, rename).
- **Where the key lives.** By default the key is written to `data/faceguard.key` beside the ciphertext so the project runs out of the box. That protects against a leaked repo, backup or cloud sync, not against someone who can read the key. In production, set `FACEGUARD_KEY` from a secrets manager so the key never touches disk.
- **Rate limiting.** Sliding window, exponential lockout up to a cap (5 failures in 300 s, 30 s doubling to 3600 s by default). Framing failures such as a dark room, a face too far away or two people in shot do not count toward lockout.
- **Audit log.** JSONL at `logs/audit.jsonl`. Rejections are logged in full, so twenty `TEXTURE_SPOOF_SUSPECTED` failures from one client in two minutes are visible. Client identifiers are hashed.
- **PIN fallback.** scrypt with per-entry salts. Unknown identities still pay the full hash cost so timing cannot enumerate users, and PIN failures share the face lockout counter. A 6-digit PIN is about 20 bits of entropy and is offered as a fallback, nothing more.

## Quick start

Tested on Python 3.11. The exact versions behind the published numbers are pinned in [`requirements.lock.txt`](requirements.lock.txt) (MediaPipe 0.10.35, DeepFace 0.0.100, TensorFlow 2.21.0, FastAPI 0.140.0).

```bash
git clone https://github.com/ishtiaq-ahmed-dev/faceguard-ai.git
cd faceguard-ai
python -m venv .venv
.venv\Scripts\activate             # Windows
# source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
python tools/fetch_models.py       # MediaPipe models into models/, checksum-verified
```

**Train the spoof classifier.** It is not shipped, because it is trained on face photos and those are not in the repo. Without `models/spoof_classifier.joblib` every verification is refused with `LIVENESS_UNAVAILABLE`, by design. Put your own images in a folder with one sub-folder per person, then:

```bash
python tools/make_splits.py --root your_faces --output splits/manifest.json
python tools/make_splits.py --verify splits/manifest.json     # identity-leakage check
python tools/attack_sim.py --manifest splits/manifest.json --output attacks
python tools/train_spoof_classifier.py --explain
```

**Run the web demo.**

```bash
python -m uvicorn web.app:app --port 8000
```

Open <http://localhost:8000>. Enrol on the **Enrol** tab, then try **Verify**. Then hold up a photo of yourself on your phone and read the refusal on the **Security Log** tab.

The server warms the Facenet512 weights (about 95 MB) at startup, so the first user does not wait for them.

> `getUserMedia` needs a secure context. Browsers allow the camera on `localhost` over plain HTTP but refuse it on a LAN address without TLS, so put the server behind HTTPS to demo from another machine.

### CLI

For development and testing. Every command prints the same structured result the API returns.

```bash
python cli.py status                                   # what is loaded, what is degraded
python cli.py enrol --name "Ada" --images "photos/*.jpg"
python cli.py verify --camera 0                        # live webcam
python cli.py verify --video clip.mp4                  # from a recording
python cli.py verify --images "frames/*.png"           # from ordered frames
python cli.py identities                               # list, or --delete NAME
python cli.py audit --limit 20                         # recent decisions
python cli.py set-pin --name "Ada"                     # PIN fallback
```

## Configuration

Every accept/reject threshold lives in [`config/default.yaml`](config/default.yaml). No decision boundary is hard-coded in the source.

The loader is strict: unknown keys, missing keys, type mismatches and contradictions all raise `ConfigError`. A typo in a threshold name must not fall back to a permissive default. It also rejects configs that cannot be satisfied (more optional signals required than exist) or that defeat themselves (inverted blink hysteresis, single-frame liveness).

To use your own copy:

```bash
FACEGUARD_CONFIG=path/to/custom.yaml python -m uvicorn web.app:app
python cli.py --config path/to/custom.yaml status
```

| Variable | Purpose |
|---|---|
| `FACEGUARD_KEY` | Fernet key for the template store. Set this in production |
| `FACEGUARD_CONFIG` | Path to a config file (default `config/default.yaml`) |
| `FACEGUARD_AUDIT_SALT` | Salt for hashing client identifiers in the audit log |

## Testing and evaluation

```bash
pytest                          # 120 tests, no face images needed
python eval/evaluate.py         # recognition and spoof metrics, reported separately
python eval/latency.py          # per-stage latency on your hardware
python eval/robustness.py       # 26 degraded conditions
```

The tests run on a fresh clone with no face images. The core geometric claims are tested on synthetic point clouds: a cloud with facial relief and the same cloud flattened test the anti-spoofing argument more directly than a photo could, because the ground truth is exact.

`make_splits.py` splits by **identity, never by image**, and refuses to write a manifest that leaks. Two photos of one person on opposite sides of a split let a model score well by recognising the person instead of learning anything about spoofing, and a random shuffle of image paths does exactly that while looking reasonable.

## Project layout

```text
faceguard-ai/
├── faceguard/
│   ├── config.py          Typed YAML config; bad or unsatisfiable policies fail at load time
│   ├── schemas.py         ReasonCode enum and result dataclasses
│   ├── detect.py          MediaPipe Tasks: box, score, 478 landmarks, 52 blendshapes, pose
│   ├── quality.py         Quality scoring, resolution normalisation, occlusion hints
│   ├── recognition.py     Facenet512 embeddings, margin check, multi-frame voting
│   ├── store.py           Fernet-encrypted template store, atomic writes
│   ├── session.py         Multi-frame state machine, rate limiting, the FaceGuard facade
│   ├── audit.py           JSONL audit log
│   ├── pin.py             scrypt PIN fallback
│   └── liveness/
│       ├── texture.py     LBP + FFT + colour features, logistic regression
│       ├── blink.py       EAR + blendshapes, blink-duration gate
│       ├── motion.py      Planar parallax, head pose, challenge-response
│       ├── depth.py       RealSense probe and the rejected planarity proxy
│       └── fusion.py      Policy engine
├── web/                   FastAPI + WebSocket backend, single-page frontend in web/static/
├── config/default.yaml    Every threshold, with notes on which direction is stricter
├── cli.py                 Development and testing CLI
├── tools/                 Model fetch, splits, attack simulation, training, parallax calibration
├── eval/                  evaluate.py, latency.py, robustness.py
└── tests/                 120 tests
```

**Why a WebSocket and not a POST endpoint?** The system will not decide from one frame, so the demo has to be a conversation: issue a random challenge, watch a sequence, and tell the user what to fix while they are still in front of the camera. A post-one-photo endpoint would be a weaker system wearing this one's UI.

## Known limitations

1. **Every anti-spoofing number comes from simulated attacks.** No printed photo has been held in front of a real webcam. Real APCER will be higher.
2. **The dataset is tiny.** 7 identities, 17 images. Confidence intervals are wide, and the identity-disjoint spoof split has 4 training identities. Do not read anything here as a benchmark.
3. **The live end-to-end pass rate is unmeasured.** The synthetic sequences are flat, so a live face correctly fails parallax in that harness. Only a real camera settles it.
4. **Parallax thresholds are calibrated on simulation.** The flat-target side is trustworthy; the live side is projected from MediaPipe's inferred depth. Run `tools/measure_parallax.py` and a real capture before treating them as an operating point.
5. **Spoof scores are not calibrated across people.** Pooled CV AUC 0.845 against a per-fold mean of 0.954.
6. **Screen replay is the weakest defended attack**, at APCER 0.143 for texture alone, and a replay that performs the right challenge is the most realistic open attack.
7. **Yaw beyond about 25° is refused.** Safe, but the challenge has to ask for a movement big enough to matter and small enough to stay detectable.
8. **No depth hardware was available.** The strongest defence is untested on real hardware, and the software substitute was measured not to work.
9. **Occlusion detection is advisory**, with 17.6% false positives on clean faces.
10. **`client_id` is the socket peer address.** Behind a proxy every client shares one address. A real deployment should key rate limiting on an authenticated session or device token. `X-Forwarded-For` is deliberately ignored, since a client can set it to anything and reset its own lockout.

**Not done yet:** ONNX export of Facenet512 (`onnxruntime` is already installed as a DeepFace dependency, and recognition at 208 ms is the clear target), adversarial robustness testing, and evaluation against real deepfakes.

### The short version

What can fairly be claimed: a layered presentation-attack detector with measured numbers against simulated attacks (AUC 0.977 and APCER 0.071 for texture, 0 of 51 attacks through the full policy), recognition separated at *d* = 4.45, and a parallax check that stops a tilted photo from faking a head turn.

What should not be claimed: "99% accurate" (that is recognition only and says nothing about spoofing), "Face ID-level security" (Face ID has a dot projector and an infrared camera), "deepfake detection" (the generative class is a proxy), or "production ready" (the key sits beside the ciphertext by default and rate limiting keys on an IP address).

The part worth showing off is the method: thresholds set from measurement, two features removed because the numbers said they did not work, and a security hole found by the project's own test harness and closed with a fix whose severity was chosen from data.

## Privacy and data

- **No face images are in this repository**, and none are shipped as samples. Bring your own and enrol through the UI. The images behind the numbers above were of consenting family members and stayed local.
- **Nothing is recorded during a session.** Frames live for the length of one attempt. Only the encrypted templates (`data/`) and the audit log (`logs/`) touch disk.
- Templates, keys, audit logs, trained models and your face folders are generated locally and are not part of the repository. Keep them out of version control: audit logs contain identity names and count as personal data.

## License

MIT. See [LICENSE](LICENSE).

Built with [MediaPipe](https://developers.google.com/mediapipe), [DeepFace](https://github.com/serengil/deepface), scikit-learn and FastAPI.

## Author

**Ishtiaq Ahmed**<br>
GitHub: [@ishtiaq-ahmed-dev](https://github.com/ishtiaq-ahmed-dev)
