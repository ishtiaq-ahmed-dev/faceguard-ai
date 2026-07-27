# FaceGuard AI

**Face verification that checks whether you are *present*, not just whether you
look right.**

A printed photograph of you is recognisable. So is a video of you on a phone
screen. Face recognition alone cannot tell either of them apart from you, which
is why a recogniser with 99% accuracy and no liveness check is not 99% secure —
it is 0% secure against a sheet of paper.

FaceGuard layers three independent liveness signals over a face recogniser,
requires a sequence of frames rather than a single photo, and refuses rather
than degrades when a check cannot run. Everything below is measured, and the
things that did not work are documented next to the things that did.

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [How the liveness check works](#how-the-liveness-check-works)
- [Measured results](#measured-results)
- [What is actually defended, and what is not](#what-is-actually-defended-and-what-is-not)
- [Things that were tried and did not work](#things-that-were-tried-and-did-not-work)
- [Security design](#security-design)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Testing and evaluation](#testing-and-evaluation)
- [Known limitations](#known-limitations)
- [If someone technical asks how secure this is](#if-someone-technical-asks-how-secure-this-is)

---

## What it does

| | |
|---|---|
| **Liveness** | Surface texture, blink, and motion parallax — three independent signals, layered so defeating one does not help defeat the next |
| **No single-frame decisions** | A minimum of 15 usable frames and 3 frames agreeing on an identity before anything is accepted |
| **Active challenge** | A randomly chosen head movement per attempt, verified for both completion and continuity |
| **Explicit refusals** | 26 machine-readable reason codes. Every rejection says *why* |
| **Separated confidences** | Detection confidence and recognition confidence are never blended into one number |
| **Encrypted templates** | Fernet at rest. Raw face images are never written to disk |
| **Rate limiting** | Sliding-window lockout with exponential backoff, shared between face and PIN |
| **Audit log** | Rejections recorded in the same detail as acceptances |
| **PIN fallback** | scrypt-hashed second factor, so a failed camera cannot lock someone out permanently |
| **Web demo** | Browser webcam capture over a WebSocket, with a CLI for development |

---

## Quick start

```bash
git clone https://github.com/ishtiaq-ahmed-dev/faceguard-ai.git
cd faceguard-ai
python -m venv .venv && .venv/Scripts/activate      # Windows
# source .venv/bin/activate                          # macOS / Linux
pip install -r requirements.txt
python tools/fetch_models.py
```

Run the demo:

```bash
python -m uvicorn web.app:app --port 8000
```

Open <http://localhost:8000>. Enrol yourself on the **Enrol** tab, then try
**Verify**. Then hold up a photo of yourself on your phone and watch the
**Security Log** tab explain the refusal.

> `getUserMedia` needs a secure context. Browsers allow camera access on
> `localhost` over plain HTTP but refuse it on a LAN address without TLS, so
> put it behind HTTPS to demo from another machine.

The first verification loads ~95 MB of Facenet512 weights; the server warms
them at startup so the first user does not pay for it.

<details>
<summary>CLI (development and testing only)</summary>

```bash
python cli.py status                                   # what is loaded, what is degraded
python cli.py enrol --name "Ada" --images "photos/*.jpg"
python cli.py verify --camera 0                        # live webcam
python cli.py verify --video clip.mp4                  # from a recording
python cli.py audit --limit 20                         # recent decisions
python cli.py set-pin --name "Ada"                     # PIN fallback
```
</details>

---

## How the liveness check works

Four signals. The important column is the last one — each is defeated by
something *different*, which is the entire argument for layering.

| Signal | Question it answers | Defeated by |
|---|---|---|
| **Texture** | Is this surface skin? | A very high quality print, or a mask of skin-like material |
| **Blink** | Do these eyes behave like eyes? | A video replay — it contains real blinks |
| **Motion** | Did the thing that moved have depth? | A cooperative victim, or a real 3-D mask |
| **Depth** | Is it flat? | Nothing cheap — but needs hardware this machine lacks |

### 1. Surface texture

Three things happen to a face between the real world and a re-captured copy of
it, and all three are measurable:

- **Detail is destroyed.** Skin has fine irregular micro-texture — pores, fine
  hair, subsurface mottling. Printing replaces it with ink dots; a screen
  replaces it with a pixel grid. Local Binary Patterns at two scales measure
  exactly this.
- **New periodic structure appears.** A display has a physical pixel lattice,
  and photographing one produces moiré — energy at specific spatial frequencies
  no real face emits. The radial FFT power spectrum makes it visible.
- **Colour and reflectance shift.** Skin is translucent and scatters light
  beneath the surface. Paper and glass do not, so chroma compresses and broad
  specular highlights appear where the room reflects.

142 features feed a logistic regression — linear on purpose, because the
training set is small enough that a stronger model would memorise it, and
because the coefficients can be read directly
(`python tools/train_spoof_classifier.py --explain`).

### 2. Blink

Two independent measures of eye closure: the classic **eye aspect ratio** from
six landmarks per eye, and MediaPipe's calibrated **blink blendshapes**, which
unlike raw EAR are robust to head rotation.

A blink counts only if the eye closes and reopens within **70–500 ms** — the
physiological range of a spontaneous human blink. Faster is landmark noise.
Slower is a squint, or something being moved across a photograph's eyes.
Neither is proof of life. There is a deliberate hysteresis band between the
"closed" and "open" thresholds, so a signal dithering around one threshold
cannot manufacture blinks.

Two details that came out of measurement rather than theory:

- EAR thresholds are a **fraction of each subject's own open-eye baseline**,
  not absolute. Measured across this project's images, a wide-open eye ranges
  from 0.245 to 0.523 between individuals — a 2× spread. Any fixed threshold is
  unreachable for one person and permanently tripped for another.
- Closure is taken from the **more open** of the two eyes, so a blink is only
  counted when *both* close. Averaging them, which is the obvious approach,
  lets a single wink register as a blink.

### 3. Motion parallax — the one that does the real work

Asking "did you turn your head" is trivially defeated: tilt a printed photo and
the pose estimator happily reports a turn. So the system asks a harder question
— **did turning reveal three-dimensional structure?**

The geometry is exact. **Rotating a flat plane in front of a camera is precisely
a homography.** So fit the best homography mapping the landmarks in one frame
onto a later, rotated frame:

- If it fits, the thing that turned was flat.
- If there is stubborn residual error, the thing that turned had depth.

In practice that alone is not enough, because re-detecting a warped image makes
the landmarker refit slightly differently, and that refit noise is not
homography-explainable either. So a second condition is required: the
per-landmark residual must **correlate with landmark depth**. Real parallax
displaces the nose far more than the cheek; refit noise does not care where a
landmark sits. Measured separation on that second condition is Cohen's
*d* = 2.17.

The challenge instruction is chosen at random from four per attempt, so a
pre-recorded clip of the victim turning one way does not satisfy the other
three. The pose trajectory is also checked for continuity — a jump larger than
a head can physically travel between frames means the target in front of the
lens was swapped, not turned.

### 4. Depth — honestly unavailable

With a real depth sensor this is the strongest signal available and the reason
Face ID is hard to beat with a photograph: a flat target has no facial relief
to measure. The code probes for one (Intel RealSense).

This machine has none. The system says so — in the API, in the UI status pill,
and on every result as `degraded: true` — and **refuses to substitute something
weaker**. See the next section for the shortcut that was tried and rejected.

---

## Measured results

All numbers produced by the scripts in `eval/`, on the hardware below. Nothing
here is quoted from a paper.

<details open>
<summary><strong>Recognition</strong> — leave-one-image-out over the enrolled gallery</summary>

| Metric | Value |
|---|---|
| Recognition accuracy | **1.000** (15 genuine trials) |
| FAR — impostors accepted | **0.000** — 95% CI **[0.000, 0.117]** |
| FRR — genuine users rejected | **0.000** — 95% CI [0.000, 0.204] |
| Genuine similarity | 0.821 ± 0.084 (min 0.651) |
| Impostor similarity | 0.322 ± 0.135 (max 0.602) |
| Separation | Cohen's *d* = **4.45** |

**Read the confidence interval, not the point estimate.** Zero false accepts in
29 impostor trials does not mean FAR = 0; it means FAR is somewhere below about
12%. With 7 identities and 17 images, that is the most this dataset can
support. The Cohen's *d* of 4.45 is the more meaningful number: genuine and
impostor score distributions are cleanly separated.
</details>

<details open>
<summary><strong>Spoof detection</strong> — identity-disjoint splits, simulated attacks</summary>

| Metric | Value |
|---|---|
| AUC | **0.977** |
| Accuracy | 0.935 |
| **APCER** — attacks accepted as live | **0.071** |
| **BPCER** — real faces rejected | 0.060 |
| ACER | 0.065 |

By attack type:

| Attack | APCER | n |
|---|---|---|
| Print | 0.071 | 28 |
| Screen replay | 0.143 | 28 |
| Generative-artefact proxy | 0.000 | 28 |

Train/val/test share **no identity**. The threshold was chosen by
leave-one-identity-out cross-validation over the pooled train and validation
identities; the test split was touched once.

Pooled CV AUC was 0.845 against a mean per-fold AUC of 0.954. That gap is
reported rather than smoothed over: the model separates live from spoof well
*within* each person but its probabilities are not calibrated *across* people,
so a single global threshold is the weak link, not the features.

These are **texture-signal-only** numbers. The deployed policy also requires
blink and parallax, which a printed photo fails independently.
</details>

<details open>
<summary><strong>End-to-end</strong> — the full layered policy on sequences</summary>

| | Result |
|---|---|
| Attacks passing all layers | **0 / 51** (APCER 0.000) |
| Print | 0 / 19 |
| Screen replay | 0 / 16 |
| Generative proxy | 0 / 16 |

Why they were rejected (attacks trip several layers at once):

| Reason | Count |
|---|---|
| `NO_BLINK_DETECTED` | 51 |
| `LIVENESS_SCORE_TOO_LOW` | 49 |
| `TEXTURE_SPOOF_SUSPECTED` | 48 |
| `FLAT_GEOMETRY` | 45 |
| `CHALLENGE_FAILED` | 39 |

**The live pass rate in this harness is 0/4, and that is expected.** Sequences
are synthesised from stills by plane-rotation homography, which is
geometrically a flat rotation — faithful for attacks, and a deliberately unfair
worst case for a live face, which *should* fail a parallax check when it is
literally flat. The live side cannot be measured without a real camera. See
[Known limitations](#known-limitations).
</details>

<details open>
<summary><strong>Latency</strong> — Windows 11, AMD64 CPU, no GPU</summary>

| Stage | Median | p95 | p99 |
|---|---|---|---|
| Detection + 478 landmarks | 12.4 ms | 12.9 ms | 13.2 ms |
| Texture features | 2.1 ms | 2.4 ms | 2.5 ms |
| Liveness fusion (per decision) | 36.5 ms | 41.5 ms | 43.9 ms |
| Recognition embedding | 208.2 ms | 214.5 ms | 218.4 ms |

Detection and texture run every frame (14.6 ms). Recognition runs on one frame
in five, adding 41.6 ms amortised.

**56.2 ms per frame → 17.8 fps sustained.** Compute for a 15-frame attempt is
0.88 s; wall-clock is longer because the challenge is paced by how fast a person
turns their head, not by the pipeline.

Recognition embedding dominates and is the obvious optimisation target.
</details>

<details>
<summary><strong>Environment robustness</strong> — 26 degraded conditions</summary>

Two failure directions are counted separately, because conflating them is
misleading: failing to match *yourself* is annoying, matching *someone else* is
dangerous.

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
| Yaw ≥ 25° | 17/17 | — | — | 0 |
| Too far (×0.35) | 11/17 | 6/6 | 0.941 | 0 |
| Too close (×2.6) | 17/17 | — | — | 0 |
| Motion blur (all) | 17/17 | — | — | 0 |
| JPEG q10 | 0/17 | 17/17 | 0.955 | 0 |

**Zero crashes.** Refusals are correct behaviour — the system said why it could
not decide.

The hand-occlusion result was a **real security hole**, and how it was closed is
in the next section.
</details>

---

## What is actually defended, and what is not

Stated precisely, because this is the section a technical reviewer will read
first.

### Defended, with evidence

| Attack | Status | Evidence |
|---|---|---|
| **Printed photo, held still** | Defended | Fails blink, parallax and challenge independently. 0/19 print attacks passed end-to-end |
| **Printed photo, tilted to fake a head turn** | Defended | The parallax check is what closes this. Rotating a plane is exactly a homography; measured separation Cohen's *d* = 2.17 |
| **Screen replay, static image** | Defended | 0/16 passed. Fails blink and texture |
| **Second face in frame** | Defended | Explicit `MULTIPLE_FACES` refusal, never a pick-the-biggest guess |
| **Brute-force retry** | Defended | Sliding-window lockout, exponential backoff, shared with PIN |
| **Template theft from disk** | Mitigated | Fernet-encrypted, authenticated. Raw images never stored |
| **Store tampering** | Defended | Fernet auth tag; a modified store fails to decrypt rather than yielding altered templates |

### Open gaps — untested or known weak

| Attack | Status |
|---|---|
| **Video replay of the victim performing the exact challenge** | **Open.** Blink and challenge both pass. Only texture and parallax stand in the way, and a high-quality screen at the right distance may beat texture. This is the most realistic attack against the system |
| **Real printed photo, physically held to a real webcam** | **Untested.** All print results are from simulation. See below |
| **Real screen replay on a real second screen** | **Untested.** Same |
| **3-D mask (resin, silicone)** | **Open.** Has real depth, so it passes parallax. Texture is the only defence and was never trained on mask materials |
| **Real deepfake video** | **Untested.** The `synthetic` class is a hand-built imitation of generative artefacts, not output from any generative model. A classifier that detects it has not been shown to detect anything real |
| **Adversarial perturbation** | **Untested.** No adversarial robustness work was done |
| **Identical twins** | **Untested.** Recognition-level problem, not liveness |

### The honest headline

Every anti-spoofing number in this repository comes from **simulated attacks**.
The simulations model the physics — halftone screening and gamut compression
for print, pixel lattice and moiré and backlight bleed for replay — and the
pipeline avoids the obvious shortcut (live images go through the same camera
noise and JPEG pass as attacks, so the classifier cannot learn "compressed
means fake"). But they are simulations. Real-world APCER will be higher,
probably substantially.

**The physical test has not been run.** It needs a printed photo and a second
screen, which is ten minutes of work and would make every claim here far more
credible:

```bash
python cli.py verify --camera 0        # you
python cli.py verify --camera 0        # then hold up a printed photo
python cli.py verify --camera 0        # then a video of yourself on a phone
python cli.py audit --limit 10         # compare the reason codes
```

---

## Things that were tried and did not work

Documented because a project that only lists its successes is not
trustworthy, and because these were expensive lessons.

### The landmark-planarity depth proxy — abandoned

MediaPipe returns a z coordinate per landmark, so the tempting shortcut is to
fit a plane to the landmark cloud and call a flat fit a photograph. It was
implemented and measured:

| | Live | Simulated spoof |
|---|---|---|
| Planarity residual | 0.2060 ± 0.0149 | 0.2115 ± 0.0207 |

**Cohen's *d* = 0.30.** That is no separation at all. MediaPipe's z is
*inferred from a 2-D image*, never measured, so a photograph of a face produces
a perfectly face-shaped landmark cloud complete with a protruding nose that is
not there.

It is still computed and reported so a reviewer can check the claim, and it
carries **zero weight** in the policy. The genuine depth-free substitute is the
parallax check, which recovers structure from *motion* instead of trusting a
model's guess about depth.

### The non-rigid deformation ratio — demoted

Align each frame on rigid landmarks (nose bridge, temples), then measure how
much the deformable ones (lips, brows) moved. A photo should score ~1.0.

It does not. A jittered still image scores **1.33 ± 0.12, up to 1.55**, because
soft-tissue landmarks sit on low-contrast edges and are inherently noisier to
localise than the nose bridge. The threshold was originally set at 1.35 — the
middle of the spoof distribution, which would have accepted roughly half of all
print attacks.

Kept as corroborating evidence, reported in the metrics, never able to grant a
pass on its own.

### Region-contrast occlusion detection — calibrated, then demoted to advisory

First attempt: flag any face region much flatter than its neighbours. It fired
on **59% of clean faces**, because facial regions are not interchangeable — the
bridge of the nose measures **four times flatter** than the face median on every
healthy face.

Normalising by measured per-region baselines brought false positives to 17.6%
and detection to ~86%. Still too unreliable to *reject* on — that would turn
away one honest user in six — so it advises rather than gates, and it excludes
frames from the recognition vote (below).

### The hand-occlusion false acceptance — found and closed

The robustness suite found a real hole. With a hand covering part of the face,
two probes matched the **wrong enrolled identity** at cosine 0.632 and 0.650.

Threshold tuning cannot fix it. The lowest *genuine* match over the same gallery
scored **0.651** — one thousandth above the worst false accept. The runner-up
margin did not block them either (gaps of 0.148 and 0.069).

Closed by **excluding suspected-occluded frames from the recognition vote**.
The severity is chosen from measurement: the detector fires on 17.6% of clean
faces, so refusing the attempt outright would turn away one honest user in six,
while dropping a frame costs only a retry. With 3 agreeing frames required and
~87% detection on hand occlusion, an attacker needs three consecutive misses.
Regression test: `tests/test_occlusion_defence.py`.

### Three bugs the test suite caught

| Bug | Consequence |
|---|---|
| Euler decomposition used the wrong rotation order | "Turn left" was satisfied by turning **right** |
| Parallax checked depth correlation before residual | A perfectly flat target reported `UNAVAILABLE` instead of `FLAT_GEOMETRY` — an unmeasurable secondary check overturning a conclusive primary one |
| Blink averaged both eyes | A single **wink** registered as a blink |

### One scale-invariance bug

Laplacian-variance sharpness is not scale-invariant: the same photo measured
**0.084 at 3120 px wide and 1.000 at 640 px**. The browser downscales before
upload so the web path was accidentally consistent; enrolling the same faces
from original files rejected every one as `LOW_QUALITY`.

Fixed by normalising resolution at every entry point rather than loosening the
threshold. That also improved spoof detection markedly, since texture features
became comparable across sources: test AUC 0.967 → **0.977**, print APCER 0.286
→ **0.071**.

---

## Security design

### No single-frame decisions

15 usable frames minimum before any verdict, 3 independent frames must agree on
an identity, and the config loader **refuses to start** if `min_frames` is set
below 2 or `min_agreeing_frames` below 2.

### Unavailable is not the same as passed

`SignalStatus` distinguishes `FAIL` from `UNAVAILABLE`. If the spoof classifier
is missing, the attempt is rejected with `LIVENESS_UNAVAILABLE` — never waved
through. Unavailable signals are *excluded* from the weighted score rather than
scored zero, because scoring them zero would penalise every honest user for a
missing depth camera and push an operator toward lowering the threshold —
weakening the system in response to a hardware gap instead of reporting it.

### Liveness is decided before identity is revealed

An attacker holding up a photo learns only that liveness failed, never whether
the face in the photo is enrolled.

### Template storage

Fernet (AES-128-CBC + HMAC-SHA256). Raw face images are never persisted.

Embeddings are treated as biometric data, not anonymous vectors: template
inversion can reconstruct a recognisable face from one, and unlike a password a
face cannot be rotated after a breach.

**What this protects against, stated honestly:** the repository, the backup, the
cloud sync, the other user of the machine. It does **not** protect against
someone who has the key. By default the key sits in `data/faceguard.key` beside
the ciphertext so the project runs out of the box — that is not a production
posture. Set `FACEGUARD_KEY` from a secrets manager and the key never touches
disk.

Writes are atomic (temp file, fsync, rename) so an interrupted save cannot lock
every enrolled user out.

### Rate limiting

Sliding window, exponential lockout to a ceiling. Framing failures — dark room,
face too far, two people in shot — are **excluded** from the lockout counter.
Punishing a usability problem as though it were an attack is the fastest way to
get a security control switched off.

### Audit log

JSONL. Rejections recorded in the same detail as acceptances — a success-only
log cannot show you twenty `TEXTURE_SPOOF_SUSPECTED` failures from one client in
two minutes, which is the only interesting thing in it. Client identifiers are
hashed so logs are shareable for debugging.

### PIN fallback

scrypt with per-entry salts. Unknown identities still pay the full hash cost so
timing cannot enumerate users. Failures share the biometric lockout counter, so
PIN guessing is not a way around face rate limiting.

A 6-digit PIN is about 20 bits of entropy — trivially brute-forceable offline if
the hash file leaks. It is a fallback, and it is not claimed to be more.

---

## Architecture

```
faceguard/
  config.py          Typed YAML config. Unknown/missing keys and unsatisfiable
                     policies raise at load time, not at 3am
  schemas.py         ReasonCode enum + result dataclasses. Detection and
                     recognition confidence kept separate throughout
  detect.py          MediaPipe Tasks: bbox + detection score + 478 landmarks
                     + 52 blendshapes + 4x4 pose, in one inference pass
  quality.py         Quality scoring, resolution normalisation, occlusion hints
  recognition.py     Facenet512 embeddings, margin + multi-frame voting
  store.py           Fernet-encrypted template store, atomic writes
  session.py         Multi-frame state machine, rate limiting, the facade
  audit.py           JSONL audit log
  pin.py             scrypt PIN fallback
  liveness/
    texture.py       LBP + FFT + colour features, logistic regression
    blink.py         EAR + blendshapes, physiological duration gate
    motion.py        Planar parallax, head pose, challenge-response
    depth.py         Hardware probe + the rejected planarity proxy
    fusion.py        Policy engine

web/                 FastAPI + WebSocket backend, dark-theme SPA frontend
cli.py               Development and testing CLI
tools/               Model fetch, splits, attack simulation, training, calibration
eval/                evaluate.py, latency.py, robustness.py
tests/               118 tests, no face images required
```

**Why a WebSocket rather than a POST endpoint:** the system refuses to decide
from a single frame, so the demo has to be a conversation — issue a random
challenge, watch a sequence, and tell the user what to fix while they are still
in front of the camera. A post-one-photo endpoint would be a weaker system
wearing this one's UI.

---

## Configuration

Every accept/reject threshold lives in [`config/default.yaml`](config/default.yaml).
Nothing in the source hard-codes a decision boundary.

The loader is strict on purpose: unknown keys, missing keys, type mismatches and
cross-field contradictions all raise `ConfigError`. A typo in a threshold name
must not silently fall back to a permissive default — that is how a "tightened"
config becomes a no-op. It also refuses configurations that are unsatisfiable
(more optional signals required than exist) or self-defeating (inverted blink
hysteresis, single-frame liveness).

```bash
FACEGUARD_CONFIG=config/strict.yaml python -m uvicorn web.app:app
python cli.py --config config/strict.yaml status
```

Environment variables:

| Variable | Purpose |
|---|---|
| `FACEGUARD_KEY` | Fernet key for the template store. Set this in production |
| `FACEGUARD_CONFIG` | Path to a config file |
| `FACEGUARD_AUDIT_SALT` | Salt for hashing client identifiers in the audit log |

---

## Testing and evaluation

```bash
pytest                                    # 118 tests, ~4 seconds, no data needed
python eval/evaluate.py                   # recognition + spoof, reported separately
python eval/latency.py                    # per-stage latency on your hardware
python eval/robustness.py                 # 26 degraded conditions
```

Tests run on a fresh clone with **no face images present** — the dataset is
personal data and is not in this repository. The core geometric claims are
tested on synthetic point clouds, where a cloud with facial relief and the same
cloud flattened test the anti-spoofing claim more directly than a photograph
could, because the ground truth is exact.

Reproduce the full pipeline on your own images:

```bash
python tools/make_splits.py --root your_faces --output splits/manifest.json
python tools/make_splits.py --verify splits/manifest.json     # leakage check
python tools/attack_sim.py --manifest splits/manifest.json --output attacks/
python tools/train_spoof_classifier.py --explain
python eval/evaluate.py
```

`make_splits.py` splits over **identities, never images**, and refuses to write
a manifest that leaks. Two photos of the same person on opposite sides of a
split let a model score well by recognising the person rather than by learning
anything about spoofing — and a random shuffle of image paths looks perfectly
reasonable while doing exactly that.

---

## Known limitations

1. **Every anti-spoofing number comes from simulated attacks.** No printed
   photo has been held in front of a real webcam. Real APCER will be higher.
2. **The dataset is tiny.** 7 identities, 17 images. Every metric has wide
   confidence intervals, and the identity-disjoint spoof split has 4 training
   identities. Nothing here should be read as a benchmark result.
3. **The live end-to-end pass rate is unmeasured.** The synthetic sequences are
   geometrically flat, so a live face correctly fails parallax in that harness.
   Only a real camera settles it.
4. **Parallax thresholds are calibrated on simulation.** The flat-target side is
   end-to-end and trustworthy; the live side is projected from MediaPipe's
   inferred depth. Run `tools/measure_parallax.py` and a real capture before
   treating the numbers as an operating point.
5. **Spoof scores are not calibrated across people.** Pooled CV AUC 0.845 vs
   mean per-fold 0.954 — a single global threshold is the weak link.
6. **Screen replay is the weakest defended attack** at APCER 0.143 for texture
   alone, and a video replay performing the right challenge is the most
   realistic open attack.
7. **Yaw beyond ~25° is refused entirely.** Safe, but it means the challenge
   asks for a movement large enough to matter and small enough to stay
   detectable — a narrow window.
8. **No depth hardware was available.** The strongest available defence is
   untested on real hardware, and the software substitute was measured not to
   work.
9. **Occlusion detection is advisory**, at 17.6% false positives on clean faces.
10. **`client_id` is the socket peer address.** Behind a proxy every client
    shares one address. A real deployment should key rate limiting on an
    authenticated session or device token. `X-Forwarded-For` is deliberately not
    trusted — a client can set it to anything and reset their own lockout.

### Not done

- **ONNX export.** `onnxruntime` is installed as a DeepFace dependency but the
  Facenet512 graph is not exported. Recognition at 208 ms is the clear
  optimisation target and this is the obvious route.
- **Adversarial robustness testing.**
- **Real deepfake evaluation.** The `synthetic` attack class is a proxy and is
  labelled as one everywhere.

---

## If someone technical asks how secure this is

Say this:

> It is a layered presentation-attack detection system with measured numbers
> against **simulated** attacks: AUC 0.977, APCER 0.071 for the texture signal,
> and 0 of 51 simulated attacks passing the full policy. Recognition separates
> genuine from impostor with Cohen's *d* = 4.45.
>
> The strongest part is the parallax check — rotating a flat surface is exactly
> a homography, so a tilted photo cannot fake the depth structure a real head
> reveals. That closes the hole in naive challenge-response.
>
> The honest gaps: I have not tested against a physical printed photo or a real
> screen replay, only simulations. A video replay of me performing the requested
> challenge would likely get through. There is no depth sensor, and I measured
> the software substitute at Cohen's *d* = 0.30 and gave it zero weight rather
> than shipping it as a defence. The dataset is 7 people, so every confidence
> interval is wide.

Do **not** say:

- "99% accurate" — that number is recognition-only and meaningless about spoofing
- "Face ID-level security" — Face ID has a dot projector and an infrared camera;
  this has neither
- "Deepfake detection" — the generative class is a hand-built proxy, never
  validated against a real generative model
- "Production ready" — the key sits next to the ciphertext by default and rate
  limiting keys on an IP address

The thing worth claiming is the **method**: thresholds derived from measurement
rather than guessed, two features deleted because measurement showed they did
not work, and a security hole found by the project's own robustness suite and
closed with a mechanism whose severity was chosen from data.

---

## Privacy and data

- **No face images are in this repository**, and `.gitignore` excludes them,
  along with templates, the encrypted store, keys, audit logs and model
  checkpoints. `.gitignore` was the first file committed, before any other file
  was tracked.
- **No sample faces are shipped.** Bring your own and enrol through the UI. Any
  images used to produce the numbers above were of consenting family members and
  stayed local.
- **Nothing is recorded during a session.** Frames exist for the length of one
  attempt. Only encrypted templates and the audit log touch disk.
- Audit logs contain identity names and are personal data — they are gitignored
  and client identifiers are hashed.

---

## Licence

MIT. See [LICENSE](LICENSE).

Built with [MediaPipe](https://developers.google.com/mediapipe),
[DeepFace](https://github.com/serengil/deepface), scikit-learn and FastAPI.
