/* FaceGuard AI - demo client.
 *
 * Captures webcam frames and streams them over a WebSocket. Deliberately thin:
 * every decision is made server-side, and nothing here can influence a verdict.
 * The client's only jobs are capture, draw and display.
 */

'use strict';

const FRAME_WIDTH = 640;       // downscaled before upload; the pipeline works at this scale
const FRAME_INTERVAL_MS = 110; // ~9 fps upstream, comfortably faster than the pipeline decides
const JPEG_QUALITY = 0.72;

const $ = (id) => document.getElementById(id);

const els = {
  video: $('video'), overlay: $('overlay'), cameraIdle: $('camera-idle'),
  challengeBanner: $('challenge-banner'), challengeText: $('challenge-text'),
  challengeIcon: $('challenge-icon'),
  progressBar: $('progress-bar'), progressLabel: $('progress-label'), hint: $('hint'),
  btnStart: $('btn-start'), btnStop: $('btn-stop'), btnPin: $('btn-pin'),
  verdict: $('verdict'), verdictIcon: $('verdict-icon'),
  verdictTitle: $('verdict-title'), verdictMessage: $('verdict-message'),
  signals: $('signals'), degraded: $('degraded'), systemPill: $('system-pill'),
  meterDetection: $('meter-detection'), valueDetection: $('value-detection'),
  meterRecognition: $('meter-recognition'), valueRecognition: $('value-recognition'),
  meterLiveness: $('meter-liveness'), valueLiveness: $('value-liveness'),
  enrolVideo: $('enrol-video'), enrolIdle: $('enrol-idle'), enrolHint: $('enrol-hint'),
  enrolName: $('enrol-name'), enrolPin: $('enrol-pin'), enrolStatus: $('enrol-status'),
  btnEnrolCamera: $('btn-enrol-camera'), btnCapture: $('btn-capture'),
  btnClear: $('btn-clear'), btnEnrol: $('btn-enrol'), thumbs: $('thumbs'),
  identityList: $('identity-list'), auditBody: $('audit-body'),
  btnRefreshAudit: $('btn-refresh-audit'), policyDump: $('policy-dump'),
  pinDialog: $('pin-dialog'), pinName: $('pin-name'), pinValue: $('pin-value'),
  pinStatus: $('pin-status'), btnPinSubmit: $('btn-pin-submit'),
};

const CHALLENGE_ICONS = {
  TURN_LEFT: '↶', TURN_RIGHT: '↷', LOOK_UP: '↑', LOOK_DOWN: '↓',
};

const state = {
  socket: null, stream: null, sending: null, running: false,
  enrolStream: null, enrolFrames: [], lastBox: null, timeout: 0,
};

/* ------------------------------------------------------------------ tabs -- */

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('is-active'));
    document.querySelectorAll('.view').forEach((v) => v.classList.remove('is-active'));
    tab.classList.add('is-active');
    $(`view-${tab.dataset.view}`).classList.add('is-active');
    if (tab.dataset.view === 'audit') loadAudit();
    if (tab.dataset.view === 'enrol') loadIdentities();
  });
});

/* ---------------------------------------------------------------- camera -- */

async function startCamera(videoEl, idleEl) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    audio: false,
  });
  videoEl.srcObject = stream;
  await videoEl.play();
  if (idleEl) idleEl.hidden = true;
  return stream;
}

function stopCamera(stream, videoEl, idleEl) {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (videoEl) videoEl.srcObject = null;
  if (idleEl) idleEl.hidden = false;
}

/** Downscale the current video frame and encode it as base64 JPEG. */
function captureFrame(videoEl) {
  if (!videoEl.videoWidth) return null;
  const scale = FRAME_WIDTH / videoEl.videoWidth;
  const canvas = document.createElement('canvas');
  canvas.width = FRAME_WIDTH;
  canvas.height = Math.round(videoEl.videoHeight * scale);
  const ctx = canvas.getContext('2d');
  // The <video> is mirrored in CSS for the user's benefit, but the server must
  // receive the true (unmirrored) image or "turn left" would be inverted.
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', JPEG_QUALITY);
}

/* ------------------------------------------------------------- rendering -- */

function drawBox(box, colour) {
  const canvas = els.overlay;
  const video = els.video;
  if (!video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!box) return;

  // Server works at FRAME_WIDTH; scale its coordinates back up to video space.
  const scale = video.videoWidth / FRAME_WIDTH;
  const [x, y, w, h] = box.map((v) => v * scale);

  ctx.strokeStyle = colour;
  ctx.lineWidth = Math.max(2, video.videoWidth / 320);
  ctx.strokeRect(x, y, w, h);

  // Corner ticks, which read as "tracking" rather than "cropping".
  const tick = Math.min(w, h) * 0.18;
  ctx.lineWidth *= 1.7;
  ctx.beginPath();
  [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]
    .forEach(([cx, cy, sx, sy]) => {
      ctx.moveTo(cx + tick * sx, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + tick * sy);
    });
  ctx.stroke();
}

function setHint(text, level) {
  els.hint.textContent = text || '';
  els.hint.className = 'hint' + (level ? ` is-${level}` : '');
}

function setVerdict(kind, title, message) {
  els.verdict.className = 'verdict' + (kind ? ` is-${kind}` : '');
  els.verdictIcon.textContent =
    kind === 'accepted' ? '✓' : kind === 'rejected' ? '✗' :
    kind === 'busy' ? '●' : '●';
  els.verdictTitle.textContent = title;
  els.verdictMessage.textContent = message;
}

function setMeter(meterEl, valueEl, value, available) {
  const pct = Math.max(0, Math.min(1, Number(value) || 0)) * 100;
  meterEl.style.width = available ? `${pct}%` : '0%';
  valueEl.textContent = available ? `${pct.toFixed(1)}%` : '—';
}

function renderSignals(liveness) {
  els.signals.innerHTML = '';
  if (!liveness || !liveness.signals || !Object.keys(liveness.signals).length) {
    els.signals.innerHTML = '<li class="signal-empty">No signals were evaluated.</li>';
    return;
  }
  // Most decisive first, so the reason for a rejection is at the top.
  const order = { FAIL: 0, UNAVAILABLE: 1, PASS: 2, PENDING: 3 };
  Object.entries(liveness.signals)
    .sort((a, b) => (order[a[1].status] ?? 9) - (order[b[1].status] ?? 9))
    .forEach(([name, signal]) => {
      const li = document.createElement('li');
      li.className = 'signal';
      const badge = signal.status.toLowerCase();
      li.innerHTML = `
        <div class="signal-head">
          <span>${escapeHtml(name)}</span>
          <span class="signal-badge badge-${badge}">${escapeHtml(signal.status)}</span>
        </div>
        <p class="signal-detail">${escapeHtml(signal.detail || '')}</p>`;
      els.signals.appendChild(li);
    });
}

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = value == null ? '' : String(value);
  return div.innerHTML;
}

/* ---------------------------------------------------------- verification -- */

async function startVerification() {
  try {
    if (!state.stream) state.stream = await startCamera(els.video, els.cameraIdle);
  } catch (error) {
    setVerdict('rejected', 'Camera unavailable', String(error.message || error));
    setHint('Grant camera permission, then try again. HTTPS is required off localhost.', 'error');
    return;
  }

  els.btnStart.disabled = true;
  els.btnStop.disabled = false;
  state.running = true;
  setVerdict('busy', 'Verifying', 'Collecting a sequence of frames.');
  renderSignals(null);
  els.degraded.hidden = true;

  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${protocol}://${location.host}/ws/verify`);
  state.socket = socket;

  socket.addEventListener('open', () => socket.send(JSON.stringify({ type: 'start' })));

  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);

    if (message.type === 'session') {
      els.challengeBanner.hidden = false;
      els.challengeText.textContent = message.instruction;
      els.challengeIcon.textContent = CHALLENGE_ICONS[message.challenge] || '●';
      els.progressLabel.textContent = `0 / ${message.frames_required} frames`;
      state.timeout = message.timeout_s;
      state.sending = setInterval(() => {
        if (socket.readyState !== WebSocket.OPEN) return;
        const frame = captureFrame(els.video);
        if (frame) socket.send(JSON.stringify({ type: 'frame', data: frame }));
      }, FRAME_INTERVAL_MS);

    } else if (message.type === 'feedback') {
      const fb = message.feedback;
      const pct = fb.frames_required ? (fb.frames_collected / fb.frames_required) * 100 : 0;
      els.progressBar.style.width = `${Math.min(100, pct)}%`;
      // Once enough frames exist, the only thing still outstanding is the
      // challenge. Show the remaining time rather than a stuck frame counter,
      // so a user who cannot complete it is not left staring at "19 / 15".
      if (fb.frames_collected >= fb.frames_required && state.timeout) {
        const left = Math.max(0, state.timeout - (fb.elapsed_s || 0));
        els.progressLabel.textContent = `${left.toFixed(0)}s left`;
      } else {
        els.progressLabel.textContent = `${fb.frames_collected} / ${fb.frames_required} frames`;
      }
      setHint(fb.hint, fb.accepted_frame ? null : 'warn');
      drawBox(fb.bbox, fb.accepted_frame ? '#22c55e' : '#f59e0b');
      setMeter(els.meterDetection, els.valueDetection, fb.detection_confidence, true);

    } else if (message.type === 'result') {
      showResult(message.result);
      stopVerification();

    } else if (message.type === 'error') {
      setHint(message.message, 'error');
    }
  });

  socket.addEventListener('close', () => { if (state.running) stopVerification(); });
  socket.addEventListener('error', () => setHint('Connection to the server failed.', 'error'));
}

function stopVerification() {
  state.running = false;
  if (state.sending) { clearInterval(state.sending); state.sending = null; }
  if (state.socket && state.socket.readyState === WebSocket.OPEN) state.socket.close();
  state.socket = null;
  els.btnStart.disabled = false;
  els.btnStop.disabled = true;
  els.challengeBanner.hidden = true;
  drawBox(null);
}

function showResult(result) {
  if (result.accepted) {
    setVerdict('accepted', `Welcome, ${result.identity}`, result.message || 'Verified.');
  } else {
    setVerdict('rejected', 'Rejected', `${result.message || ''} (${result.reason})`);
  }
  setHint(
    `Decided in ${Math.round(result.elapsed_ms || 0)} ms from ${result.frames_used || 0} frames.`
  );
  setMeter(els.meterDetection, els.valueDetection, result.detection_confidence, true);
  setMeter(els.meterRecognition, els.valueRecognition,
    result.recognition_confidence, Boolean(result.liveness_passed));
  setMeter(els.meterLiveness, els.valueLiveness, result.liveness_score, true);
  renderSignals(result.liveness);

  if (result.degraded && result.degraded_detail) {
    els.degraded.hidden = false;
    els.degraded.innerHTML =
      `<strong>Running degraded.</strong> ${escapeHtml(result.degraded_detail)}`;
  }
  els.progressBar.style.width = '100%';
}

els.btnStart.addEventListener('click', startVerification);
els.btnStop.addEventListener('click', () => {
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify({ type: 'finish' }));
  }
  stopVerification();
  setVerdict(null, 'Stopped', 'The attempt was cancelled.');
});

/* ---------------------------------------------------------------- enrol -- */

els.btnEnrolCamera.addEventListener('click', async () => {
  try {
    state.enrolStream = await startCamera(els.enrolVideo, els.enrolIdle);
    els.btnCapture.disabled = false;
    els.btnEnrolCamera.textContent = 'Camera on';
    els.btnEnrolCamera.disabled = true;
  } catch (error) {
    els.enrolStatus.textContent = `Camera unavailable: ${error.message || error}`;
  }
});

els.btnCapture.addEventListener('click', () => {
  const frame = captureFrame(els.enrolVideo);
  if (!frame) return;
  state.enrolFrames.push(frame);
  const img = document.createElement('img');
  img.src = frame;
  els.thumbs.appendChild(img);
  els.enrolHint.textContent =
    `Captured ${state.enrolFrames.length} frame(s). Vary your angle slightly between shots.`;
  els.btnClear.disabled = false;
  updateEnrolButton();
});

els.btnClear.addEventListener('click', () => {
  state.enrolFrames = [];
  els.thumbs.innerHTML = '';
  els.enrolHint.textContent = 'Captured 0 frames.';
  els.btnClear.disabled = true;
  updateEnrolButton();
});

els.enrolName.addEventListener('input', updateEnrolButton);

function updateEnrolButton() {
  els.btnEnrol.disabled = !(state.enrolFrames.length > 0 && els.enrolName.value.trim());
}

els.btnEnrol.addEventListener('click', async () => {
  els.btnEnrol.disabled = true;
  els.enrolStatus.textContent = 'Enrolling…';
  try {
    const response = await fetch('/api/enrol', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: els.enrolName.value.trim(),
        frames: state.enrolFrames,
        pin: els.enrolPin.value.trim() || null,
      }),
    });
    const body = await response.json();
    if (!response.ok) {
      const detail = body.detail;
      els.enrolStatus.textContent =
        typeof detail === 'string' ? detail : (detail?.message || 'Enrolment failed.');
    } else {
      els.enrolStatus.textContent =
        `Enrolled ${body.identity}: ${body.templates_added} template(s) stored ` +
        `(${body.total_templates} total)${body.pin_set ? ', PIN set' : ''}.`;
      els.btnClear.click();
      els.enrolPin.value = '';
      loadIdentities();
    }
  } catch (error) {
    els.enrolStatus.textContent = `Enrolment failed: ${error.message || error}`;
  }
  updateEnrolButton();
});

async function loadIdentities() {
  try {
    const { identities } = await (await fetch('/api/identities')).json();
    els.identityList.innerHTML = '';
    if (!identities.length) {
      els.identityList.innerHTML = '<li class="signal-empty">Nobody is enrolled yet.</li>';
      return;
    }
    identities.forEach((identity) => {
      const li = document.createElement('li');
      li.className = 'identity-item';
      li.innerHTML = `
        <span>${escapeHtml(identity.name)}</span>
        <span class="identity-meta">${identity.templates} template(s)${identity.has_pin ? ' · PIN' : ''}</span>
        <button class="icon-btn" title="Delete">✕</button>`;
      li.querySelector('button').addEventListener('click', async () => {
        if (!confirm(`Delete all templates for ${identity.name}?`)) return;
        await fetch(`/api/identities/${encodeURIComponent(identity.name)}`, { method: 'DELETE' });
        loadIdentities();
      });
      els.identityList.appendChild(li);
    });
  } catch { /* panel simply stays as it was */ }
}

/* ---------------------------------------------------------------- audit -- */

async function loadAudit() {
  try {
    const { records } = await (await fetch('/api/audit')).json();
    els.auditBody.innerHTML = '';
    if (!records.length) {
      els.auditBody.innerHTML = '<tr><td colspan="7" class="signal-empty">No records yet.</td></tr>';
      return;
    }
    records.forEach((record) => {
      const accepted = record.accepted;
      const row = document.createElement('tr');
      row.innerHTML = `
        <td class="cell-mono">${escapeHtml((record.timestamp || '').slice(11, 19))}</td>
        <td>${escapeHtml(record.event || '')}</td>
        <td class="${accepted ? 'result-ok' : 'result-bad'}">
          ${accepted === undefined ? '—' : accepted ? 'ACCEPT' : 'REJECT'}</td>
        <td class="cell-mono">${escapeHtml(record.reason || record.triggered_by || '')}</td>
        <td>${escapeHtml(record.identity || '')}</td>
        <td>${record.liveness_passed === undefined ? '—' : (record.liveness_passed ? 'yes' : 'no')}</td>
        <td class="cell-mono">${record.elapsed_ms ?? ''}</td>`;
      els.auditBody.appendChild(row);
    });
  } catch { /* leave the previous contents in place */ }
}

els.btnRefreshAudit.addEventListener('click', loadAudit);

/* ------------------------------------------------------------------ pin -- */

els.btnPin.addEventListener('click', () => {
  els.pinStatus.textContent = '';
  els.pinDialog.showModal();
});

els.btnPinSubmit.addEventListener('click', async () => {
  els.pinStatus.textContent = 'Checking…';
  try {
    const response = await fetch('/api/pin/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: els.pinName.value.trim(), pin: els.pinValue.value }),
    });
    const body = await response.json();
    els.pinStatus.textContent = body.message || '';
    if (body.accepted) {
      els.pinDialog.close();
      setVerdict('accepted', `Welcome, ${body.identity}`,
        'Verified with the PIN fallback — no liveness check was performed.');
      renderSignals(null);
      els.degraded.hidden = false;
      els.degraded.innerHTML =
        '<strong>PIN fallback used.</strong> This path performs no biometric or ' +
        'liveness verification at all. It exists so a failed camera cannot lock ' +
        'someone out permanently.';
    }
    els.pinValue.value = '';
  } catch (error) {
    els.pinStatus.textContent = `Failed: ${error.message || error}`;
  }
});

/* --------------------------------------------------------------- status -- */

async function loadStatus() {
  try {
    const status = await (await fetch('/api/status')).json();

    if (!status.spoof_classifier.loaded) {
      els.systemPill.className = 'pill pill-danger';
      els.systemPill.textContent = 'Spoof model missing';
    } else if (!status.ready) {
      els.systemPill.className = 'pill pill-muted';
      els.systemPill.textContent = 'Loading models…';
    } else if (!status.depth.available) {
      els.systemPill.className = 'pill pill-warn';
      els.systemPill.textContent = 'Software-only (no depth)';
    } else {
      els.systemPill.className = 'pill pill-ok';
      els.systemPill.textContent = 'Depth sensor active';
    }

    const test = status.spoof_classifier.test_metrics;
    els.policyDump.textContent = [
      `Required signals    : ${status.policy.required_signals.join(', ')}`,
      `Optional signals    : ${status.policy.optional_signals.join(', ')} ` +
        `(need ${status.policy.min_optional})`,
      `Challenge required  : ${status.policy.challenge_required}`,
      `Minimum frames      : ${status.policy.min_frames}  (no single-frame decisions)`,
      `Min fused score     : ${status.policy.min_fused_score}`,
      '',
      `Recognition model   : ${status.recognition.model}`,
      `Match threshold     : ${status.recognition.match_threshold}`,
      `Runner-up margin    : ${status.recognition.margin_threshold}`,
      `Frames must agree   : ${status.recognition.min_agreeing_frames}`,
      '',
      `Depth hardware      : ${status.depth.available ? status.depth.backend : 'none'}`,
      `                      ${status.depth.detail}`,
      '',
      `Spoof classifier    : ${status.spoof_classifier.loaded ? 'loaded' : 'MISSING'}`,
      test
        ? `Measured (simulated): AUC ${status.spoof_classifier.test_auc?.toFixed(3) ?? 'n/a'}  ` +
          `APCER ${test.apcer.toFixed(3)}  BPCER ${test.bpcer.toFixed(3)}  ACER ${test.acer.toFixed(3)}`
        : 'Measured            : not available',
      `                      ${status.spoof_classifier.detail}`,
    ].join('\n');

    if (!status.ready) setTimeout(loadStatus, 2500);
  } catch {
    els.systemPill.className = 'pill pill-danger';
    els.systemPill.textContent = 'Server unreachable';
    setTimeout(loadStatus, 4000);
  }
}

loadStatus();
loadIdentities();
