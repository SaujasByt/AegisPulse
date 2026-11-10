/**
 * AegisPulse AI - Simplified Clinical Command Center Controller
 */

const state = {
  currentTab: 'queue',
  soundEnabled: true,
  status: {},
  queue: [],
  doctors: [],
  beds: [],
  wardSummary: [],
  selectedWardFilter: 'ALL'
};

// Web Audio Clinical Sound Synthesizer
const audio = {
  ctx: null,
  init() {
    if (!this.ctx && (window.AudioContext || window.webkitAudioContext)) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
  },
  play(freq = 440, type = 'sine', duration = 0.12) {
    if (!state.soundEnabled) return;
    try {
      this.init();
      if (!this.ctx) return;
      if (this.ctx.state === 'suspended') this.ctx.resume();
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
      gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch (e) {}
  },
  alert() {
    this.play(880, 'sine', 0.15);
    setTimeout(() => this.play(660, 'sine', 0.2), 120);
  },
  chime() {
    this.play(523, 'triangle', 0.1);
    setTimeout(() => this.play(659, 'triangle', 0.15), 80);
    setTimeout(() => this.play(784, 'triangle', 0.22), 160);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  initClock();
  initTabs();
  initSoundToggle();
  initIntakeForm();
  fetchAllData();

  // Poll every 4 seconds for real-time updates
  setInterval(() => {
    fetchStatus();
    fetchQueue();
    if (state.currentTab === 'beds') {
      fetchBeds();
    }
  }, 4000);
});

// Live Clock
function initClock() {
  const clockEl = document.getElementById('live-clock');
  function updateTime() {
    if (!clockEl) return;
    const now = new Date();
    const pad = n => String(n).padStart(2, '0');
    clockEl.textContent = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
  }
  updateTime();
  setInterval(updateTime, 1000);
}

// Navigation Tabs
function initTabs() {
  document.querySelectorAll('[data-tab-target]').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tabTarget;
      state.currentTab = target;

      document.querySelectorAll('[data-tab-target]').forEach(b => {
        const isCurrent = b.dataset.tabTarget === target;
        b.classList.toggle('bg-sky-500', isCurrent);
        b.classList.toggle('text-white', isCurrent);
        b.classList.toggle('text-slate-400', !isCurrent);
      });

      document.querySelectorAll('[data-tab-content]').forEach(sec => {
        sec.classList.toggle('hidden', sec.dataset.tabContent !== target);
      });

      if (target === 'beds') {
        fetchBeds();
      }
      audio.play(480, 'sine', 0.05);
    });
  });
}

// Sound toggle
function initSoundToggle() {
  const btn = document.getElementById('sound-toggle-btn');
  const icon = document.getElementById('sound-icon');
  if (!btn) return;

  btn.addEventListener('click', () => {
    state.soundEnabled = !state.soundEnabled;
    btn.classList.toggle('text-sky-400', state.soundEnabled);
    btn.classList.toggle('text-slate-500', !state.soundEnabled);
    if (icon) {
      icon.setAttribute('data-lucide', state.soundEnabled ? 'volume-2' : 'volume-x');
      lucide.createIcons();
    }
    if (state.soundEnabled) audio.chime();
  });
}

// -------------------------------------------------------------
// Data Fetching & Rendering
// -------------------------------------------------------------

async function fetchAllData() {
  await Promise.all([fetchStatus(), fetchQueue(), fetchBeds()]);
  lucide.createIcons();
}

async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    state.status = data;

    const waiting = data.active_waiting_patients ?? 0;
    const freeBeds = (data.total_hospital_beds ?? 120) - (data.occupied_hospital_beds ?? 0);

    setText('kpi-queue-count', `${waiting} Patients`);
    setText('kpi-avg-wait', `${data.average_wait_time_minutes ?? 0} mins`);
    setText('kpi-beds-free', `${freeBeds} Free`);
    setText('nav-queue-count', waiting);
  } catch (e) {
    console.error("Status fetch error", e);
  }
}

async function fetchQueue() {
  try {
    const res = await fetch('/api/queue');
    const data = await res.json();
    state.queue = data.queue || [];
    state.doctors = data.doctors || [];
    renderQueueCards(state.queue);
    renderDoctorRooms(state.doctors);
  } catch (e) {
    console.error("Queue fetch error", e);
  }
}

function renderQueueCards(queue) {
  const container = document.getElementById('queue-cards-list');
  if (!container) return;

  if (queue.length === 0) {
    container.innerHTML = `
      <div class="p-8 text-center text-slate-400 clean-card">
        <i data-lucide="check-circle" class="w-10 h-10 mx-auto text-emerald-400 mb-2"></i>
        <div class="text-sm font-semibold text-white">OPD Queue is Empty</div>
        <div class="text-xs text-slate-400 mt-0.5">All arriving patients have been attended by doctors.</div>
      </div>
    `;
    lucide.createIcons();
    return;
  }

  container.innerHTML = queue.map((p, idx) => {
    const isConsulting = p.status === 'IN_CONSULTATION';
    const urgency = p.predicted_urgency_level || 3;

    const urgencyLabels = {
      1: { text: "Emergency (ESI 1)", bg: "bg-rose-500/15 text-rose-400 border-rose-500/30", dot: "bg-rose-500 pulse-beacon-red" },
      2: { text: "Very Urgent (ESI 2)", bg: "bg-orange-500/15 text-orange-400 border-orange-500/30", dot: "bg-orange-400" },
      3: { text: "Urgent (ESI 3)", bg: "bg-yellow-500/15 text-yellow-400 border-yellow-500/30", dot: "bg-yellow-400" },
      4: { text: "Standard (ESI 4)", bg: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30", dot: "bg-emerald-400" },
      5: { text: "Routine (ESI 5)", bg: "bg-sky-500/15 text-sky-400 border-sky-500/30", dot: "bg-sky-400" }
    };
    const badge = urgencyLabels[urgency] || urgencyLabels[3];

    // Vitals micro-badge styling (Red for severe/critical, Amber for elevated)
    let hrBadge = '';
    if (p.heart_rate_bpm > 120) {
      hrBadge = `<span class="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/40 font-bold">${p.heart_rate_bpm} bpm</span>`;
    } else if (p.heart_rate_bpm >= 105 || p.heart_rate_bpm <= 50) {
      hrBadge = `<span class="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 font-semibold">${p.heart_rate_bpm} bpm</span>`;
    } else {
      hrBadge = `<strong class="text-white">${p.heart_rate_bpm} bpm</strong>`;
    }

    let bpBadge = '';
    if (p.systolic_bp >= 170 || p.diastolic_bp >= 105) {
      bpBadge = `<span class="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/40 font-bold">${p.systolic_bp}/${p.diastolic_bp}</span>`;
    } else if (p.systolic_bp >= 140 || p.diastolic_bp >= 90 || p.systolic_bp <= 95) {
      bpBadge = `<span class="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 font-semibold">${p.systolic_bp}/${p.diastolic_bp}</span>`;
    } else {
      bpBadge = `<strong class="text-white">${p.systolic_bp}/${p.diastolic_bp}</strong>`;
    }

    let spo2Badge = '';
    if (p.spo2_percent <= 92) {
      spo2Badge = `<span class="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/40 font-bold">${p.spo2_percent}%</span>`;
    } else if (p.spo2_percent <= 95) {
      spo2Badge = `<span class="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 font-semibold">${p.spo2_percent}%</span>`;
    } else {
      spo2Badge = `<span class="text-emerald-400 font-semibold">${p.spo2_percent}%</span>`;
    }

    let painBadge = '';
    if (p.pain_score >= 8) {
      painBadge = `<span class="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/40 font-bold">${p.pain_score}/10</span>`;
    } else if (p.pain_score >= 5) {
      painBadge = `<span class="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 font-semibold">${p.pain_score}/10</span>`;
    } else {
      painBadge = `<strong class="text-white">${p.pain_score}/10</strong>`;
    }

    return `
      <div class="p-4 rounded-xl clean-card priority-strip-${urgency} flex flex-col md:flex-row md:items-center justify-between gap-4">
        
        <!-- Patient Information -->
        <div class="space-y-1">
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs font-bold px-2.5 py-0.5 rounded-full border ${badge.bg} flex items-center gap-1.5">
              <span class="w-2 h-2 rounded-full ${badge.dot}"></span>
              ${badge.text}
            </span>

            <span class="text-xs font-semibold text-white">${p.patient_name}</span>
            <span class="text-xs text-slate-400">(${p.age}y / ${p.gender})</span>
            <span class="text-xs font-medium text-sky-400">• ${p.department}</span>

            ${isConsulting ? `
              <span class="text-[11px] font-bold px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-300 border border-sky-500/40">
                In Consultation (${p.room_assigned || 'Active'})
              </span>
            ` : ''}
          </div>

          <div class="text-xs text-slate-300 italic">
            "${p.chief_complaint}"
          </div>

          <div class="flex items-center gap-3 text-[11px] text-slate-400 font-mono pt-1 flex-wrap">
            <span class="flex items-center gap-1">HR: ${hrBadge}</span>
            <span class="flex items-center gap-1">BP: ${bpBadge}</span>
            <span class="flex items-center gap-1">SpO2: ${spo2Badge}</span>
            <span class="flex items-center gap-1">Pain: ${painBadge}</span>
          </div>
        </div>

        <!-- Right Side: Wait time & Action button -->
        <div class="flex items-center justify-between md:justify-end gap-4 shrink-0">
          <div class="text-right">
            <div class="text-xs font-mono font-bold text-sky-400">
              Score: ${p.dps} pts
            </div>
            <div class="text-[10px] text-slate-400 font-mono">
              ${p.score_breakdown || `Base: ${p.base_acuity_score || 0} + Risk: ${p.risk_score_component || 0}`}
            </div>
            <div class="text-xs text-slate-300 mt-1">
              Wait: <strong class="text-white font-mono">${p.elapsed_wait_min}m</strong>
            </div>
            <div class="text-[10px] ${urgency === 1 ? 'text-rose-400 font-bold' : 'text-slate-400'}">
              Target SLA: ${urgency === 1 ? 'Immediate (&lt; 1m)' : `${p.target_wait_minutes}m`}
            </div>
          </div>

          <div>
            ${isConsulting ? `
              <button onclick="finishConsultation('${p.patient_id}')" class="px-3.5 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold shadow-sm transition flex items-center gap-1.5">
                <i data-lucide="check" class="w-3.5 h-3.5"></i>
                Complete
              </button>
            ` : `
              <button onclick="callSpecificPatient('${p.patient_id}')" class="px-3.5 py-2 rounded-lg ${idx === 0 ? 'bg-sky-500 hover:bg-sky-400 text-white font-bold' : 'bg-slate-800 hover:bg-slate-700 text-slate-300'} text-xs shadow-sm transition flex items-center gap-1.5">
                <i data-lucide="phone-call" class="w-3.5 h-3.5"></i>
                Call In
              </button>
            `}
          </div>
        </div>

      </div>
    `;
  }).join('');
  lucide.createIcons();
}

function renderDoctorRooms(doctors) {
  const container = document.getElementById('doctor-rooms-list');
  if (!container) return;

  container.innerHTML = doctors.slice(0, 4).map(doc => {
    const isAvail = doc.status === 'AVAILABLE';
    return `
      <div class="p-3 rounded-lg clean-card ${isAvail ? 'border-emerald-500/30' : 'border-amber-500/30'} flex items-center justify-between">
        <div>
          <div class="text-xs font-bold text-white">${doc.room}</div>
          <div class="text-[11px] text-slate-400">${doc.name}</div>
        </div>
        <span class="text-[10px] font-bold px-2 py-0.5 rounded-full ${isAvail ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'}">
          ${isAvail ? 'FREE' : 'BUSY'}
        </span>
      </div>
    `;
  }).join('');
}

// Actions
async function callTopPatient() {
  try {
    const res = await fetch('/api/queue/call-next', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      audio.alert();
      showToast(data.message, 'success');
      fetchAllData();
    } else {
      showToast(data.message, 'warning');
    }
  } catch (e) {
    showToast("Error calling patient", 'error');
  }
}

async function callSpecificPatient(patientId) {
  await callTopPatient();
}

async function finishConsultation(patientId) {
  try {
    const res = await fetch('/api/queue/finish-consultation', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_id: patientId })
    });
    const data = await res.json();
    if (data.success) {
      audio.chime();
      showToast(data.message, 'success');
      fetchAllData();
    }
  } catch (e) {
    showToast("Error completing consultation", 'error');
  }
}

// -------------------------------------------------------------
// Simplified Triage Intake & Scenarios
// -------------------------------------------------------------

function initIntakeForm() {
  const form = document.getElementById('simplified-intake-form');
  if (!form) return;

  // Trigger live predict on ANY field change (vitals, symptoms, demographics)
  const inputFields = [
    'form-name', 'form-age', 'form-gender', 'form-dept',
    'form-complaint', 'form-hr', 'form-bp', 'form-spo2', 'form-pain'
  ];

  inputFields.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', debouncePredict);
      el.addEventListener('change', debouncePredict);
      el.addEventListener('keyup', debouncePredict);
    }
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    await submitPatientForm();
  });

  // Run initial prediction so UI matches starting form values immediately
  setTimeout(runPredict, 100);
}

let predictTimer = null;
function debouncePredict() {
  clearTimeout(predictTimer);
  predictTimer = setTimeout(runPredict, 150);
}

function parseBP(bpStr) {
  const parts = String(bpStr || '120/80').split('/');
  return {
    sbp: parseFloat(parts[0]) || 120,
    dbp: parseFloat(parts[1]) || 80
  };
}

async function runPredict() {
  const bp = parseBP(document.getElementById('form-bp')?.value || '120/80');
  const payload = {
    patient_name: document.getElementById('form-name')?.value || 'New Patient',
    age: parseFloat(document.getElementById('form-age')?.value || 45),
    gender: document.getElementById('form-gender')?.value || 'Female',
    department: document.getElementById('form-dept')?.value || 'General Medicine',
    chief_complaint: document.getElementById('form-complaint')?.value || 'General Consultation',
    heart_rate_bpm: parseFloat(document.getElementById('form-hr')?.value || 75),
    systolic_bp: bp.sbp,
    diastolic_bp: bp.dbp,
    spo2_percent: parseFloat(document.getElementById('form-spo2')?.value || 98),
    pain_score: parseFloat(document.getElementById('form-pain')?.value || 0),
    respiratory_rate: 18,
    temperature_c: 37.0,
    gcs: 15
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      renderAIPrediction(data.prediction, payload);
    }
  } catch (e) {}
}

function generateClinicalLogic(payload, pred) {
  const flags = [];
  const sbp = payload.systolic_bp;
  const dbp = payload.diastolic_bp;
  const hr = payload.heart_rate_bpm;
  const spo2 = payload.spo2_percent;
  const pain = payload.pain_score;
  const complaint = (payload.chief_complaint || '').trim();

  if (hr >= 120) flags.push(`tachycardia (${hr} bpm)`);
  else if (hr <= 50) flags.push(`severe bradycardia (${hr} bpm)`);

  // Clinical respiratory status evaluation (Never call SpO2 <= 92% stable)
  let respDesc = '';
  if (spo2 <= 88) {
    respDesc = `severe hypoxia / desaturation (SpO2 ${spo2}%)`;
    flags.push(respDesc);
  } else if (spo2 <= 92) {
    respDesc = `hypoxia / desaturation (SpO2 ${spo2}%)`;
    flags.push(respDesc);
  } else if (spo2 <= 95) {
    respDesc = `mild hypoxemia (SpO2 ${spo2}%)`;
    flags.push(respDesc);
  } else {
    respDesc = `stable oxygen saturation (SpO2 ${spo2}%)`;
  }

  if (sbp >= 180 || dbp >= 110) flags.push(`hypertensive emergency (${sbp}/${dbp} mmHg)`);
  else if (sbp >= 160 || dbp >= 100) flags.push(`elevated blood pressure (${sbp}/${dbp} mmHg)`);
  else if (sbp <= 85) flags.push(`hypotensive shock (${sbp} mmHg)`);

  if (pain >= 8) flags.push(`acute severe pain (${pain}/10)`);
  else if (pain >= 5) flags.push(`moderate pain (${pain}/10)`);

  const complaintStr = complaint ? `"${complaint}"` : 'Reported symptoms';

  let explanation = '';
  if (pred.predicted_urgency_level === 1) {
    const reasons = flags.length > 0 ? flags.join(', ') : 'critical physiological presentation';
    explanation = `Presentation of ${complaintStr} combined with ${reasons} indicates acute physiological distress. Immediate resuscitation prioritized (Target SLA: Immediate < 1m) with bed recommendation for ${pred.recommended_ward}.`;
  } else if (pred.predicted_urgency_level === 2) {
    const reasons = flags.length > 0 ? flags.join(' and ') : 'high-risk clinical indicators';
    explanation = `${complaintStr} presenting with ${reasons} generates a clinical deterioration risk of ${pred.deterioration_risk_score}%. Patient requires rapid emergent evaluation within 10 minutes (${pred.recommended_ward}).`;
  } else if (pred.predicted_urgency_level === 3) {
    explanation = `${complaintStr} with ${respDesc} and HR ${hr} bpm is classified as Urgent (Target SLA: <= 30 mins). Estimated physician time: ~${pred.predicted_service_minutes}m.`;
  } else if (pred.predicted_urgency_level === 4) {
    explanation = `${complaintStr} presenting with ${respDesc} and BP ${sbp}/${dbp} indicates lower acute risk. Standard OPD consultation without bed admission.`;
  } else {
    explanation = `${complaintStr} with normal vital parameters (HR ${hr} bpm, ${respDesc}, Pain ${pain}/10) is categorized as routine care. Fast-track outpatient discharge expected.`;
  }

  return explanation;
}

function renderAIPrediction(pred, payload) {
  const badgeEl = document.getElementById('ai-acuity-badge');
  const slaEl = document.getElementById('ai-sla-wait');
  const riskEl = document.getElementById('ai-risk-val');
  const durEl = document.getElementById('ai-duration-val');
  const admEl = document.getElementById('ai-admission-verdict');
  const wardEl = document.getElementById('ai-ward-val');
  const logicEl = document.getElementById('ai-decision-logic-text');

  if (badgeEl) {
    badgeEl.textContent = `${pred.acuity_badge}: ${pred.acuity_label}`;
    badgeEl.style.color = pred.acuity_color;
  }

  // Label Polish: Change 'Target SLA: 0m' for ESI 1 cases to display 'Target SLA: Immediate (< 1m)'
  if (slaEl) {
    if (pred.predicted_urgency_level === 1) {
      slaEl.textContent = 'Target SLA: Immediate (< 1m)';
      slaEl.className = 'text-xs font-semibold text-rose-300';
    } else {
      slaEl.textContent = `Target SLA: <= ${pred.target_wait_minutes} mins`;
      slaEl.className = 'text-xs text-slate-400';
    }
  }

  if (riskEl) {
    riskEl.textContent = `${pred.deterioration_risk_score}% (${pred.deterioration_risk_score >= 70 ? 'Critical' : (pred.deterioration_risk_score >= 40 ? 'Moderate' : 'Low')})`;
    riskEl.className = pred.deterioration_risk_score >= 70 ? 'font-bold text-rose-400' : 'font-bold text-emerald-400';
  }
  if (durEl) durEl.textContent = `~${pred.predicted_service_minutes} minutes`;
  if (admEl) {
    admEl.textContent = pred.admission_recommended ? 'Hospital Bed Needed' : 'Outpatient (No bed needed)';
    admEl.className = pred.admission_recommended ? 'font-bold text-amber-400' : 'font-bold text-emerald-400';
  }
  if (wardEl) wardEl.textContent = pred.recommended_ward;

  // Reactive State: Dynamically update AI decision logic text based on exact typed vitals and complaint
  if (logicEl && payload) {
    logicEl.textContent = generateClinicalLogic(payload, pred);
  }
}

async function submitPatientForm() {
  const bp = parseBP(document.getElementById('form-bp')?.value || '120/80');
  const payload = {
    patient_name: document.getElementById('form-name')?.value || 'New Patient',
    age: parseFloat(document.getElementById('form-age')?.value || 45),
    gender: document.getElementById('form-gender')?.value || 'Female',
    department: document.getElementById('form-dept')?.value || 'General Medicine',
    chief_complaint: document.getElementById('form-complaint')?.value || 'General Consultation',
    heart_rate_bpm: parseFloat(document.getElementById('form-hr')?.value || 75),
    systolic_bp: bp.sbp,
    diastolic_bp: bp.dbp,
    spo2_percent: parseFloat(document.getElementById('form-spo2')?.value || 98),
    pain_score: parseFloat(document.getElementById('form-pain')?.value || 0),
    respiratory_rate: 18,
    temperature_c: 37.0,
    gcs: 15
  };

  try {
    const res = await fetch('/api/queue/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      audio.chime();
      showToast(data.message, 'success');
      if (window.confetti) {
        window.confetti({ particleCount: 35, spread: 50, origin: { y: 0.7 } });
      }
      await fetchAllData();
      // Switch back to queue tab
      document.querySelector('[data-tab-target="queue"]')?.click();
    }
  } catch (e) {
    showToast("Failed to add patient", 'error');
  }
}

// Preset clinical test scenarios
function applyScenario(type) {
  if (type === 'cardiac') {
    setVal('form-name', 'Eleanor Sterling');
    setVal('form-age', 68);
    setVal('form-gender', 'Female');
    setVal('form-dept', 'Cardiology');
    setVal('form-complaint', 'Severe crushing chest pain radiating to left jaw & sweating');
    setVal('form-hr', 122);
    setVal('form-bp', '188/108');
    setVal('form-spo2', 90);
    setVal('form-pain', 9);
  } else if (type === 'breathing') {
    setVal('form-name', 'David Miller');
    setVal('form-age', 72);
    setVal('form-gender', 'Male');
    setVal('form-dept', 'Pulmonology');
    setVal('form-complaint', 'Acute shortness of breath, severe wheezing & cyanosis');
    setVal('form-hr', 130);
    setVal('form-bp', '152/94');
    setVal('form-spo2', 83);
    setVal('form-pain', 6);
  } else if (type === 'injury') {
    setVal('form-name', 'Sophia Bennett');
    setVal('form-age', 28);
    setVal('form-gender', 'Female');
    setVal('form-dept', 'Orthopedics');
    setVal('form-complaint', 'Inversion ankle sprain with lateral swelling, able to bear partial weight');
    setVal('form-hr', 76);
    setVal('form-bp', '118/76');
    setVal('form-spo2', 99);
    setVal('form-pain', 4);
  } else if (type === 'checkup') {
    setVal('form-name', 'Arthur Pendelton');
    setVal('form-age', 60);
    setVal('form-gender', 'Male');
    setVal('form-dept', 'General Medicine');
    setVal('form-complaint', 'Routine 3-month blood pressure review and prescription renewal');
    setVal('form-hr', 68);
    setVal('form-bp', '122/78');
    setVal('form-spo2', 99);
    setVal('form-pain', 0);
  }

  runPredict();
  showToast(`Loaded scenario: ${type.toUpperCase()}`, 'info');
}

// -------------------------------------------------------------
// Hospital Beds Management
// -------------------------------------------------------------

async function fetchBeds() {
  try {
    const res = await fetch('/api/beds');
    const data = await res.json();
    state.beds = data.beds || [];
    state.wardSummary = data.ward_summary || [];
    renderWardCards(state.wardSummary);
    renderOccupiedBeds(state.beds);
  } catch (e) {}
}

function renderWardCards(wards) {
  const container = document.getElementById('ward-cards-grid');
  if (!container) return;

  container.innerHTML = wards.map(w => {
    const total = w.total_beds;
    const occupied = w.occupied_beds;
    // Strict bed availability math: free = max(0, total - occupied)
    const free = Math.max(0, total - occupied);
    const occPct = total > 0 ? Math.round((occupied / total) * 100) : 0;
    const isSelected = state.selectedWardFilter === w.ward_name;

    return `
      <div onclick="filterBedsByWard('${w.ward_name}')" 
           class="p-3.5 rounded-xl clean-card cursor-pointer transition ${isSelected ? 'border-sky-500 bg-sky-950/30 ring-1 ring-sky-500/50' : 'hover:border-sky-500/40 hover:bg-slate-900/80'}"
           title="Click to filter ${w.ward_name} beds below">
        <div class="flex items-center justify-between mb-1">
          <span class="text-xs font-bold ${isSelected ? 'text-sky-300' : 'text-white'} flex items-center gap-1.5">
            ${isSelected ? '<span class="w-1.5 h-1.5 rounded-full bg-sky-400"></span>' : ''}
            ${w.ward_name}
          </span>
          <span class="text-[10px] font-bold px-2 py-0.5 rounded-full ${free > 0 ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'}">
            ${free} Free
          </span>
        </div>
        <div class="text-[11px] text-slate-400">${occupied}/${total} beds filled (${occPct}%)</div>
        <div class="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-2">
          <div class="h-full ${occPct >= 85 ? 'bg-rose-500' : 'bg-sky-500'} rounded-full" style="width: ${occPct}%"></div>
        </div>
      </div>
    `;
  }).join('');
}

function onWardFilterChange(wardName) {
  state.selectedWardFilter = wardName;
  renderWardCards(state.wardSummary);
  renderOccupiedBeds(state.beds);
}

function filterBedsByWard(wardName) {
  if (state.selectedWardFilter === wardName) {
    state.selectedWardFilter = 'ALL';
  } else {
    state.selectedWardFilter = wardName;
  }
  const dropdown = document.getElementById('bed-ward-filter');
  if (dropdown) dropdown.value = state.selectedWardFilter;
  renderWardCards(state.wardSummary);
  renderOccupiedBeds(state.beds);
}

function renderOccupiedBeds(beds) {
  const container = document.getElementById('occupied-beds-table');
  if (!container) return;

  const currentFilter = state.selectedWardFilter || 'ALL';

  let filtered = beds.filter(b => b.status === 'OCCUPIED' || b.status === 'RESERVED');
  if (currentFilter !== 'ALL') {
    filtered = filtered.filter(b => b.ward === currentFilter);
  }

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="p-6 text-center text-slate-400 clean-card">
        <i data-lucide="check-circle" class="w-8 h-8 mx-auto text-emerald-400 mb-2"></i>
        <div class="text-xs font-semibold text-white">No Occupied Beds in ${currentFilter === 'ALL' ? 'Hospital' : currentFilter}</div>
        <div class="text-[11px] text-slate-500 mt-0.5">All beds in this care area are currently available for clinical assignment.</div>
      </div>
    `;
    lucide.createIcons();
    return;
  }

  container.innerHTML = filtered.map(b => {
    const isReserved = b.status === 'RESERVED';
    return `
      <div class="p-3 rounded-lg clean-card flex items-center justify-between gap-3 text-xs hover:border-slate-700 transition">
        <div class="flex items-center gap-2.5 flex-wrap">
          <span class="font-mono font-bold text-sky-400 bg-sky-950/40 px-2 py-0.5 rounded border border-sky-500/20">${b.bed_id}</span>
          <span class="text-xs font-semibold text-white">${b.patient_name || (isReserved ? 'Reserved Patient' : 'Inpatient')}</span>
          <span class="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-medium">${b.ward}</span>
          ${b.has_telemetry ? '<span class="text-[10px] px-1.5 py-0.2 rounded bg-indigo-500/20 text-indigo-300 font-mono">TELEMETRY</span>' : ''}
          ${b.has_ventilator ? '<span class="text-[10px] px-1.5 py-0.2 rounded bg-rose-500/20 text-rose-300 font-mono">VENT</span>' : ''}
          ${b.is_isolation ? '<span class="text-[10px] px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 font-mono">ISOLATION</span>' : ''}
        </div>

        <div class="flex items-center gap-3 shrink-0">
          <span class="text-slate-400 font-mono text-[11px]">~${b.hours_until_discharge}h remaining</span>
          <button onclick="dischargeBed('${b.bed_id}')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-rose-600 hover:text-white text-slate-300 transition text-[11px] font-medium flex items-center gap-1">
            <i data-lucide="log-out" class="w-3 h-3"></i>
            <span>Discharge</span>
          </button>
        </div>
      </div>
    `;
  }).join('');
  lucide.createIcons();
}

async function dischargeBed(bedId) {
  try {
    const res = await fetch('/api/beds/discharge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bed_id: bedId })
    });
    const data = await res.json();
    if (data.success) {
      audio.chime();
      showToast(data.message, 'success');
      fetchBeds();
      fetchStatus();
    }
  } catch (e) {
    showToast("Discharge failed", 'error');
  }
}

// -------------------------------------------------------------
// Utilities
// -------------------------------------------------------------

function showToast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'p-3 rounded-xl bg-slate-900 border border-slate-700 text-slate-200 text-xs shadow-xl flex items-center gap-2 max-w-sm';
  toast.innerHTML = `<i data-lucide="info" class="w-4 h-4 text-sky-400 shrink-0"></i><span>${msg}</span>`;

  container.appendChild(toast);
  lucide.createIcons();

  setTimeout(() => {
    toast.remove();
  }, 3500);
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}
