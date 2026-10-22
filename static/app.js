// ==========================================
// NOVA FRONTEND CLIENT APPLICATION
// ==========================================

let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let microphoneStream = null;
let animationFrameId = null;
let isAudioMuted = false;
let currentTab = 'assistant';
let websocket = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();
  initWebSocket();
  loadStatus();
  loadMemoryVault();
  initCanvas();
});

// ==========================================
// 1. TAB NAVIGATION
// ==========================================

function switchTab(tab) {
  currentTab = tab;
  ['assistant', 'memory', 'security'].forEach(t => {
    const el = document.getElementById(`tab-${t}`);
    const btn = document.getElementById(`tab-btn-${t}`);
    if (t === tab) {
      el.classList.remove('hidden');
      el.classList.add('flex');
      btn.className = "px-3.5 py-1.5 rounded-lg bg-indigo-600 text-white shadow-sm flex items-center gap-1.5 transition";
    } else {
      el.classList.add('hidden');
      el.classList.remove('flex');
      btn.className = "px-3.5 py-1.5 rounded-lg text-slate-400 hover:text-white flex items-center gap-1.5 transition";
    }
  });

  if (tab === 'memory') {
    loadMemoryVault();
  }
}

// ==========================================
// 2. ORB & STATE CONTROLLER
// ==========================================

function setOrbState(state, title, desc) {
  const orb = document.getElementById('nova-orb');
  const orbMicIcon = document.getElementById('orb-mic-icon');
  const titleEl = document.getElementById('assistant-status-title');
  const descEl = document.getElementById('assistant-status-desc');
  const recordBtnText = document.getElementById('record-btn-text');

  orb.className = `orb ${state} cursor-pointer relative flex items-center justify-center`;

  if (title) titleEl.innerText = title;
  if (desc) descEl.innerHTML = desc;

  if (state === 'listening') {
    recordBtnText.innerText = "Listening... (Tap to Send)";
    orbMicIcon.className = "w-10 h-10 text-white animate-pulse";
  } else if (state === 'thinking') {
    recordBtnText.innerText = "Nova is thinking...";
    orbMicIcon.className = "w-10 h-10 text-white animate-spin-slow";
  } else if (state === 'speaking') {
    recordBtnText.innerText = "Nova is speaking...";
    orbMicIcon.className = "w-10 h-10 text-white";
  } else {
    recordBtnText.innerText = "Tap to Speak";
    orbMicIcon.className = "w-10 h-10 text-white drop-shadow-md";
  }
}

// ==========================================
// 3. WEBSOCKET REAL-TIME EVENTS
// ==========================================

function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  websocket = new WebSocket(wsUrl);

  websocket.onopen = () => {
    console.log("[WS] Connected to Nova real-time bus");
  };

  websocket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'memory_updated') {
        loadMemoryVault();
        loadStatus();
      } else if (msg.type === 'status_changed') {
        loadStatus();
      }
    } catch (e) {
      console.warn("[WS] Error parsing message:", e);
    }
  };

  websocket.onclose = () => {
    // Retry reconnecting after 3 seconds
    setTimeout(initWebSocket, 3000);
  };
}

// ==========================================
// 4. SMART AUDIO CAPTURE & FAST SPEECH RECOGNITION
// ==========================================

let speechRecognizer = null;
let speechRecognitionActive = false;
let userHasSpoken = false;
let silenceTimeout = null;

async function toggleMicrophone() {
  if (isRecording || speechRecognitionActive) {
    stopRecording();
  } else {
    startListening();
  }
}

async function startListening() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  // Option A: Ultra-Fast Browser SpeechRecognition (Zero-latency direct streaming)
  if (SpeechRecognition) {
    try {
      startSpeechRecognitionEngine(SpeechRecognition);
      return;
    } catch (e) {
      console.warn("SpeechRecognition fallback to MediaRecorder:", e);
    }
  }

  // Option B: MediaRecorder with Auto-VAD Silence Detection
  startMediaRecorderEngine();
}

function startSpeechRecognitionEngine(SpeechRecognition) {
  speechRecognizer = new SpeechRecognition();
  speechRecognizer.lang = 'en-US';
  speechRecognizer.interimResults = true;
  speechRecognizer.continuous = false;
  speechRecognizer.maxAlternatives = 1;

  let finalTranscript = '';
  speechRecognitionActive = true;
  userHasSpoken = false;

  speechRecognizer.onstart = () => {
    isRecording = true;
    setOrbState('listening', 'Nova is listening...', 'Speak naturally (auto-sends on pause)');
  };

  speechRecognizer.onresult = (event) => {
    let interim = '';
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript;
      } else {
        interim += event.results[i][0].transcript;
      }
    }

    const currentText = (finalTranscript + ' ' + interim).trim();
    if (currentText.length > 0) {
      userHasSpoken = true;
      const descEl = document.getElementById('assistant-status-desc');
      if (descEl) {
        descEl.innerHTML = `Hearing: <span class="text-sky-300 font-semibold italic">"${escapeHtml(currentText)}"</span>`;
      }
    }
  };

  speechRecognizer.onspeechend = () => {
    setOrbState('thinking', 'Nova is thinking...', 'Processing your request...');
  };

  speechRecognizer.onerror = (event) => {
    console.warn("Speech recognition error:", event.error);
    speechRecognitionActive = false;
    isRecording = false;
    if (event.error === 'not-allowed') {
      alert("Microphone permission denied. Please allow microphone access.");
      setOrbState('idle', 'Standing by...', 'Say <span class="text-sky-300 font-semibold">"Hey Nova"</span> or tap the orb');
    } else if (userHasSpoken && finalTranscript) {
      sendRecognizedText(finalTranscript);
    } else {
      setOrbState('idle', 'Standing by...', 'Say <span class="text-sky-300 font-semibold">"Hey Nova"</span> or tap the orb');
    }
  };

  speechRecognizer.onend = () => {
    speechRecognitionActive = false;
    isRecording = false;
    const recognized = finalTranscript.trim();
    if (recognized.length > 0) {
      sendRecognizedText(recognized);
    } else {
      setOrbState('idle', 'Standing by...', 'Say <span class="text-sky-300 font-semibold">"Hey Nova"</span> or tap the orb');
    }
  };

  speechRecognizer.start();
}

async function sendRecognizedText(text) {
  appendMessage('user', text);
  setOrbState('thinking', 'Nova is thinking...', 'Formulating fast response...');

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });

    if (!resp.ok) throw new Error("Server error");
    const data = await resp.json();

    appendMessage('nova', data.reply);
    speakAloud(data.reply);
  } catch (err) {
    console.error(err);
    appendMessage('nova', "Sorry, I had trouble processing that. Please try again.");
    setOrbState('idle', 'Standing by...', 'Say <span class="text-sky-300 font-semibold">"Hey Nova"</span> or tap the orb');
  }
}

async function startMediaRecorderEngine() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    microphoneStream = stream;
    audioChunks = [];
    userHasSpoken = false;

    // Set up Web Audio Analyser for live visualizer & VAD silence detection
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 64;
    source.connect(analyser);

    startVisualizer();

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      stopVisualizer();
      if (audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        await sendAudioPayload(audioBlob);
      }
    };

    mediaRecorder.start(100);
    isRecording = true;
    setOrbState('listening', 'Listening to you...', 'Speak naturally (auto-detects end of speech)');
  } catch (err) {
    console.error("Microphone access denied:", err);
    alert("Microphone permission denied. Please allow microphone access in your browser.");
  }
}

function stopRecording() {
  if (speechRecognizer && speechRecognitionActive) {
    try {
      speechRecognizer.stop();
    } catch (e) {}
    speechRecognitionActive = false;
  }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  if (microphoneStream) {
    microphoneStream.getTracks().forEach(track => track.stop());
    microphoneStream = null;
  }
  isRecording = false;
  setOrbState('thinking', 'Processing speech...', 'Analyzing query with Gemini');
}

async function sendAudioPayload(blob) {
  const formData = new FormData();
  formData.append('file', blob, 'user_voice.webm');

  try {
    const resp = await fetch('/api/audio', {
      method: 'POST',
      body: formData
    });

    if (!resp.ok) throw new Error("Audio upload failed");
    const data = await resp.json();

    if (data.transcript) {
      appendMessage('user', data.transcript);
    }
    appendMessage('nova', data.reply);
    speakAloud(data.reply);
  } catch (err) {
    console.error(err);
    appendMessage('nova', "Sorry, I had trouble processing that audio. Please try again.");
    setOrbState('idle', 'Standing by...', 'Say <span class="text-sky-300 font-semibold">"Hey Nova"</span> or tap the orb');
  }
}

// ==========================================
// 5. WAVEFORM VISUALIZER
// ==========================================

function initCanvas() {
  const canvas = document.getElementById('waveform-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function startVisualizer() {
  const canvas = document.getElementById('waveform-canvas');
  const ctx = canvas.getContext('2d');
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  let silenceStartTime = null;

  function draw() {
    animationFrameId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const barWidth = (canvas.width / bufferLength) * 2.2;
    let x = 0;
    let sum = 0;

    for (let i = 0; i < bufferLength; i++) {
      const val = dataArray[i];
      sum += val;
      const barHeight = (val / 255) * canvas.height * 0.9;

      const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
      gradient.addColorStop(0, '#4f46e5');
      gradient.addColorStop(1, '#38bdf8');

      ctx.fillStyle = gradient;
      ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
      x += barWidth;
    }

    // Auto-detect speech and trailing silence to automatically stop recording
    const avgVolume = sum / bufferLength;
    if (avgVolume > 14) {
      userHasSpoken = true;
      silenceStartTime = null;
    } else if (userHasSpoken) {
      if (!silenceStartTime) {
        silenceStartTime = performance.now();
      } else if (performance.now() - silenceStartTime > 700) {
        // Automatically stop recording after 700ms silence
        stopRecording();
      }
    }
  }

  draw();
}

function stopVisualizer() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  initCanvas();
}

// ==========================================
// 6. TEXT CONVERSATION & CHAT FEED
// ==========================================

async function handleTextSubmit(e) {
  e.preventDefault();
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  appendMessage('user', text);
  setOrbState('thinking', 'Nova is thinking...', 'Generating contextual grounded response');

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });

    if (!resp.ok) throw new Error("Server error");
    const data = await resp.json();

    appendMessage('nova', data.reply);
    speakAloud(data.reply);
  } catch (err) {
    console.error(err);
    appendMessage('nova', "Sorry, I encountered an issue processing your request.");
    setOrbState('idle', 'Standing by...', 'Say <span class="text-sky-300 font-semibold">"Hey Nova"</span> or tap the orb');
  }
}

function appendMessage(sender, text) {
  const container = document.getElementById('messages-container');
  const msgEl = document.createElement('div');
  msgEl.className = "flex items-start gap-3 transition-all animate-fade-in";

  if (sender === 'user') {
    msgEl.innerHTML = `
      <div class="ml-auto flex items-start gap-3 justify-end max-w-lg">
        <div class="bg-indigo-600 text-white rounded-2xl rounded-tr-none p-4 text-sm leading-relaxed shadow-md">
          ${escapeHtml(text)}
        </div>
        <div class="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0 text-slate-300 font-semibold text-xs">
          You
        </div>
      </div>
    `;
  } else {
    msgEl.innerHTML = `
      <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-600 to-sky-400 flex items-center justify-center shrink-0 shadow-md shadow-indigo-600/30">
        <i data-lucide="sparkles" class="w-4 h-4 text-white"></i>
      </div>
      <div class="bg-slate-900/90 border border-slate-800 rounded-2xl rounded-tl-none p-4 max-w-lg text-sm leading-relaxed text-slate-200 shadow-sm">
        ${escapeHtml(text)}
      </div>
    `;
  }

  container.appendChild(msgEl);
  container.scrollTop = container.scrollHeight;
  lucide.createIcons();
}

function clearChatFeed() {
  const container = document.getElementById('messages-container');
  container.innerHTML = `
    <div class="flex items-start gap-3">
      <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-600 to-sky-400 flex items-center justify-center shrink-0">
        <i data-lucide="sparkles" class="w-4 h-4 text-white"></i>
      </div>
      <div class="bg-slate-900/90 border border-slate-800 rounded-2xl rounded-tl-none p-4 max-w-lg text-sm leading-relaxed text-slate-200">
        Chat feed cleared. Ready for your next request!
      </div>
    </div>
  `;
  lucide.createIcons();
}

// ==========================================
// 7. SPEECH SYNTHESIS (SPOKEN AUDIO)
// ==========================================

function speakAloud(text) {
  if (isAudioMuted || !('speechSynthesis' in window)) {
    setOrbState('idle', 'Standing by...', 'Say <span class="text-sky-300 font-semibold">"Hey Nova"</span> or tap the orb');
    return;
  }

  // Cancel prior utterances
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.15;
  utterance.pitch = 1.0;

  // Try to pick a natural English voice
  const voices = window.speechSynthesis.getVoices();
  const naturalVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Siri') || v.name.includes('Samantha')));
  if (naturalVoice) {
    utterance.voice = naturalVoice;
  }

  utterance.onstart = () => {
    setOrbState('speaking', 'Nova is speaking...', 'Audio reply in progress');
  };

  utterance.onend = () => {
    setOrbState('idle', 'Standing by...', 'Say <span class="text-sky-300 font-semibold">"Hey Nova"</span> or tap the orb');
  };

  utterance.onerror = () => {
    setOrbState('idle', 'Standing by...', 'Say <span class="text-sky-300 font-semibold">"Hey Nova"</span> or tap the orb');
  };

  window.speechSynthesis.speak(utterance);
}

function toggleAudioMute() {
  isAudioMuted = !isAudioMuted;
  const icon = document.getElementById('mute-icon');
  const label = document.getElementById('mute-label');

  if (isAudioMuted) {
    window.speechSynthesis.cancel();
    label.innerText = "Muted";
    icon.className = "w-3.5 h-3.5 text-rose-400";
    setOrbState('idle');
  } else {
    label.innerText = "Audio On";
    icon.className = "w-3.5 h-3.5 text-sky-400";
  }
}

// ==========================================
// 8. MEMORY VAULT MANAGEMENT
// ==========================================

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    const greeting = document.getElementById('user-greeting');
    if (data.user_name) {
      greeting.innerText = `Speaking with ${data.user_name}`;
      document.getElementById('profile-name').innerText = data.user_name;
    }

    const privToggle = document.getElementById('privacy-mode-toggle');
    if (privToggle) {
      privToggle.checked = data.security.privacy_mode;
    }

    const encBadge = document.getElementById('enc-badge');
    if (encBadge) {
      encBadge.innerText = data.security.encryption ? "AES-256 Encrypted" : "Standard JSON";
      encBadge.className = data.security.encryption 
        ? "text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 font-medium"
        : "text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700 font-medium";
    }
  } catch (e) {
    console.error("Status fetch failed", e);
  }
}

async function loadMemoryVault() {
  try {
    const res = await fetch('/api/memory');
    const data = await res.json();

    // Profile card
    const profile = data.profile;
    document.getElementById('profile-name').innerText = profile.name || "Unspecified";
    document.getElementById('profile-style').innerText = profile.communication_style || "Concise and friendly";

    // Preferences
    const prefsContainer = document.getElementById('profile-preferences');
    if (profile.preferences && profile.preferences.length > 0) {
      prefsContainer.innerHTML = profile.preferences.map(p => `
        <span class="px-2.5 py-1 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs">
          ${escapeHtml(p)}
        </span>
      `).join('');
    } else {
      prefsContainer.innerHTML = `<span class="text-xs text-slate-500 italic">No preferences stored yet</span>`;
    }

    // Projects
    const projectsContainer = document.getElementById('profile-projects');
    if (profile.projects && profile.projects.length > 0) {
      projectsContainer.innerHTML = profile.projects.map(pr => `
        <span class="px-2.5 py-1 rounded-lg bg-purple-500/10 border border-purple-500/20 text-purple-300 text-xs">
          ${escapeHtml(pr)}
        </span>
      `).join('');
    } else {
      projectsContainer.innerHTML = `<span class="text-xs text-slate-500 italic">No active projects logged yet</span>`;
    }

    // LTM Table
    const ltmTable = document.getElementById('ltm-table-body');
    const ltmList = data.long_term_memories || [];
    document.getElementById('ltm-badge-count').innerText = `${ltmList.length} Items`;

    if (ltmList.length === 0) {
      ltmTable.innerHTML = `<tr><td colspan="5" class="py-8 text-center text-slate-500 italic">No long-term memories stored yet.</td></tr>`;
    } else {
      ltmTable.innerHTML = ltmList.map(m => {
        const catBadgeColors = {
          'personal': 'bg-sky-500/10 text-sky-300 border-sky-500/20',
          'preference': 'bg-rose-500/10 text-rose-300 border-rose-500/20',
          'work_project': 'bg-purple-500/10 text-purple-300 border-purple-500/20',
          'technical': 'bg-indigo-500/10 text-indigo-300 border-indigo-500/20'
        };
        const colorClass = catBadgeColors[m.category] || 'bg-slate-800 text-slate-300 border-slate-700';
        const dateStr = m.created_at ? new Date(m.created_at).toLocaleDateString() : 'N/A';

        return `
          <tr class="hover:bg-slate-900/40 transition">
            <td class="py-3 px-4">
              <span class="px-2 py-0.5 rounded text-[10px] font-medium border ${colorClass}">${escapeHtml(m.category || 'general')}</span>
            </td>
            <td class="py-3 px-4 font-medium text-slate-200">${escapeHtml(m.content)}</td>
            <td class="py-3 px-4 text-slate-400 text-[11px]">${(m.tags || []).join(', ')}</td>
            <td class="py-3 px-4 text-slate-500 text-[11px]">${dateStr}</td>
            <td class="py-3 px-4 text-right">
              <button onclick="forgetMemoryItem('${escapeHtml(m.id || m.content)}')" class="p-1.5 rounded-lg hover:bg-rose-500/10 text-slate-400 hover:text-rose-400 transition" title="Forget this fact">
                <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
              </button>
            </td>
          </tr>
        `;
      }).join('');
    }

    // Episodic history
    const epContainer = document.getElementById('episodic-history-list');
    const episodes = data.episodic_history || [];
    if (episodes.length === 0) {
      epContainer.innerHTML = `<p class="text-xs text-slate-500 italic">No session episodes summarized yet.</p>`;
    } else {
      epContainer.innerHTML = episodes.map(ep => {
        const timeStr = ep.timestamp ? new Date(ep.timestamp).toLocaleString() : '';
        return `
          <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3.5 text-xs">
            <div class="text-[10px] text-slate-500 mb-1 flex items-center gap-1.5">
              <i data-lucide="clock" class="w-3 h-3 text-amber-400"></i> ${timeStr} (${ep.turns || 0} turns)
            </div>
            <p class="text-slate-300 leading-relaxed">${escapeHtml(ep.summary)}</p>
          </div>
        `;
      }).join('');
    }

    lucide.createIcons();
  } catch (e) {
    console.error("Memory load failed", e);
  }
}

async function forgetMemoryItem(queryOrId) {
  if (!confirm(`Are you sure you want Nova to forget this memory?`)) return;

  try {
    const res = await fetch('/api/memory/forget', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: queryOrId })
    });
    if (res.ok) {
      loadMemoryVault();
    }
  } catch (e) {
    console.error("Forget failed", e);
  }
}

async function confirmResetMemory() {
  if (!confirm("⚠️ WARNING: This will permanently delete ALL stored memories, profile facts, and conversation episodes. Do you want to proceed?")) return;

  try {
    const res = await fetch('/api/memory/reset', { method: 'POST' });
    if (res.ok) {
      alert("All memory has been cleared.");
      loadMemoryVault();
      loadStatus();
    }
  } catch (e) {
    console.error("Reset failed", e);
  }
}

function exportMemoryReport() {
  window.open('/api/memory/export', '_blank');
}

// Add memory modal
function openAddMemoryModal() {
  document.getElementById('add-memory-modal').classList.remove('hidden');
}

function closeAddMemoryModal() {
  document.getElementById('add-memory-modal').classList.add('hidden');
}

async function handleAddMemorySubmit(e) {
  e.preventDefault();
  const category = document.getElementById('new-mem-cat').value;
  const content = document.getElementById('new-mem-content').value.trim();
  if (!content) return;

  try {
    const res = await fetch('/api/memory/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, category })
    });
    if (res.ok) {
      document.getElementById('new-mem-content').value = '';
      closeAddMemoryModal();
      loadMemoryVault();
    }
  } catch (e) {
    console.error("Add memory failed", e);
  }
}

async function togglePrivacyModeUI(active) {
  try {
    await fetch('/api/security/privacy-mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ active })
    });
  } catch (e) {
    console.error("Privacy mode toggle failed", e);
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}
