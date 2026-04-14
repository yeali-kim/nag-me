let monitoring = false;
let intervalId = null;
let voices = [];
let nagCounter = 1;
let activeStream = null;
let currentAudio = null;
let isNagging = false;

// set up drawing Media Pipe detection results
const canvasEl = document.getElementById("output_canvas");
const ctx = canvasEl.getContext("2d");
let handLandmarks = null;
let faceLandmarks = null;

function paintFrame() {
  if (!document.getElementById("liveViewToggle").checked) return;
  const video = document.getElementById("webcam");
  canvasEl.width = video.videoWidth;
  canvasEl.height = video.videoHeight;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  if (handLandmarks) {
    drawConnectors(ctx, handLandmarks, HAND_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 2,
    });
    drawLandmarks(ctx, handLandmarks, {
      color: "#FF0000",
      lineWidth: 1,
      radius: 3,
    });
  }
  if (faceLandmarks) {
    drawConnectors(ctx, faceLandmarks, FACEMESH_TESSELATION, {
      color: "#C0C0C070",
      lineWidth: 1,
    });
    drawConnectors(ctx, faceLandmarks, FACEMESH_LIPS, {
      color: "#E0E0E0",
      lineWidth: 2,
    });
  }
}

const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});
hands.onResults((results) => {
  handLandmarks = results.multiHandLandmarks?.[0];
  paintFrame();
});

const faceMesh = new FaceMesh({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
});
faceMesh.setOptions({
  maxNumFaces: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});
faceMesh.onResults((results) => {
  faceLandmarks = results.multiFaceLandmarks?.[0];
  paintFrame();
});

// camera use
async function startCamera() {
  if (!activeStream) {
    activeStream = await navigator.mediaDevices.getUserMedia({
      video: true,
    });
    document.getElementById("webcam").srcObject = activeStream;
  }
}

async function stopCamera() {
  if (activeStream) {
    activeStream.getTracks().forEach((t) => t.stop());
    activeStream = null;
  }
}

async function toggleCameraView() {
  const container = document.getElementById("cameraContainer");
  if (document.getElementById("liveViewToggle").checked) {
    await startCamera();
    container.style.display = "block";
    const video = document.getElementById("webcam");
    const camera = new Camera(video, {
      onFrame: async () => {
        await hands.send({ image: video });
        await faceMesh.send({ image: video });
      },
      width: 640,
      height: 480,
    });
    camera.start();
  } else {
    container.style.display = "none";
    if (!monitoring) stopCamera();
  }
}

// start monitoring!
function startNagMe() {
  const btn = document.getElementById("startBtn");
  const status = document.getElementById("statusText");
  const circle = document.getElementById("mainCircle");

  if (!monitoring) {
    const activeVoices = voices.filter((v) => v.enabled);
    if (activeVoices.length === 0) {
      alert(
        "Please add at least one voice profile before monitoring. Use Record New or Upload."
      );
      return;
    }
    startCamera();
    monitoring = true;
    btn.textContent = "STOP MONITORING";
    status.textContent = "Monitoring Active...";
    circle.classList.add("pulse-active");
    intervalId = setInterval(sendFrame, 500);
  } else {
    monitoring = false;
    btn.textContent = "START MONITORING";
    status.textContent = "System Standby";
    circle.classList.remove("pulse-active");
    clearInterval(intervalId);
  }
}

async function sendFrame() {
  const video = document.getElementById("webcam");
  if (!activeStream || video.readyState < 2) return;

  const canvas = document.createElement("canvas");
  canvas.width = 300;
  canvas.height = 225;
  canvas
    .getContext("2d")
    .drawImage(video, 0, 0, canvas.width, canvas.height);
  const frame = canvas.toDataURL("image/jpeg", 0.5);

  //send to server for prediction
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame }),
    });
    const data = await res.json();
    if (data.prediction === 1) {
      console.log("Nag detected! Playing audio...");
      playNag();
    }
  } catch (err) {
    console.error("Server error:", err);
  }
}

// audio playback
function playNag() {
  if (isNagging) return;

  const activeVoices = voices.filter((v) => v.enabled);
  if (activeVoices.length === 0) return;

  isNagging = true;

  const randomVoice =
    activeVoices[Math.floor(Math.random() * activeVoices.length)];
  currentAudio = new Audio(randomVoice.url);

  currentAudio.onended = () => {
    setTimeout(() => {
      isNagging = false;
    }, 1000);
  };

  currentAudio.play().catch((err) => {
    isNagging = false;
    console.error("Audio play failed:", err);
  });
}

// voice profile management
function renderVoices() {
  const container = document.querySelector(".voices-container");
  const actions = document.querySelector(".actions");

  document.querySelectorAll(".custom-card").forEach((c) => c.remove());
  document.getElementById("emptyState")?.remove();

  const count = voices.length;
  document.getElementById("profileCount").textContent = `${count} ${count === 1 ? "Profile" : "Profiles"
    }`;

  if (voices.length === 0) {
    const empty = document.createElement("p");
    empty.id = "emptyState";
    empty.style.cssText =
      "text-align:center; color:var(--gray-muted); padding: 20px 0;";
    empty.textContent = "No nag profiles yet. Record or upload one!";
    container.insertBefore(empty, actions);
    return;
  }

  voices.forEach((v, index) => {
    const div = document.createElement("div");
    div.className = "voice-card custom-card";
    div.innerHTML = `
      <div style="background: #d1e3ff; padding: 10px; border-radius: 10px;">
        <i class="fa-regular fa-face-kiss-wink-heart"></i>
      </div>
      <div class="voice-info">
        <span class="voice-name">${v.name}</span>
      </div>
      <div style="display: flex; align-items: center; gap: 15px;">
        <i class="fas fa-trash-alt" 
           style="color: #ff4444; cursor: pointer; font-size: 16px;" 
           onclick="deleteVoice(${index})"></i>
        
        <label class="switch">
          <input type="checkbox" ${v.enabled ? "checked" : ""}
            onchange="voices[${index}].enabled = this.checked; saveVoicesToStorage()">
          <span class="slider"></span>
        </label>
      </div>
    `;
    container.insertBefore(div, actions);
  });
}

function saveVoicesToStorage() {
  const promises = voices.map(async (v) => {
    if (v.base64)
      return { name: v.name, enabled: v.enabled, base64: v.base64 };
    const res = await fetch(v.url);
    const blob = await res.blob();
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () =>
        resolve({
          name: v.name,
          enabled: v.enabled,
          base64: reader.result,
        });
      reader.readAsDataURL(blob);
    });
  });
  Promise.all(promises).then((stored) => {
    localStorage.setItem("nagme_voices", JSON.stringify(stored));
  });
}

function deleteVoice(index) {
  if (confirm(`Are you sure you want to delete "${voices[index].name}"?`)) {
    voices.splice(index, 1);
    renderVoices();
    saveVoicesToStorage();
  }
}

function loadVoicesFromStorage() {
  const stored = localStorage.getItem("nagme_voices");
  if (!stored) {
    renderVoices();
    return;
  }
  const parsed = JSON.parse(stored);
  voices = parsed.map((v) => ({
    name: v.name,
    enabled: v.enabled,
    base64: v.base64,
    url: v.base64,
  }));
  renderVoices();
}

// voice recording
let mediaRecorder;
let audioChunks = [];

function openRecordModal() {
  document.getElementById("recordModal").style.display = "flex";
  document.getElementById("voiceNameInput").value = `Unnamed Nag ${nagCounter}`;
  document.getElementById("recordPreview").style.display = "none";
  document.getElementById("saveVoiceBtn").style.display = "none";
  document.getElementById("micBtn").textContent = "Start Recording";
  document.getElementById("micBtn").classList.remove("btn-record-active");
}

function closeModal() {
  document.getElementById("recordModal").style.display = "none";
}

async function startRecordingLogic() {
  const btn = document.getElementById("micBtn");
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const rawBlob = new Blob(audioChunks, { type: "audio/wav" });
      const trimmedBlob = await stripSilence(rawBlob);
      const url = URL.createObjectURL(trimmedBlob);
      const preview = document.getElementById("recordPreview");
      preview.src = url;
      preview.style.display = "block";
      document.getElementById("saveVoiceBtn").style.display = "block";
      document.getElementById("saveVoiceBtn")._blobUrl = url;
    };
    mediaRecorder.start();
    btn.textContent = "Stop Recording";
    btn.classList.add("btn-record-active");
  } else {
    mediaRecorder.stop();
    btn.textContent = "Record Again";
    btn.classList.remove("btn-record-active");
  }
}

function saveVoiceLogic() {
  const name = document.getElementById("voiceNameInput").value;
  const saveBtn = document.getElementById("saveVoiceBtn");
  const url = saveBtn._blobUrl || document.getElementById("recordPreview").src;
  voices.push({ name, url, enabled: true });
  renderVoices();
  saveVoicesToStorage();
  nagCounter++;
  closeModal();
}

function uploadVoice() {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "audio/*";
  input.onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const name = file.name.replace(/\.[^/.]+$/, "");
    voices.push({ name, url, enabled: true });
    renderVoices();
    saveVoicesToStorage();
  };
  input.click();
}

// remove silence from start/end of recording
async function stripSilence(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new AudioContext();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const data = audioBuffer.getChannelData(0);
  const threshold = 0.01;

  let start = 0;
  for (let i = 0; i < data.length; i++) {
    if (Math.abs(data[i]) > threshold) {
      start = i;
      break;
    }
  }
  let end = data.length - 1;
  for (let i = data.length - 1; i >= 0; i--) {
    if (Math.abs(data[i]) > threshold) {
      end = i;
      break;
    }
  }

  const trimmed = audioCtx.createBuffer(
    audioBuffer.numberOfChannels,
    end - start,
    audioBuffer.sampleRate
  );
  for (let c = 0; c < audioBuffer.numberOfChannels; c++) {
    trimmed.copyToChannel(
      audioBuffer.getChannelData(c).slice(start, end),
      c
    );
  }

  const offlineCtx = new OfflineAudioContext(
    trimmed.numberOfChannels,
    trimmed.length,
    trimmed.sampleRate
  );
  const source = offlineCtx.createBufferSource();
  source.buffer = trimmed;
  source.connect(offlineCtx.destination);
  source.start();
  const rendered = await offlineCtx.startRendering();
  return audioBufferToWav(rendered);
}

function audioBufferToWav(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const samples = buffer.getChannelData(0);
  const dataLength = samples.length * bytesPerSample;
  const bufferArray = new ArrayBuffer(44 + dataLength);
  const view = new DataView(bufferArray);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++)
      view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, "data");
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([bufferArray], { type: "audio/wav" });
}



window.toggleCameraView = toggleCameraView;
window.startNagMe = startNagMe;
window.closeModal = closeModal;
window.deleteVoice = deleteVoice;
window.saveVoicesToStorage = saveVoicesToStorage;

function init() {
  document.getElementById("recordBtn").onclick = openRecordModal;
  document.getElementById("uploadBtn").onclick = uploadVoice;
  document.getElementById("micBtn").onclick = startRecordingLogic;
  document.getElementById("saveVoiceBtn").onclick = saveVoiceLogic;

  if (document.getElementById("startBtn")) {
    document.getElementById("startBtn").onclick = startNagMe;
  }
  if (document.getElementById("liveViewToggle")) {
    document.getElementById("liveViewToggle").onchange = toggleCameraView;
  }
  if (document.getElementById("cancelModalBtn")) {
    document.getElementById("cancelModalBtn").onclick = closeModal;
  }

  loadVoicesFromStorage();
}

init();