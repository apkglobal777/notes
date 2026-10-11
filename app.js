const SPEECH_LANG = "en-IN"; // change to "en-US", "hi-IN", etc.

const chatBox = document.getElementById("chat");
const statusEl = document.getElementById("status");
const micBtn = document.getElementById("mic");
const stopBtn = document.getElementById("stop");
const typeForm = document.getElementById("typeForm");
const typedInput = document.getElementById("typed");
const gradeSel = document.getElementById("grade");
const subjectSel = document.getElementById("subject");

let history = [];          // the conversation sent to the tutor
let recorder = null;       // MediaRecorder
let chunks = [];           // pieces of recorded audio
let recording = false;
let busy = false;

function setStatus(text) { statusEl.textContent = text; }

function addBubble(text, who) {
  const div = document.createElement("div");
  div.className = "bubble " + who;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function setBusy(value) {
  busy = value;
  micBtn.disabled = value;
  typeForm.querySelector("button").disabled = value;
}

// ---------- Step 1: record the student's voice ----------
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop()); // turn the mic off
      sendAudio(new Blob(chunks, { type: recorder.mimeType }));
    };
    recorder.start();
    recording = true;
    micBtn.classList.add("recording");
    micBtn.setAttribute("aria-label", "Stop speaking");
    setStatus("Listening... press again when you finish");
    window.speechSynthesis.cancel(); // stop Tara talking while the student speaks
  } catch (err) {
    setStatus("Microphone blocked. Allow microphone access, or type your question.");
  }
}

function stopRecording() {
  recorder.stop();
  recording = false;
  micBtn.classList.remove("recording");
  micBtn.setAttribute("aria-label", "Start speaking");
}

micBtn.addEventListener("click", () => {
  if (busy) return;
  recording ? stopRecording() : startRecording();
});

// ---------- Step 2: voice -> text (Groq Whisper via Flask) ----------
async function sendAudio(blob) {
  setBusy(true);
  setStatus("Understanding your voice...");
  const form = new FormData();
  form.append("audio", blob, "speech.webm");
  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    if (!data.text) throw new Error("I could not hear anything. Try again.");
    await askTutor(data.text);
  } catch (err) {
    setStatus(err.message);
    setBusy(false);
  }
}

// ---------- Step 3: text -> answer (Groq Llama via Flask) ----------
async function askTutor(question) {
  addBubble(question, "student");
  history.push({ role: "user", content: question });
  setBusy(true);
  setStatus("Tara is thinking...");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: history,
        grade: gradeSel.value,
        subject: subjectSel.value,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    history.push({ role: "assistant", content: data.reply });
    addBubble(data.reply, "tutor");
    speak(data.reply);
  } catch (err) {
    setStatus(err.message);
  } finally {
    setBusy(false);
  }
}

// ---------- Step 4: text -> voice (built into the browser) ----------
function speak(text) {
  if (!("speechSynthesis" in window)) { setStatus("Ready"); return; }
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = SPEECH_LANG;
  utter.rate = 0.95;
  utter.onstart = () => setStatus("Tara is speaking...");
  utter.onend = () => setStatus("Ready - press the button to ask more");
  window.speechSynthesis.speak(utter);
}

stopBtn.addEventListener("click", () => {
  window.speechSynthesis.cancel();
  setStatus("Ready");
});

// Typing works too
typeForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = typedInput.value.trim();
  if (!q || busy) return;
  typedInput.value = "";
  askTutor(q);
});
