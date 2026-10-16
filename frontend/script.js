"use strict";

const MAX_BYTES = 15 * 1024 * 1024;
const TIMEOUT_MS = 180000;

const MODES = {
  "analyze": {
    accept: "image/*,audio/*",
    hint: "Images are checked for deepfakes. Audio is checked for voice spoofing.",
  },
  "analyze-image": {
    accept: "image/*",
    hint: "Deepfake verdict plus what the image appears to show.",
  },
  "classify-image": {
    accept: "image/*",
    hint: "General image recognition (ImageNet labels). No deepfake verdict.",
  },
};

const $ = (id) => document.getElementById(id);
let mode = "analyze";
let lastResult = "";

// ---------- Helpers ----------
function toast(message) {
  const el = document.createElement("div");
  el.className = "toast-msg";
  el.textContent = message;
  $("toast-area").appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

function show(id, visible) {
  $(id).classList.toggle("d-none", !visible);
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = busy ? "Analyzing…" : label;
}

async function fetchWithTimeout(url, opts = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    return await fetch(url, { ...opts, signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function parseResponse(res) {
  const text = await res.text();
  try {
    return { ok: res.ok, data: JSON.parse(text) };
  } catch {
    return { ok: false, data: { error: `Server returned HTTP ${res.status}` } };
  }
}

// ---------- Mode tabs ----------
document.querySelectorAll("[data-mode]").forEach((tab) => {
  tab.addEventListener("click", () => {
    mode = tab.dataset.mode;
    document.querySelectorAll("[data-mode]").forEach((t) => {
      const active = t === tab;
      t.classList.toggle("active", active);
      t.setAttribute("aria-selected", String(active));
    });
    $("file-input").accept = MODES[mode].accept;
    $("mode-hint").textContent = MODES[mode].hint;
  });
});

// ---------- File selection ----------
function validFile(file) {
  if (!file) return false;
  const imageOnly = mode !== "analyze";
  const isImage = file.type.startsWith("image/");
  const isAudio = file.type.startsWith("audio/");
  if (imageOnly ? !isImage : !(isImage || isAudio)) {
    toast(imageOnly ? "Please choose an image file." : "Please choose an image or audio file.");
    return false;
  }
  if (file.size > MAX_BYTES) {
    toast("File is larger than 15 MB.");
    return false;
  }
  return true;
}

function onFileChosen() {
  const file = $("file-input").files[0];
  if (!validFile(file)) {
    $("file-input").value = "";
    $("drop-label").textContent = "Drop a file here or click to choose";
    return;
  }
  $("drop-label").textContent = file.name;
}

const zone = $("drop-zone");
zone.addEventListener("click", () => $("file-input").click());
zone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    $("file-input").click();
  }
});
zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
zone.addEventListener("drop", (e) => {
  e.preventDefault();
  zone.classList.remove("drag");
  if (e.dataTransfer.files?.length) {
    $("file-input").files = e.dataTransfer.files;
    onFileChosen();
  }
});
$("file-input").addEventListener("change", onFileChosen);

// ---------- Results ----------
const VERDICTS = {
  likely_fake: { title: "Likely fake", icon: "fa-triangle-exclamation", cls: "fake" },
  likely_real: { title: "Likely authentic", icon: "fa-circle-check", cls: "real" },
  inconclusive: { title: "Inconclusive", icon: "fa-circle-question", cls: "unknown" },
};

function renderLabels(title, preds) {
  const list = $("labels-list");
  list.replaceChildren();
  $("labels-title").textContent = title;
  (preds || []).forEach((p) => {
    const li = document.createElement("li");
    li.className = "list-group-item d-flex justify-content-between";
    const label = document.createElement("span");
    label.textContent = p.label;
    const score = document.createElement("span");
    score.className = "text-muted";
    score.textContent = `${(Number(p.score) * 100).toFixed(1)}%`;
    li.append(label, score);
    list.appendChild(li);
  });
  show("labels-block", (preds || []).length > 0);
}

function renderResult(ok, data) {
  lastResult = JSON.stringify(data, null, 2);
  $("raw-output").textContent = lastResult;
  show("verdict-card", false);
  show("labels-block", false);
  show("error-block", false);

  if (!ok || data.error) {
    $("error-block").textContent = data.error || "Something went wrong.";
    show("error-block", true);
  } else if (data.verdict) {
    const v = VERDICTS[data.verdict] || VERDICTS.inconclusive;
    const card = $("verdict-card");
    card.className = `verdict mb-4 ${v.cls}`;
    $("verdict-icon").className = `fas ${v.icon}`;
    $("verdict-title").textContent = v.title;
    $("verdict-summary").textContent = data.summary || "";
    const pct = Math.max(0, Math.min(100, Number(data.confidence) * 100 || 0));
    $("confidence-text").textContent = `${pct.toFixed(1)}%`;
    $("confidence-fill").style.width = `${pct}%`;
    const contents = data.contents?.predictions;
    renderLabels(contents ? "What the image shows" : "Model scores", contents || data.predictions);
  } else {
    renderLabels("What the image shows", data.predictions);
  }

  show("upload-view", false);
  show("results-view", true);
}

async function submit(body, headers, button, label) {
  setBusy(button, true, label);
  try {
    const res = await fetchWithTimeout(`/${mode}`, { method: "POST", body, headers });
    const { ok, data } = await parseResponse(res);
    renderResult(ok, data);
  } catch (e) {
    renderResult(false, { error: e.name === "AbortError" ? "Request timed out." : "Could not reach the server." });
  } finally {
    setBusy(button, false, label);
  }
}

$("btn-upload").addEventListener("click", () => {
  const file = $("file-input").files[0];
  if (!file) { toast("Choose a file first."); return; }
  if (!validFile(file)) return;
  const fd = new FormData();
  fd.append("file", file, file.name);
  submit(fd, undefined, $("btn-upload"), "Analyze file");
});

$("url-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const url = $("url-input").value.trim();
  if (!/^https?:\/\//i.test(url)) { toast("Enter an http(s) URL."); return; }
  submit(JSON.stringify({ url }), { "Content-Type": "application/json" }, $("btn-url"), "Analyze URL");
});

$("btn-try-again").addEventListener("click", () => {
  show("results-view", false);
  show("upload-view", true);
});

// ---------- Footer ----------
async function checkHealth() {
  const badge = $("health-status");
  badge.textContent = "checking…";
  badge.className = "badge bg-warning";
  try {
    const res = await fetchWithTimeout("/health");
    const data = await res.json();
    badge.textContent = data.ok ? `online (${data.inference_mode})` : "error";
    badge.className = `badge ${data.ok ? "bg-success" : "bg-danger"}`;
  } catch {
    badge.textContent = "offline";
    badge.className = "badge bg-danger";
  }
}
$("btn-health").addEventListener("click", checkHealth);

$("btn-copy").addEventListener("click", async () => {
  if (!lastResult) { toast("Nothing to copy yet."); return; }
  try {
    await navigator.clipboard.writeText(lastResult);
    toast("Copied.");
  } catch {
    toast("Could not copy.");
  }
});

checkHealth();
