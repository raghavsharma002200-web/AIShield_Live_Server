// ============================================================
// AIShield dashboard client logic
// ============================================================

// --- Tab switching ------------------------------------------------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// --- Train Now button (lets the app be trained with no Shell access) ---
const trainBtn = document.getElementById("train-now-btn");
const trainBanner = document.getElementById("train-banner");

if (trainBtn) {
  trainBtn.addEventListener("click", async () => {
    trainBtn.disabled = true;
    trainBtn.textContent = "Training…";
    trainBanner.classList.remove("hidden", "error", "success");
    trainBanner.textContent = "Training the hybrid ensemble on a starter synthetic dataset — this takes roughly 15-30 seconds…";

    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n_samples: 8000 }),
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        trainBanner.classList.add("error");
        trainBanner.textContent = data.error || "Training failed.";
        trainBtn.disabled = false;
        trainBtn.textContent = "⚡ Train Now";
        return;
      }

      trainBanner.classList.add("success");
      const m = data.metrics;
      trainBanner.textContent =
        `Model trained! Accuracy ${Math.round(m.accuracy * 100)}%, ` +
        `F1 ${Math.round(m.f1 * 100)}% on held-out data. Reloading…`;
      setTimeout(() => window.location.reload(), 1200);
    } catch (err) {
      console.error(err);
      trainBanner.classList.add("error");
      trainBanner.textContent = "Network error while training.";
      trainBtn.disabled = false;
      trainBtn.textContent = "⚡ Train Now";
    }
  });
}

// --- Single record: sample loading ---------------------------------
document.querySelectorAll(".chip[data-sample]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const kind = btn.dataset.sample;
    try {
      const res = await fetch(`/sample/${kind}`);
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      Object.entries(data).forEach(([field, value]) => {
        const el = document.getElementById(field);
        if (el) el.value = value;
      });
    } catch (err) {
      console.error(err);
      alert("Could not load sample record.");
    }
  });
});

// --- Single record: submit form -------------------------------------
const predictForm = document.getElementById("predict-form");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const resultError = document.getElementById("result-error");

function pct(x) {
  return `${Math.round(x * 100)}%`;
}

function renderResult(data) {
  resultEmpty.classList.add("hidden");
  resultError.classList.add("hidden");
  resultContent.classList.remove("hidden");

  const badge = document.getElementById("verdict-badge");
  const isAttack = data.verdict === "ATTACK";
  badge.textContent = isAttack ? "🚨 ATTACK DETECTED" : "✅ NORMAL TRAFFIC";
  badge.className = "verdict-badge " + (isAttack ? "attack" : "normal");

  document.getElementById("final-score-fill").style.width = pct(data.final_score);
  document.getElementById("final-score-text").textContent = pct(data.final_score);
  document.getElementById("supervised-score-fill").style.width = pct(data.supervised_score);
  document.getElementById("supervised-score-text").textContent = pct(data.supervised_score);
  document.getElementById("anomaly-score-fill").style.width = pct(data.anomaly_score);
  document.getElementById("anomaly-score-text").textContent = pct(data.anomaly_score);

  const modelNames = ["random_forest", "xgboost", "svm", "neural_network"];
  const tbody = document.getElementById("votes-body");
  tbody.innerHTML = "";
  modelNames.forEach((name) => {
    const score = data[`${name}_score`];
    const vote = data[`${name}_vote`];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${name.replace("_", " ")}</td>
      <td>${pct(score)}</td>
      <td class="${vote === "ATTACK" ? "attack" : "normal"}">${vote}</td>
    `;
    tbody.appendChild(tr);
  });
}

if (predictForm) {
  predictForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(predictForm);
    const payload = {};
    formData.forEach((value, key) => { payload[key] = value; });

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        resultEmpty.classList.add("hidden");
        resultContent.classList.add("hidden");
        resultError.classList.remove("hidden");
        resultError.textContent = data.error || "Prediction failed.";
        return;
      }
      renderResult(data);
    } catch (err) {
      console.error(err);
      resultError.classList.remove("hidden");
      resultError.textContent = "Network error while contacting AIShield.";
    }
  });
}

// --- Batch CSV upload -------------------------------------------------
const csvForm = document.getElementById("csv-form");
const batchSummary = document.getElementById("batch-summary");
const batchTableBody = document.querySelector("#batch-table tbody");
const batchTruncatedNote = document.getElementById("batch-truncated-note");
const batchError = document.getElementById("batch-error");

if (csvForm) {
  csvForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("csv-file");
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    batchError.classList.add("hidden");
    batchSummary.classList.add("hidden");
    batchTruncatedNote.classList.add("hidden");
    batchTableBody.innerHTML = "";

    try {
      const res = await fetch("/predict_csv", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok || data.error) {
        batchError.classList.remove("hidden");
        batchError.textContent = data.error || "Batch prediction failed.";
        return;
      }

      batchSummary.classList.remove("hidden");
      batchSummary.innerHTML = `
        <span>Total: <strong>${data.n_rows}</strong></span>
        <span class="n-attack">🚨 Attack: ${data.n_attack}</span>
        <span class="n-normal">✅ Normal: ${data.n_normal}</span>
      `;

      data.rows.forEach((row, i) => {
        const tr = document.createElement("tr");
        const isAttack = row.verdict === "ATTACK";
        tr.innerHTML = `
          <td>${i + 1}</td>
          <td class="${isAttack ? "attack" : "normal"}">${row.verdict}</td>
          <td>${pct(row.final_score)}</td>
          <td>${pct(row.supervised_score)}</td>
          <td>${pct(row.anomaly_score)}</td>
        `;
        batchTableBody.appendChild(tr);
      });

      if (data.truncated) {
        batchTruncatedNote.classList.remove("hidden");
        batchTruncatedNote.textContent =
          `Showing first 200 of ${data.n_rows} rows.`;
      }
    } catch (err) {
      console.error(err);
      batchError.classList.remove("hidden");
      batchError.textContent = "Network error while uploading CSV.";
    }
  });
}


// --- Live Server monitor ---------------------------------------------
const liveTotal = document.getElementById("live-total");
const liveNormal = document.getElementById("live-normal");
const liveSuspicious = document.getElementById("live-suspicious");
const liveBadge = document.getElementById("live-status-badge");
const liveToggle = document.getElementById("live-toggle-btn");
const liveClear = document.getElementById("live-clear-btn");
const liveBody = document.querySelector("#live-table tbody");

let liveEnabled = true;
let liveTimer = null;

function renderLive(data) {
  if (!liveTotal) return;

  liveEnabled = !!data.monitoring;
  liveTotal.textContent = data.total;
  liveNormal.textContent = data.normal;
  liveSuspicious.textContent = data.suspicious;

  liveBadge.textContent = liveEnabled ? "● Monitoring" : "● Paused";
  liveBadge.className = "live-status " + (liveEnabled ? "on" : "off");
  liveToggle.textContent = liveEnabled ? "Pause" : "Resume";

  liveBody.innerHTML = "";
  data.events.forEach((e) => {
    const tr = document.createElement("tr");
    const suspicious = e.verdict === "SUSPICIOUS";
    tr.innerHTML = `
      <td>${e.time}</td>
      <td><code>${e.method}</code></td>
      <td><code>${e.path}</code></td>
      <td>${e.status}</td>
      <td>${e.response_ms} ms</td>
      <td>${e.risk}%</td>
      <td class="${suspicious ? "attack" : "normal"}">${e.verdict}</td>
      <td>${e.reason}</td>
    `;
    liveBody.appendChild(tr);
  });
}

async function refreshLive() {
  if (!liveBody) return;
  try {
    const res = await fetch("/api/live/status", { cache: "no-store" });
    const data = await res.json();
    if (res.ok) renderLive(data);
  } catch (err) {
    console.error("Live monitor:", err);
  }
}

if (liveToggle) {
  liveToggle.addEventListener("click", async () => {
    const enabled = !liveEnabled;
    try {
      const res = await fetch("/api/live/toggle", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({enabled}),
      });
      const data = await res.json();
      liveEnabled = !!data.monitoring;
      refreshLive();
    } catch (err) {
      console.error(err);
    }
  });
}

if (liveClear) {
  liveClear.addEventListener("click", async () => {
    try {
      await fetch("/api/live/clear", {method: "POST"});
      refreshLive();
    } catch (err) {
      console.error(err);
    }
  });
}

if (document.getElementById("tab-live")) {
  refreshLive();
  liveTimer = setInterval(refreshLive, 1500);
}
