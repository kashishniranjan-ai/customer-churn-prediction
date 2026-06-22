/* ═══════════════════════════════════════════════════════════════════════════
   Customer Churn Prediction System – script.js
═══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── Feather icons ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  feather.replace();
  loadSidebarData();
  setupForm();
  setupDarkMode();
  setupReset();
});

// ── Dark mode toggle ──────────────────────────────────────────────────────
function setupDarkMode() {
  const btn  = document.getElementById("darkToggle");
  const icon = document.getElementById("darkIcon");
  const root = document.documentElement;

  const saved = localStorage.getItem("theme") || "light";
  root.setAttribute("data-theme", saved);
  icon.setAttribute("data-feather", saved === "dark" ? "sun" : "moon");
  feather.replace();

  btn.addEventListener("click", () => {
    const current = root.getAttribute("data-theme");
    const next    = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    icon.setAttribute("data-feather", next === "dark" ? "sun" : "moon");
    feather.replace();
    if (featureChartInstance) updateChartTheme();
  });
}

// ── Form validation & submission ──────────────────────────────────────────
function setupForm() {
  const form      = document.getElementById("churnForm");
  const btn       = document.getElementById("predictBtn");
  const errBox    = document.getElementById("formError");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Validate numerics
    const tenure   = parseFloat(form.tenure.value);
    const monthly  = parseFloat(form.MonthlyCharges.value);
    const total    = parseFloat(form.TotalCharges.value);

    if (isNaN(tenure) || tenure < 0 || tenure > 120) {
      showError("Tenure must be between 0 and 120 months."); return;
    }
    if (isNaN(monthly) || monthly < 0) {
      showError("Monthly Charges must be a non-negative number."); return;
    }
    if (isNaN(total) || total < 0) {
      showError("Total Charges must be a non-negative number."); return;
    }
    hideError();

    // Show loading
    showLoading();
    btn.disabled = true;

    try {
      const formData = new FormData(form);
      const res  = await fetch("/predict", { method: "POST", body: formData });
      const data = await res.json();

      if (data.error) throw new Error(data.error);

      showResult(data);
    } catch (err) {
      showError("Prediction failed: " + err.message);
      showIdle();
    } finally {
      btn.disabled = false;
    }
  });
}

function showError(msg) {
  const box = document.getElementById("formError");
  box.textContent = "⚠ " + msg;
  box.style.display = "block";
}
function hideError() {
  document.getElementById("formError").style.display = "none";
}

// ── Reset ─────────────────────────────────────────────────────────────────
function setupReset() {
  document.getElementById("resetBtn").addEventListener("click", () => {
    document.getElementById("churnForm").reset();
    hideError();
    showIdle();
  });
}

// ── State management ──────────────────────────────────────────────────────
function showIdle() {
  document.getElementById("resultIdle").style.display    = "";
  document.getElementById("resultLoading").style.display = "none";
  document.getElementById("resultOutput").style.display  = "none";
}
function showLoading() {
  document.getElementById("resultIdle").style.display    = "none";
  document.getElementById("resultLoading").style.display = "";
  document.getElementById("resultOutput").style.display  = "none";
}
function showResultPanel() {
  document.getElementById("resultIdle").style.display    = "none";
  document.getElementById("resultLoading").style.display = "none";
  document.getElementById("resultOutput").style.display  = "";
}

// ── Render prediction result ──────────────────────────────────────────────
function showResult(data) {
  showResultPanel();

  const pct    = data.probability;
  const status = data.status;        // "success" or "danger"

  // Badge
  const badge = document.getElementById("resultBadge");
  badge.className = "result-badge " + status;
  document.getElementById("resultLabel").textContent = data.label;
  const iconEl = document.getElementById("resultIcon");
  iconEl.setAttribute("data-feather", status === "danger" ? "alert-triangle" : "check-circle");
  feather.replace();

  // Gauge animation
  animateGauge(pct);

  // Risk bar
  animateRiskBar(pct);

  // Probability text
  document.getElementById("probValue").textContent = pct.toFixed(1) + "%";

  // Hint text
  const hints = {
    danger : pct > 70
      ? "High churn risk. Proactive retention actions strongly recommended."
      : "Elevated churn risk. Consider targeted offers or follow-ups.",
    success: pct < 20
      ? "Low churn risk. Customer appears satisfied — maintain engagement."
      : "Moderate retention probability. Monitor and nurture this account.",
  };
  document.getElementById("resultHint").textContent =
    status === "danger" ? hints.danger : hints.success;
}

// ── Gauge ─────────────────────────────────────────────────────────────────
function animateGauge(pct) {
  const path = document.getElementById("gaugeFill");
  const text = document.getElementById("gaugeText");
  const totalLen = 283;           // arc length for the semicircle at r=90

  // pct → dashoffset: 0% = full offset (empty), 100% = no offset (full)
  const offset = totalLen - (pct / 100) * totalLen;

  // Color: green → amber → red
  const color = pct < 40 ? "#059669" : pct < 65 ? "#d97706" : "#dc2626";
  path.style.stroke = color;

  // Kick the transition
  requestAnimationFrame(() => {
    path.style.strokeDashoffset = offset;
  });

  // Counter animation for text
  let start = null;
  const duration = 1200;
  function step(ts) {
    if (!start) start = ts;
    const progress = Math.min((ts - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    text.textContent = Math.round(ease * pct) + "%";
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Risk bar ──────────────────────────────────────────────────────────────
function animateRiskBar(pct) {
  const thumb = document.getElementById("riskThumb");
  const fill  = document.getElementById("riskFill");

  // Delay to allow display:block to register before transition
  setTimeout(() => {
    thumb.style.left = pct + "%";
    fill.style.width = pct + "%";
  }, 50);
}

// ── Feature importance chart ──────────────────────────────────────────────
let featureChartInstance = null;

async function loadSidebarData() {
  await Promise.all([loadFeatureChart(), loadConfusionMatrix()]);
}

async function loadFeatureChart() {
  try {
    const res  = await fetch("/feature-importances");
    const data = await res.json();

    const labels = Object.keys(data).map(k => k.replace(/_/g, " "));
    const values = Object.values(data).map(v => +(v * 100).toFixed(2));

    const ctx = document.getElementById("featureChart").getContext("2d");

    featureChartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Importance (%)",
          data: values,
          backgroundColor: values.map((_, i) =>
            `hsla(${245 - i * 8}, 80%, 60%, 0.82)`),
          borderRadius: 5,
          borderSkipped: false,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.parsed.x.toFixed(2)}%`,
            },
          },
        },
        scales: {
          x: {
            grid:  { color: getComputedStyle(document.documentElement).getPropertyValue("--border").trim() },
            ticks: { color: getComputedStyle(document.documentElement).getPropertyValue("--text-secondary").trim(), font: { size: 11 } },
          },
          y: {
            grid:  { display: false },
            ticks: { color: getComputedStyle(document.documentElement).getPropertyValue("--text-primary").trim(), font: { size: 11 } },
          },
        },
      },
    });
    // Set canvas height proportional to items
    document.getElementById("featureChart").style.height = (labels.length * 22 + 30) + "px";
  } catch (err) {
    console.warn("Could not load feature importances", err);
  }
}

function updateChartTheme() {
  if (!featureChartInstance) return;
  const border = getComputedStyle(document.documentElement).getPropertyValue("--border").trim();
  const textSec = getComputedStyle(document.documentElement).getPropertyValue("--text-secondary").trim();
  const textPri = getComputedStyle(document.documentElement).getPropertyValue("--text-primary").trim();
  featureChartInstance.options.scales.x.grid.color  = border;
  featureChartInstance.options.scales.x.ticks.color = textSec;
  featureChartInstance.options.scales.y.ticks.color = textPri;
  featureChartInstance.update();
}

// ── Confusion matrix ──────────────────────────────────────────────────────
async function loadConfusionMatrix() {
  try {
    const res  = await fetch("/confusion-matrix");
    const data = await res.json();

    document.querySelector("#cmTN .cm-val").textContent = data.tn;
    document.querySelector("#cmFP .cm-val").textContent = data.fp;
    document.querySelector("#cmFN .cm-val").textContent = data.fn;
    document.querySelector("#cmTP .cm-val").textContent = data.tp;
    document.getElementById("cmAccuracy").textContent =
      `Model Test Accuracy: ${(data.accuracy * 100).toFixed(2)}%`;
  } catch (err) {
    console.warn("Could not load confusion matrix", err);
  }
}
