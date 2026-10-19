/**
 * Interactive Feature Selection Studio — Kaggle 50 Startups
 * Inspired by interactive-ml.com design & UX
 */

// =========================================================================
// 1. DATASETS & MODEL PARAMETERS (Pre-calculated via Scikit-Learn)
// =========================================================================

const ALGORITHM_DATA = {
  sfs: {
    name: "Sequential Forward Selection (SFS)",
    description: "Greedy search that iteratively adds the feature yielding the largest error reduction.",
    optimalK: 2,
    records: [
      {
        k: 1,
        features: ["R&D Spend"],
        rmse: 8274.868018,
        mae: 6772.453280,
        r2: 0.946459,
        modelKey: "R&D Spend"
      },
      {
        k: 2,
        features: ["R&D Spend", "Marketing Spend"],
        rmse: 8198.797191,
        mae: 6886.594588,
        r2: 0.947439,
        modelKey: "Marketing Spend__R&D Spend"
      },
      {
        k: 3,
        features: ["R&D Spend", "Marketing Spend", "State_New York"],
        rmse: 8309.059683,
        mae: 7028.435332,
        r2: 0.946015,
        modelKey: "Marketing Spend__R&D Spend__State_New York"
      },
      {
        k: 4,
        features: ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida"],
        rmse: 8409.916714,
        mae: 7109.161998,
        r2: 0.944697,
        modelKey: "Marketing Spend__R&D Spend__State_Florida__State_New York"
      },
      {
        k: 5,
        features: ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida", "Administration"],
        rmse: 9137.990153,
        mae: 7514.293660,
        r2: 0.934707,
        modelKey: "Administration__Marketing Spend__R&D Spend__State_Florida__State_New York"
      }
    ]
  },
  rfe: {
    name: "Recursive Feature Elimination (RFE)",
    description: "Starts with the complete feature set and iteratively eliminates the least informative predictor.",
    optimalK: 3,
    records: [
      {
        k: 1,
        features: ["R&D Spend"],
        rmse: 8274.868018,
        mae: 6772.453280,
        r2: 0.946459,
        modelKey: "R&D Spend"
      },
      {
        k: 2,
        features: ["R&D Spend", "State_Florida"],
        rmse: 8217.446448,
        mae: 6772.403035,
        r2: 0.947199,
        modelKey: "R&D Spend__State_Florida"
      },
      {
        k: 3,
        features: ["R&D Spend", "Administration", "State_Florida"],
        rmse: 8171.645989,
        mae: 6730.785004,
        r2: 0.947786,
        modelKey: "Administration__R&D Spend__State_Florida"
      },
      {
        k: 4,
        features: ["R&D Spend", "Administration", "State_Florida", "State_New York"],
        rmse: 8178.369215,
        mae: 6751.890337,
        r2: 0.947700,
        modelKey: "Administration__R&D Spend__State_Florida__State_New York"
      },
      {
        k: 5,
        features: ["R&D Spend", "Administration", "Marketing Spend", "State_Florida", "State_New York"],
        rmse: 9137.990153,
        mae: 7514.293660,
        r2: 0.934707,
        modelKey: "Administration__Marketing Spend__R&D Spend__State_Florida__State_New York"
      }
    ]
  },
  exhaustive: {
    name: "Exhaustive Best Subset Selection",
    description: "Evaluates every combination of size k to guarantee discovery of the mathematical global optimum.",
    optimalK: 3,
    records: [
      {
        k: 1,
        features: ["R&D Spend"],
        rmse: 8274.868018,
        mae: 6772.453280,
        r2: 0.946459,
        modelKey: "R&D Spend"
      },
      {
        k: 2,
        features: ["R&D Spend", "Marketing Spend"],
        rmse: 8198.797191,
        mae: 6886.594588,
        r2: 0.947439,
        modelKey: "Marketing Spend__R&D Spend"
      },
      {
        k: 3,
        features: ["R&D Spend", "Administration", "State_Florida"],
        rmse: 8171.645989,
        mae: 6730.785004,
        r2: 0.947786,
        modelKey: "Administration__R&D Spend__State_Florida"
      },
      {
        k: 4,
        features: ["R&D Spend", "Administration", "State_Florida", "State_New York"],
        rmse: 8178.369215,
        mae: 6751.890337,
        r2: 0.947700,
        modelKey: "Administration__R&D Spend__State_Florida__State_New York"
      },
      {
        k: 5,
        features: ["R&D Spend", "Administration", "Marketing Spend", "State_Florida", "State_New York"],
        rmse: 9137.990153,
        mae: 7514.293660,
        r2: 0.934707,
        modelKey: "Administration__Marketing Spend__R&D Spend__State_Florida__State_New York"
      }
    ]
  }
};

// Fitted OLS regression equations & weights
const MODEL_COEFFICIENTS = {
  "R&D Spend": {
    intercept: 48416.30,
    coef: { "R&D Spend": 0.8516 }
  },
  "Marketing Spend__R&D Spend": {
    intercept: 46346.34,
    coef: { "R&D Spend": 0.7886, "Marketing Spend": 0.0326 }
  },
  "Marketing Spend__R&D Spend__State_New York": {
    intercept: 46100.29,
    coef: { "R&D Spend": 0.7859, "Marketing Spend": 0.0333, "State_New York": 864.00 }
  },
  "Marketing Spend__R&D Spend__State_Florida__State_New York": {
    intercept: 46329.06,
    coef: { "R&D Spend": 0.7851, "Marketing Spend": 0.0340, "State_New York": 554.66, "State_Florida": -826.47 }
  },
  "Administration__Marketing Spend__R&D Spend__State_Florida__State_New York": {
    intercept: 42554.17,
    coef: { "R&D Spend": 0.7735, "Administration": 0.0329, "Marketing Spend": 0.0366, "State_Florida": -959.28, "State_New York": 699.37 }
  },
  "R&D Spend__State_Florida": {
    intercept: 48319.70,
    coef: { "R&D Spend": 0.8516, "State_Florida": 399.18 }
  },
  "Administration__R&D Spend__State_Florida": {
    intercept: 48524.92,
    coef: { "R&D Spend": 0.8519, "Administration": -0.0019, "State_Florida": 404.02 }
  },
  "Administration__R&D Spend__State_Florida__State_New York": {
    intercept: 48444.53,
    coef: { "R&D Spend": 0.8516, "Administration": -0.0017, "State_Florida": 484.73, "State_New York": 173.20 }
  }
};

// Held-out test set observations (10 startups from 20% test split)
const TEST_STARTUPS = [
  { rd: 66051.52, admin: 182645.56, marketing: 118148.20, state_ny: 0, state_fl: 1, profit: 103282.38 },
  { rd: 100671.96, admin: 91790.61, marketing: 249744.55, state_ny: 0, state_fl: 0, profit: 144259.40 },
  { rd: 101913.08, admin: 110594.11, marketing: 229160.95, state_ny: 0, state_fl: 1, profit: 146121.95 },
  { rd: 27892.92, admin: 84710.77, marketing: 164470.71, state_ny: 0, state_fl: 1, profit: 77798.83 },
  { rd: 153441.51, admin: 101145.55, marketing: 407934.54, state_ny: 0, state_fl: 1, profit: 191050.39 },
  { rd: 72107.60, admin: 127864.55, marketing: 353183.81, state_ny: 1, state_fl: 0, profit: 105008.31 },
  { rd: 20229.59, admin: 65947.93, marketing: 185265.10, state_ny: 1, state_fl: 0, profit: 81229.06 },
  { rd: 61136.38, admin: 152701.92, marketing: 88218.23, state_ny: 1, state_fl: 0, profit: 97483.56 },
  { rd: 73994.56, admin: 122782.75, marketing: 303319.26, state_ny: 0, state_fl: 1, profit: 110352.25 },
  { rd: 142107.34, admin: 91391.77, marketing: 366168.42, state_ny: 0, state_fl: 1, profit: 166187.94 }
];


// =========================================================================
// 2. APPLICATION STATE
// =========================================================================

let currentAlgo = "sfs";
let currentK = 2;

let rmseChartInstance = null;
let r2ChartInstance = null;
let scatterChartInstance = null;


// =========================================================================
// 3. INITIALIZATION & LIFECYCLE
// =========================================================================

document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  renderAll();
  updateProfitPrediction();
});


// =========================================================================
// 4. CHART RENDERING (Chart.js)
// =========================================================================

function initCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(26, 32, 44, 0.95)',
        titleFont: { size: 13, weight: 'bold', family: 'Inter' },
        bodyFont: { size: 12, family: 'Inter' },
        padding: 10,
        cornerRadius: 8,
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(226, 232, 240, 0.6)' },
        ticks: { font: { family: 'Inter', weight: 600 } },
        title: {
          display: true,
          text: 'Number of Features (k)',
          font: { family: 'Inter', weight: 700, size: 11 },
          color: '#4a5568'
        }
      },
      y: {
        grid: { color: 'rgba(226, 232, 240, 0.6)' },
        ticks: { font: { family: 'Inter' } }
      }
    }
  };

  // RMSE Chart Setup
  const ctxRmse = document.getElementById("rmseChart").getContext("2d");
  rmseChartInstance = new Chart(ctxRmse, {
    type: 'line',
    data: {
      labels: [1, 2, 3, 4, 5],
      datasets: [
        {
          label: 'RMSE ($)',
          data: [],
          borderColor: '#1f77b4',
          backgroundColor: '#1f77b4',
          borderWidth: 2.2,
          pointRadius: 6,
          pointHoverRadius: 9,
          tension: 0.15
        }
      ]
    },
    options: {
      ...commonOptions,
      onClick: (e, elements) => {
        if (elements.length > 0) {
          const clickedK = elements[0].index + 1;
          setK(clickedK);
        }
      },
      plugins: {
        ...commonOptions.plugins,
        tooltip: {
          ...commonOptions.plugins.tooltip,
          callbacks: {
            label: (ctx) => `RMSE: $${ctx.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
          }
        }
      },
      scales: {
        ...commonOptions.scales,
        y: {
          ...commonOptions.scales.y,
          title: {
            display: true,
            text: 'RMSE ($)',
            font: { family: 'Inter', weight: 700, size: 11 },
            color: '#4a5568'
          }
        }
      }
    }
  });

  // R-squared Chart Setup
  const ctxR2 = document.getElementById("r2Chart").getContext("2d");
  r2ChartInstance = new Chart(ctxR2, {
    type: 'line',
    data: {
      labels: [1, 2, 3, 4, 5],
      datasets: [
        {
          label: 'R-squared',
          data: [],
          borderColor: '#1f77b4',
          backgroundColor: '#1f77b4',
          borderWidth: 2.2,
          pointRadius: 6,
          pointHoverRadius: 9,
          tension: 0.15
        }
      ]
    },
    options: {
      ...commonOptions,
      onClick: (e, elements) => {
        if (elements.length > 0) {
          const clickedK = elements[0].index + 1;
          setK(clickedK);
        }
      },
      plugins: {
        ...commonOptions.plugins,
        tooltip: {
          ...commonOptions.plugins.tooltip,
          callbacks: {
            label: (ctx) => `R²: ${ctx.parsed.y.toFixed(6)}`
          }
        }
      },
      scales: {
        ...commonOptions.scales,
        y: {
          ...commonOptions.scales.y,
          title: {
            display: true,
            text: 'R-squared (R²)',
            font: { family: 'Inter', weight: 700, size: 11 },
            color: '#4a5568'
          }
        }
      }
    }
  });

  // Scatter & Fit Setup
  const ctxScatter = document.getElementById("scatterChart").getContext("2d");
  scatterChartInstance = new Chart(ctxScatter, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Test Startups',
          data: [],
          backgroundColor: '#4338ca',
          borderColor: '#312e81',
          pointRadius: 7,
          pointHoverRadius: 10
        },
        {
          label: 'Perfect Fit (y = x)',
          data: [
            { x: 70000, y: 70000 },
            { x: 200000, y: 200000 }
          ],
          type: 'line',
          borderColor: '#e53e3e',
          borderDash: [5, 5],
          borderWidth: 2,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { font: { family: 'Inter', weight: 600 } }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              if (ctx.datasetIndex === 0) {
                return `Actual: $${ctx.parsed.x.toLocaleString()} | Predicted: $${ctx.parsed.y.toLocaleString()}`;
              }
              return 'Perfect Prediction Reference';
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(226, 232, 240, 0.6)' },
          title: {
            display: true,
            text: 'Actual Profit ($)',
            font: { family: 'Inter', weight: 700, size: 11 },
            color: '#4a5568'
          }
        },
        y: {
          grid: { color: 'rgba(226, 232, 240, 0.6)' },
          title: {
            display: true,
            text: 'Predicted Profit ($)',
            font: { family: 'Inter', weight: 700, size: 11 },
            color: '#4a5568'
          }
        }
      }
    }
  });
}

function updateCharts() {
  const algo = ALGORITHM_DATA[currentAlgo];
  const records = algo.records;

  const rmseData = records.map(r => r.rmse);
  const r2Data = records.map(r => r.r2);

  // Dynamic point colors: red highlight for active k, gold for optimal k
  const pointBgColors = records.map(r => {
    if (r.k === currentK) return '#e53e3e';
    if (r.k === algo.optimalK) return '#d97706';
    return '#1f77b4';
  });

  const pointRadii = records.map(r => (r.k === currentK ? 9 : 6));

  // Update RMSE Chart
  rmseChartInstance.data.datasets[0].data = rmseData;
  rmseChartInstance.data.datasets[0].pointBackgroundColor = pointBgColors;
  rmseChartInstance.data.datasets[0].pointRadius = pointRadii;
  rmseChartInstance.update();

  // Update R2 Chart
  r2ChartInstance.data.datasets[0].data = r2Data;
  r2ChartInstance.data.datasets[0].pointBackgroundColor = pointBgColors;
  r2ChartInstance.data.datasets[0].pointRadius = pointRadii;
  r2ChartInstance.update();

  // Compute Scatter Predictions for active subset
  const currentRecord = records.find(r => r.k === currentK);
  const model = MODEL_COEFFICIENTS[currentRecord.modelKey];

  const scatterPoints = TEST_STARTUPS.map(startup => {
    let pred = model.intercept;
    if (model.coef["R&D Spend"]) pred += model.coef["R&D Spend"] * startup.rd;
    if (model.coef["Administration"]) pred += model.coef["Administration"] * startup.admin;
    if (model.coef["Marketing Spend"]) pred += model.coef["Marketing Spend"] * startup.marketing;
    if (model.coef["State_New York"]) pred += model.coef["State_New York"] * startup.state_ny;
    if (model.coef["State_Florida"]) pred += model.coef["State_Florida"] * startup.state_fl;
    return { x: Math.round(startup.profit), y: Math.round(pred) };
  });

  scatterChartInstance.data.datasets[0].data = scatterPoints;
  scatterChartInstance.update();

  document.getElementById("model-status-badge").textContent = `${algo.name.split(' ')[0]} (k=${currentK})`;
}


// =========================================================================
// 5. UI UPDATES & EVENT HANDLERS
// =========================================================================

function renderAll() {
  const algo = ALGORITHM_DATA[currentAlgo];
  const activeRecord = algo.records.find(r => r.k === currentK);
  const model = MODEL_COEFFICIENTS[activeRecord.modelKey];

  // 1. Update Algo Title
  document.getElementById("current-algo-name").textContent = algo.name;

  // 2. Update Slider and Labels
  document.getElementById("k-slider").value = currentK;
  document.getElementById("k-value-display").textContent = `k = ${currentK}`;

  // 3. Update Included Features Tags
  const tagsContainer = document.getElementById("feature-tags-list");
  tagsContainer.innerHTML = "";
  activeRecord.features.forEach(feat => {
    const chip = document.createElement("span");
    chip.className = "feature-tag active";
    chip.innerHTML = `✓ ${feat}`;
    tagsContainer.appendChild(chip);
  });

  // 4. Update Equation Box
  let eqStr = `Profit = ${Math.round(model.intercept).toLocaleString()}`;
  for (const [col, weight] of Object.entries(model.coef)) {
    const sign = weight >= 0 ? "+" : "−";
    const absWeight = Math.abs(weight).toFixed(4);
    eqStr += ` ${sign} (${absWeight} × ${col})`;
  }
  document.getElementById("equation-display").textContent = eqStr;

  // 5. Update Metrics Grid
  document.getElementById("metric-rmse").textContent = `$${activeRecord.rmse.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  document.getElementById("metric-r2").textContent = activeRecord.r2.toFixed(4);
  document.getElementById("metric-mae").textContent = `$${activeRecord.mae.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  const overfitBadge = document.getElementById("metric-overfit");
  if (currentK === algo.optimalK) {
    overfitBadge.textContent = "Optimal ★";
    overfitBadge.className = "metric-value highlight-good";
  } else if (currentK > algo.optimalK) {
    overfitBadge.textContent = "Overfitting ⚠";
    overfitBadge.className = "metric-value highlight-primary";
  } else {
    overfitBadge.textContent = "Underfitting";
    overfitBadge.className = "metric-value";
  }

  // 6. Update Table
  renderTable();

  // 7. Update Charts
  updateCharts();

  // 8. Update Calculator Description
  updateProfitPrediction();
}

function renderTable() {
  const algo = ALGORITHM_DATA[currentAlgo];
  const tbody = document.getElementById("table-body");
  tbody.innerHTML = "";

  algo.records.forEach(rec => {
    const tr = document.createElement("tr");
    if (rec.k === currentK) tr.classList.add("active-row");
    if (rec.k === algo.optimalK) tr.classList.add("optimal-row");

    tr.onclick = () => setK(rec.k);

    const isOptimal = rec.k === algo.optimalK;
    const optimalBadgeHtml = isOptimal ? `<span class="optimal-badge">★ Optimal</span>` : "";

    tr.innerHTML = `
      <td><strong>${rec.k}</strong> ${optimalBadgeHtml}</td>
      <td><code>${JSON.stringify(rec.features).replace(/"/g, "'")}</code></td>
      <td>${rec.rmse.toFixed(6)}</td>
      <td>${rec.mae.toFixed(6)}</td>
      <td><strong>${rec.r2.toFixed(6)}</strong></td>
    `;

    tbody.appendChild(tr);
  });
}

function switchAlgorithm(algoKey) {
  currentAlgo = algoKey;

  // Update tabs visual state
  document.querySelectorAll(".algo-tab-btn").forEach(btn => btn.classList.remove("active"));
  document.getElementById(`tab-${algoKey}`).classList.add("active");

  // Reset to optimal k for this algorithm
  currentK = ALGORITHM_DATA[algoKey].optimalK;

  renderAll();
}

function onKSliderChange(val) {
  currentK = parseInt(val, 10);
  renderAll();
}

function adjustK(delta) {
  const nextK = Math.max(1, Math.min(5, currentK + delta));
  if (nextK !== currentK) {
    currentK = nextK;
    renderAll();
  }
}

function setK(k) {
  currentK = k;
  renderAll();
}

function selectOptimalSubset() {
  currentK = ALGORITHM_DATA[currentAlgo].optimalK;
  renderAll();
}

function resetControls() {
  currentAlgo = "sfs";
  currentK = 2;
  document.querySelectorAll(".algo-tab-btn").forEach(btn => btn.classList.remove("active"));
  document.getElementById("tab-sfs").classList.add("active");

  document.getElementById("calc-rd-slider").value = 100000;
  document.getElementById("calc-admin-slider").value = 120000;
  document.getElementById("calc-mkt-slider").value = 200000;
  document.getElementById("calc-state-select").value = "California";

  renderAll();
}


// =========================================================================
// 6. LIVE PROFIT PREDICTION PLAYGROUND
// =========================================================================

function updateProfitPrediction() {
  const rd = parseFloat(document.getElementById("calc-rd-slider").value);
  const admin = parseFloat(document.getElementById("calc-admin-slider").value);
  const mkt = parseFloat(document.getElementById("calc-mkt-slider").value);
  const state = document.getElementById("calc-state-select").value;

  document.getElementById("calc-rd-val").textContent = `$${rd.toLocaleString()}`;
  document.getElementById("calc-admin-val").textContent = `$${admin.toLocaleString()}`;
  document.getElementById("calc-mkt-val").textContent = `$${mkt.toLocaleString()}`;

  const algo = ALGORITHM_DATA[currentAlgo];
  const activeRecord = algo.records.find(r => r.k === currentK);
  const model = MODEL_COEFFICIENTS[activeRecord.modelKey];

  let profit = model.intercept;
  if (model.coef["R&D Spend"]) profit += model.coef["R&D Spend"] * rd;
  if (model.coef["Administration"]) profit += model.coef["Administration"] * admin;
  if (model.coef["Marketing Spend"]) profit += model.coef["Marketing Spend"] * mkt;
  if (model.coef["State_New York"] && state === "New York") profit += model.coef["State_New York"];
  if (model.coef["State_Florida"] && state === "Florida") profit += model.coef["State_Florida"];

  document.getElementById("calc-predicted-profit").textContent = `$${profit.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  document.getElementById("calc-formula-desc").textContent = `Model: ${algo.name.split(' ')[0]} (k=${currentK}) using ${JSON.stringify(activeRecord.features).replace(/"/g, "'")}`;
}


// =========================================================================
// 7. EXPORT DATA (CSV)
// =========================================================================

function exportResultsCSV() {
  const algo = ALGORITHM_DATA[currentAlgo];
  let csvContent = "data:text/csv;charset=utf-8,Number of Features,Selected Features,RMSE,MAE,R-squared\n";

  algo.records.forEach(r => {
    const featStr = `"${r.features.join('; ')}"`;
    csvContent += `${r.k},${featStr},${r.rmse.toFixed(6)},${r.mae.toFixed(6)},${r.r2.toFixed(6)}\n`;
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `${currentAlgo}_feature_selection_benchmark.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
