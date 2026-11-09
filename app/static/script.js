/* ============================================================
   Explainable Malicious URL Detection — Frontend State Machine
   ES2020+, no external framework or CDN dependency
   ============================================================ */

'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let _lastURL = '';
let _lastPrediction = null;   // last /predict response
let _lastExplanation = null;   // last /explain response keyed by method
let _activeMethod = 'ig';
let _activeModel = 'full';
let _researchData = null;
let _requestSerial = 0;
let _explainCache = {};     // { 'ig': {...}, 'lime': {...}, 'shap': {...} }

const METHOD_INFO = {
  ig: {
    label: 'Integrated Gradients',
    badge: 'IG',
    badgeClass: 'ig',
    note: 'Integrated Gradients over PAD-token embedding baseline. Attributions are logit-space; scale is model-specific.',
  },
  lime: {
    label: 'LIME',
    badge: 'LIME',
    badgeClass: 'lime',
    note: 'Local linear surrogate (150 samples). Weights are local linear coefficients; scale differs from IG/SHAP.',
  },
  shap: {
    label: 'SHAP',
    badge: 'SHAP',
    badgeClass: 'shap',
    note: 'Shapley value partition explainer. Values are probability-space contributions; scale differs from IG/LIME.',
  },
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const urlInput = () => document.getElementById('url-input');
const btnAnalyze = () => document.getElementById('btn-analyze');
const analyzeError = () => document.getElementById('analyze-error');
const predCard = () => document.getElementById('prediction-card');
const predLabel = () => document.getElementById('pred-label');
const predConf = () => document.getElementById('pred-confidence');
const barBenign = () => document.getElementById('bar-benign');
const barMalicious = () => document.getElementById('bar-malicious');
const valBenign = () => document.getElementById('val-benign');
const valMalicious = () => document.getElementById('val-malicious');
const btnExplain = () => document.getElementById('btn-explain');
const explainError = () => document.getElementById('explain-error');
const explPanel = () => document.getElementById('explanation-panel');
const methodBadge = () => document.getElementById('method-badge');
const methodNote = () => document.getElementById('method-note');
const tokenHeatmap = () => document.getElementById('token-heatmap');
const compChart = () => document.getElementById('component-chart');
const explWarning = () => document.getElementById('explanation-warning');

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function showError(el, msg) {
  el.textContent = msg;
  el.hidden = false;
}
function hideError(el) {
  el.hidden = true;
  el.textContent = '';
}
function setLoading(btn, loading) {
  const text = btn.querySelector('.btn-text');
  const spinner = btn.querySelector('.spinner');
  btn.disabled = loading;
  if (text) text.hidden = loading;
  if (spinner) spinner.hidden = !loading;
}

// ---------------------------------------------------------------------------
// Tab Management
// ---------------------------------------------------------------------------
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const method = tab.dataset.method;
    if (method === _activeMethod) return;
    _activeMethod = method;

    document.querySelectorAll('.tab').forEach(t => {
      t.classList.toggle('active', t.dataset.method === method);
      t.setAttribute('aria-selected', String(t.dataset.method === method));
    });

    // If we have a cached explanation for this method, show it immediately
    if (_explainCache[_activeModel + ':' + method]) {
      renderExplanation(_explainCache[_activeModel + ':' + method]);
    } else if (explPanel().hidden === false) {
      // Panel was visible — re-fetch for new method
      explainURL();
    }
  });
});

// ---------------------------------------------------------------------------
// analyzeURL() — POST /predict
// ---------------------------------------------------------------------------
async function analyzeURL() {
  const url = urlInput().value.trim();
  if (!url) { showError(analyzeError(), 'Please enter a URL.'); return; }

  hideError(analyzeError());
  hideError(explainError());
  setLoading(btnAnalyze(), true);
  predCard().hidden = true;
  explPanel().hidden = true;
  _explainCache = {};
  _lastURL = '';
  _lastPrediction = null;
  const serial = ++_requestSerial;
  const selectedModel = _activeModel;

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, model: selectedModel }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    if (serial !== _requestSerial) return;
    _lastURL = url;
    _lastPrediction = data;
    renderPrediction(data);
    loadComparison(url, serial);
  } catch (err) {
    showError(analyzeError(), 'Prediction error: ' + err.message);
  } finally {
    setLoading(btnAnalyze(), false);
  }
}

function renderPrediction(data) {
  const isMal = data.predicted_class === 1;
  const lbl = predLabel();
  lbl.textContent = data.prediction;
  lbl.className = 'pred-label ' + (isMal ? 'malicious' : 'benign');
  predConf().textContent = (data.confidence * 100).toFixed(2) + '%';

  const bP = data.probabilities.benign;
  const mP = data.probabilities.malicious;
  barBenign().style.width = (bP * 100).toFixed(1) + '%';
  barMalicious().style.width = (mP * 100).toFixed(1) + '%';
  valBenign().textContent = bP.toFixed(4);
  valMalicious().textContent = mP.toFixed(4);

  predCard().hidden = false;
}

// ---------------------------------------------------------------------------
// explainURL() — POST /explain
// ---------------------------------------------------------------------------
async function explainURL() {
  const url = urlInput().value.trim();
  if (!url || url !== _lastURL || !_lastPrediction) {
    showError(explainError(), 'Click Analyze URL first for this exact URL.'); return;
  }

  hideError(explainError());
  setLoading(btnExplain(), true);

  try {
    const res = await fetch('/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, method: _activeMethod, model: _activeModel }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    if (url !== _lastURL || data.model_id !== _activeModel) return;
    _explainCache[_activeModel + ':' + _activeMethod] = data;
    renderExplanation(data);
  } catch (err) {
    showError(explainError(), 'Explanation error: ' + err.message);
    explPanel().hidden = true;
  } finally {
    setLoading(btnExplain(), false);
  }
}

// ---------------------------------------------------------------------------
// renderExplanation()
// ---------------------------------------------------------------------------
function renderExplanation(data) {
  const info = METHOD_INFO[_activeMethod] || METHOD_INFO.ig;

  // Method badge
  const badge = methodBadge();
  badge.textContent = info.badge;
  badge.className = 'method-badge ' + info.badgeClass;
  methodNote().textContent = info.note;

  // Heatmap
  renderTokenHeatmap(data.tokens || []);

  // Component chart
  renderComponentChart(data.components || {});

  // Warning
  explWarning().textContent = data.warning || '';
  explWarning().hidden = !data.warning;

  // Also update prediction card if /explain returned prediction info
  if (data.prediction && predCard().hidden) {
    renderPrediction(data);
  }

  explPanel().hidden = false;
}

// ---------------------------------------------------------------------------
// renderTokenHeatmap()
// ---------------------------------------------------------------------------
function renderTokenHeatmap(tokens) {
  const container = tokenHeatmap();
  container.replaceChildren();

  if (!tokens.length) {
    container.textContent = 'No tokens available.';
    return;
  }

  // Normalize by method's own max absolute attribution
  const maxAbs = Math.max(1e-9, ...tokens.map(t => Math.abs(t.attribution)));

  tokens.forEach(tok => {
    const v = Number(tok.attribution) || 0;
    const t = Math.abs(v) / maxAbs;      // 0–1

    // Theme colors: positive -> orange, negative -> teal (white at zero)
    const hi = v >= 0 ? [255, 122, 47] : [47, 127, 143];
    const r = Math.round(255 - (255 - hi[0]) * t);
    const g = Math.round(255 - (255 - hi[1]) * t);
    const b = Math.round(255 - (255 - hi[2]) * t);

    const el = document.createElement('span');
    el.className = 'hm-token';
    el.textContent = tok.text || tok.token || '?';
    el.style.backgroundColor = `rgb(${r},${g},${b})`;
    el.style.color = t > 0.6 ? '#fff' : '#1b1917';
    el.title = `Token: "${tok.token}"  Attribution: ${v.toFixed(6)}`;
    container.appendChild(el);
  });
}

// ---------------------------------------------------------------------------
// renderComponentChart() — signed horizontal SVG bar chart
// ---------------------------------------------------------------------------
function renderComponentChart(components) {
  const container = compChart();
  container.replaceChildren();

  const entries = Object.entries(components).filter(([, v]) => v !== undefined);
  if (!entries.length) {
    container.textContent = 'No component data.';
    return;
  }

  const maxAbs = Math.max(1e-9, ...entries.map(([, v]) => Math.abs(v)));

  entries.forEach(([name, value]) => {
    const row = document.createElement('div');
    row.className = 'comp-row';

    const nameEl = document.createElement('span');
    nameEl.className = 'comp-name';
    nameEl.textContent = name;

    // SVG bar
    const svgWrap = document.createElement('div');
    svgWrap.className = 'comp-svg-wrap';

    const W = 300, H = 28, MID = W / 2;
    const barLen = Math.abs(value) / maxAbs * (MID - 4);
    const isPos = value >= 0;
    const barX = isPos ? MID : MID - barLen;
    const color = isPos ? '#ff7a2f' : '#2f7f8f';
    const labelX = isPos ? MID + barLen + 4 : MID - barLen - 4;
    const anchor = isPos ? 'start' : 'end';

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('preserveAspectRatio', 'none');

    // Background track
    const track = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    track.setAttribute('x', '0'); track.setAttribute('y', '9');
    track.setAttribute('width', String(W)); track.setAttribute('height', '10');
    track.setAttribute('rx', '5'); track.setAttribute('fill', '#dedbd5');
    svg.appendChild(track);

    // Center line
    const cline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    cline.setAttribute('x1', String(MID)); cline.setAttribute('y1', '4');
    cline.setAttribute('x2', String(MID)); cline.setAttribute('y2', String(H - 4));
    cline.setAttribute('stroke', '#6f6a64'); cline.setAttribute('stroke-width', '1');
    svg.appendChild(cline);

    // Value bar
    if (barLen > 0) {
      const bar = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      bar.setAttribute('x', String(barX)); bar.setAttribute('y', '9');
      bar.setAttribute('width', String(barLen)); bar.setAttribute('height', '10');
      bar.setAttribute('rx', '3'); bar.setAttribute('fill', color);
      svg.appendChild(bar);
    }

    // Value label inside SVG
    const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    txt.setAttribute('x', String(labelX));
    txt.setAttribute('y', String(H / 2 + 4));
    txt.setAttribute('text-anchor', anchor);
    txt.setAttribute('font-size', '9');
    txt.setAttribute('fill', '#6f6a64');
    txt.textContent = value.toFixed(4);
    svg.appendChild(txt);

    svgWrap.appendChild(svg);

    const valEl = document.createElement('span');
    valEl.className = 'comp-val';
    valEl.textContent = value >= 0 ? '+' + value.toFixed(4) : value.toFixed(4);
    valEl.style.color = isPos ? '#c2410c' : '#2f7f8f';

    row.append(nameEl, svgWrap, valEl);
    container.appendChild(row);
  });
}

// ---------------------------------------------------------------------------
// fetchResearchResults() and renderResearchTable()
// ---------------------------------------------------------------------------
async function fetchResearchResults() {
  const loading = document.getElementById('research-loading');
  const errEl = document.getElementById('research-error');
  try {
    const res = await fetch('/research_results');
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    _researchData = data;
    loading.hidden = true;
    if (data.test_metrics) {
      const m = data.test_metrics;
      document.getElementById('test-metrics').textContent =
        `Full DistilBERT held-out test (${m.test_samples || '—'} URLs): Accuracy ${((m.accuracy || 0) * 100).toFixed(2)}%, F1 ${((m.f1 || 0) * 100).toFixed(2)}%, Recall ${((m.recall || 0) * 100).toFixed(2)}%, ROC-AUC ${((m.roc_auc || 0) * 100).toFixed(2)}%.`;
    }
    showResearchModel();
  } catch (err) { loading.hidden = true; showError(errEl, 'Could not load research results: ' + err.message); }
}

function showResearchModel() {
  if (!_researchData) return;
  const key = document.getElementById('research-model').value;
  const data = _researchData.xai?.[key];
  const wrap = document.getElementById('research-table-wrap');
  const err = document.getElementById('research-error');
  if (!data) {
    wrap.hidden = true;
    showError(err, `No completed 200-URL XAI summary found for DistilBERT ${key}. Finish the offline evaluation first.`);
    return;
  }
  hideError(err); wrap.hidden = false;
  document.getElementById('research-source').textContent = 'Source: ' + _researchData.sources[key];
  renderResearchTable(data); renderResearchCharts(data);
}

function renderResearchTable(data) {
  const tbody = document.getElementById('research-tbody');
  tbody.replaceChildren();

  const methods = ['IG', 'LIME', 'SHAP'];

  // Columns: [key_mean, key_std, higher_is_better]
  const cols = [
    ['occlusion_spearman_mean', 'occlusion_spearman_std', true],
    ['deletion_probability_drop_mean', 'deletion_probability_drop_std', true],
    ['mean_cosine_similarity_mean', 'mean_cosine_similarity_std', true],
    ['explain_seconds_mean', 'explain_seconds_std', false],
    ['top_component_stability_mean', 'top_component_stability_std', true],
  ];

  // Find best value per column
  const bestVals = cols.map(([mk, , hib]) => {
    const vals = methods.map(m => data[m]?.[mk] ?? null).filter(v => v !== null);
    if (!vals.length) return null;
    return hib ? Math.max(...vals) : Math.min(...vals);
  });

  methods.forEach(m => {
    const row = document.createElement('tr');
    const d = data[m] || {};

    // Method cell
    const methodTd = document.createElement('td');
    methodTd.className = `method-cell method-${m.toLowerCase()}`;
    methodTd.textContent = m;
    row.appendChild(methodTd);

    cols.forEach(([mk, sk], ci) => {
      const td = document.createElement('td');
      const val = d[mk] ?? null;
      const std = d[sk] ?? null;
      if (val === null) {
        td.textContent = '—';
      } else {
        td.textContent = std !== null
          ? `${val.toFixed(4)} ± ${std.toFixed(4)}`
          : val.toFixed(4);
        if (ci !== 1 && bestVals[ci] !== null && Math.abs(val - bestVals[ci]) < 1e-9) {
          td.classList.add('best');
          td.title = 'Best value in this column';
        }
      }
      row.appendChild(td);
    });

    tbody.appendChild(row);
  });
}

// ---------------------------------------------------------------------------
// Offline research charts — values are loaded from /research_results CSV
// ---------------------------------------------------------------------------
function renderResearchCharts(data) {
  const charts = [
    { id: 'chart-occlusion', key: 'occlusion_spearman_mean', max: 1, digits: 4 },
    { id: 'chart-stability', key: 'mean_cosine_similarity_mean', max: 1, digits: 4 },
    { id: 'chart-runtime', key: 'explain_seconds_mean', max: null, digits: 3 },
  ];
  const colors = { IG: '#ff7a2f', LIME: '#f2b705', SHAP: '#1b1917' };
  charts.forEach(chart => {
    const container = document.getElementById(chart.id);
    if (!container) return;
    container.replaceChildren();
    const vals = ['IG', 'LIME', 'SHAP'].map(m => ({ method: m, raw: data[m]?.[chart.key] }))
      .filter(x => x.raw !== null && x.raw !== undefined && Number.isFinite(Number(x.raw)))
      .map(x => ({ method: x.method, value: Number(x.raw) }));
    if (!vals.length) { container.textContent = 'No saved values available.'; return; }
    const scale = chart.max ?? Math.max(0.001, ...vals.map(x => x.value));
    vals.forEach(({ method, value }) => {
      const row = document.createElement('div'); row.className = 'research-bar-row';
      const label = document.createElement('span'); label.className = 'research-bar-label'; label.textContent = method;
      const track = document.createElement('div'); track.className = 'research-bar-track';
      const fill = document.createElement('div'); fill.className = 'research-bar-fill';
      fill.style.backgroundColor = colors[method];
      fill.style.width = `${Math.max(0, Math.min(100, value / scale * 100))}%`;
      track.appendChild(fill);
      const number = document.createElement('span'); number.className = 'research-bar-value';
      number.textContent = value.toFixed(chart.digits) + (chart.id === 'chart-runtime' ? ' s' : '');
      row.append(label, track, number); container.appendChild(row);
    });
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  // Enter key in URL input triggers analyze
  urlInput().addEventListener('keydown', e => {
    if (e.key === 'Enter') analyzeURL();
  });

  fetchResearchResults();
});


async function loadComparison(url, serial) {
  const panel = document.getElementById('comparison-panel');
  const tbody = document.getElementById('comparison-tbody');
  tbody.replaceChildren(); panel.hidden = false;
  const row = document.createElement('tr');
  const cell = document.createElement('td'); cell.colSpan = 4; cell.textContent = 'Comparing four models…'; row.appendChild(cell); tbody.appendChild(row);
  try {
    const res = await fetch('/compare_models', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url }) });
    const data = await res.json(); if (!res.ok) throw new Error(data.detail || 'Comparison failed');
    if (serial !== _requestSerial) return;
    tbody.replaceChildren();
    for (const [key, name] of [['rf', 'Random Forest'], ['xgb', 'XGBoost'], ['10k', 'DistilBERT 10k'], ['full', 'DistilBERT Full']]) {
      const d = data.results[key] || {};
      const tr = document.createElement('tr');
      const vals = [name, d.error ? 'Unavailable: ' + d.error : d.prediction,
        d.error ? '—' : (100 * d.probabilities.malicious).toFixed(2) + '%',
        d.error ? '—' : String(d.inference_time_ms)];
      vals.forEach(v => { const td = document.createElement('td'); td.textContent = v; tr.appendChild(td) });
      tbody.appendChild(tr);
    }
  } catch (err) { tbody.replaceChildren(); const tr = document.createElement('tr'); const td = document.createElement('td'); td.colSpan = 4; td.textContent = 'Comparison unavailable: ' + err.message; tr.appendChild(td); tbody.appendChild(tr); }
}

document.getElementById('model-select').addEventListener('change', e => {
  _activeModel = e.target.value;
  _requestSerial++;
  _lastURL = ''; _lastPrediction = null; _explainCache = {};
  predCard().hidden = true; explPanel().hidden = true;
  document.getElementById('comparison-panel').hidden = true;
  hideError(explainError());
});
document.getElementById('research-model').addEventListener('change', showResearchModel);
