"""
CS550 Project Demo - Cora Citation Network Explorer
Web application: node classification & link prediction inference
"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np
import scipy.sparse as sp
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import (load_cora, build_adjacency, normalize_features,
                                 normalize_adjacency, split_nodes_classification,
                                 split_edges_link_prediction)
from gcn_model import train_gcn, accuracy
from link_prediction import (get_gcn_embeddings, edge_features_hadamard,
                              train_link_predictor, predict_link_gcn,
                              compute_heuristic_scores)
from node_classification import label_propagation

app = Flask(__name__)

# ── Global state (loaded once at startup) ─────────────────────
STATE = {}

CLASS_NAMES = ['Case_Based','Genetic_Algorithms','Neural_Networks',
               'Probabilistic_Methods','Reinforcement_Learning',
               'Rule_Learning','Theory']

def startup():
    print("[Demo] Loading dataset and training model...")
    labels, edges, features = load_cora()
    N = len(labels)
    A_full        = build_adjacency(edges, N)
    features_norm = normalize_features(features)
    X_dense       = features_norm.toarray()
    A_norm_full   = normalize_adjacency(A_full)

    link_splits = split_edges_link_prediction(edges, N, test_ratio=0.2, val_ratio=0.1)
    train_idx, val_idx, test_idx = split_nodes_classification(labels)

    gcn_model, history = train_gcn(
        A_norm_full, X_dense, labels, train_idx, val_idx, test_idx,
        n_hidden=64, lr=0.01, epochs=300, weight_decay=5e-4,
        dropout=0.5, patience=30, verbose=False
    )

    preds_full, Z = gcn_model.predict(A_norm_full, X_dense)
    test_acc = accuracy(preds_full, labels, test_idx)
    print(f"[Demo] GCN Test Accuracy: {test_acc:.4f}")

    # Train link predictor
    A_train       = link_splits['train_A']
    A_norm_train  = normalize_adjacency(A_train)
    emb = get_gcn_embeddings(gcn_model, A_norm_full, X_dense)
    link_clf = train_link_predictor(emb, link_splits['train']['pos'], link_splits['train']['neg'])

    STATE.update({
        'labels': labels, 'edges': edges, 'X_dense': X_dense,
        'A_full': A_full, 'A_norm': A_norm_full,
        'gcn_model': gcn_model, 'Z': Z, 'preds': preds_full,
        'embeddings': emb, 'link_clf': link_clf, 'link_splits': link_splits,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
        'history': history, 'test_acc': test_acc,
        'n_nodes': N, 'n_edges': len(edges),
    })
    print("[Demo] Ready!")

# ── HTML Template ──────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cora Citation Network — CS550 Demo</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --card: #22263a;
    --border: #2d3250; --accent: #4e7cff; --accent2: #7c4eff;
    --green: #38d9a9; --orange: #ffa94d; --red: #ff6b6b;
    --text: #e8eaf6; --muted: #8892b0; --white: #ffffff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

  header {
    background: linear-gradient(135deg, var(--surface) 0%, #1e2235 100%);
    border-bottom: 1px solid var(--border);
    padding: 20px 40px;
    display: flex; align-items: center; gap: 16px;
  }
  header .logo { font-size: 28px; }
  header h1 { font-size: 20px; font-weight: 700; color: var(--white); }
  header p  { font-size: 13px; color: var(--muted); margin-top: 2px; }
  .badge {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white; border-radius: 8px; padding: 4px 12px;
    font-size: 12px; font-weight: 600; margin-left: auto;
  }

  .stats-bar {
    display: flex; gap: 0; border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .stat-item {
    flex: 1; padding: 14px 20px; text-align: center;
    border-right: 1px solid var(--border);
  }
  .stat-item:last-child { border-right: none; }
  .stat-val  { font-size: 24px; font-weight: 800; color: var(--accent); }
  .stat-label { font-size: 11px; color: var(--muted); margin-top: 2px; letter-spacing: .5px; text-transform: uppercase; }

  .main { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; padding: 28px 40px; max-width: 1400px; margin: 0 auto; }
  @media (max-width: 900px) { .main { grid-template-columns: 1fr; padding: 16px; } }

  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; overflow: hidden;
  }
  .card-header {
    padding: 18px 22px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px;
    background: linear-gradient(to right, rgba(78,124,255,.08), transparent);
  }
  .card-header .icon { font-size: 20px; }
  .card-header h2   { font-size: 15px; font-weight: 700; color: var(--white); }
  .card-body { padding: 22px; }

  label { font-size: 12px; color: var(--muted); font-weight: 600;
          text-transform: uppercase; letter-spacing: .5px; display: block; margin-bottom: 6px; }
  input[type=number], select {
    width: 100%; background: var(--surface); border: 1px solid var(--border);
    color: var(--text); padding: 10px 14px; border-radius: 8px;
    font-size: 14px; outline: none; margin-bottom: 14px;
    transition: border-color .2s;
  }
  input:focus, select:focus { border-color: var(--accent); }

  .btn {
    width: 100%; padding: 12px; border-radius: 10px; border: none; cursor: pointer;
    font-size: 14px; font-weight: 700; letter-spacing: .3px;
    transition: all .2s; margin-top: 4px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
  }
  .btn:hover { opacity: .88; transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }

  .result-box {
    margin-top: 18px; padding: 18px; border-radius: 12px;
    background: var(--surface); border: 1px solid var(--border);
    display: none;
  }
  .result-box.show { display: block; }

  .pred-label {
    font-size: 22px; font-weight: 800; color: var(--accent);
    margin-bottom: 6px;
  }
  .conf-text { font-size: 13px; color: var(--muted); margin-bottom: 16px; }

  .prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .prob-name { font-size: 12px; color: var(--muted); width: 170px; flex-shrink: 0; }
  .prob-bar-wrap { flex: 1; background: var(--bg); border-radius: 4px; height: 8px; overflow: hidden; }
  .prob-bar { height: 100%; border-radius: 4px;
    background: linear-gradient(to right, var(--accent), var(--accent2)); transition: width .5s; }
  .prob-val { font-size: 12px; color: var(--text); width: 44px; text-align: right; }

  .link-result {
    display: flex; align-items: center; gap: 14px;
    font-size: 15px; font-weight: 700;
  }
  .link-dot { width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }
  .link-yes  { background: var(--green); }
  .link-no   { background: var(--red); }
  .link-score { font-size: 13px; color: var(--muted); margin-top: 6px; }

  .methods-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
  .method-card {
    background: var(--bg); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px;
  }
  .method-name  { font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; margin-bottom: 4px; }
  .method-pred  { font-size: 14px; font-weight: 700; }

  .results-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
  .results-table th {
    background: var(--bg); padding: 10px 12px; text-align: left;
    font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: .5px; border-bottom: 1px solid var(--border);
  }
  .results-table td { padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,.04); }
  .results-table tr:last-child td { border-bottom: none; }
  .metric-val { font-weight: 700; color: var(--accent); }
  .best { background: rgba(78,124,255,.08); }

  .info-pair { display: flex; justify-content: space-between; align-items: center;
               padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,.05); font-size: 13px; }
  .info-pair:last-child { border-bottom: none; }
  .info-key { color: var(--muted); }
  .info-val { font-weight: 600; }

  .spinner { display: inline-block; width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,.2); border-top-color: white;
    border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .tag {
    display: inline-block; background: rgba(78,124,255,.15); color: var(--accent);
    border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: 600;
    margin-right: 4px; margin-top: 4px;
  }
  .error-box { color: var(--red); font-size: 13px; margin-top: 10px; }
</style>
</head>
<body>

<header>
  <div class="logo">🕸️</div>
  <div>
    <h1>Cora Citation Network Explorer</h1>
    <p>CS550 Project &mdash; Option 2: Social Networks &mdash; GCN Demo</p>
  </div>
  <div class="badge">Live Inference</div>
</header>

<div class="stats-bar">
  <div class="stat-item"><div class="stat-val" id="s-nodes">2708</div><div class="stat-label">Nodes (Papers)</div></div>
  <div class="stat-item"><div class="stat-val" id="s-edges">5429</div><div class="stat-label">Edges (Citations)</div></div>
  <div class="stat-item"><div class="stat-val">7</div><div class="stat-label">Classes</div></div>
  <div class="stat-item"><div class="stat-val">1433</div><div class="stat-label">Features</div></div>
  <div class="stat-item"><div class="stat-val" id="s-acc">—</div><div class="stat-label">GCN Test Acc</div></div>
</div>

<div class="main">

  <!-- Node Classification -->
  <div class="card">
    <div class="card-header">
      <span class="icon">🔬</span>
      <h2>Node Classification</h2>
    </div>
    <div class="card-body">
      <p style="font-size:13px;color:var(--muted);margin-bottom:18px;">
        Predict the research topic (class) of a paper given its node ID.
        Uses the trained GCN, Label Propagation, and Logistic Regression.
      </p>
      <label>Node ID <span style="color:var(--muted);font-weight:400">(0 – 2707)</span></label>
      <input type="number" id="node-id" min="0" max="2707" value="42">
      <button class="btn" onclick="classifyNode()">🚀 Classify Node</button>
      <div class="error-box" id="nc-error"></div>
      <div class="result-box" id="nc-result">
        <div class="pred-label" id="nc-pred">—</div>
        <div class="conf-text" id="nc-conf">—</div>
        <div id="nc-bars"></div>
        <div style="margin-top:16px;font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;">All Methods</div>
        <div class="methods-grid" id="nc-methods"></div>
      </div>
    </div>
  </div>

  <!-- Link Prediction -->
  <div class="card">
    <div class="card-header">
      <span class="icon">🔗</span>
      <h2>Link Prediction</h2>
    </div>
    <div class="card-body">
      <p style="font-size:13px;color:var(--muted);margin-bottom:18px;">
        Predict whether a citation link exists between two papers
        using GCN embeddings, Adamic-Adar, and Common Neighbors.
      </p>
      <label>Source Node ID</label>
      <input type="number" id="lp-src" min="0" max="2707" value="100">
      <label>Target Node ID</label>
      <input type="number" id="lp-dst" min="0" max="2707" value="200">
      <button class="btn" onclick="predictLink()">🔍 Predict Link</button>
      <div class="error-box" id="lp-error"></div>
      <div class="result-box" id="lp-result">
        <div class="link-result">
          <div class="link-dot" id="lp-dot"></div>
          <div id="lp-verdict">—</div>
        </div>
        <div class="link-score" id="lp-score"></div>
        <div class="methods-grid" id="lp-methods" style="margin-top:14px;grid-template-columns:1fr 1fr 1fr;"></div>
      </div>
    </div>
  </div>

  <!-- Overall Results -->
  <div class="card">
    <div class="card-header">
      <span class="icon">📊</span>
      <h2>Node Classification Results</h2>
    </div>
    <div class="card-body">
      <table class="results-table" id="nc-table">
        <thead>
          <tr><th>Method</th><th>Accuracy</th><th>Macro P</th><th>Macro R</th><th>Macro F1</th></tr>
        </thead>
        <tbody id="nc-table-body"><tr><td colspan="5" style="color:var(--muted);text-align:center">Loading...</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- Link Prediction Results -->
  <div class="card">
    <div class="card-header">
      <span class="icon">📈</span>
      <h2>Link Prediction Results</h2>
    </div>
    <div class="card-body">
      <table class="results-table" id="lp-table">
        <thead>
          <tr><th>Method</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr>
        </thead>
        <tbody id="lp-table-body"><tr><td colspan="5" style="color:var(--muted);text-align:center">Loading...</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- Model Info -->
  <div class="card" style="grid-column:1/-1">
    <div class="card-header">
      <span class="icon">🧠</span>
      <h2>Model Architecture & Dataset Info</h2>
    </div>
    <div class="card-body" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px">
      <div>
        <div style="font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px">GCN Architecture</div>
        <div class="info-pair"><span class="info-key">Type</span><span class="info-val">2-Layer GCN</span></div>
        <div class="info-pair"><span class="info-key">Framework</span><span class="info-val">Pure NumPy</span></div>
        <div class="info-pair"><span class="info-key">Hidden dim</span><span class="info-val">64</span></div>
        <div class="info-pair"><span class="info-key">Dropout</span><span class="info-val">0.5</span></div>
        <div class="info-pair"><span class="info-key">Optimizer</span><span class="info-val">Adam</span></div>
        <div class="info-pair"><span class="info-key">Norm</span><span class="info-val">D⁻½(A+I)D⁻½</span></div>
      </div>
      <div>
        <div style="font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px">Cora Dataset</div>
        <div class="info-pair"><span class="info-key">Nodes</span><span class="info-val">2,708</span></div>
        <div class="info-pair"><span class="info-key">Edges</span><span class="info-val">5,429</span></div>
        <div class="info-pair"><span class="info-key">Features</span><span class="info-val">1,433 (BoW)</span></div>
        <div class="info-pair"><span class="info-key">Classes</span><span class="info-val">7 topics</span></div>
        <div class="info-pair"><span class="info-key">Train split</span><span class="info-val">20 per class</span></div>
        <div class="info-pair"><span class="info-key">Homophily</span><span class="info-val">~0.85</span></div>
      </div>
      <div>
        <div style="font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px">Class Labels</div>
        {% for c in classes %}
        <span class="tag">{{ c }}</span>
        {% endfor %}
      </div>
    </div>
  </div>

</div>

<script>
const CLASSES = {{ classes|tojson }};
const COLORS  = ['#4e7cff','#7c4eff','#38d9a9','#ffa94d','#ff6b6b','#74c0fc','#a9e34b'];

async function classifyNode() {
  const nodeId = parseInt(document.getElementById('node-id').value);
  const errEl  = document.getElementById('nc-error');
  errEl.textContent = '';
  if (isNaN(nodeId) || nodeId < 0 || nodeId > 2707) {
    errEl.textContent = 'Node ID must be between 0 and 2707.'; return;
  }
  const btn = document.querySelector('#nc-result').previousElementSibling;
  btn.innerHTML = '<span class="spinner"></span> Running...';
  btn.disabled = true;

  try {
    const res  = await fetch('/classify', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({node_id: nodeId})});
    const data = await res.json();
    if (data.error) { errEl.textContent = data.error; return; }

    document.getElementById('nc-pred').textContent = data.gcn_pred;
    document.getElementById('nc-conf').textContent = `GCN Confidence: ${(data.gcn_probs[data.gcn_idx]*100).toFixed(1)}%  |  True label: ${data.true_label}`;

    // Probability bars
    let barsHTML = '';
    const sorted = data.gcn_probs.map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p);
    for (const {p, i} of sorted) {
      barsHTML += `<div class="prob-row">
        <div class="prob-name">${CLASSES[i]}</div>
        <div class="prob-bar-wrap"><div class="prob-bar" style="width:${(p*100).toFixed(1)}%;background:${COLORS[i]}"></div></div>
        <div class="prob-val">${(p*100).toFixed(1)}%</div>
      </div>`;
    }
    document.getElementById('nc-bars').innerHTML = barsHTML;

    // Methods grid
    const methods = [
      {name:'GCN', pred: data.gcn_pred, color:'var(--accent)'},
      {name:'Label Prop', pred: data.lp_pred, color:'var(--green)'},
      {name:'Log. Reg.', pred: data.lr_pred, color:'var(--orange)'},
    ];
    document.getElementById('nc-methods').innerHTML = methods.map(m => `
      <div class="method-card">
        <div class="method-name">${m.name}</div>
        <div class="method-pred" style="color:${m.color}">${m.pred}</div>
      </div>`).join('');

    document.getElementById('nc-result').classList.add('show');
  } catch(e) {
    errEl.textContent = 'Request failed.';
  } finally {
    btn.innerHTML = '🚀 Classify Node'; btn.disabled = false;
  }
}

async function predictLink() {
  const src = parseInt(document.getElementById('lp-src').value);
  const dst = parseInt(document.getElementById('lp-dst').value);
  const errEl = document.getElementById('lp-error');
  errEl.textContent = '';
  if (isNaN(src)||isNaN(dst)||src<0||src>2707||dst<0||dst>2707) {
    errEl.textContent = 'Both node IDs must be between 0 and 2707.'; return;
  }
  if (src === dst) { errEl.textContent = 'Source and target must differ.'; return; }

  const btn = document.querySelector('#lp-result').previousElementSibling;
  btn.innerHTML = '<span class="spinner"></span> Running...';
  btn.disabled = true;

  try {
    const res  = await fetch('/predict_link', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({src, dst})});
    const data = await res.json();
    if (data.error) { errEl.textContent = data.error; return; }

    const exists = data.link_exists;
    const dot = document.getElementById('lp-dot');
    dot.className = 'link-dot ' + (exists ? 'link-yes' : 'link-no');
    document.getElementById('lp-verdict').textContent = exists
      ? '✅ Link Likely Exists'  : '❌ Link Likely Does Not Exist';
    document.getElementById('lp-score').innerHTML =
      `GCN+LR probability: <b>${(data.gcn_prob*100).toFixed(1)}%</b> &nbsp;|&nbsp; `+
      `Actual: <b>${data.actual_link ? 'Yes' : 'No'}</b>`;

    document.getElementById('lp-methods').innerHTML = [
      {name:'GCN + LR', val: `${(data.gcn_prob*100).toFixed(1)}%`},
      {name:'Adamic-Adar', val: data.aa_score.toFixed(3)},
      {name:'Common N.', val: data.cn_score},
    ].map(m => `<div class="method-card"><div class="method-name">${m.name}</div>
       <div class="method-pred" style="color:var(--accent)">${m.val}</div></div>`).join('');

    document.getElementById('lp-result').classList.add('show');
  } catch(e) {
    errEl.textContent = 'Request failed.';
  } finally {
    btn.innerHTML = '🔍 Predict Link'; btn.disabled = false;
  }
}

// Load summary tables
async function loadSummary() {
  try {
    const res  = await fetch('/summary');
    const data = await res.json();

    document.getElementById('s-acc').textContent = (data.test_acc * 100).toFixed(1) + '%';

    // NC table
    const NC_LABELS = {logistic_regression:'Logistic Regression', label_propagation:'Label Propagation', gcn:'GCN (ours)'};
    let best_nc_f1 = 0;
    const nc_rows = data.nc_results;
    for (const m of Object.keys(nc_rows)) if (nc_rows[m].macro_f1 > best_nc_f1) best_nc_f1 = nc_rows[m].macro_f1;
    document.getElementById('nc-table-body').innerHTML = Object.entries(nc_rows).map(([m, r]) =>
      `<tr class="${r.macro_f1===best_nc_f1?'best':''}">
        <td>${NC_LABELS[m]||m}</td>
        <td class="metric-val">${(r.accuracy*100).toFixed(1)}%</td>
        <td class="metric-val">${r.macro_precision.toFixed(4)}</td>
        <td class="metric-val">${r.macro_recall.toFixed(4)}</td>
        <td class="metric-val">${r.macro_f1.toFixed(4)}</td>
      </tr>`).join('');

    // LP table
    const LP_LABELS = {common_neighbors:'Common Neighbors', adamic_adar:'Adamic-Adar', jaccard:'Jaccard', gcn_lr:'GCN + LR (ours)'};
    let best_lp_auc = 0;
    const lp_rows = data.lp_results;
    for (const m of Object.keys(lp_rows)) if (lp_rows[m].auc > best_lp_auc) best_lp_auc = lp_rows[m].auc;
    document.getElementById('lp-table-body').innerHTML = Object.entries(lp_rows).map(([m, r]) =>
      `<tr class="${r.auc===best_lp_auc?'best':''}">
        <td>${LP_LABELS[m]||m}</td>
        <td class="metric-val">${r.precision.toFixed(4)}</td>
        <td class="metric-val">${r.recall.toFixed(4)}</td>
        <td class="metric-val">${r.f1.toFixed(4)}</td>
        <td class="metric-val">${r.auc.toFixed(4)}</td>
      </tr>`).join('');
  } catch(e) { console.error('Summary load failed', e); }
}
loadSummary();
</script>
</body>
</html>
"""

# ── Routes ─────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML, classes=CLASS_NAMES)


@app.route('/summary')
def summary():
    nc_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'node_classification_results.json')
    lp_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'link_prediction_results.json')
    nc_results, lp_results = {}, {}
    if os.path.exists(nc_path):
        with open(nc_path) as f: nc_results = json.load(f)
    if os.path.exists(lp_path):
        with open(lp_path) as f: lp_results = json.load(f)
    return jsonify({'test_acc': float(STATE.get('test_acc', 0)), 'nc_results': nc_results, 'lp_results': lp_results})


@app.route('/classify', methods=['POST'])
def classify():
    data    = request.get_json()
    node_id = int(data.get('node_id', 0))
    if node_id < 0 or node_id >= STATE['n_nodes']:
        return jsonify({'error': f'Node ID out of range (0–{STATE["n_nodes"]-1})'})

    labels    = STATE['labels']
    gcn_model = STATE['gcn_model']
    A_norm    = STATE['A_norm']
    X_dense   = STATE['X_dense']
    A_full    = STATE['A_full']

    # GCN prediction
    Z     = STATE['Z']
    probs = Z[node_id].tolist()
    pred_gcn = int(np.argmax(Z[node_id]))

    # Label Propagation
    from node_classification import label_propagation
    lp_preds, _ = label_propagation(A_full, labels, STATE['train_idx'], 7)
    pred_lp = int(lp_preds[node_id])

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize
    X_norm = normalize(X_dense, norm='l1')
    clf_lr = LogisticRegression(max_iter=300, C=1.0)
    clf_lr.fit(X_norm[STATE['train_idx']], labels[STATE['train_idx']])
    pred_lr = int(clf_lr.predict(X_norm[node_id:node_id+1])[0])

    return jsonify({
        'node_id':    node_id,
        'true_label': CLASS_NAMES[labels[node_id]],
        'gcn_pred':   CLASS_NAMES[pred_gcn],
        'gcn_probs':  probs,
        'gcn_idx':    pred_gcn,
        'lp_pred':    CLASS_NAMES[pred_lp],
        'lr_pred':    CLASS_NAMES[pred_lr],
    })


@app.route('/predict_link', methods=['POST'])
def predict_link():
    data = request.get_json()
    src  = int(data.get('src', 0))
    dst  = int(data.get('dst', 1))
    N    = STATE['n_nodes']
    if src < 0 or src >= N or dst < 0 or dst >= N:
        return jsonify({'error': f'Node IDs must be between 0 and {N-1}'})

    A_full  = STATE['A_full']
    emb     = STATE['embeddings']
    clf     = STATE['link_clf']
    edges   = STATE['edges']
    edge_set = set(map(tuple, [( min(u,v), max(u,v)) for u,v in edges]))

    # GCN + LR
    pair = np.array([[src, dst]])
    gcn_prob = float(predict_link_gcn(clf, emb, pair)[0])
    link_exists = gcn_prob >= 0.5

    # Adamic-Adar
    from link_prediction import adamic_adar_score, common_neighbors_score
    aa_score = adamic_adar_score(A_full, src, dst)
    cn_score = common_neighbors_score(A_full, src, dst)

    actual = (min(src,dst), max(src,dst)) in edge_set

    return jsonify({
        'src': src, 'dst': dst,
        'gcn_prob': gcn_prob,
        'link_exists': link_exists,
        'aa_score': aa_score,
        'cn_score': int(cn_score),
        'actual_link': actual,
    })


if __name__ == '__main__':
    startup()
    app.run(host='0.0.0.0', port=5050, debug=False)
