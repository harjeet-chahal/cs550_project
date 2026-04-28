"""
CS550 Project Demo - Cora Citation Network Explorer
Web application: node classification & link prediction inference
"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np
import scipy.sparse as sp
import sys, os, json, pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import (load_cora, build_adjacency, normalize_features,
                                 normalize_adjacency)
from gcn_model import GCN
from link_prediction import predict_link_gcn
from explainability import explain_node_prediction, explain_link_prediction

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

app = Flask(__name__)

# ── Global state (loaded once at startup) ─────────────────────
STATE = {}

CLASS_NAMES = ['Case_Based','Genetic_Algorithms','Neural_Networks',
               'Probabilistic_Methods','Reinforcement_Learning',
               'Rule_Learning','Theory']

REQUIRED_ARTIFACTS = {
    'state':      'demo_state.pkl',
    'splits':     'splits.npz',
    'gcn':        'gcn_weights.npz',
    'embeddings': 'gcn_embeddings.npy',
    'probs':      'node_probabilities.npy',
    'link_clf':   'link_classifier.pkl',
    'lp_preds':   'lp_predictions.npy',
    'lr_preds':   'lr_predictions.npy',
}


def _check_artifacts():
    """Return ([] | [missing_paths]). Empty list ⇒ all good."""
    paths = {k: os.path.join(RESULTS_DIR, fname)
             for k, fname in REQUIRED_ARTIFACTS.items()}
    missing = [(k, p) for k, p in paths.items() if not os.path.exists(p)]
    return paths, missing


def startup():
    """
    Load all artifacts produced by `python src/main.py`. The demo does
    NOT retrain anything: weights are loaded into reconstructed GCN
    objects, embeddings / probabilities / baseline predictions come from
    saved NumPy arrays, and the sklearn link classifier is unpickled.

    If any required artifact is missing we exit immediately with a
    clear, actionable error message.
    """
    paths, missing = _check_artifacts()
    if missing:
        print("\n[Demo] ERROR — required trained artifacts are missing:")
        for _, p in missing:
            print(f"          {p}")
        print("\n  Train and save them first:")
        print("          python src/main.py\n")
        sys.exit(1)

    print("[Demo] Loading saved artifacts (no retraining)...")

    # Cora data is fast to reload and stays the source of truth for
    # labels / edges / raw features (none of which we persist).
    labels, edges, features = load_cora()
    N = len(labels)
    A_full        = build_adjacency(edges, N)
    features_norm = normalize_features(features)
    X_dense       = features_norm.toarray()
    A_norm_full   = normalize_adjacency(A_full)

    # ── Metadata + splits ────────────────────────────────────
    with open(paths['state'], 'rb') as f:
        meta = pickle.load(f)
    splits = np.load(paths['splits'])
    train_idx = splits['train_idx']
    val_idx   = splits['val_idx']
    test_idx  = splits['test_idx']
    train_pos = splits['train_pos']

    # ── GCN backbones reconstructed from saved weights ───────
    weights = np.load(paths['gcn'])
    gcn_model = GCN(meta['n_features'], meta['n_hidden'], meta['n_classes'])
    gcn_model.W0 = weights['W0']
    gcn_model.W1 = weights['W1']
    gcn_lp_model = GCN(meta['n_features'], meta['n_hidden'], meta['n_classes'])
    gcn_lp_model.W0 = weights['W0_lp']
    gcn_lp_model.W1 = weights['W1_lp']

    # ── Per-node arrays ──────────────────────────────────────
    Z             = np.load(paths['probs'])
    emb           = np.load(paths['embeddings'])
    lp_preds_full = np.load(paths['lp_preds'])
    lr_preds_full = np.load(paths['lr_preds'])

    # ── sklearn link classifier ──────────────────────────────
    with open(paths['link_clf'], 'rb') as f:
        link_clf = pickle.load(f)

    # Build A_train from the saved train_pos (cheap; not persisted as
    # a sparse matrix to keep things in pure NumPy / scipy.sparse).
    A_train      = build_adjacency(train_pos, N)
    A_norm_train = normalize_adjacency(A_train)

    preds_full = np.argmax(Z, axis=1)
    test_acc   = float(np.mean(preds_full[test_idx] == labels[test_idx]))
    print(f"[Demo] GCN Test Accuracy (from saved Z): {test_acc:.4f}")
    print(f"[Demo] LP threshold (from saved state):  {meta['link_threshold']:.4f}")

    # `n_edges_unique` is `len(edges)` after load_cora dedups
    # (A→B, B→A) pairs and exact duplicates from cora.cites. The raw
    # count comes from main.py's saved state (5,429 in stock Cora).
    n_edges_unique = len(edges)
    n_edges_raw    = int(meta.get('n_edges_raw', n_edges_unique))

    STATE.update({
        'labels': labels, 'edges': edges, 'X_dense': X_dense,
        'A_full': A_full, 'A_norm': A_norm_full,
        'A_train': A_train, 'A_norm_train': A_norm_train,
        'gcn_model': gcn_model, 'gcn_lp_model': gcn_lp_model,
        'Z': Z, 'preds': preds_full,
        'embeddings': emb, 'link_clf': link_clf,
        'link_threshold': float(meta['link_threshold']),
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
        'test_acc': test_acc,
        'n_nodes': N,
        'n_edges_unique': n_edges_unique,
        'n_edges_raw':    n_edges_raw,
        'lp_preds_full': lp_preds_full,
        'lr_preds_full': lr_preds_full,
    })
    print(f"[Demo] Edges: {n_edges_raw} raw citations → {n_edges_unique} unique undirected edges.")
    print("[Demo] Ready (loaded from disk, no retraining).")

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
  <div class="stat-item"><div class="stat-val" id="s-nodes">2,708</div><div class="stat-label">Nodes (Papers)</div></div>
  <div class="stat-item"><div class="stat-val" id="s-edges-unique">5,278</div><div class="stat-label">Unique Edges Used</div></div>
  <div class="stat-item"><div class="stat-val" id="s-edges-raw">5,429</div><div class="stat-label">Raw Citations</div></div>
  <div class="stat-item"><div class="stat-val">7</div><div class="stat-label">Classes</div></div>
  <div class="stat-item"><div class="stat-val">1,433</div><div class="stat-label">Features</div></div>
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

        <div style="margin-top:18px;padding-top:16px;border-top:1px solid var(--border);font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;">Prediction Explanation</div>
        <div id="nc-explain-text" style="margin-top:8px;font-size:13px;line-height:1.5;color:var(--text);"></div>

        <div style="margin-top:14px;font-size:11px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;">Top Influential Neighbors</div>
        <table class="results-table" id="nc-neighbors">
          <thead><tr><th>Node</th><th>Class</th><th>Δ Conf</th></tr></thead>
          <tbody></tbody>
        </table>

        <div style="margin-top:14px;font-size:11px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;">Top Important Features</div>
        <table class="results-table" id="nc-features">
          <thead><tr><th>Feature index</th><th>Δ Conf</th></tr></thead>
          <tbody></tbody>
        </table>
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
        <div class="methods-grid" id="lp-methods" style="margin-top:14px;grid-template-columns:1fr 1fr 1fr 1fr;"></div>

        <div style="margin-top:18px;padding-top:16px;border-top:1px solid var(--border);font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.5px;">Link Explanation</div>
        <div id="lp-explain-text" style="margin-top:8px;font-size:13px;line-height:1.5;color:var(--text);"></div>
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
        <div class="info-pair"><span class="info-key">Raw citations</span><span class="info-val" id="s-info-edges-raw">5,429</span></div>
        <div class="info-pair"><span class="info-key">Unique edges</span><span class="info-val" id="s-info-edges-unique">5,278</span></div>
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

function isIntegerString(s) {
  return /^-?\d+$/.test(String(s).trim());
}

async function classifyNode() {
  const raw    = document.getElementById('node-id').value;
  const errEl  = document.getElementById('nc-error');
  errEl.textContent = '';
  if (!isIntegerString(raw)) {
    errEl.textContent = 'Node ID must be an integer.'; return;
  }
  const nodeId = parseInt(raw, 10);
  if (nodeId < 0 || nodeId > 2707) {
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

    // ── Explanation panel ──
    document.getElementById('nc-explain-text').textContent = data.explanation || '';

    const nbrRows = (data.top_influential_neighbors || []).map(n => `
      <tr>
        <td>${n.neighbor_id}</td>
        <td>${n.neighbor_true_label || '—'}</td>
        <td class="metric-val">${(n.confidence_drop >= 0 ? '+' : '') + n.confidence_drop.toFixed(4)}</td>
      </tr>`).join('');
    document.querySelector('#nc-neighbors tbody').innerHTML =
      nbrRows || '<tr><td colspan="3" style="color:var(--muted);text-align:center">No neighbors</td></tr>';

    const featRows = (data.top_important_features || []).map(f => `
      <tr>
        <td>${f.feature_idx}</td>
        <td class="metric-val">${(f.confidence_drop >= 0 ? '+' : '') + f.confidence_drop.toFixed(4)}</td>
      </tr>`).join('');
    document.querySelector('#nc-features tbody').innerHTML =
      featRows || '<tr><td colspan="2" style="color:var(--muted);text-align:center">No active features</td></tr>';

    document.getElementById('nc-result').classList.add('show');
  } catch(e) {
    errEl.textContent = 'Request failed.';
  } finally {
    btn.innerHTML = '🚀 Classify Node'; btn.disabled = false;
  }
}

async function predictLink() {
  const rawSrc = document.getElementById('lp-src').value;
  const rawDst = document.getElementById('lp-dst').value;
  const errEl  = document.getElementById('lp-error');
  errEl.textContent = '';
  if (!isIntegerString(rawSrc) || !isIntegerString(rawDst)) {
    errEl.textContent = 'Inputs must be integers.'; return;
  }
  const src = parseInt(rawSrc, 10);
  const dst = parseInt(rawDst, 10);
  if (src < 0 || src > 2707 || dst < 0 || dst > 2707) {
    errEl.textContent = 'Both node IDs must be between 0 and 2707.'; return;
  }
  if (src === dst) {
    errEl.textContent = 'Source and target must be different.'; return;
  }

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

    // Borderline = the GCN probability sits within 0.05 of the
    // validation-tuned decision threshold. Show the threshold next
    // to the probability and label borderline cases explicitly.
    const threshold  = (typeof data.threshold === 'number') ? data.threshold : 0.5;
    const margin     = data.gcn_prob - threshold;
    const borderline = Math.abs(margin) < 0.05;
    let verdict;
    if (exists) {
      verdict = borderline ? '⚠️ Predicted Link: Yes (Borderline)'
                           : '✅ Predicted Link: Yes';
    } else {
      verdict = borderline ? '⚠️ Predicted Link: No (Borderline)'
                           : '❌ Predicted Link: No';
    }
    document.getElementById('lp-verdict').textContent = verdict;
    document.getElementById('lp-score').innerHTML =
      `GCN+LR probability: <b>${(data.gcn_prob*100).toFixed(1)}%</b> &nbsp;|&nbsp; `+
      `Decision threshold: <b>${(threshold*100).toFixed(1)}%</b> &nbsp;|&nbsp; `+
      `Actual: <b>${data.actual_link ? 'Yes' : 'No'}</b>`;

    document.getElementById('lp-methods').innerHTML = [
      {name:'GCN + LR',    val: `${(data.gcn_prob*100).toFixed(1)}%`},
      {name:'Adamic-Adar', val: data.aa_score.toFixed(3)},
      {name:'Common N.',   val: data.cn_score},
      {name:'Jaccard',     val: (data.jaccard ?? 0).toFixed(3)},
    ].map(m => `<div class="method-card"><div class="method-name">${m.name}</div>
       <div class="method-pred" style="color:var(--accent)">${m.val}</div></div>`).join('');

    // Embedding cosine + natural-language summary.
    let explanationText = data.explanation || '';
    if (borderline) {
      const direction = margin >= 0 ? 'above' : 'below';
      explanationText += ` The probability is only slightly ${direction} `+
                         `the validation-tuned threshold, so this is a `+
                         `borderline decision.`;
    }
    document.getElementById('lp-explain-text').innerHTML =
      `<div style="margin-bottom:8px;color:var(--muted)">`+
      `Embedding cosine similarity: <b style="color:var(--text)">${(data.embedding_cosine ?? 0).toFixed(3)}</b></div>`+
      explanationText;

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
    const fmt = n => (n || 0).toLocaleString();
    if (data.n_nodes) {
      document.getElementById('s-nodes').textContent = fmt(data.n_nodes);
    }
    if (data.n_edges_unique) {
      document.getElementById('s-edges-unique').textContent      = fmt(data.n_edges_unique);
      document.getElementById('s-info-edges-unique').textContent = fmt(data.n_edges_unique);
    }
    if (data.n_edges_raw) {
      document.getElementById('s-edges-raw').textContent      = fmt(data.n_edges_raw);
      document.getElementById('s-info-edges-raw').textContent = fmt(data.n_edges_raw);
    }

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
    return jsonify({
        'test_acc':       float(STATE.get('test_acc', 0)),
        'n_nodes':        int(STATE.get('n_nodes', 0)),
        'n_edges_raw':    int(STATE.get('n_edges_raw', 0)),
        'n_edges_unique': int(STATE.get('n_edges_unique', 0)),
        'nc_results': nc_results,
        'lp_results': lp_results,
    })


def _coerce_int(value):
    """Best-effort int coercion that rejects floats with fractional parts
    and non-numeric strings. Returns (int, None) or (None, error_msg)."""
    if isinstance(value, bool):
        return None, 'must be an integer'
    if isinstance(value, int):
        return int(value), None
    if isinstance(value, float):
        if not value.is_integer():
            return None, 'must be an integer (no decimals)'
        return int(value), None
    if isinstance(value, str):
        s = value.strip()
        if s and (s.lstrip('-').isdigit()):
            return int(s), None
        return None, 'must be an integer'
    return None, 'must be an integer'


@app.route('/classify', methods=['POST'])
def classify():
    data    = request.get_json(silent=True) or {}
    N       = STATE['n_nodes']

    node_id, err = _coerce_int(data.get('node_id'))
    if err is not None:
        return jsonify({'error': f'Node ID {err}.'})
    if node_id < 0 or node_id >= N:
        return jsonify({'error': f'Node ID must be between 0 and {N - 1}.'})

    labels    = STATE['labels']
    gcn_model = STATE['gcn_model']
    A_norm    = STATE['A_norm']
    A_full    = STATE['A_full']
    X_dense   = STATE['X_dense']

    # GCN softmax row from the cached startup forward.
    Z        = STATE['Z']
    probs    = Z[node_id].tolist()
    pred_gcn = int(np.argmax(Z[node_id]))

    # Cached baseline predictions (computed once at startup).
    pred_lp = int(STATE['lp_preds_full'][node_id])
    pred_lr = int(STATE['lr_preds_full'][node_id])

    # Occlusion-based explanation (top neighbors + features + summary).
    explanation = explain_node_prediction(
        gcn_model, node_id, X_dense,
        adjacency_norm=A_norm, adjacency_raw=A_full,
        labels=labels, class_names=CLASS_NAMES, top_k=5,
    )

    return jsonify({
        'node_id':    node_id,
        'true_label': CLASS_NAMES[int(labels[node_id])],
        'gcn_pred':   CLASS_NAMES[pred_gcn],
        'gcn_probs':  probs,
        'gcn_idx':    pred_gcn,
        'lp_pred':    CLASS_NAMES[pred_lp],
        'lr_pred':    CLASS_NAMES[pred_lr],
        # New explanation fields.
        'confidence':                explanation.get('confidence'),
        'class_probabilities':       explanation.get('class_probabilities', []),
        'top_influential_neighbors': explanation.get('top_influential_neighbors', []),
        'top_important_features':    explanation.get('top_important_features', []),
        'explanation':               explanation.get('explanation', ''),
    })


@app.route('/predict_link', methods=['POST'])
def predict_link():
    data = request.get_json(silent=True) or {}
    N    = STATE['n_nodes']

    src, err_s = _coerce_int(data.get('src'))
    if err_s is not None:
        return jsonify({'error': f'Source ID {err_s}.'})
    dst, err_d = _coerce_int(data.get('dst'))
    if err_d is not None:
        return jsonify({'error': f'Target ID {err_d}.'})
    if src < 0 or src >= N or dst < 0 or dst >= N:
        return jsonify({'error': f'Node IDs must be between 0 and {N - 1}.'})
    if src == dst:
        return jsonify({'error': 'Source and target must be different.'})

    A_train   = STATE['A_train']
    A_full    = STATE['A_full']
    emb       = STATE['embeddings']         # extracted via A_norm_train
    clf       = STATE['link_clf']
    labels    = STATE['labels']
    threshold = STATE.get('link_threshold', 0.5)

    # Delegate everything to the link explainer (heuristics use A_train,
    # full adjacency only used for `edge_exists`).
    expl = explain_link_prediction(
        src, dst,
        embeddings=emb,
        adjacency_train=A_train,
        adjacency_full=A_full,
        link_model=clf,
        class_names=CLASS_NAMES,
        labels=labels,
    )
    if 'error' in expl:
        return jsonify({'error': expl['error']})

    gcn_prob = float(expl['gcn_link_probability'])
    return jsonify({
        'src':              src,
        'dst':              dst,
        'gcn_prob':         gcn_prob,
        # Strict `>` matches the rule used by tune_threshold /
        # evaluate_scores_with_threshold so a probability sitting
        # exactly on the threshold isn't auto-classified as positive.
        'link_exists':      gcn_prob > threshold,
        'threshold':        float(threshold),
        'aa_score':         expl['adamic_adar'],
        'cn_score':         int(expl['common_neighbors']),
        'jaccard':          expl['jaccard'],
        'embedding_cosine': expl['embedding_cosine'],
        'actual_link':      bool(expl['edge_exists']),
        'src_label':        expl.get('src_label'),
        'dst_label':        expl.get('dst_label'),
        'explanation':      expl['explanation'],
    })


if __name__ == '__main__':
    startup()
    app.run(host='0.0.0.0', port=5050, debug=False)
