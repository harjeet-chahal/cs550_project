"""
CS550 Project — Robustness experiments
======================================

Evaluate how Cora node-classification and link-prediction performance
change when the input graph and features are perturbed.

Perturbations
-------------
1. Random EDGE REMOVAL          — drop x% of training edges.
2. Random FAKE EDGE ADDITION    — inject x% non-existent edges.
3. FEATURE NOISE                — flip x% of binary feature entries (0↔1).

For each perturbation type and rate ∈ {5%, 10%, 20%}:
  - retrain the node-classification GCN on the perturbed full graph,
  - retrain the link-prediction GCN on a perturbed train edge set
    (val / test edges are FIXED across rates, so F1 changes reflect
    the perturbation alone — not split variance),
  - report Precision / Recall / F1 for both tasks
  - and `robustness_drop = baseline_f1 − perturbed_f1`.

Reproducibility
---------------
All perturbation samplers take a `seed` argument; the orchestrator
re-seeds NumPy and `random` before each GCN training so every rate
sees the same Glorot init and the only varying input is the edges /
features. None of the perturbation functions mutate their inputs —
they always return a fresh array / sparse matrix / edge list.

Style note
----------
This module deliberately stays in the project's NumPy / scipy /
sklearn lane. No PyTorch, no PyG, no DGL.
"""

import os
import json
import random
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support

from data_preprocessing import (
    build_adjacency, normalize_features, normalize_adjacency,
    split_edges_link_prediction,
)
from gcn_model import train_gcn
from link_prediction import run_link_prediction


# ─────────────────────────────────────────────────────────────────
# Perturbation samplers (stateless, reproducible, non-mutating)
# ─────────────────────────────────────────────────────────────────

def remove_random_edges(edges, removal_rate, seed=42):
    """
    Return a copy of `edges` with a random `removal_rate` fraction
    of rows removed. The original `edges` array is NOT mutated.

    `removal_rate` of 0 returns a copy of all edges; rates ≥ 1
    return an empty edge list.
    """
    rng = np.random.default_rng(seed)
    edges = np.asarray(edges)
    n = len(edges)
    if n == 0 or removal_rate <= 0:
        return edges.copy()
    n_remove = int(round(n * removal_rate))
    if n_remove >= n:
        return np.empty((0, 2), dtype=edges.dtype)
    keep = rng.permutation(n)[n_remove:]
    return edges[keep].copy()


def add_random_fake_edges(edges, num_nodes, add_rate, seed=42):
    """
    Append `add_rate * len(edges)` brand-new edges sampled uniformly
    over node pairs (u, v), with u ≠ v and (u, v) not already in the
    graph. Returns a NEW edge array; `edges` is not mutated.

    Defensive cap on attempts so a saturated graph cannot loop forever.
    """
    rng = np.random.default_rng(seed)
    edges = np.asarray(edges)
    n_add = int(round(len(edges) * add_rate))
    if n_add <= 0:
        return edges.copy()

    # Canonical undirected key set for fast membership checks.
    edge_set = set(
        (int(min(u, v)), int(max(u, v))) for u, v in edges
    )
    fakes, attempts, max_attempts = [], 0, max(n_add * 100, 10_000)
    while len(fakes) < n_add and attempts < max_attempts:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        attempts += 1
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in edge_set:
            continue
        edge_set.add(key)
        fakes.append([u, v])

    if not fakes:
        return edges.copy()
    fake_arr = np.array(fakes, dtype=edges.dtype)
    return np.vstack([edges.copy(), fake_arr])


def flip_feature_noise(features, noise_rate, seed=42):
    """
    Flip `noise_rate` × N×F binary feature entries (0 → 1, 1 → 0).
    Accepts either a sparse CSR or a dense ndarray; always returns
    a NEW sparse CSR matrix. Original `features` is not mutated.

    Cora's feature matrix is binary BoW — flipping is meaningful
    here. For non-binary input, the value is replaced by `1 - x`
    which still toggles {0, 1} values correctly and would compute
    `1 - x` for arbitrary floats (rarely the desired semantics; we
    document this limitation).
    """
    rng = np.random.default_rng(seed)
    if hasattr(features, 'toarray'):
        X = features.toarray().astype(np.float32)
    else:
        X = np.asarray(features, dtype=np.float32).copy()

    n_total = X.size
    n_flip = int(round(n_total * noise_rate))
    if n_flip <= 0:
        return sp.csr_matrix(X)

    # Sample positions without replacement (cheap for Cora-scale
    # tensors at the rates used here).
    flat_idx = rng.choice(n_total, size=n_flip, replace=False)
    rows, cols = np.unravel_index(flat_idx, X.shape)
    X[rows, cols] = 1.0 - X[rows, cols]
    return sp.csr_matrix(X)


# ─────────────────────────────────────────────────────────────────
# Re-evaluation helpers (one (edges, features) → one set of numbers)
# ─────────────────────────────────────────────────────────────────

def _set_seed(seed):
    """Re-seed both numpy and random so GCN init / shuffles are
    deterministic across rates."""
    np.random.seed(seed)
    random.seed(seed)


def _eval_node_classification(edges, features, labels,
                              train_idx, val_idx, test_idx, seed=42):
    """Train a fresh NC-GCN on the (possibly perturbed) full graph
    and return macro precision / recall / F1 on test_idx."""
    _set_seed(seed)
    N = len(labels)
    A = build_adjacency(edges, N) if len(edges) > 0 else sp.csr_matrix((N, N))
    feats_norm = normalize_features(features)
    X = feats_norm.toarray()
    A_n = normalize_adjacency(A)

    model, _ = train_gcn(
        A_n, X, labels, train_idx, val_idx, test_idx,
        n_hidden=64, lr=0.01, epochs=300,
        weight_decay=5e-4, dropout=0.5, patience=30, verbose=False,
    )
    preds, _ = model.predict(A_n, X)
    p, r, f1, _ = precision_recall_fscore_support(
        labels[test_idx], preds[test_idx], average='macro', zero_division=0,
    )
    return {'precision': float(p), 'recall': float(r), 'f1': float(f1)}


def _eval_link_prediction(train_pos, link_splits, features, labels,
                          train_idx, val_idx, test_idx, seed=42):
    """
    Train a fresh LP-only GCN on the (possibly perturbed) train edges
    and report the GCN+LR head-of-method metrics. The val / test
    positive / negative pairs in `link_splits` are kept FIXED across
    rates so the perturbation is the only varying input.
    """
    _set_seed(seed)
    N = len(labels)
    feats_norm = normalize_features(features)
    X = feats_norm.toarray()

    A_train = build_adjacency(train_pos, N) if len(train_pos) > 0 else sp.csr_matrix((N, N))
    A_norm_train = normalize_adjacency(A_train)

    gcn_lp, _ = train_gcn(
        A_norm_train, X, labels, train_idx, val_idx, test_idx,
        n_hidden=64, lr=0.01, epochs=300,
        weight_decay=5e-4, dropout=0.5, patience=30, verbose=False,
    )

    splits = {
        'train':   {'pos': train_pos, 'neg': link_splits['train']['neg']},
        'val':     link_splits['val'],
        'test':    link_splits['test'],
        'train_A': A_train,
    }
    res = run_link_prediction(
        A_train, splits,
        gcn_model=gcn_lp, A_norm_train=A_norm_train, X_dense=X,
    )
    g = res['gcn_lr']
    return {
        'precision': float(g['precision']),
        'recall':    float(g['recall']),
        'f1':        float(g['f1']),
        'auc':       float(g['auc']),
    }


# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

def run_robustness_experiments(labels, edges, features,
                               train_idx, val_idx, test_idx,
                               rates=(0.05, 0.10, 0.20),
                               seed=42, save_dir=None, verbose=True):
    """
    Run all three perturbations at every rate. Returns a JSON-friendly
    dict and (optionally) writes `robustness_results.json` and
    `robustness_comparison.png` into `save_dir`.

    The link-prediction val/test split is built ONCE here and reused
    for every rate, so LP F1 changes reflect the perturbation alone.
    """
    N = len(labels)

    # Fixed LP split shared across all rates.
    _set_seed(seed)
    base_splits = split_edges_link_prediction(
        edges, N, test_ratio=0.2, val_ratio=0.1,
    )
    base_train_pos = base_splits['train']['pos']

    if verbose:
        print("[Robustness] Computing baseline (no perturbation)...")
    baseline = {
        'nc': _eval_node_classification(
            edges, features, labels, train_idx, val_idx, test_idx, seed=seed,
        ),
        'lp': _eval_link_prediction(
            base_train_pos, base_splits, features, labels,
            train_idx, val_idx, test_idx, seed=seed,
        ),
    }
    base_nc_f1 = baseline['nc']['f1']
    base_lp_f1 = baseline['lp']['f1']
    if verbose:
        print(f"  Baseline → NC F1={base_nc_f1:.4f}  LP F1={base_lp_f1:.4f}")

    results = {
        'baseline':      baseline,
        'edge_removal':  {},
        'edge_addition': {},
        'feature_noise': {},
    }

    def _record(bucket, rate, nc, lp):
        results[bucket][f'{int(round(rate * 100))}%'] = {
            'rate': float(rate),
            'nc':   nc,
            'lp':   lp,
            'nc_robustness_drop': float(base_nc_f1 - nc['f1']),
            'lp_robustness_drop': float(base_lp_f1 - lp['f1']),
        }

    for rate in rates:
        # ── 1. Edge removal ──────────────────────────────────
        if verbose:
            print(f"[Robustness] Edge removal {rate * 100:.0f}%...")
        edges_pruned     = remove_random_edges(edges, rate, seed=seed)
        train_pos_pruned = remove_random_edges(base_train_pos, rate, seed=seed)
        nc = _eval_node_classification(
            edges_pruned, features, labels,
            train_idx, val_idx, test_idx, seed=seed,
        )
        lp = _eval_link_prediction(
            train_pos_pruned, base_splits, features, labels,
            train_idx, val_idx, test_idx, seed=seed,
        )
        _record('edge_removal', rate, nc, lp)
        if verbose:
            print(f"    NC F1={nc['f1']:.4f} (Δ={base_nc_f1-nc['f1']:+.4f})  "
                  f"LP F1={lp['f1']:.4f} (Δ={base_lp_f1-lp['f1']:+.4f})")

        # ── 2. Fake edge addition ────────────────────────────
        if verbose:
            print(f"[Robustness] Fake edge addition {rate * 100:.0f}%...")
        edges_added     = add_random_fake_edges(edges, N, rate, seed=seed)
        train_pos_added = add_random_fake_edges(base_train_pos, N, rate, seed=seed)
        nc = _eval_node_classification(
            edges_added, features, labels,
            train_idx, val_idx, test_idx, seed=seed,
        )
        lp = _eval_link_prediction(
            train_pos_added, base_splits, features, labels,
            train_idx, val_idx, test_idx, seed=seed,
        )
        _record('edge_addition', rate, nc, lp)
        if verbose:
            print(f"    NC F1={nc['f1']:.4f} (Δ={base_nc_f1-nc['f1']:+.4f})  "
                  f"LP F1={lp['f1']:.4f} (Δ={base_lp_f1-lp['f1']:+.4f})")

        # ── 3. Feature noise ─────────────────────────────────
        if verbose:
            print(f"[Robustness] Feature noise {rate * 100:.0f}%...")
        features_noisy = flip_feature_noise(features, rate, seed=seed)
        nc = _eval_node_classification(
            edges, features_noisy, labels,
            train_idx, val_idx, test_idx, seed=seed,
        )
        # Also probe LP under the same noisy features (the LP-GCN reads
        # X too, so this is informative even though the spec calls out NC).
        lp = _eval_link_prediction(
            base_train_pos, base_splits, features_noisy, labels,
            train_idx, val_idx, test_idx, seed=seed,
        )
        _record('feature_noise', rate, nc, lp)
        if verbose:
            print(f"    NC F1={nc['f1']:.4f} (Δ={base_nc_f1-nc['f1']:+.4f})  "
                  f"LP F1={lp['f1']:.4f} (Δ={base_lp_f1-lp['f1']:+.4f})")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, 'robustness_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        plot_path = os.path.join(save_dir, 'robustness_comparison.png')
        plot_robustness(results, plot_path)
        if verbose:
            print(f"[Robustness] Saved → {json_path}")
            print(f"[Robustness] Saved → {plot_path}")

    return results


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────

def plot_robustness(results, save_path):
    """Three-panel line plot: NC F1 and LP F1 vs perturbation rate."""
    panels = [
        ('edge_removal',  'Edge Removal'),
        ('edge_addition', 'Fake Edge Addition'),
        ('feature_noise', 'Feature Noise'),
    ]
    base_nc_f1 = results['baseline']['nc']['f1']
    base_lp_f1 = results['baseline']['lp']['f1']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title) in zip(axes, panels):
        bucket = results[key]
        levels = sorted(bucket.keys(), key=lambda k: bucket[k]['rate'])
        rates_pct = [0.0] + [bucket[k]['rate'] * 100 for k in levels]
        nc_f1     = [base_nc_f1] + [bucket[k]['nc']['f1'] for k in levels]
        lp_f1     = [base_lp_f1] + [bucket[k]['lp']['f1'] for k in levels]

        ax.plot(rates_pct, nc_f1, '-o', color='#4C72B0',
                linewidth=2, markersize=7, label='Node Class. (macro F1)')
        ax.plot(rates_pct, lp_f1, '-o', color='#DD8452',
                linewidth=2, markersize=7, label='Link Pred. (F1)')
        ax.set_xlabel('Perturbation rate (%)', fontsize=11)
        ax.set_ylabel('F1', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='lower left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ─────────────────────────────────────────────────────────────────
# Defense comparison: normal GCN vs edge-dropout-trained GCN
# ─────────────────────────────────────────────────────────────────

def _eval_model(model, A_norm, X_dense, labels, test_idx):
    """Forward through (possibly perturbed) inputs and return P/R/F1."""
    Z = model.forward(A_norm, X_dense, training=False)
    preds = np.argmax(Z, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels[test_idx], preds[test_idx], average='macro', zero_division=0,
    )
    return {'precision': float(p), 'recall': float(r), 'f1': float(f1)}


def run_defense_comparison(labels, edges, features,
                           train_idx, val_idx, test_idx,
                           edge_dropout_rate=0.10, attack_rate=0.10,
                           seed=42, save_dir=None, verbose=True):
    """
    Compare two GCNs under three test-time attacks (10% by default):
      • Normal       — trained on the clean graph (edge_dropout_rate=0).
      • Defended     — trained with DropEdge (edge_dropout_rate=0.10):
                       every epoch a different random fraction of edges
                       is masked out, so the model is regularized
                       against any single edge being decisive.

    Both models are then evaluated on:
      • clean graph,
      • graph with 10% of edges removed,
      • graph with 10% fake edges added,
      • clean graph but with 10% of binary features flipped.

    Results JSON has, for each model:
      clean / edge_removal_<r> / edge_addition_<r> / feature_noise_<r>
      → {precision, recall, f1}, and a `robustness_drops` block giving
      `clean_f1 − attacked_f1` for each attack.
    """
    N = len(labels)

    # ── Shared clean inputs ───────────────────────────────────
    A_full     = build_adjacency(edges, N)
    A_norm     = normalize_adjacency(A_full)
    feats_norm = normalize_features(features)
    X_clean    = feats_norm.toarray()

    # ── Train normal GCN (rate=0; identical to main pipeline) ─
    if verbose:
        print("[Defense] Training NORMAL GCN (edge_dropout_rate=0.0)...")
    _set_seed(seed)
    normal_model, _ = train_gcn(
        A_norm, X_clean, labels, train_idx, val_idx, test_idx,
        n_hidden=64, lr=0.01, epochs=300,
        weight_decay=5e-4, dropout=0.5, patience=30, verbose=False,
        edge_dropout_rate=0.0,
    )

    # ── Train defended GCN with edge dropout ──────────────────
    if verbose:
        print(f"[Defense] Training DEFENDED GCN "
              f"(edge_dropout_rate={edge_dropout_rate})...")
    _set_seed(seed)
    defended_model, _ = train_gcn(
        A_norm, X_clean, labels, train_idx, val_idx, test_idx,
        n_hidden=64, lr=0.01, epochs=300,
        weight_decay=5e-4, dropout=0.5, patience=30, verbose=False,
        edge_dropout_rate=edge_dropout_rate, A_raw=A_full,
        edge_dropout_seed=seed,
    )

    # ── Pre-build attacked inputs once (shared by both models) ─
    edges_removed  = remove_random_edges(edges, attack_rate, seed=seed)
    edges_added    = add_random_fake_edges(edges, N, attack_rate, seed=seed)
    features_noisy = flip_feature_noise(features, attack_rate, seed=seed)

    A_norm_removed = normalize_adjacency(build_adjacency(edges_removed, N))
    A_norm_added   = normalize_adjacency(build_adjacency(edges_added, N))
    X_noisy        = normalize_features(features_noisy).toarray()

    rate_tag = f'{int(round(attack_rate * 100))}'
    conditions = [
        ('clean',                    A_norm,         X_clean),
        (f'edge_removal_{rate_tag}', A_norm_removed, X_clean),
        (f'edge_addition_{rate_tag}',A_norm_added,   X_clean),
        (f'feature_noise_{rate_tag}',A_norm,         X_noisy),
    ]

    def evaluate_all(model, tag):
        out = {}
        for name, A_n, X in conditions:
            out[name] = _eval_model(model, A_n, X, labels, test_idx)
            if verbose:
                print(f"  [{tag:<8}] {name:<22} F1={out[name]['f1']:.4f}")
        clean_f1 = out['clean']['f1']
        out['robustness_drops'] = {
            name: float(clean_f1 - out[name]['f1'])
            for name, _, _ in conditions if name != 'clean'
        }
        return out

    normal_results   = evaluate_all(normal_model,   'normal')
    defended_results = evaluate_all(defended_model, 'defended')

    out = {
        'edge_dropout_rate': float(edge_dropout_rate),
        'attack_rate':       float(attack_rate),
        'seed':              int(seed),
        'normal':            normal_results,
        'defended':          defended_results,
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, 'defense_comparison.json')
        with open(json_path, 'w') as f:
            json.dump(out, f, indent=2)
        plot_path = os.path.join(save_dir, 'defense_comparison.png')
        plot_defense_comparison(out, plot_path)
        if verbose:
            print(f"[Defense] Saved → {json_path}")
            print(f"[Defense] Saved → {plot_path}")

    return out


def plot_defense_comparison(results, save_path):
    """Grouped bar chart: normal vs defended F1 across the four conditions."""
    rate_tag = f'{int(round(results["attack_rate"] * 100))}'
    conditions = [
        ('clean',                     'Clean'),
        (f'edge_removal_{rate_tag}',  f'Edge removal {rate_tag}%'),
        (f'edge_addition_{rate_tag}', f'Fake edges {rate_tag}%'),
        (f'feature_noise_{rate_tag}', f'Feature noise {rate_tag}%'),
    ]
    keys = [c[0] for c in conditions]
    xticks = [c[1] for c in conditions]
    normal_f1   = [results['normal'][k]['f1']   for k in keys]
    defended_f1 = [results['defended'][k]['f1'] for k in keys]

    x = np.arange(len(keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, normal_f1, width,
                   color='#4C72B0', label='Normal GCN', alpha=0.9)
    bars2 = ax.bar(x + width / 2, defended_f1, width,
                   color='#55A868',
                   label=f"Defended (DropEdge {results['edge_dropout_rate']*100:.0f}%)",
                   alpha=0.9)

    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(xticks, fontsize=10)
    ax.set_ylabel('Macro F1', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('GCN Robustness: Normal vs Edge-Dropout-Trained',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=10, loc='lower left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path
