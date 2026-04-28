"""
CS550 Project - Visualization & Plots
Generates all figures for the project report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.sparse as sp
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = [
    'Case_Based', 'Genetic_Alg', 'Neural_Nets',
    'Prob_Methods', 'RL', 'Rule_Learning', 'Theory'
]
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52',
          '#8172B2', '#937860', '#DA8BC3']


def plot_gcn_training(history, save_path=None):
    """Plot training loss and validation/test accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], color='#4C72B0', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('GCN Training Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['val_acc'],  color='#DD8452', linewidth=2, label='Val Acc')
    ax2.plot(epochs, history['test_acc'], color='#55A868', linewidth=2, label='Test Acc')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('GCN Validation & Test Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, 'gcn_training.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_link_prediction_comparison(link_results, save_path=None):
    """Bar chart comparing link prediction methods."""
    methods = list(link_results.keys())
    method_labels = {
        'common_neighbors': 'Common\nNeighbors',
        'adamic_adar':      'Adamic-Adar',
        'jaccard':          'Jaccard',
        'gcn_lr':           'GCN + LR',
    }

    metrics = ['precision', 'recall', 'f1', 'auc']
    metric_labels = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    x = np.arange(len(methods))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        vals = [link_results[m][metric] for m in methods]
        bars = ax.bar(x + i * width, vals, width, label=mlabel, color=COLORS[i], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([method_labels.get(m, m) for m in methods], fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title('Link Prediction: Method Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, 'link_prediction_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_node_classification_comparison(nc_results, save_path=None):
    """Bar chart comparing node classification methods."""
    methods = list(nc_results.keys())
    method_labels = {
        'logistic_regression': 'Logistic\nRegression',
        'label_propagation':   'Label\nPropagation',
        'gcn':                 'GCN',
    }

    metrics = ['accuracy', 'macro_p', 'macro_r', 'macro_f1']
    metric_labels = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']

    x = np.arange(len(methods))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        vals = [nc_results[m][metric] for m in methods]
        ax.bar(x + i * width, vals, width, label=mlabel, color=COLORS[i], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([method_labels.get(m, m) for m in methods], fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title('Node Classification: Method Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, 'node_classification_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_confusion_matrix(cm, title='Confusion Matrix', save_path=None):
    """Plot normalized confusion matrix heatmap."""
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(CLASS_NAMES)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [c[:10] for c in CLASS_NAMES]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, 'confusion_matrix_gcn.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_per_class_f1(nc_results, save_path=None):
    """Per-class F1 scores for each method."""
    methods = [m for m in ['logistic_regression', 'label_propagation', 'gcn'] if m in nc_results]
    method_labels = {'logistic_regression': 'LR', 'label_propagation': 'LP', 'gcn': 'GCN'}

    n_classes = 7
    x = np.arange(n_classes)
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, method in enumerate(methods):
        f1_vals = nc_results[method]['per_class']['f1']
        ax.bar(x + i * width, f1_vals, width,
               label=method_labels[method], color=COLORS[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([c[:12] for c in CLASS_NAMES], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-Class F1 Score by Method', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, 'per_class_f1.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_graph_statistics(edges, labels, save_path=None):
    """Plot basic graph statistics: degree distribution and class distribution."""
    n_nodes = len(labels)
    n_classes = int(labels.max()) + 1
    # Degree distribution
    deg = np.zeros(n_nodes, dtype=int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Degree distribution
    deg_counts = np.bincount(deg)
    ax1.bar(range(min(30, len(deg_counts))), deg_counts[:30],
            color='#4C72B0', alpha=0.8)
    ax1.set_xlabel('Node Degree', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Degree Distribution (top 30)', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.text(0.6, 0.9, f'Mean deg: {deg.mean():.1f}\nMax deg: {deg.max()}',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Class distribution
    class_counts = np.bincount(labels, minlength=n_classes)
    bars = ax2.bar(range(n_classes), class_counts, color=COLORS[:n_classes], alpha=0.85)
    ax2.set_xticks(range(n_classes))
    ax2.set_xticklabels([c[:12] for c in CLASS_NAMES], rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Number of Nodes', fontsize=12)
    ax2.set_title('Class Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    for bar, cnt in zip(bars, class_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(cnt), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, 'graph_statistics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_node_explanation(explanation, save_path=None):
    """Three-panel figure for one node's explanation:
       1. class probability bars,
       2. top influential neighbors (Δ confidence after edge occlusion),
       3. top important active features (Δ confidence after feature occlusion).

    `explanation` is the dict returned by
    `explainability.explain_node_prediction`."""
    probs = explanation.get('class_probabilities', [])
    nbrs  = explanation.get('top_influential_neighbors', [])
    feats = explanation.get('top_important_features', [])

    node_id = explanation.get('node_id', '?')
    pred    = explanation.get('predicted_class', '?')
    true_   = explanation.get('true_class', '?')
    conf    = explanation.get('confidence', 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ── Panel 1: class probabilities (sorted desc by the explainer) ──
    ax = axes[0]
    if probs:
        names = [p.get('class_name', f"C{p['class_idx']}") for p in probs]
        vals  = [p['probability'] for p in probs]
        bars = ax.bar(range(len(vals)), vals,
                      color=COLORS[:len(vals)], alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([n[:12] for n in names],
                           rotation=30, ha='right', fontsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('Class probabilities', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.10)
    ax.grid(True, axis='y', alpha=0.3)

    # ── Panel 2: neighbor influence (Δ conf after edge occlusion) ──
    ax = axes[1]
    if nbrs:
        labels_n = [
            f"{n['neighbor_id']}\n({(n.get('neighbor_true_label') or '?')[:10]})"
            for n in nbrs
        ]
        drops = [n['confidence_drop'] for n in nbrs]
        # Green when removing the neighbor lowers confidence (it helped),
        # red when it raises confidence (it argued against the prediction).
        bar_colors = ['#55A868' if d >= 0 else '#C44E52' for d in drops]
        bars = ax.bar(range(len(drops)), drops, color=bar_colors, alpha=0.85)
        ax.set_xticks(range(len(drops)))
        ax.set_xticklabels(labels_n, fontsize=9)
        for b, v in zip(bars, drops):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + (0.002 if v >= 0 else -0.002),
                    f'{v:+.3f}',
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    ax.set_ylabel('Δ confidence', fontsize=11)
    ax.set_title('Top influential neighbors', fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)

    # ── Panel 3: feature importance (Δ conf after feature occlusion) ──
    ax = axes[2]
    if feats:
        labels_f = [f"feat {f['feature_idx']}" for f in feats]
        drops = [f['confidence_drop'] for f in feats]
        bar_colors = ['#55A868' if d >= 0 else '#C44E52' for d in drops]
        bars = ax.bar(range(len(drops)), drops, color=bar_colors, alpha=0.85)
        ax.set_xticks(range(len(drops)))
        ax.set_xticklabels(labels_f, rotation=30, ha='right', fontsize=9)
        for b, v in zip(bars, drops):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + (0.002 if v >= 0 else -0.002),
                    f'{v:+.3f}',
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    ax.set_ylabel('Δ confidence', fontsize=11)
    ax.set_title('Top important active features', fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)

    suptitle = (f"Node {node_id} explanation — predicted: {pred} "
                f"({conf * 100:.1f}%) | true: {true_}")
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, 'explain_node_example.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_link_explanation(explanation, save_path=None):
    """Two-panel summary for a (u, v) link explanation. The metrics live
    on different scales (Jaccard / cosine / GCN prob ∈ [0, 1] but CN is a
    raw count and Adamic-Adar is unbounded), so we split them.

    `explanation` is the dict returned by
    `explainability.explain_link_prediction`."""
    src = explanation.get('src', '?')
    dst = explanation.get('dst', '?')
    actual = explanation.get('edge_exists', False)

    cn  = explanation.get('common_neighbors', 0)
    jc  = explanation.get('jaccard', 0.0)
    aa  = explanation.get('adamic_adar', 0.0)
    cos = explanation.get('embedding_cosine', 0.0)
    gcn = explanation.get('gcn_link_probability')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel 1: bounded scores in [0, 1] ──
    bounded = [('Jaccard', jc), ('Cosine', cos)]
    if gcn is not None:
        bounded.append(('GCN prob', float(gcn)))
    names1, vals1 = zip(*bounded)
    bars = ax1.bar(range(len(vals1)), vals1,
                   color=COLORS[:len(vals1)], alpha=0.85)
    ax1.set_xticks(range(len(vals1)))
    ax1.set_xticklabels(names1, fontsize=10)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_ylim(0, 1.10)
    ax1.set_title('Bounded scores ([0, 1])', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    for b, v in zip(bars, vals1):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # ── Panel 2: counts (CN integer) and Adamic-Adar (unbounded) ──
    counts = [('Common Neighbors', float(cn)), ('Adamic-Adar', float(aa))]
    names2, vals2 = zip(*counts)
    bars = ax2.bar(range(len(vals2)), vals2,
                   color=[COLORS[3], COLORS[4]], alpha=0.85)
    ax2.set_xticks(range(len(vals2)))
    ax2.set_xticklabels(names2, fontsize=10)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Heuristic scores (unbounded)', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    headroom = max(vals2) * 0.05 if max(vals2) > 0 else 0.05
    ax2.set_ylim(0, max(vals2) + headroom * 4 if max(vals2) > 0 else 1)
    for b, v in zip(bars, vals2):
        # CN is conceptually an integer; render it as such.
        label = f'{int(v)}' if abs(v - round(v)) < 1e-9 else f'{v:.3f}'
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + headroom,
                 label, ha='center', va='bottom', fontsize=9)

    edge_msg = ("edge present in graph" if actual
                else "edge absent from graph")
    suptitle = f"Link explanation: ({src}, {dst}) — {edge_msg}"
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, 'explain_link_example.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_robustness_f1(robustness_results, task='nc', save_path=None):
    """Two-panel figure for one task ('nc' or 'lp'):
       top    — F1 vs perturbation rate, one line per perturbation type;
       bottom — F1 drop (baseline − perturbed) vs rate, same lines.

    `robustness_results` is the dict returned by
    `robustness.run_robustness_experiments`."""
    if task not in ('nc', 'lp'):
        raise ValueError("task must be 'nc' or 'lp'")

    panels = [
        ('edge_removal',  'Edge removal'),
        ('edge_addition', 'Fake edge addition'),
        ('feature_noise', 'Feature noise'),
    ]
    base_f1 = robustness_results['baseline'][task]['f1']

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for i, (key, name) in enumerate(panels):
        bucket = robustness_results[key]
        # Sort the rate-keyed sub-dict ascending.
        levels = sorted(bucket.keys(), key=lambda k: bucket[k]['rate'])
        rates_pct = [0.0] + [bucket[k]['rate'] * 100 for k in levels]
        f1_vals   = [base_f1] + [bucket[k][task]['f1'] for k in levels]
        drops     = [0.0]     + [base_f1 - bucket[k][task]['f1'] for k in levels]

        ax_top.plot(rates_pct, f1_vals, '-o', color=COLORS[i],
                    linewidth=2, markersize=7, label=name)
        ax_bot.plot(rates_pct, drops,   '-o', color=COLORS[i],
                    linewidth=2, markersize=7, label=name)

    # Baseline reference on the F1 panel.
    ax_top.axhline(base_f1, color='gray', linestyle='--', alpha=0.5,
                   label=f'Baseline F1 = {base_f1:.3f}')

    title_task = 'Node Classification' if task == 'nc' else 'Link Prediction'
    y_label    = 'Macro F1' if task == 'nc' else 'F1'

    ax_top.set_ylabel(y_label, fontsize=11)
    ax_top.set_ylim(0, 1.05)
    ax_top.set_title(f'{title_task} F1 vs perturbation level',
                     fontsize=12, fontweight='bold')
    ax_top.legend(fontsize=9, loc='lower left')
    ax_top.grid(True, alpha=0.3)

    ax_bot.axhline(0, color='gray', linewidth=0.5)
    ax_bot.set_xlabel('Perturbation rate (%)', fontsize=11)
    ax_bot.set_ylabel('F1 drop (baseline − attacked)', fontsize=11)
    ax_bot.set_title(f'{title_task} robustness drop',
                     fontsize=12, fontweight='bold')
    ax_bot.legend(fontsize=9, loc='upper left')
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = 'node' if task == 'nc' else 'link'
    path = save_path or os.path.join(RESULTS_DIR, f'robustness_{suffix}_f1.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_embedding_tsne(embeddings, labels, save_path=None, title='GCN Node Embeddings (t-SNE)'):
    """2D t-SNE visualization of node embeddings colored by class."""
    from sklearn.manifold import TSNE
    from gcn_model import safe_blas, _assert_finite
    _assert_finite(embeddings, 't-SNE input embeddings')
    print("  Running t-SNE on embeddings (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=300)
    # t-SNE internally does many matmuls on potentially zero-heavy
    # arrays — same Accelerate BLAS spurious-FPE issue as sklearn's LR.
    with safe_blas():
        emb_2d = tsne.fit_transform(embeddings)
    _assert_finite(emb_2d, 't-SNE output')

    n_classes = int(labels.max()) + 1
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(n_classes):
        idx = labels == c
        ax.scatter(emb_2d[idx, 0], emb_2d[idx, 1],
                   c=COLORS[c], label=CLASS_NAMES[c], alpha=0.6, s=15)

    ax.legend(fontsize=9, loc='best', framealpha=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, 'embedding_tsne.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path
