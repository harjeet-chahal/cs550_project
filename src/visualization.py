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


def plot_embedding_tsne(embeddings, labels, save_path=None, title='GCN Node Embeddings (t-SNE)'):
    """2D t-SNE visualization of node embeddings colored by class."""
    from sklearn.manifold import TSNE
    print("  Running t-SNE on embeddings (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=300)
    emb_2d = tsne.fit_transform(embeddings)

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
