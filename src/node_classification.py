"""
CS550 Project - Option 2: Social Networks
Step 3: Node Classification

Algorithms implemented:
  1. Logistic Regression (baseline, on raw features)
  2. Label Propagation (graph-based semi-supervised)
  3. GCN (Graph Convolutional Network) - main method

Evaluation: Per-class & macro Precision, Recall, F-measure
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    confusion_matrix, accuracy_score
)

from gcn_model import safe_blas, _assert_finite


# ─────────────────────────────────────────────────────────────────
# Baseline: Logistic Regression on raw features
# ─────────────────────────────────────────────────────────────────

def run_logistic_regression(X, labels, train_idx, test_idx):
    """Baseline: Logistic Regression on node feature vectors."""
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = np.array(X)
    _assert_finite(X_dense, 'LR baseline input X')

    # See `safe_blas` docstring — sklearn's matmul triggers spurious
    # BLAS FPE warnings on macOS Accelerate. Inputs are finite-checked.
    with safe_blas():
        clf = LogisticRegression(max_iter=500, solver='lbfgs', C=1.0)
        clf.fit(X_dense[train_idx], labels[train_idx])
        preds = clf.predict(X_dense[test_idx])
    return preds, clf


# ─────────────────────────────────────────────────────────────────
# Label Propagation
# ─────────────────────────────────────────────────────────────────

def label_propagation(A, labels, train_idx, n_classes, alpha=0.85, max_iter=100, tol=1e-6):
    """
    Semi-supervised label propagation.
    F* = (1-α)(I - αD^{-1}A)^{-1} Y  (approximated by iterative updates)

    Iterative: F^{t+1} = α * D^{-1} A F^t + (1-α) Y_0
    """
    N = A.shape[0]
    # Row-normalize adjacency
    deg = np.array(A.sum(1)).flatten()
    deg[deg == 0] = 1
    D_inv = sp.diags(1.0 / deg)
    P = D_inv.dot(A)   # row-stochastic transition matrix

    # Initial label matrix Y_0 (one-hot, 0 for unlabeled)
    Y0 = np.zeros((N, n_classes))
    Y0[train_idx, labels[train_idx]] = 1.0

    F = Y0.copy()
    for i in range(max_iter):
        F_new = alpha * P.dot(F) + (1 - alpha) * Y0
        delta = np.linalg.norm(F_new - F, 'fro')
        F = F_new
        if delta < tol:
            break

    preds = np.argmax(F, axis=1)
    return preds, F


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
    'Probabilistic_Methods', 'Reinforcement_Learning',
    'Rule_Learning', 'Theory'
]


def evaluate_node_classification(preds, labels, test_idx, method_name='', n_classes=7):
    """
    Full evaluation: accuracy, per-class precision/recall/F1, macro averages.
    Returns results dict.
    """
    y_true = labels[test_idx]
    y_pred = preds[test_idx] if len(preds) == len(labels) else preds

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )

    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Macro  Precision:  {macro_p:.4f}   Recall: {macro_r:.4f}   F1: {macro_f1:.4f}")
    print(f"  Micro  Precision:  {micro_p:.4f}   Recall: {micro_r:.4f}   F1: {micro_f1:.4f}")
    print(f"\n  Per-Class Results:")
    print(f"  {'Class':<28} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*56}")
    for i in range(n_classes):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class_{i}'
        print(f"  {name:<28} {precision[i]:>6.4f} {recall[i]:>6.4f} {f1[i]:>6.4f} {support[i]:>8}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    return {
        'method':    method_name,
        'accuracy':  acc,
        'macro_p':   macro_p,
        'macro_r':   macro_r,
        'macro_f1':  macro_f1,
        'micro_p':   micro_p,
        'micro_r':   micro_r,
        'micro_f1':  micro_f1,
        'per_class': {
            'precision': precision,
            'recall':    recall,
            'f1':        f1,
            'support':   support,
        },
        'confusion_matrix': cm,
        'preds': y_pred,
        'labels': y_true,
    }


def run_node_classification(A, X, X_dense, labels, train_idx, val_idx, test_idx,
                            gcn_model=None, A_norm=None):
    """
    Run all node classification methods and return results.
    """
    results = {}
    n_classes = int(labels.max()) + 1

    # ── Baseline: Logistic Regression ──────────────────────────
    print("\n[Node Classification] Running Logistic Regression (baseline)...")
    lr_preds, _ = run_logistic_regression(X, labels, train_idx, test_idx)
    # lr_preds is only for test_idx, evaluate directly
    y_true = labels[test_idx]
    acc = accuracy_score(y_true, lr_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, lr_preds, average='macro', zero_division=0
    )
    per_p, per_r, per_f1, per_s = precision_recall_fscore_support(
        y_true, lr_preds, labels=list(range(n_classes)), average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, lr_preds, labels=list(range(n_classes)))
    results['logistic_regression'] = {
        'method': 'Logistic Regression',
        'accuracy': acc, 'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
        'per_class': {'precision': per_p, 'recall': per_r, 'f1': per_f1, 'support': per_s},
        'confusion_matrix': cm, 'preds': lr_preds, 'labels': y_true
    }
    print(f"  Accuracy: {acc:.4f}  |  Macro F1: {macro_f1:.4f}")

    # ── Label Propagation ──────────────────────────────────────
    print("\n[Node Classification] Running Label Propagation...")
    lp_preds_full, lp_probs = label_propagation(A, labels, train_idx, n_classes)
    res_lp = evaluate_node_classification(lp_preds_full, labels, test_idx,
                                          method_name='Label Propagation',
                                          n_classes=n_classes)
    results['label_propagation'] = res_lp

    # ── GCN ───────────────────────────────────────────────────
    if gcn_model is not None and A_norm is not None:
        print("\n[Node Classification] Evaluating GCN...")
        preds_full, _ = gcn_model.predict(A_norm, X_dense)
        res_gcn = evaluate_node_classification(preds_full, labels, test_idx,
                                               method_name='GCN',
                                               n_classes=n_classes)
        results['gcn'] = res_gcn

    return results
