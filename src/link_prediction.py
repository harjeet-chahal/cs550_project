"""
CS550 Project - Option 2: Social Networks
Step 2: Link Prediction

Algorithms implemented:
  1. Common Neighbors (CN)       - Jaccard Coefficient variant
  2. Adamic-Adar (AA)            - weighted common neighbors
  3. GCN-based Link Prediction   - uses node embeddings from GCN

Evaluation: Precision, Recall, F-measure (at threshold 0.5 & top-K)
            AUC-ROC
"""

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
from sklearn.linear_model import LogisticRegression


# ─────────────────────────────────────────────────────────────────
# Heuristic methods
# ─────────────────────────────────────────────────────────────────

def common_neighbors_score(A, u, v):
    """Number of common neighbors between u and v."""
    neighbors_u = set(A[u].nonzero()[1])
    neighbors_v = set(A[v].nonzero()[1])
    return len(neighbors_u & neighbors_v)


def adamic_adar_score(A, u, v):
    """Adamic-Adar: sum of 1/log(degree) over common neighbors."""
    neighbors_u = set(A[u].nonzero()[1])
    neighbors_v = set(A[v].nonzero()[1])
    common = neighbors_u & neighbors_v
    deg = np.array(A.sum(1)).flatten()
    score = 0.0
    for w in common:
        d = deg[w]
        if d > 1:
            score += 1.0 / np.log(d)
    return score


def jaccard_coefficient_score(A, u, v):
    """Jaccard: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|"""
    neighbors_u = set(A[u].nonzero()[1])
    neighbors_v = set(A[v].nonzero()[1])
    union = neighbors_u | neighbors_v
    if len(union) == 0:
        return 0.0
    return len(neighbors_u & neighbors_v) / len(union)


def compute_heuristic_scores(A, edge_pairs, method='adamic_adar'):
    """Compute link-prediction scores for all edge pairs using chosen heuristic."""
    scores = []
    fn_map = {
        'common_neighbors': common_neighbors_score,
        'adamic_adar':      adamic_adar_score,
        'jaccard':          jaccard_coefficient_score,
    }
    fn = fn_map[method]
    for u, v in edge_pairs:
        scores.append(fn(A, int(u), int(v)))
    return np.array(scores, dtype=float)


# ─────────────────────────────────────────────────────────────────
# GCN-based link prediction (embedding dot-product)
# ─────────────────────────────────────────────────────────────────

def get_gcn_embeddings(model, A_norm, X_dense):
    """
    Extract the hidden-layer embeddings (H1) from a trained GCN.
    These are used as node representations for link prediction.
    """
    Z = model.forward(A_norm, X_dense, training=False)
    # H1 is cached after forward pass
    return model._cache['H1']   # (N x hidden)


def edge_features_dot(emb, edge_pairs):
    """Score edges by dot product of endpoint embeddings."""
    u_emb = emb[edge_pairs[:, 0]]   # (M x d)
    v_emb = emb[edge_pairs[:, 1]]   # (M x d)
    return (u_emb * v_emb).sum(axis=1)


def edge_features_hadamard(emb, edge_pairs):
    """Concatenate hadamard product features for logistic regression."""
    u_emb = emb[edge_pairs[:, 0]]
    v_emb = emb[edge_pairs[:, 1]]
    return u_emb * v_emb   # element-wise product (M x d)


def train_link_predictor(emb, train_pos, train_neg):
    """
    Train a logistic regression on top of GCN embeddings.
    Features: hadamard product of endpoint embeddings.
    """
    X_pos = edge_features_hadamard(emb, train_pos)
    X_neg = edge_features_hadamard(emb, train_neg)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*len(train_pos) + [0]*len(train_neg))

    clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    clf.fit(X, y)
    return clf


def predict_link_gcn(clf, emb, edge_pairs):
    """Predict link probability using logistic regression over GCN embeddings."""
    X = edge_features_hadamard(emb, edge_pairs)
    return clf.predict_proba(X)[:, 1]   # probability of link=1


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_link_prediction(scores_pos, scores_neg, threshold=None):
    """
    Evaluate link prediction given scores for positive and negative edges.

    Args:
        scores_pos: scores for true edges
        scores_neg: scores for non-edges
        threshold: decision threshold; if None, uses median of all scores

    Returns:
        dict with precision, recall, f1, auc, threshold used
    """
    all_scores = np.concatenate([scores_pos, scores_neg])
    all_labels = np.array([1]*len(scores_pos) + [0]*len(scores_neg))

    # Normalize scores to [0,1] for heuristic methods
    s_min, s_max = all_scores.min(), all_scores.max()
    if s_max > s_min:
        all_scores_norm = (all_scores - s_min) / (s_max - s_min)
    else:
        all_scores_norm = all_scores.copy()

    if threshold is None:
        threshold = np.median(all_scores_norm)

    preds = (all_scores_norm >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average='binary', zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_scores_norm)
    except Exception:
        auc = float('nan')

    return {
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'auc':       auc,
        'threshold': threshold,
        'n_pos':     len(scores_pos),
        'n_neg':     len(scores_neg),
    }


def run_link_prediction(A_train, link_splits, gcn_model=None, A_norm=None, X_dense=None):
    """
    Run all link prediction algorithms and return results dict.

    Methods:
      - Common Neighbors
      - Adamic-Adar
      - Jaccard Coefficient
      - GCN + Logistic Regression (if gcn_model provided)
    """
    results = {}
    test_pos = link_splits['test']['pos']
    test_neg = link_splits['test']['neg']
    val_pos  = link_splits['val']['pos']
    val_neg  = link_splits['val']['neg']

    # ── Heuristic methods ──────────────────────────────────────
    for method in ['common_neighbors', 'adamic_adar', 'jaccard']:
        print(f"  Running {method}...")
        test_edges = np.vstack([test_pos, test_neg])
        scores_all = compute_heuristic_scores(A_train, test_edges, method=method)

        n_pos = len(test_pos)
        scores_p = scores_all[:n_pos]
        scores_n = scores_all[n_pos:]

        res = evaluate_link_prediction(scores_p, scores_n)
        results[method] = res
        print(f"    Precision={res['precision']:.4f}  Recall={res['recall']:.4f}  "
              f"F1={res['f1']:.4f}  AUC={res['auc']:.4f}")

    # ── GCN-based method ──────────────────────────────────────
    if gcn_model is not None and A_norm is not None and X_dense is not None:
        print("  Running GCN + Logistic Regression...")
        emb = get_gcn_embeddings(gcn_model, A_norm, X_dense)

        train_pos = link_splits['train']['pos']
        train_neg = link_splits['train']['neg']
        clf = train_link_predictor(emb, train_pos, train_neg)

        # Evaluate on test
        test_all   = np.vstack([test_pos, test_neg])
        scores_all = predict_link_gcn(clf, emb, test_all)
        n_pos      = len(test_pos)
        res        = evaluate_link_prediction(scores_all[:n_pos], scores_all[n_pos:], threshold=0.5)
        results['gcn_lr'] = res
        print(f"    Precision={res['precision']:.4f}  Recall={res['recall']:.4f}  "
              f"F1={res['f1']:.4f}  AUC={res['auc']:.4f}")

    return results
