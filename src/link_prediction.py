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

from gcn_model import safe_blas, _assert_finite


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
    _assert_finite(X, 'edge_features (LR fit input)')

    # sklearn's LogisticRegression internally calls `X @ weights`, which
    # on macOS Accelerate emits spurious BLAS FPE warnings. Inputs are
    # finite-checked above, so we scope-suppress the warning ONLY for
    # this library call (not globally).
    with safe_blas():
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        clf.fit(X, y)
    return clf


def predict_link_gcn(clf, emb, edge_pairs):
    """Predict link probability using logistic regression over GCN embeddings."""
    X = edge_features_hadamard(emb, edge_pairs)
    _assert_finite(X, 'edge_features (LR predict input)')
    with safe_blas():
        probs = clf.predict_proba(X)[:, 1]   # probability of link=1
    _assert_finite(probs, 'LR predict_proba output')
    return probs


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

def tune_threshold(y_true_val, scores_val, eps=1e-6):
    """
    Pick the F1-best decision threshold on VALIDATION scores using a
    *strict* `>` rule.

    Why strict `>` and not `>=`:
        With `>=`, threshold = 0 trivially classifies *every* zero-score
        pair as positive. On Cora's neighbor-counting heuristics the
        majority of pairs (positive AND negative) score 0, so `>=` plus
        an F1-greedy search collapses to "predict-everything-positive"
        with P=0.5, R=1, F1=0.667 — a useless classifier dressed up as
        a tuned one. The strict `>` rule excludes that degeneracy by
        construction.

    Candidate set:
        * midpoints between consecutive unique val scores,
        * a tiny `eps` candidate (so "any non-zero score is positive"
          is always considered),
        * an "above-max" candidate at `max(scores) + eps` (predicts
          nothing positive — kept for completeness, never wins on
          F1-discriminative data).

    Tie-breaking on equal F1:
        higher precision → higher balanced accuracy → higher threshold.
        Implemented via lexicographic tuple comparison
        `(f1, precision, balanced_acc, threshold)`.
    """
    y = np.asarray(y_true_val).astype(int)
    s = np.asarray(scores_val, dtype=float)
    if s.size == 0:
        return float(eps)

    unique = np.unique(s)
    if len(unique) == 1:
        # All val scores identical → no usable separator. Returning
        # just-above means we predict nothing positive (F1=0; honest).
        return float(unique[0] + eps)

    midpoints = (unique[:-1] + unique[1:]) / 2.0
    tiny      = float(eps)
    above_max = float(unique[-1] + eps)
    candidates = np.unique(np.concatenate([midpoints, [tiny, above_max]]))

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    best = None  # (f1, precision, balanced_acc, threshold)
    for t in candidates:
        preds = (s > t).astype(int)        # STRICT inequality
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = n_pos - tp
        tn = n_neg - fp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = 0.5 * (rec + spec)
        score = (f1, prec, bal_acc, float(t))
        if best is None or score > best:
            best = score
    return float(best[3])


def evaluate_scores_with_threshold(y_true_test, scores_test, threshold):
    """
    Apply a val-tuned threshold to TEST scores using the same strict
    `>` rule that `tune_threshold` uses. AUC-ROC is also reported
    because it captures ranking quality independent of any threshold.
    """
    y = np.asarray(y_true_test).astype(int)
    s = np.asarray(scores_test, dtype=float)
    preds = (s > threshold).astype(int)    # STRICT inequality

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average='binary', zero_division=0
    )
    try:
        auc = roc_auc_score(y, s)
    except Exception:
        auc = float('nan')

    return {
        'precision':        float(precision),
        'recall':           float(recall),
        'f1':               float(f1),
        'auc':              float(auc),
        'chosen_threshold': float(threshold),
        'threshold':        float(threshold),
        'n_pos':            int((y == 1).sum()),
        'n_neg':            int((y == 0).sum()),
    }


def run_link_prediction(A_train, link_splits, gcn_model=None, A_norm_train=None, X_dense=None):
    """
    Run all link prediction algorithms and return results dict.

    Methods:
      - Common Neighbors
      - Adamic-Adar
      - Jaccard Coefficient
      - GCN + Logistic Regression (if gcn_model provided)

    Data-leakage contract (link prediction):
      - A_train is built from TRAINING positive edges only
        (see split_edges_link_prediction). Heuristics score test pairs
        against A_train, never the full graph.
      - The GCN backbone passed in (`gcn_model`) MUST have been trained
        on A_norm_train (the train-only normalized adjacency). Embeddings
        are extracted by forwarding through A_norm_train as well. This
        guarantees that held-out test edges never appear in the message-
        passing structure at training OR inference time.
      - The Logistic Regression edge classifier is fit on TRAIN positive
        + TRAIN negative pairs only.
      - The decision threshold is tuned on VAL pos/neg edges.
      - Final P/R/F1/AUC are reported on TEST pos/neg edges only.
    """
    results = {}
    test_pos  = link_splits['test']['pos']
    test_neg  = link_splits['test']['neg']
    val_pos   = link_splits['val']['pos']
    val_neg   = link_splits['val']['neg']
    train_pos = link_splits['train']['pos']
    train_neg = link_splits['train']['neg']

    # Pre-stack val / test pairs once; same arrays are reused per method.
    val_pairs   = np.vstack([val_pos,  val_neg])
    test_pairs  = np.vstack([test_pos, test_neg])
    y_val       = np.array([1]*len(val_pos)  + [0]*len(val_neg))
    y_test      = np.array([1]*len(test_pos) + [0]*len(test_neg))

    # ── Heuristic methods (use A_train; never see test edges) ──
    # Each heuristic now picks its OWN threshold by F1-maximization
    # on validation scores, then we evaluate on test at that threshold.
    for method in ['common_neighbors', 'adamic_adar', 'jaccard']:
        print(f"  Running {method}...")

        # Heuristic scores on val and test pairs (against A_train).
        val_scores  = compute_heuristic_scores(A_train, val_pairs,  method=method)
        test_scores = compute_heuristic_scores(A_train, test_pairs, method=method)

        # Tune threshold on VAL only, evaluate on TEST only.
        best_t = tune_threshold(y_val, val_scores)
        res    = evaluate_scores_with_threshold(y_test, test_scores, best_t)
        results[method] = res

        print(f"    Tuned threshold (val): {best_t:.4f}")
        print(f"    Precision={res['precision']:.4f}  Recall={res['recall']:.4f}  "
              f"F1={res['f1']:.4f}  AUC={res['auc']:.4f}")

    # ── GCN + LR (leak-free) ───────────────────────────────────
    if gcn_model is not None and A_norm_train is not None and X_dense is not None:
        print("  Running GCN + Logistic Regression...")

        # Embeddings come from forwarding through A_norm_train (train-only
        # adjacency). The caller is responsible for ensuring `gcn_model`
        # was also TRAINED on A_norm_train, not the full graph — otherwise
        # the trained weights would already encode information about the
        # held-out test edges via message passing.
        emb = get_gcn_embeddings(gcn_model, A_norm_train, X_dense)

        # Fit the edge classifier on TRAIN edges only.
        clf = train_link_predictor(emb, train_pos, train_neg)

        # Score val and test pairs in raw probability space.
        val_scores  = predict_link_gcn(clf, emb, val_pairs)
        test_scores = predict_link_gcn(clf, emb, test_pairs)

        # Tune threshold on VAL only, evaluate on TEST only — same
        # helpers as the heuristics.
        best_t = tune_threshold(y_val, val_scores)
        res    = evaluate_scores_with_threshold(y_test, test_scores, best_t)
        results['gcn_lr'] = res

        print(f"    Tuned threshold (val): {best_t:.4f}")
        print(f"    Precision={res['precision']:.4f}  Recall={res['recall']:.4f}  "
              f"F1={res['f1']:.4f}  AUC={res['auc']:.4f}")

    return results
