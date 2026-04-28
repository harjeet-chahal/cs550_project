"""Tests for explainability — node + link explanation contracts."""

import numpy as np
import scipy.sparse as sp


def _toy_setup(n=10, n_features=4, n_classes=3, n_hidden=4, seed=0):
    """Small random graph + untrained GCN. We don't train — these tests
    only check the explanation contract (keys, shapes, error handling),
    not prediction quality."""
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.35:
                pairs.append([i, j])
    if not pairs:                      # very unlikely but keep deterministic
        pairs = [[0, 1], [1, 2], [2, 3]]
    edges = np.array(pairs)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    A_raw = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A_hat = A_raw + sp.eye(n)
    deg = np.array(A_hat.sum(1)).flatten()
    D_inv = sp.diags(np.power(np.maximum(deg, 1e-12), -0.5))
    A_norm = D_inv.dot(A_hat).dot(D_inv)

    X = (rng.random((n, n_features)) > 0.5).astype(float)
    labels = rng.integers(0, n_classes, n)

    from gcn_model import GCN
    model = GCN(n_features, n_hidden, n_classes)
    return model, A_norm, A_raw, X, labels


# ── Node explanation ────────────────────────────────────────────

NODE_REQUIRED_KEYS = {
    'node_id', 'predicted_class_idx', 'predicted_class', 'true_class_idx',
    'true_class', 'confidence', 'class_probabilities',
    'top_influential_neighbors', 'top_important_features', 'explanation',
}


def test_explain_node_returns_required_keys():
    from explainability import explain_node_prediction
    model, A_norm, A_raw, X, labels = _toy_setup()
    out = explain_node_prediction(
        model, 0, X,
        adjacency_norm=A_norm, adjacency_raw=A_raw,
        labels=labels, class_names=['A', 'B', 'C'], top_k=3,
    )
    assert NODE_REQUIRED_KEYS.issubset(out.keys())
    # Sanity: probabilities are sorted descending and sum to ~1.
    probs = [p['probability'] for p in out['class_probabilities']]
    assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))
    assert abs(sum(probs) - 1.0) < 1e-6


def test_explain_node_invalid_id_returns_error():
    from explainability import explain_node_prediction
    model, A_norm, A_raw, X, labels = _toy_setup()
    n = A_norm.shape[0]

    bad_high = explain_node_prediction(model, n + 100, X, A_norm, A_raw)
    assert 'error' in bad_high

    bad_low = explain_node_prediction(model, -1, X, A_norm, A_raw)
    assert 'error' in bad_low


# ── Link explanation ────────────────────────────────────────────

LINK_REQUIRED_KEYS = {
    'src', 'dst', 'gcn_link_probability', 'common_neighbors',
    'jaccard', 'adamic_adar', 'embedding_cosine', 'edge_exists',
    'explanation',
}


def test_explain_link_returns_required_keys():
    from explainability import explain_link_prediction
    model, A_norm, A_raw, X, labels = _toy_setup()
    # H1 from a forward pass acts as our embeddings.
    model.forward(A_norm, X, training=False)
    emb = model._cache['H1']

    out = explain_link_prediction(
        0, 1,
        embeddings=emb,
        adjacency_train=A_raw,
        adjacency_full=A_raw,
        link_model=None,
        class_names=['A', 'B', 'C'],
        labels=labels,
    )
    assert LINK_REQUIRED_KEYS.issubset(out.keys())
    assert isinstance(out['common_neighbors'], int)
    assert isinstance(out['edge_exists'], bool)


def test_explain_link_self_loop_rejected():
    from explainability import explain_link_prediction
    model, A_norm, A_raw, X, labels = _toy_setup()
    model.forward(A_norm, X, training=False)
    emb = model._cache['H1']

    out = explain_link_prediction(
        0, 0, embeddings=emb,
        adjacency_train=A_raw, adjacency_full=A_raw,
    )
    assert 'error' in out


def test_explain_link_out_of_range_rejected():
    from explainability import explain_link_prediction
    model, A_norm, A_raw, X, labels = _toy_setup()
    model.forward(A_norm, X, training=False)
    emb = model._cache['H1']

    n = A_norm.shape[0]
    out = explain_link_prediction(
        0, n + 5, embeddings=emb,
        adjacency_train=A_raw, adjacency_full=A_raw,
    )
    assert 'error' in out
