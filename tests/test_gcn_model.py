"""GCN forward-pass tests on a tiny toy graph (no training)."""

import numpy as np
import scipy.sparse as sp


def _toy_graph(n=5, n_features=3, seed=0):
    """Tiny path-like graph with one extra cross-edge."""
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 2]])
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A_hat = A + sp.eye(n)
    deg = np.array(A_hat.sum(1)).flatten()
    D_inv = sp.diags(np.power(np.maximum(deg, 1e-12), -0.5))
    A_norm = D_inv.dot(A_hat).dot(D_inv)
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_features)).astype(float)
    return A_norm, X


def test_forward_output_shape():
    from gcn_model import GCN
    A_norm, X = _toy_graph()
    n_classes = 4
    model = GCN(n_features=X.shape[1], n_hidden=6, n_classes=n_classes)
    Z = model.forward(A_norm, X, training=False)
    assert Z.shape == (X.shape[0], n_classes)


def test_forward_softmax_rows_sum_to_one():
    from gcn_model import GCN
    A_norm, X = _toy_graph()
    model = GCN(n_features=X.shape[1], n_hidden=6, n_classes=3)
    Z = model.forward(A_norm, X, training=False)
    np.testing.assert_allclose(Z.sum(axis=1), np.ones(Z.shape[0]), atol=1e-6)


def test_forward_no_nans():
    from gcn_model import GCN
    A_norm, X = _toy_graph()
    model = GCN(n_features=X.shape[1], n_hidden=6, n_classes=3)
    Z = model.forward(A_norm, X, training=False)
    assert not np.isnan(Z).any()
    assert not np.isinf(Z).any()
