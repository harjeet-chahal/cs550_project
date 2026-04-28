"""Tests for the link-prediction heuristics + threshold tuning."""

import math
import numpy as np
import scipy.sparse as sp


def _toy_adjacency():
    """4-node graph:
        edges: 0-1, 1-2, 0-2, 1-3
        neighbours:
          N(0) = {1, 2}
          N(1) = {0, 2, 3}
          N(2) = {0, 1}
          N(3) = {1}
        For pair (0, 3): common = {1}, union = {1, 2, 3}.
    """
    edges = np.array([[0, 1], [1, 2], [0, 2], [1, 3]])
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(4, 4))


def test_common_neighbors_known_value():
    from link_prediction import common_neighbors_score
    A = _toy_adjacency()
    # |N(0) ∩ N(3)| = |{1}| = 1
    assert common_neighbors_score(A, 0, 3) == 1
    # |N(0) ∩ N(1)| = |{2}| = 1   (0 and 1 are themselves connected via edge)
    assert common_neighbors_score(A, 0, 1) == 1


def test_jaccard_known_value():
    from link_prediction import jaccard_coefficient_score
    A = _toy_adjacency()
    # N(0) = {1, 2}, N(3) = {1}.
    # Intersection = {1} (size 1), Union = {1, 2} (size 2). Jaccard = 1/2.
    val = jaccard_coefficient_score(A, 0, 3)
    assert math.isclose(val, 0.5, rel_tol=1e-9, abs_tol=1e-12)


def test_adamic_adar_nonnegative():
    from link_prediction import adamic_adar_score
    A = _toy_adjacency()
    for u, v in [(0, 3), (0, 1), (2, 3), (1, 1)]:
        assert adamic_adar_score(A, u, v) >= 0


def test_tune_threshold_returns_in_range_and_sensible():
    from link_prediction import tune_threshold, evaluate_scores_with_threshold

    rng = np.random.default_rng(0)
    pos = rng.uniform(0.6, 1.0, 100)
    neg = rng.uniform(0.0, 0.4, 100)
    scores = np.concatenate([pos, neg])
    y = np.array([1] * 100 + [0] * 100)

    t = tune_threshold(y, scores)
    # Returned threshold must be a finite float in the observed range.
    assert isinstance(t, float)
    assert math.isfinite(t)
    assert scores.min() <= t <= scores.max()

    res = evaluate_scores_with_threshold(y, scores, t)
    for key in ('precision', 'recall', 'f1', 'auc', 'chosen_threshold'):
        assert key in res
    # Well-separated synthetic data ⇒ tuning should pick a high-F1 threshold.
    assert res['f1'] > 0.9
    assert 0.0 <= res['precision'] <= 1.0
    assert 0.0 <= res['recall']    <= 1.0
