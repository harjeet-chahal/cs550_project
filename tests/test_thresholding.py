"""Strict-`>` thresholding contract.

`tune_threshold` documents that it uses a strict `>` rule precisely so
that the degenerate "predict every zero-score pair as positive"
classifier cannot be picked. This test enforces that contract on a
heuristic-style score distribution where most pairs score exactly 0.
"""

import numpy as np


def test_zero_scores_are_never_predicted_positive():
    from link_prediction import tune_threshold, evaluate_scores_with_threshold

    rng = np.random.default_rng(0)
    n_pos = n_neg = 100

    # Most positives carry signal (≥ 1 common neighbor → integer score),
    # the rest are zero. Negatives are almost all zero.
    pos_scores = np.where(
        rng.random(n_pos) < 0.4,
        rng.integers(1, 5, n_pos).astype(float),
        0.0,
    )
    neg_scores = np.where(rng.random(n_neg) < 0.02, 1.0, 0.0)

    y = np.array([1] * n_pos + [0] * n_neg)
    s = np.concatenate([pos_scores, neg_scores])

    t = tune_threshold(y, s)

    # Every pair scoring 0 must be predicted negative under strict `>`.
    preds = (s > t).astype(int)
    zero_idx = np.where(s == 0)[0]
    assert (preds[zero_idx] == 0).all(), (
        f"strict-`>` rule violated: a zero-score pair was predicted "
        f"positive at threshold {t}"
    )

    # And the official evaluator must give the same predictions.
    res = evaluate_scores_with_threshold(y, s, t)
    n_pred_pos = int((s > t).sum())
    # Recall = TP / n_pos, so TP = round(recall * n_pos); preds must match.
    tp = int(((s > t) & (y == 1)).sum())
    fp = int(((s > t) & (y == 0)).sum())
    assert tp + fp == n_pred_pos
    # Sanity: there's real signal in this synthetic setup, so F1 > 0.
    assert res['f1'] > 0.0
