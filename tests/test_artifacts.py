"""Saved-artifact integrity tests.

These check that any artifacts left in `results/` by `python src/main.py`
contain no NaN / Inf entries — the demo loads them as-is, so a single
non-finite cell would corrupt every downstream prediction. If the
artifacts haven't been generated yet (fresh checkout), the tests skip
rather than fail so the suite stays green for first-time contributors.
"""

import os
import numpy as np
import pytest

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results',
)


def _skip_if_missing(path):
    if not os.path.exists(path):
        pytest.skip(
            f"artifact not found: {path} "
            f"(run `python src/main.py` to generate)"
        )


def test_gcn_embeddings_finite():
    p = os.path.join(RESULTS_DIR, 'gcn_embeddings.npy')
    _skip_if_missing(p)
    arr = np.load(p)
    assert np.all(np.isfinite(arr)), \
        "gcn_embeddings.npy contains NaN/Inf"


def test_node_probabilities_finite():
    p = os.path.join(RESULTS_DIR, 'node_probabilities.npy')
    _skip_if_missing(p)
    arr = np.load(p)
    assert np.all(np.isfinite(arr)), \
        "node_probabilities.npy contains NaN/Inf"


def test_gcn_weights_finite():
    p = os.path.join(RESULTS_DIR, 'gcn_weights.npz')
    _skip_if_missing(p)
    with np.load(p) as f:
        for k in f.files:
            assert np.all(np.isfinite(f[k])), \
                f"gcn_weights.npz['{k}'] contains NaN/Inf"
