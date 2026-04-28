"""Leakage tests for the link-prediction split.

The central no-leakage contract is that the adjacency used for link-
prediction message passing (`train_A`) must contain ONLY training
positive edges — never validation or test positives. If this contract
ever broke, both the heuristic scorers and the GCN+LR backbone would
silently see the held-out edges through message passing.
"""


def _edge_in_adjacency(A, u, v):
    """True iff the undirected edge (u, v) is present in sparse A."""
    return bool(A[int(u), int(v)] != 0 or A[int(v), int(u)] != 0)


def test_train_adjacency_excludes_val_and_test_positives(cora):
    from data_preprocessing import split_edges_link_prediction
    labels, edges, _ = cora
    N = len(labels)
    splits = split_edges_link_prediction(edges, N)
    A_train = splits['train_A']

    for u, v in splits['val']['pos']:
        assert not _edge_in_adjacency(A_train, u, v), (
            f"validation positive edge ({int(u)}, {int(v)}) leaked into "
            f"train adjacency"
        )
    for u, v in splits['test']['pos']:
        assert not _edge_in_adjacency(A_train, u, v), (
            f"test positive edge ({int(u)}, {int(v)}) leaked into "
            f"train adjacency"
        )
