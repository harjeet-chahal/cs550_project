"""Tests for split_edges_link_prediction and split_nodes_classification."""


def _to_set(pair_array):
    """Canonicalize an (M, 2) array of edges into a set of sorted-int tuples."""
    return {(int(min(u, v)), int(max(u, v))) for u, v in pair_array}


def test_link_pos_splits_disjoint(cora):
    from data_preprocessing import split_edges_link_prediction
    labels, edges, _ = cora
    splits = split_edges_link_prediction(edges, len(labels))

    train_pos = _to_set(splits['train']['pos'])
    val_pos   = _to_set(splits['val']['pos'])
    test_pos  = _to_set(splits['test']['pos'])

    assert train_pos.isdisjoint(val_pos)
    assert train_pos.isdisjoint(test_pos)
    assert val_pos.isdisjoint(test_pos)
    # Together they should cover every original positive edge.
    assert (train_pos | val_pos | test_pos) == _to_set(edges)


def test_negative_edges_are_not_real(cora):
    from data_preprocessing import split_edges_link_prediction
    labels, edges, _ = cora
    splits = split_edges_link_prediction(edges, len(labels))

    real_edges = _to_set(edges)
    for split_name in ('train', 'val', 'test'):
        for u, v in splits[split_name]['neg']:
            key = (int(min(u, v)), int(max(u, v)))
            assert key not in real_edges, (
                f"negative edge {key} in {split_name} is actually a real edge"
            )
            assert int(u) != int(v), "negative edge must not be a self-loop"


def test_node_splits_disjoint(cora):
    from data_preprocessing import split_nodes_classification
    labels, _, _ = cora
    train_idx, val_idx, test_idx = split_nodes_classification(labels)

    train = set(int(i) for i in train_idx)
    val   = set(int(i) for i in val_idx)
    test  = set(int(i) for i in test_idx)

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # All sampled indices must land in [0, N).
    N = len(labels)
    assert all(0 <= i < N for i in train | val | test)
