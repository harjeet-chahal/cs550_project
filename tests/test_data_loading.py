"""Tests for data_preprocessing.load_cora — structural sanity checks."""


def test_node_count(cora):
    labels, _, _ = cora
    assert len(labels) == 2708


def test_feature_dimension(cora):
    _, _, features = cora
    assert features.shape[1] == 1433


def test_seven_classes(cora):
    labels, _, _ = cora
    # Labels are 0..6 inclusive.
    assert int(labels.max()) + 1 == 7
    assert int(labels.min()) == 0


def test_edges_nonempty(cora):
    _, edges, _ = cora
    assert len(edges) > 0
    # Each row is an undirected (u, v) pair with two int columns.
    assert edges.shape[1] == 2
