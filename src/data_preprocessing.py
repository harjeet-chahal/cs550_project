"""
CS550 Project - Option 2: Social Networks
Step 1: Data Preprocessing — REAL CORA DATASET

Cora file format (inside data/cora/):
  cora.content : <paper_id>  <1433 binary features...>  <class_label>
  cora.cites   : <cited_paper_id>  <citing_paper_id>
"""

import numpy as np
import scipy.sparse as sp
import os, random

random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CORA_DIR = os.path.join(DATA_DIR, 'cora')

CLASS_MAP = {
    'Case_Based':             0,
    'Genetic_Algorithms':     1,
    'Neural_Networks':        2,
    'Probabilistic_Methods':  3,
    'Reinforcement_Learning': 4,
    'Rule_Learning':          5,
    'Theory':                 6,
}


def count_raw_citations(cites_path=None):
    """Count valid (non-empty, ≥2-token) lines in cora.cites — i.e. the
    raw citation links *before* undirected deduplication. The raw file
    has 5,429 such rows; after collapsing (A→B, B→A) duplicates and
    exact repeats only 5,278 unique undirected edges remain. Both
    numbers are reported throughout the project (console, demo,
    README) so there is no ambiguity about which one is being used."""
    if cites_path is None:
        cites_path = os.path.join(CORA_DIR, 'cora.cites')
    n = 0
    with open(cites_path, 'r') as f:
        for line in f:
            if len(line.strip().split()) >= 2:
                n += 1
    return n


def load_cora():
    content_path = os.path.join(CORA_DIR, 'cora.content')
    cites_path   = os.path.join(CORA_DIR, 'cora.cites')

    if not os.path.exists(content_path):
        raise FileNotFoundError(
            f"\n[ERROR] Could not find: {content_path}\n"
            f"  Run:\n"
            f"    curl -L 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz' -o data/cora.tgz\n"
            f"    tar -xzf data/cora.tgz -C data/\n"
        )

    paper_ids, feat_rows, label_list = [], [], []
    with open(content_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            paper_ids.append(parts[0])
            feat_rows.append([int(x) for x in parts[1:-1]])
            label_list.append(CLASS_MAP[parts[-1]])

    N = len(paper_ids)
    paper_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    labels = np.array(label_list, dtype=np.int64)

    rows, cols = [], []
    for i, feat_vec in enumerate(feat_rows):
        for j, val in enumerate(feat_vec):
            if val != 0:
                rows.append(i)
                cols.append(j)
    F = len(feat_rows[0])
    features = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(N, F), dtype=np.float32
    )

    edge_set, edge_list = set(), []
    with open(cites_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            cited, citing = parts[0], parts[1]
            if cited not in paper_to_idx or citing not in paper_to_idx:
                continue
            u, v = paper_to_idx[cited], paper_to_idx[citing]
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                edge_set.add(key)
                edge_list.append(key)

    edges = np.array(edge_list, dtype=np.int64)

    n_raw = count_raw_citations(cites_path)
    print(f"[Data] Nodes: {N}  |  Raw citations: {n_raw}  |  "
          f"Unique undirected edges: {len(edges)}  |  "
          f"Features: {F}  |  Classes: {len(CLASS_MAP)}")
    print(f"[Data] Class distribution: {np.bincount(labels)}")
    return labels, edges, features


def build_adjacency(edges, n_nodes):
    row  = np.concatenate([edges[:, 0], edges[:, 1]])
    col  = np.concatenate([edges[:, 1], edges[:, 0]])
    A    = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n_nodes, n_nodes))
    return A


def normalize_features(features):
    """Row-l1 normalize a sparse feature matrix.

    Zero-row guard: rows with no active features become 1 in the
    denominator so the output row stays all-zero (instead of NaN /
    Inf). Result is asserted to be finite.
    """
    rowsum = np.asarray(features.sum(1)).flatten().astype(np.float64)
    rowsum[rowsum == 0] = 1.0
    inv = 1.0 / rowsum
    if not np.all(np.isfinite(inv)):
        raise FloatingPointError(
            "normalize_features: 1 / rowsum produced non-finite values"
        )
    out = sp.diags(inv).dot(features)
    if not np.all(np.isfinite(out.data)):
        raise FloatingPointError(
            "normalize_features output contains NaN/Inf entries"
        )
    return out


def normalize_adjacency(A):
    """Symmetric normalization with self-loops: D^{-1/2}(A + I)D^{-1/2}.

    Self-loops guarantee `deg >= 1` for every row, so `deg ** -0.5`
    is finite. The clamp `max(deg, 1e-12)` is a defensive belt for
    the case where an isolated graph is fed in without self-loops.
    Result is asserted to be finite.
    """
    n      = A.shape[0]
    A_hat  = A + sp.eye(n)
    deg    = np.asarray(A_hat.sum(1)).flatten().astype(np.float64)
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
    if not np.all(np.isfinite(deg_inv_sqrt)):
        raise FloatingPointError(
            "normalize_adjacency: D^{-1/2} produced non-finite values"
        )
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    out = D_inv_sqrt.dot(A_hat).dot(D_inv_sqrt)
    if not np.all(np.isfinite(out.data)):
        raise FloatingPointError(
            "normalize_adjacency output contains NaN/Inf entries"
        )
    return out


def split_edges_link_prediction(edges, n_nodes, test_ratio=0.2, val_ratio=0.1):
    all_edges = list(map(tuple, edges))
    random.shuffle(all_edges)
    n_test  = int(len(all_edges) * test_ratio)
    n_val   = int(len(all_edges) * val_ratio)
    n_train = len(all_edges) - n_test - n_val

    train_pos = np.array(all_edges[:n_train])
    val_pos   = np.array(all_edges[n_train:n_train + n_val])
    test_pos  = np.array(all_edges[n_train + n_val:])
    train_A   = build_adjacency(train_pos, n_nodes)
    edge_set  = set(map(tuple, all_edges))

    def sample_negatives(n_samples):
        negs = []
        while len(negs) < n_samples:
            u = random.randint(0, n_nodes - 1)
            v = random.randint(0, n_nodes - 1)
            if u != v and (u,v) not in edge_set and (v,u) not in edge_set:
                negs.append((u, v))
                edge_set.add((u, v))
        return np.array(negs)

    train_neg = sample_negatives(n_train)
    val_neg   = sample_negatives(n_val)
    test_neg  = sample_negatives(n_test)

    print(f"[Link Split] Train pos/neg: {len(train_pos)}/{len(train_neg)} | "
          f"Val pos/neg: {len(val_pos)}/{len(val_neg)} | "
          f"Test pos/neg: {len(test_pos)}/{len(test_neg)}")
    return {
        'train':   {'pos': train_pos, 'neg': train_neg},
        'val':     {'pos': val_pos,   'neg': val_neg},
        'test':    {'pos': test_pos,  'neg': test_neg},
        'train_A': train_A,
    }


def split_nodes_classification(labels, train_per_class=20, val_size=500, test_size=1000):
    n, classes = len(labels), np.unique(labels)
    train_idx  = []
    for c in classes:
        c_idx  = np.where(labels == c)[0]
        chosen = np.random.choice(c_idx, min(train_per_class, len(c_idx)), replace=False)
        train_idx.extend(chosen.tolist())
    train_idx = np.array(train_idx)
    remaining = np.setdiff1d(np.arange(n), train_idx)
    np.random.shuffle(remaining)
    val_idx  = remaining[:val_size]
    test_idx = remaining[val_size:val_size + test_size]
    print(f"[Node Split] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx
