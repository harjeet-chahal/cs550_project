"""
Graph Convolutional Network (GCN) - Pure NumPy Implementation
Based on: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)

Architecture:
  Z = softmax( A_norm * ReLU( A_norm * X * W0 ) * W1 )

where:
  - X     : normalized feature matrix  (N x F)
  - A_norm: symmetrically normalized adjacency with self-loops  (N x N)
  - W0, W1: learnable weight matrices
  - Z     : predicted class probabilities  (N x C)
"""

import numpy as np
import scipy.sparse as sp
import contextlib


@contextlib.contextmanager
def safe_blas():
    """Targeted suppression of NumPy 2.x BLAS spurious-FPE warnings.

    On macOS Accelerate (and some other BLAS builds) NumPy ≥ 2.0 raises
    "divide by zero / overflow / invalid value encountered in matmul"
    RuntimeWarnings whenever an operand has many explicit zeros — which
    happens for every sparse-feature graph and every BoW classifier
    fit. The math is correct (we verify with `_assert_finite` on inputs
    and outputs), only the wrapper is noisy. We swap our own `@` to
    `np.dot` to dodge it inside the GCN, but inside library code
    (sklearn, t-SNE) we have no choice but to scope-suppress.

    USE THIS ONLY AROUND LIBRARY CALLS, never around your own code.
    Wrap with `_assert_finite(...)` checks on inputs/outputs so that
    a real numerical issue (NaN/Inf) is still caught.
    """
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        yield


def _assert_finite(arr, name):
    """Raise FloatingPointError with a useful message on NaN / Inf.

    Used at every key intermediate (AX, H0, H1, AH1, logits, Z, embeddings,
    weights, gradients) so any genuine numerical blow-up — gradient
    explosion, weight overflow, dtype clash, sparse-to-dense mishap — is
    caught immediately at the offending step rather than producing
    silent garbage downstream.
    """
    if hasattr(arr, 'data'):  # sparse matrix
        data = arr.data
    else:
        data = np.asarray(arr)
    if not np.all(np.isfinite(data)):
        n_nan = int(np.isnan(data).sum())
        n_inf = int(np.isinf(data).sum())
        raise FloatingPointError(
            f"non-finite values in '{name}': nan={n_nan}, inf={n_inf}, "
            f"dtype={data.dtype}, shape={getattr(arr, 'shape', None)}"
        )


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def softmax(x):
    # Numerically stable row-wise softmax
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, labels, idx):
    """Cross-entropy loss on subset idx."""
    eps = 1e-10
    n = len(idx)
    return -np.mean(np.log(probs[idx, labels[idx]] + eps))


class GCN:
    """
    Two-layer GCN for semi-supervised node classification.

    Forward pass:
        H1 = ReLU( A_norm @ X @ W0 )          # (N x hidden)
        Z  = softmax( A_norm @ H1 @ W1 )       # (N x n_classes)

    Trained with mini-batch SGD (full graph, masked loss on train nodes).
    """

    def __init__(self, n_features, n_hidden, n_classes, lr=0.01, weight_decay=5e-4, dropout=0.5):
        self.lr           = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout

        # Glorot uniform initialization
        def glorot(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, (fan_in, fan_out))

        self.W0 = glorot(n_features, n_hidden)
        self.W1 = glorot(n_hidden, n_classes)

        # Adam optimizer state
        self.m_W0 = np.zeros_like(self.W0)
        self.v_W0 = np.zeros_like(self.W0)
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.t = 0

        # Cache for backward pass
        self._cache = {}

    def forward(self, A_norm, X, training=True):
        """
        A_norm: dense or sparse (N x N)
        X     : dense or sparse feature matrix (N x F)
        Returns probability matrix Z (N x C)

        Numerical-stability notes
        -------------------------
        Dense @ dense is performed via `np.dot` rather than `@` /
        `np.matmul`. On macOS Accelerate (and some other BLAS builds)
        NumPy 2.x's matmul wrapper raises spurious "divide by zero /
        overflow / invalid value encountered in matmul" RuntimeWarnings
        when one operand has many explicit zero entries (which
        AX = A_norm @ X always does for sparse-feature graphs like
        Cora). The output is mathematically correct but the warning
        chatter is noise — `np.dot` uses BLAS gemm directly without
        that wrapper, so the warnings disappear while the results stay
        identical. Softmax is computed on row-max-subtracted logits
        (see `softmax`).
        """
        # Layer 1: A @ X @ W0
        if hasattr(X, 'toarray'):
            AX = A_norm.dot(X).toarray()
        else:
            AX = A_norm.dot(X) if hasattr(A_norm, 'dot') else A_norm @ X
        AX = np.ascontiguousarray(AX)
        _assert_finite(AX, 'AX')

        H0 = np.dot(AX, self.W0)           # (N x hidden) — uses BLAS gemm
        _assert_finite(H0, 'H0')
        H1 = relu(H0)                      # (N x hidden)
        _assert_finite(H1, 'H1')

        # Dropout on hidden layer (training only)
        if training and self.dropout_rate > 0:
            mask = (np.random.rand(*H1.shape) > self.dropout_rate).astype(float)
            mask /= (1.0 - self.dropout_rate + 1e-10)   # inverted dropout
            H1_drop = H1 * mask
        else:
            H1_drop = H1
            mask = np.ones_like(H1)

        # Layer 2: A @ H1 @ W1
        if hasattr(A_norm, 'dot'):
            AH1 = A_norm.dot(H1_drop)
        else:
            AH1 = A_norm @ H1_drop
        AH1 = np.ascontiguousarray(AH1)
        _assert_finite(AH1, 'AH1')

        logits = np.dot(AH1, self.W1)      # (N x C)
        _assert_finite(logits, 'logits')
        Z = softmax(logits)                # (N x C); stable (row-max subtracted)
        _assert_finite(Z, 'Z')

        # Cache for backprop
        self._cache = {
            'AX': AX, 'H0': H0, 'H1': H1, 'H1_drop': H1_drop,
            'mask': mask, 'AH1': AH1, 'logits': logits, 'Z': Z,
            'A_norm': A_norm
        }
        return Z

    def backward(self, labels, train_idx):
        """
        Backprop through GCN.
        Gradient flows only through train_idx nodes for loss, but
        the graph convolution propagates gradients to all nodes.

        Dense @ dense is again routed through `np.dot` (see forward).
        Each gradient is finite-checked so a genuine numerical blow-up
        (e.g. activation explosion or NaN-producing dropout mask)
        surfaces immediately at the offending matmul.
        """
        c = self._cache
        N = c['Z'].shape[0]
        C = c['Z'].shape[1]

        # Gradient of cross-entropy + softmax at train nodes
        dZ = np.zeros_like(c['Z'])                          # (N x C)
        dZ[train_idx] = c['Z'][train_idx].copy()
        dZ[train_idx, labels[train_idx]] -= 1
        dZ /= len(train_idx)
        _assert_finite(dZ, 'dZ')

        # Backprop through layer 2: Z = softmax(AH1 @ W1)
        # dL/d(AH1) = dZ @ W1^T
        dAH1 = np.dot(dZ, self.W1.T)                        # (N x hidden)
        _assert_finite(dAH1, 'dAH1')
        dW1  = np.dot(c['AH1'].T, dZ)                       # (hidden x C)
        _assert_finite(dW1, 'dW1')

        # Backprop through A_norm multiplication: AH1 = A @ H1_drop
        A_norm = c['A_norm']
        if hasattr(A_norm, 'T'):
            dH1_drop = A_norm.T.dot(dAH1)
        else:
            dH1_drop = A_norm.T @ dAH1
        dH1_drop = np.ascontiguousarray(dH1_drop)

        # Backprop through dropout
        dH1 = dH1_drop * c['mask']

        # Backprop through ReLU
        dH0 = dH1 * relu_grad(c['H0'])                     # (N x hidden)

        # Backprop through layer 1: H0 = AX @ W0
        dW0 = np.dot(c['AX'].T, dH0)                       # (F x hidden)
        _assert_finite(dW0, 'dW0')

        # L2 regularization
        dW0 += self.weight_decay * self.W0
        dW1 += self.weight_decay * self.W1

        return dW0, dW1

    def _adam_update(self, W, m, v, grad, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 0.5   # incremented twice per step (W0, W1), so half each
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** self.t)
        v_hat = v / (1 - beta2 ** self.t)
        W -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        return W, m, v

    def step(self, dW0, dW1):
        self.W0, self.m_W0, self.v_W0 = self._adam_update(self.W0, self.m_W0, self.v_W0, dW0)
        self.W1, self.m_W1, self.v_W1 = self._adam_update(self.W1, self.m_W1, self.v_W1, dW1)

    def train_epoch(self, A_norm, X, labels, train_idx):
        Z    = self.forward(A_norm, X, training=True)
        loss = cross_entropy_loss(Z, labels, train_idx)
        dW0, dW1 = self.backward(labels, train_idx)
        self.step(dW0, dW1)
        return loss, Z

    def predict(self, A_norm, X):
        Z = self.forward(A_norm, X, training=False)
        return np.argmax(Z, axis=1), Z


def accuracy(preds, labels, idx):
    return np.mean(preds[idx] == labels[idx])


def _build_dropedge_adjacency(edge_pairs, n_nodes, drop_rate, rng):
    """
    Sample a fresh sub-graph by independently dropping each undirected
    edge with probability `drop_rate`, mirror it (since the original is
    symmetric), and return the symmetrically normalized adjacency
    D^{-1/2}(Â+I)D^{-1/2}. Self-loops are restored by the normalization
    step, so isolated nodes still receive their own feature.
    """
    n = len(edge_pairs)
    keep_mask = rng.random(n) >= drop_rate
    kept = edge_pairs[keep_mask]
    rows = np.concatenate([kept[:, 0], kept[:, 1]])
    cols = np.concatenate([kept[:, 1], kept[:, 0]])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(n_nodes, n_nodes))
    # Self-loops + symmetric normalization, identical to
    # data_preprocessing.normalize_adjacency.
    A_hat = A + sp.eye(n_nodes)
    deg = np.array(A_hat.sum(1)).flatten()
    D_inv_sqrt = sp.diags(np.power(np.maximum(deg, 1e-12), -0.5))
    return D_inv_sqrt.dot(A_hat).dot(D_inv_sqrt)


def train_gcn(A_norm, X_dense, labels, train_idx, val_idx, test_idx,
              n_hidden=64, lr=0.01, epochs=200, weight_decay=5e-4,
              dropout=0.5, patience=20, verbose=True,
              edge_dropout_rate=0.0, A_raw=None, edge_dropout_seed=None):
    """
    Full training loop with early stopping on validation accuracy.
    Returns trained model and history dict.

    Edge dropout (DropEdge — Rong et al. 2020) — *defense / regularization*
    -----------------------------------------------------------------------
    When `edge_dropout_rate > 0`, each epoch independently drops that
    fraction of edges from `A_raw`, re-normalizes (self-loops preserved),
    and uses the resulting smaller graph for THAT epoch's forward and
    backward pass. The model therefore sees many slightly-different
    graphs during training, which:
      • reduces over-reliance on any single edge or short path,
      • acts as graph-level data augmentation, and
      • measurably improves robustness when test-time inputs are
        perturbed (random edge add / drop, feature noise, …).
    Validation / test accuracy are still measured on the FULL clean
    graph each epoch so early stopping reflects true generalization.

    With the default `edge_dropout_rate=0.0` (and `A_raw=None`), the
    training loop is byte-identical to the original implementation.
    """
    n_feat    = X_dense.shape[1]
    n_classes = int(labels.max()) + 1

    model = GCN(n_feat, n_hidden, n_classes, lr=lr,
                weight_decay=weight_decay, dropout=dropout)

    use_edge_dropout = (edge_dropout_rate > 0.0) and (A_raw is not None)
    if use_edge_dropout:
        # Pre-extract the upper-triangular edge list ONCE so we can
        # cheaply resample per epoch without rebuilding from sparse.
        coo = A_raw.tocoo()
        upper = coo.row < coo.col
        edge_pairs = np.stack([coo.row[upper], coo.col[upper]], axis=1)
        n_nodes = A_raw.shape[0]
        rng = np.random.default_rng(edge_dropout_seed)

    best_val_acc = 0
    best_W0 = model.W0.copy()
    best_W1 = model.W1.copy()
    wait = 0

    history = {'train_loss': [], 'val_acc': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        # When edge dropout is active, build a fresh per-epoch
        # adjacency. Otherwise reuse the static A_norm from the caller.
        if use_edge_dropout:
            A_norm_epoch = _build_dropedge_adjacency(
                edge_pairs, n_nodes, edge_dropout_rate, rng,
            )
        else:
            A_norm_epoch = A_norm

        loss, Z = model.train_epoch(A_norm_epoch, X_dense, labels, train_idx)

        if use_edge_dropout:
            # For early stopping & history, evaluate on the FULL clean
            # graph so val/test accuracy reflect generalization rather
            # than this epoch's particular sub-graph.
            Z_eval = model.forward(A_norm, X_dense, training=False)
            preds = np.argmax(Z_eval, axis=1)
        else:
            preds = np.argmax(Z, axis=1)

        val_acc  = accuracy(preds, labels, val_idx)
        test_acc = accuracy(preds, labels, test_idx)

        history['train_loss'].append(loss)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_W0 = model.W0.copy()
            best_W1 = model.W1.copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    # Restore best weights
    model.W0 = best_W0
    model.W1 = best_W1

    return model, history
