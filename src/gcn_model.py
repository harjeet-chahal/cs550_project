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
        """
        # Layer 1: A @ X @ W0
        if hasattr(X, 'toarray'):
            AX = A_norm.dot(X).toarray()
        else:
            AX = A_norm.dot(X) if hasattr(A_norm, 'dot') else A_norm @ X

        H0 = AX @ self.W0                  # (N x hidden)
        H1 = relu(H0)                      # (N x hidden)

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

        logits = AH1 @ self.W1             # (N x C)
        Z = softmax(logits)                # (N x C)

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
        """
        c = self._cache
        N = c['Z'].shape[0]
        C = c['Z'].shape[1]

        # Gradient of cross-entropy + softmax at train nodes
        dZ = np.zeros_like(c['Z'])                          # (N x C)
        dZ[train_idx] = c['Z'][train_idx].copy()
        dZ[train_idx, labels[train_idx]] -= 1
        dZ /= len(train_idx)

        # Backprop through layer 2: Z = softmax(AH1 @ W1)
        # dL/d(AH1) = dZ @ W1^T
        dAH1 = dZ @ self.W1.T                               # (N x hidden)
        dW1  = c['AH1'].T @ dZ                              # (hidden x C)

        # Backprop through A_norm multiplication: AH1 = A @ H1_drop
        A_norm = c['A_norm']
        if hasattr(A_norm, 'T'):
            dH1_drop = A_norm.T.dot(dAH1)
        else:
            dH1_drop = A_norm.T @ dAH1

        # Backprop through dropout
        dH1 = dH1_drop * c['mask']

        # Backprop through ReLU
        dH0 = dH1 * relu_grad(c['H0'])                     # (N x hidden)

        # Backprop through layer 1: H0 = AX @ W0
        dW0 = c['AX'].T @ dH0                              # (F x hidden)

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


def train_gcn(A_norm, X_dense, labels, train_idx, val_idx, test_idx,
              n_hidden=64, lr=0.01, epochs=200, weight_decay=5e-4,
              dropout=0.5, patience=20, verbose=True):
    """
    Full training loop with early stopping on validation accuracy.
    Returns trained model and history dict.
    """
    n_feat    = X_dense.shape[1]
    n_classes = int(labels.max()) + 1

    model = GCN(n_feat, n_hidden, n_classes, lr=lr,
                weight_decay=weight_decay, dropout=dropout)

    best_val_acc = 0
    best_W0 = model.W0.copy()
    best_W1 = model.W1.copy()
    wait = 0

    history = {'train_loss': [], 'val_acc': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        loss, Z = model.train_epoch(A_norm, X_dense, labels, train_idx)
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
