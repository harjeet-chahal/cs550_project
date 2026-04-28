"""
CS550 Project — Explainability for the NumPy GCN
=================================================

Three occlusion-based explanations for a single node prediction.

Occlusion intuition
-------------------
Occlusion attribution asks "what input was the model relying on?" by
*removing* one piece of input at a time and observing how the predicted-
class confidence changes. A large drop ⇒ that input was important
evidence for the prediction. A small drop (or an increase) ⇒ that input
contributed little (or argued against the prediction).

We apply this to:
  1. The full softmax over classes (no occlusion — just the prediction).
  2. Edges: for each neighbor v of the target node u, drop edge (u,v),
     re-normalize the adjacency, and re-forward.
  3. Features: Cora's bag-of-words are binary, so for each active word
     j on the target node we flip X[u,j] = 1 → 0 and re-forward.

All occlusions are performed on COPIES of the adjacency / feature
matrices. The original objects passed in are never permanently changed.
"""

import numpy as np
import scipy.sparse as sp

# Reuse the existing heuristic / classifier code so there is exactly one
# implementation of CN, AA, Jaccard, and the GCN+LR scorer in the project.
from link_prediction import (
    common_neighbors_score,
    adamic_adar_score,
    jaccard_coefficient_score,
    predict_link_gcn,
)


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _predicted_class_prob(model, A_norm, X_dense, node_id):
    """Forward through the model and return (pred_class, p_pred, all_probs)."""
    Z = model.forward(A_norm, X_dense, training=False)
    probs = Z[node_id]
    pred = int(np.argmax(probs))
    return pred, float(probs[pred]), probs


def _renormalize(A_raw):
    """
    Symmetric normalization with self-loops, matching
    `data_preprocessing.normalize_adjacency`. We re-normalize per
    edge-occlusion because removing an edge changes degrees of both
    endpoints, which changes D^{-1/2} on those rows/columns.
    """
    n = A_raw.shape[0]
    A_hat = A_raw + sp.eye(n)
    deg = np.array(A_hat.sum(1)).flatten()
    # Guard against zero-degree (shouldn't happen with self-loops, but safe).
    deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
    D = sp.diags(deg_inv_sqrt)
    return D.dot(A_hat).dot(D)


def get_neighbors(A_raw, node_id):
    """Return sorted unique neighbor node IDs (excluding the node itself)."""
    if sp.issparse(A_raw):
        nbrs = A_raw[node_id].nonzero()[1]
    else:
        nbrs = np.nonzero(np.asarray(A_raw)[node_id])[0]
    return sorted({int(n) for n in nbrs if int(n) != int(node_id)})


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def get_top_neighbor_influences(model, node_id, X_dense, A_raw,
                                labels=None, class_names=None,
                                pred_class=None, base_conf=None,
                                top_k=5):
    """
    Edge-occlusion explanation.

    For each neighbor v of `node_id`, drop the undirected edge (u, v)
    from a COPY of the raw adjacency, re-normalize, re-forward, and
    record:

        influence(v) = base_conf − P(predicted_class | u, A \\ {(u,v)})

    Higher influence ⇒ removing v's message hurts the original
    prediction more ⇒ v contributed more.

    The original `A_raw` is never mutated.
    """
    # Convert once to LIL for cheap element-wise editing; keep the
    # caller's matrix untouched.
    A_lil = sp.lil_matrix(A_raw)
    neighbors = get_neighbors(A_lil, node_id)

    if base_conf is None or pred_class is None:
        A_norm0 = _renormalize(A_lil.tocsr())
        pred_class, base_conf, _ = _predicted_class_prob(model, A_norm0, X_dense, node_id)

    influences = []
    for v in neighbors:
        # Cora citations are stored symmetrically; remove both directions
        # so a single "edge" really disappears from message passing.
        Am = A_lil.copy()
        Am[node_id, v] = 0
        Am[v, node_id] = 0
        Am_csr = Am.tocsr()
        Am_csr.eliminate_zeros()
        A_norm_m = _renormalize(Am_csr)

        Z_m = model.forward(A_norm_m, X_dense, training=False)
        new_conf = float(Z_m[node_id, pred_class])
        drop = float(base_conf - new_conf)

        item = {
            'neighbor_id':     int(v),
            'confidence_drop': drop,
            'new_confidence':  new_conf,
        }
        if labels is not None:
            lab = int(labels[v])
            item['neighbor_true_label_idx'] = lab
            if class_names is not None and 0 <= lab < len(class_names):
                item['neighbor_true_label'] = class_names[lab]
        influences.append(item)

    influences.sort(key=lambda x: x['confidence_drop'], reverse=True)
    return influences[:top_k]


def get_top_feature_importances(model, node_id, X_dense, A_norm,
                                pred_class=None, base_conf=None,
                                top_k=5):
    """
    Feature-occlusion explanation.

    Cora's features are binary BoW (1433 dims). For each active feature
    j on the target node (X[u, j] == 1), flip it to 0 on a COPY,
    re-forward through the SAME normalized adjacency, and record:

        importance(j) = base_conf − P(predicted_class | u, X with X[u,j]=0)

    Note on semantics: only the TARGET node's copy of the feature is
    zeroed; neighbors keep theirs. So this measures the marginal
    contribution of u's *own* token-j to its prediction (after graph
    convolution mixes in the neighbors). This is the standard "input
    occlusion" baseline.

    Edits are restored in place after each forward pass, so total
    memory stays O(F) rather than O(F * N * F).
    """
    if base_conf is None or pred_class is None:
        pred_class, base_conf, _ = _predicted_class_prob(model, A_norm, X_dense, node_id)

    # Private working copy of the feature matrix; caller's X_dense is
    # never permanently changed.
    X = X_dense.copy()
    active_feats = np.where(X[node_id] != 0)[0]

    importances = []
    for j in active_feats:
        original_val = X[node_id, j]
        X[node_id, j] = 0
        Z_m = model.forward(A_norm, X, training=False)
        new_conf = float(Z_m[node_id, pred_class])
        # Restore so the next iteration only varies feature j+1.
        X[node_id, j] = original_val

        importances.append({
            'feature_idx':     int(j),
            'confidence_drop': float(base_conf - new_conf),
            'new_confidence':  new_conf,
        })

    importances.sort(key=lambda x: x['confidence_drop'], reverse=True)
    return importances[:top_k]


def _format_node_explanation(node_id, pred_name, conf, true_name,
                             top_neighbors, top_features):
    """One-sentence summary for a node prediction (used by demo)."""
    if pred_name and true_name and pred_name == true_name:
        verdict = f"correctly classified as {pred_name}"
    elif pred_name and true_name:
        verdict = f"predicted as {pred_name} (true: {true_name})"
    elif pred_name:
        verdict = f"predicted as {pred_name}"
    else:
        verdict = "classified"

    parts = [f"Node {node_id} was {verdict} with {conf * 100:.1f}% confidence."]

    if top_neighbors:
        n = top_neighbors[0]
        nbr_lbl = n.get('neighbor_true_label')
        nbr_str = f"node {n['neighbor_id']}"
        if nbr_lbl:
            nbr_str += f" ({nbr_lbl})"
        parts.append(f"The most influential neighbor was {nbr_str} "
                     f"(Δconf={n['confidence_drop']:+.3f}).")

    if top_features:
        f0 = top_features[0]
        parts.append(f"The most important active feature was index "
                     f"{f0['feature_idx']} (Δconf={f0['confidence_drop']:+.3f}).")

    return ' '.join(parts)


def explain_node_prediction(model, node_id, features, adjacency_norm,
                            adjacency_raw, labels=None, class_names=None,
                            top_k=5):
    """
    Produce a JSON-serializable explanation for ONE node.

    Args:
        model:          trained NumPy GCN (with `.forward`)
        node_id:        node index to explain (0 ≤ node_id < N)
        features:       dense or sparse feature matrix (N x F)
        adjacency_norm: SYMMETRICALLY-NORMALIZED adjacency (A_norm).
                        Used for the unperturbed forward and feature
                        occlusion (where the graph structure stays
                        fixed).
        adjacency_raw:  the RAW (un-normalized) adjacency. Required
                        for edge occlusion — we re-normalize per edge.
        labels:         optional 1-D int array of true labels.
        class_names:    optional list of human-readable class names.
        top_k:          how many neighbors / features to return.

    Returns a dict with the schema described in the module docstring.
    On invalid node_id, returns {'error': '...'}.
    """
    # ── Input validation ──────────────────────────────────────
    N = adjacency_norm.shape[0]
    try:
        node_id = int(node_id)
    except (TypeError, ValueError):
        return {'error': 'node_id must be an integer'}
    if node_id < 0 or node_id >= N:
        return {'error': f'node_id must be in [0, {N - 1}]'}

    # Densify features once for occlusion; never mutate caller's data.
    X_dense = (features.toarray() if hasattr(features, 'toarray')
               else np.array(features, copy=True))

    # ── 1. Confidence explanation ─────────────────────────────
    pred, conf, probs = _predicted_class_prob(model, adjacency_norm, X_dense, node_id)
    sorted_idx = np.argsort(-probs)
    class_probabilities = []
    for i in sorted_idx:
        entry = {'class_idx': int(i), 'probability': float(probs[i])}
        if class_names is not None and 0 <= int(i) < len(class_names):
            entry['class_name'] = class_names[int(i)]
        class_probabilities.append(entry)

    pred_name = (class_names[pred]
                 if (class_names is not None and 0 <= pred < len(class_names))
                 else None)
    if labels is not None:
        true_class_idx = int(labels[node_id])
        true_class_name = (class_names[true_class_idx]
                           if (class_names is not None
                               and 0 <= true_class_idx < len(class_names))
                           else None)
    else:
        true_class_idx, true_class_name = None, None

    # ── 2. Neighbor influence (occlusion over edges) ──────────
    top_neighbors = get_top_neighbor_influences(
        model, node_id, X_dense, adjacency_raw,
        labels=labels, class_names=class_names,
        pred_class=pred, base_conf=conf, top_k=top_k,
    )

    # ── 3. Feature importance (occlusion over the node's BoW) ─
    top_features = get_top_feature_importances(
        model, node_id, X_dense, adjacency_norm,
        pred_class=pred, base_conf=conf, top_k=top_k,
    )

    explanation = _format_node_explanation(
        node_id, pred_name, conf, true_class_name, top_neighbors, top_features,
    )

    return {
        'node_id':                   node_id,
        'predicted_class_idx':       int(pred),
        'predicted_class':           pred_name,
        'true_class_idx':            true_class_idx,
        'true_class':                true_class_name,
        'confidence':                float(conf),
        'class_probabilities':       class_probabilities,
        'top_influential_neighbors': top_neighbors,
        'top_important_features':    top_features,
        'explanation':               explanation,
    }


# ─────────────────────────────────────────────────────────────────
# Link prediction explainer
# ─────────────────────────────────────────────────────────────────

def _cosine_similarity(a, b):
    """Cosine similarity of two 1-D vectors; 0 when either is zero-length."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _qualify_cosine(c):
    if c >= 0.7:  return 'highly similar'
    if c >= 0.4:  return 'similar'
    if c >= 0.1:  return 'weakly similar'
    if c >= -0.1: return 'roughly orthogonal'
    return 'dissimilar'


def _qualify_prob(p):
    if p >= 0.7: return 'high'
    if p >= 0.4: return 'moderate'
    return 'low'


def _format_link_explanation(u, v, cn, jc, aa, cos_sim, gcn_prob, edge_exists,
                             u_label=None, v_label=None):
    """Concise human-readable summary built from the same fields the
    function returns. Kept short so it fits in a demo card.

    The phrasing branches on whether structural heuristics actually
    fire on this pair. Cora's training graph is sparse, so it is common
    for two papers to have zero common neighbors (CN/JC/AA all 0) yet
    still receive a high GCN+LR probability via embedding similarity —
    in that case we say so explicitly rather than listing zeros as if
    they supported the link.
    """
    sim_phrase = _qualify_cosine(cos_sim)
    cn_word = 'common neighbor' + ('' if int(cn) == 1 else 's')
    structural_zero = (int(cn) == 0 and float(jc) == 0.0 and float(aa) == 0.0)

    if gcn_prob is None:
        # Heuristics-only descriptive summary (no model prediction to qualify).
        if structural_zero:
            head = (f"Papers {u} and {v} share 0 common neighbors and have "
                    f"Jaccard/Adamic-Adar scores of 0; their GCN embeddings "
                    f"are {sim_phrase} (cosine={cos_sim:.3f}).")
        else:
            head = (f"Papers {u} and {v} share {int(cn)} {cn_word}, "
                    f"with a Jaccard score of {jc:.3f} and an Adamic-Adar "
                    f"score of {aa:.3f}; their GCN embeddings are "
                    f"{sim_phrase} (cosine={cos_sim:.3f}).")
    else:
        verdict = _qualify_prob(gcn_prob)
        prob_pct = f"{gcn_prob * 100:.1f}%"
        high_prob = gcn_prob >= 0.7

        if structural_zero and high_prob:
            head = (f"The model predicts a high-probability link "
                    f"({prob_pct}) between papers {u} and {v} mainly because "
                    f"the learned GCN embeddings are {sim_phrase} "
                    f"(cosine={cos_sim:.3f}). Traditional structural "
                    f"heuristics do not strongly support this link, since "
                    f"the papers share 0 common neighbors and have "
                    f"Jaccard/Adamic-Adar scores of 0.")
        elif structural_zero:
            # Low/moderate prob with no structural overlap: just describe.
            head = (f"The model predicts a {verdict}-probability link "
                    f"({prob_pct}) between papers {u} and {v}. The papers "
                    f"share 0 common neighbors and have Jaccard/Adamic-Adar "
                    f"scores of 0; their GCN embeddings are {sim_phrase} "
                    f"(cosine={cos_sim:.3f}).")
        elif high_prob:
            head = (f"The model predicts a high-probability link "
                    f"({prob_pct}) between papers {u} and {v}. The "
                    f"prediction is supported by both learned embedding "
                    f"similarity and structural overlap: {int(cn)} "
                    f"{cn_word}, Jaccard {jc:.3f}, Adamic-Adar {aa:.3f}, "
                    f"with embeddings that are {sim_phrase} "
                    f"(cosine={cos_sim:.3f}).")
        else:
            # Structural positive but GCN moderate/low — descriptive only.
            head = (f"The model predicts a {verdict}-probability link "
                    f"({prob_pct}) between papers {u} and {v}. The papers "
                    f"share {int(cn)} {cn_word}, with a Jaccard score of "
                    f"{jc:.3f} and an Adamic-Adar score of {aa:.3f}; their "
                    f"GCN embeddings are {sim_phrase} (cosine={cos_sim:.3f}).")

    truth = (" This edge IS present in the original citation graph."
             if edge_exists else
             " This edge is NOT present in the original citation graph.")

    label_note = ''
    if u_label is not None and v_label is not None:
        label_note = (f" Source class: {u_label}; target class: {v_label}.")

    return head + label_note + truth


def explain_link_prediction(u, v, embeddings, adjacency_train, adjacency_full,
                            link_model=None, class_names=None, labels=None):
    """
    Explain why the model would predict a link (or not) for the (u, v) pair.

    Args:
        u, v:            node indices (must differ; both in [0, N))
        embeddings:      (N x d) array of GCN node embeddings used by the
                         link classifier. Should come from a backbone
                         trained / forwarded on the TRAIN-only adjacency
                         to remain leak-free; this function does not
                         re-derive them.
        adjacency_train: TRAIN-only raw adjacency. Heuristic scores
                         (CN / Jaccard / Adamic-Adar) are computed here
                         so the explanation never peeks at held-out edges.
        adjacency_full:  Full raw adjacency. Used ONLY to look up whether
                         the (u, v) edge is in the original graph — i.e.
                         for the `edge_exists` ground-truth flag.
        link_model:      Optional fitted classifier (e.g. the LR fit on
                         train-edge embeddings). When provided we report
                         its probability for (u, v). When None, the
                         GCN+LR field is omitted and the natural-language
                         summary uses heuristics only.
        class_names:     Optional list of human-readable class names.
        labels:          Optional 1-D int label array (for u/v classes).

    Returns a JSON-serializable dict with:
        src, dst,
        gcn_link_probability (float or None),
        common_neighbors (int),
        jaccard (float),
        adamic_adar (float),
        embedding_cosine (float),
        edge_exists (bool),
        src_label, dst_label (str or None),
        explanation (str),
        error (str)  ← only on invalid input.
    """
    # ── Input validation ──────────────────────────────────────
    N = adjacency_train.shape[0]
    try:
        u = int(u); v = int(v)
    except (TypeError, ValueError):
        return {'error': 'u and v must be integers'}
    if not (0 <= u < N and 0 <= v < N):
        return {'error': f'u and v must be in [0, {N - 1}]'}
    if u == v:
        return {'error': 'u and v must differ (self-loops are not predicted)'}
    if embeddings is None:
        return {'error': 'embeddings are required for link explanation'}
    if embeddings.shape[0] != N:
        return {'error': f'embeddings.shape[0] ({embeddings.shape[0]}) '
                         f'must match adjacency size ({N})'}

    # ── Heuristic scores on the TRAIN adjacency only ──────────
    # Reuses the same scorers as the main link-prediction pipeline so
    # the numbers shown here match what an evaluator would compute.
    cn = int(common_neighbors_score(adjacency_train, u, v))
    jc = float(jaccard_coefficient_score(adjacency_train, u, v))
    aa = float(adamic_adar_score(adjacency_train, u, v))

    # ── Embedding cosine similarity ───────────────────────────
    cos_sim = _cosine_similarity(embeddings[u], embeddings[v])

    # ── Optional GCN+LR probability via the existing predictor ─
    gcn_prob = None
    if link_model is not None:
        pair = np.array([[u, v]])
        gcn_prob = float(predict_link_gcn(link_model, embeddings, pair)[0])

    # ── Ground-truth presence in the FULL graph (read-only) ──
    if sp.issparse(adjacency_full):
        edge_exists = bool(adjacency_full[u, v] != 0 or adjacency_full[v, u] != 0)
    else:
        A = np.asarray(adjacency_full)
        edge_exists = bool(A[u, v] != 0 or A[v, u] != 0)

    # ── Optional class labels for u and v ─────────────────────
    src_label = dst_label = None
    if labels is not None:
        u_idx = int(labels[u]); v_idx = int(labels[v])
        if class_names is not None:
            if 0 <= u_idx < len(class_names): src_label = class_names[u_idx]
            if 0 <= v_idx < len(class_names): dst_label = class_names[v_idx]

    explanation = _format_link_explanation(
        u, v, cn, jc, aa, cos_sim, gcn_prob, edge_exists,
        u_label=src_label, v_label=dst_label,
    )

    return {
        'src':                  u,
        'dst':                  v,
        'gcn_link_probability': gcn_prob,
        'common_neighbors':     cn,
        'jaccard':              jc,
        'adamic_adar':          aa,
        'embedding_cosine':     cos_sim,
        'edge_exists':          edge_exists,
        'src_label':            src_label,
        'dst_label':            dst_label,
        'explanation':          explanation,
    }
