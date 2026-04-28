"""
CS550 Project - Option 2: Social Networks (Cora Citation Network)
Main orchestrator: runs all steps end-to-end and saves results.

Steps:
  1. Data preprocessing
  2. Link prediction  (Common Neighbors, Adamic-Adar, Jaccard, GCN+LR)
  3. Node classification (Logistic Regression, Label Propagation, GCN)
  4. Generate all figures
  5. Save numeric results to CSV
"""

import os, sys, json, time, argparse, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def parse_args(argv=None):
    """Command-line interface for the offline pipeline.

    Defaults match the previous hard-coded values, so calling
    `python src/main.py` with no flags reproduces the original run
    bit-for-bit.
    """
    p = argparse.ArgumentParser(
        prog='main.py',
        description='CS550 Cora pipeline: GCN node classification + '
                    'link prediction + (optional) robustness / defense.',
    )

    # ── Training hyperparameters ───────────────────────────────────
    p.add_argument('--epochs',       type=int,   default=300,
                   help='Max epochs for GCN training (default: 300).')
    p.add_argument('--hidden-dim',   type=int,   default=64,
                   help='GCN hidden layer width (default: 64).')
    p.add_argument('--lr',           type=float, default=0.01,
                   help='Adam learning rate (default: 0.01).')
    p.add_argument('--dropout',      type=float, default=0.5,
                   help='Hidden-layer dropout probability (default: 0.5).')
    p.add_argument('--weight-decay', type=float, default=5e-4,
                   help='L2 weight decay (default: 5e-4).')
    p.add_argument('--patience',     type=int,   default=30,
                   help='Early-stopping patience on val acc (default: 30).')
    p.add_argument('--seed',         type=int,   default=42,
                   help='Global random seed (default: 42).')

    # ── Run modes ──────────────────────────────────────────────────
    # Long-form `--run-X`; short alias `--X` kept for backward compat.
    p.add_argument('--run-robustness', '--robustness',
                   dest='run_robustness', action='store_true',
                   help='Run STEP 7 robustness sweep (~50s extra).')
    p.add_argument('--run-defense', '--defense',
                   dest='run_defense', action='store_true',
                   help='Run STEP 8 DropEdge defense comparison (~10s extra).')
    # Artifact saving defaults to ON to preserve existing behavior.
    # Use `--no-save-artifacts` to skip the save block.
    p.add_argument('--save-artifacts',
                   action=argparse.BooleanOptionalAction, default=True,
                   dest='save_artifacts',
                   help='Persist demo artifacts to results/ '
                        '(default: on; use --no-save-artifacts to skip).')

    return p.parse_args(argv)


def _set_global_seed(seed):
    """Reseed numpy and random.

    `data_preprocessing` reseeds to 42 at module-load time; calling this
    after import lets the CLI seed override it for both the edge / node
    splits and Glorot initialization inside train_gcn.
    """
    np.random.seed(seed)
    random.seed(seed)

from data_preprocessing import (
    load_cora, build_adjacency, normalize_features,
    normalize_adjacency, split_edges_link_prediction,
    split_nodes_classification
)
from gcn_model import train_gcn, accuracy
from link_prediction import run_link_prediction
from node_classification import run_node_classification
from visualization import (
    plot_gcn_training, plot_link_prediction_comparison,
    plot_node_classification_comparison, plot_confusion_matrix,
    plot_per_class_f1, plot_graph_statistics, plot_embedding_tsne,
    plot_node_explanation, plot_link_explanation, plot_robustness_f1,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def banner(text):
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"{'='*65}")


def save_results_json(data, name):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    with open(path, 'w') as f:
        json.dump(convert(data), f, indent=2)
    print(f"  Saved results → {path}")


def main(argv=None):
    args = parse_args(argv)
    _set_global_seed(args.seed)
    t0 = time.time()

    # ──────────────────────────────────────────────────────────
    # STEP 1: Data Preprocessing
    # ──────────────────────────────────────────────────────────
    banner("STEP 1: Data Preprocessing")

    labels, edges, features = load_cora()
    N = len(labels)

    # Build adjacency matrices
    A_full = build_adjacency(edges, N)

    # Normalize features (row-normalize)
    features_norm = normalize_features(features)

    # Dense feature matrix for GCN
    X_dense = features_norm.toarray()

    # Graph statistics plot
    print("\n[Plot] Graph statistics...")
    plot_graph_statistics(edges, labels)

    # Edge splits (link prediction)
    link_splits = split_edges_link_prediction(edges, N, test_ratio=0.2, val_ratio=0.1)

    # Node splits (classification)
    train_idx, val_idx, test_idx = split_nodes_classification(labels)

    # Two adjacency matrices — keep these strictly separate:
    #   A_norm       : FULL graph (all edges). Used ONLY by the node-
    #                  classification GCN backbone. This is the standard
    #                  transductive setting for Cora node classification
    #                  and does not leak anything for that task because
    #                  test EDGES are not the prediction target there.
    #   A_norm_train : TRAIN edges only (built from link_splits['train_A']).
    #                  Used by every link-prediction step — heuristics
    #                  (CN/AA/Jaccard score against A_train) AND the
    #                  link-prediction GCN backbone (training + inference).
    #                  This guarantees that held-out test edges never enter
    #                  the message-passing graph used to produce LP
    #                  embeddings.
    A_train      = link_splits['train_A']
    A_norm_train = normalize_adjacency(A_train)
    A_norm       = normalize_adjacency(A_full)

    from data_preprocessing import count_raw_citations
    n_raw_citations = count_raw_citations()
    print(f"\n  Dataset summary:")
    print(f"    Nodes:                    {N}")
    print(f"    Raw citation links:       {n_raw_citations}")
    print(f"    Unique undirected edges:  {len(edges)}")
    print(f"    Feature dim:    {features.shape[1]}")
    print(f"    Classes:        {int(labels.max())+1}")
    print(f"    Train nodes:    {len(train_idx)}")
    print(f"    Val nodes:      {len(val_idx)}")
    print(f"    Test nodes:     {len(test_idx)}")
    print(f"    Train edges:    {len(link_splits['train']['pos'])}")
    print(f"    Test pos/neg edges: {len(link_splits['test']['pos'])}/{len(link_splits['test']['neg'])}")

    # ──────────────────────────────────────────────────────────
    # STEP 2: Train GCN backbone for NODE CLASSIFICATION
    # Uses A_norm (full graph) — appropriate for the transductive
    # node-classification task on Cora.
    # ──────────────────────────────────────────────────────────
    banner("STEP 2: Training GCN Backbone (node classification, full graph)")

    print("\n  Hyperparameters:")
    print(f"    Hidden dim:     {args.hidden_dim}")
    print(f"    Learning rate:  {args.lr}")
    print(f"    Weight decay:   {args.weight_decay}")
    print(f"    Dropout:        {args.dropout}")
    print(f"    Max epochs:     {args.epochs}")
    print(f"    Early stop:     {args.patience}")
    print(f"    Seed:           {args.seed}\n")

    gcn_model, history = train_gcn(
        A_norm, X_dense, labels,
        train_idx, val_idx, test_idx,
        n_hidden=args.hidden_dim, lr=args.lr, epochs=args.epochs,
        weight_decay=args.weight_decay, dropout=args.dropout,
        patience=args.patience, verbose=True,
    )

    # Final GCN test accuracy
    preds_full, Z = gcn_model.predict(A_norm, X_dense)
    final_train_acc = accuracy(preds_full, labels, train_idx)
    final_val_acc   = accuracy(preds_full, labels, val_idx)
    final_test_acc  = accuracy(preds_full, labels, test_idx)
    print(f"\n  Final GCN Accuracy → Train: {final_train_acc:.4f}  |  Val: {final_val_acc:.4f}  |  Test: {final_test_acc:.4f}")

    print("\n[Plot] GCN training curves...")
    plot_gcn_training(history)

    # ──────────────────────────────────────────────────────────
    # STEP 2b: Train a SEPARATE GCN backbone for LINK PREDICTION
    # on the train-only graph (A_norm_train).
    #
    # Why a second backbone? The node-classification GCN above was
    # trained on the FULL adjacency, so its weights have already
    # observed every edge — including the held-out link-prediction
    # test edges — through message passing. Reusing those weights
    # for link prediction would leak test-edge structure into the
    # embeddings even if we forwarded with A_norm_train at inference
    # time. Training afresh on A_norm_train ensures the LP backbone
    # never sees test edges.
    # ──────────────────────────────────────────────────────────
    banner("STEP 2b: Training GCN Backbone (link prediction, TRAIN-only graph)")
    gcn_lp_model, _ = train_gcn(
        A_norm_train, X_dense, labels,
        train_idx, val_idx, test_idx,
        n_hidden=args.hidden_dim, lr=args.lr, epochs=args.epochs,
        weight_decay=args.weight_decay, dropout=args.dropout,
        patience=args.patience, verbose=False,
    )
    lp_preds, _ = gcn_lp_model.predict(A_norm_train, X_dense)
    print(f"  LP-backbone node-class val/test acc (A_train only): "
          f"{accuracy(lp_preds, labels, val_idx):.4f} / "
          f"{accuracy(lp_preds, labels, test_idx):.4f}")

    # ──────────────────────────────────────────────────────────
    # STEP 3: Link Prediction
    # Heuristics use A_train. GCN+LR uses the LP-only backbone
    # forwarded through A_norm_train, with LR fit on TRAIN edges,
    # threshold tuned on VAL edges, metrics on TEST edges.
    # ──────────────────────────────────────────────────────────
    banner("STEP 3: Link Prediction")
    print()
    link_results = run_link_prediction(
        A_train, link_splits,
        gcn_model=gcn_lp_model, A_norm_train=A_norm_train, X_dense=X_dense
    )

    print("\n[Plot] Link prediction comparison...")
    plot_link_prediction_comparison(link_results)

    # Save link prediction results (including the val-tuned threshold
    # picked for each baseline / model).
    lp_summary = {}
    for method, res in link_results.items():
        thr = res.get('chosen_threshold', res.get('threshold'))
        lp_summary[method] = {
            'precision':        round(res['precision'], 4),
            'recall':           round(res['recall'],    4),
            'f1':               round(res['f1'],        4),
            'auc':              round(res['auc'],       4),
            'chosen_threshold': round(float(thr), 4) if thr is not None else None,
        }
    save_results_json(lp_summary, 'link_prediction_results')

    # ──────────────────────────────────────────────────────────
    # STEP 4: Node Classification
    # ──────────────────────────────────────────────────────────
    banner("STEP 4: Node Classification")

    nc_results = run_node_classification(
        A_full, features_norm, X_dense, labels,
        train_idx, val_idx, test_idx,
        gcn_model=gcn_model, A_norm=A_norm
    )

    print("\n[Plot] Node classification comparison...")
    plot_node_classification_comparison(nc_results)
    plot_per_class_f1(nc_results)

    if 'gcn' in nc_results:
        plot_confusion_matrix(
            nc_results['gcn']['confusion_matrix'],
            title='GCN Confusion Matrix (Normalized)',
        )

    # Save node classification results
    nc_summary = {}
    for method, res in nc_results.items():
        nc_summary[method] = {
            'accuracy': round(res['accuracy'], 4),
            'macro_precision': round(res['macro_p'], 4),
            'macro_recall':    round(res['macro_r'], 4),
            'macro_f1':        round(res['macro_f1'], 4),
        }
    save_results_json(nc_summary, 'node_classification_results')

    # ──────────────────────────────────────────────────────────
    # STEP 5: Embedding Visualization
    # ──────────────────────────────────────────────────────────
    banner("STEP 5: Embedding Visualization (t-SNE)")
    embeddings = gcn_model._cache['H1']
    plot_embedding_tsne(embeddings, labels)

    # ──────────────────────────────────────────────────────────
    # STEP 6: Explainability smoke check
    # Run after t-SNE because occlusion does many forward passes
    # which overwrite gcn_model._cache (no longer needed past here).
    # ──────────────────────────────────────────────────────────
    banner("STEP 6: Explainability (occlusion smoke check)")
    from explainability import explain_node_prediction
    CLASS_NAMES = [
        'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
        'Probabilistic_Methods', 'Reinforcement_Learning',
        'Rule_Learning', 'Theory',
    ]
    sample_node = int(test_idx[0])
    explanation = explain_node_prediction(
        gcn_model, sample_node, X_dense,
        adjacency_norm=A_norm, adjacency_raw=A_full,
        labels=labels, class_names=CLASS_NAMES, top_k=5,
    )
    save_results_json(explanation, f'explanation_node_{sample_node}')
    plot_node_explanation(explanation)

    print(f"  Node {sample_node}:")
    print(f"    Predicted: {explanation['predicted_class']}  "
          f"(p={explanation['confidence']:.3f})")
    print(f"    True:      {explanation['true_class']}")
    print("    Top neighbor influences:")
    for n in explanation['top_influential_neighbors']:
        lbl = n.get('neighbor_true_label', '?')
        print(f"      node {n['neighbor_id']:>4} "
              f"(class={lbl:<22}) Δconf={n['confidence_drop']:+.4f}")
    print("    Top feature importances:")
    for f in explanation['top_important_features']:
        print(f"      feat {f['feature_idx']:>4}            "
              f"             Δconf={f['confidence_drop']:+.4f}")

    # Also verify graceful handling of an invalid node_id.
    bad = explain_node_prediction(
        gcn_model, -1, X_dense,
        adjacency_norm=A_norm, adjacency_raw=A_full,
        labels=labels, class_names=CLASS_NAMES,
    )
    assert 'error' in bad, "explain_node_prediction should reject invalid node_id"
    print(f"  Invalid node_id check OK → {bad['error']}")

    # ── Link prediction explainer smoke check ─────────────────
    # Use the LP-only backbone's embeddings (forwarded through the
    # train-only adjacency) so the explanation is leak-free.
    from explainability import explain_link_prediction
    from link_prediction import train_link_predictor as _train_link_predictor
    _ = gcn_lp_model.forward(A_norm_train, X_dense, training=False)
    emb_lp = gcn_lp_model._cache['H1']

    # Train the GCN+LR link classifier ONCE here so:
    #   (a) the offline link-explanation JSON below reports the same
    #       GCN+LR probability the demo would compute for this pair
    #       (no more `gcn_link_probability: null` in the saved JSON),
    #   (b) the artifact-persistence step below reuses the same
    #       classifier — no double-fit, no drift between offline and
    #       demo numbers.
    # Leak-free contract preserved: trained ONLY on train positive +
    # train negative pairs; the test edge being explained below is
    # never seen during this fit.
    _link_clf = _train_link_predictor(
        emb_lp, link_splits['train']['pos'], link_splits['train']['neg'],
    )

    sample_edge = link_splits['test']['pos'][0]
    u_e, v_e = int(sample_edge[0]), int(sample_edge[1])
    link_expl = explain_link_prediction(
        u_e, v_e,
        embeddings=emb_lp,
        adjacency_train=A_train,
        adjacency_full=A_full,
        link_model=_link_clf,
        class_names=CLASS_NAMES,
        labels=labels,
    )
    save_results_json(link_expl, f'explanation_link_{u_e}_{v_e}')
    plot_link_explanation(link_expl)
    print(f"  Link explanation for test edge ({u_e}, {v_e}):")
    print(f"    CN={link_expl['common_neighbors']}  "
          f"Jaccard={link_expl['jaccard']:.3f}  "
          f"AA={link_expl['adamic_adar']:.3f}  "
          f"cos={link_expl['embedding_cosine']:.3f}  "
          f"edge_exists={link_expl['edge_exists']}")
    print(f"    {link_expl['explanation']}")

    # Validation guards.
    self_loop = explain_link_prediction(
        0, 0, embeddings=emb_lp,
        adjacency_train=A_train, adjacency_full=A_full,
    )
    assert 'error' in self_loop, "self-loops should be rejected"
    out_of_range = explain_link_prediction(
        -1, 0, embeddings=emb_lp,
        adjacency_train=A_train, adjacency_full=A_full,
    )
    assert 'error' in out_of_range, "out-of-range ids should be rejected"
    print(f"  Self-loop check OK → {self_loop['error']}")

    # ──────────────────────────────────────────────────────────
    # Persist artifacts so demo/app.py can start without retraining.
    # Layout (per project conventions):
    #   - NumPy arrays via np.save / np.savez
    #   - sklearn classifier via pickle (per the project rule)
    #   - small scalar/string metadata via pickle as well
    # Skipped entirely when --no-save-artifacts is passed.
    # ──────────────────────────────────────────────────────────
    if not args.save_artifacts:
        banner("Skipping artifact persistence (--no-save-artifacts)")
    else:
        banner("Persisting artifacts for demo")
        import pickle as _pickle
        from link_prediction import (predict_link_gcn as _predict_link_gcn,
                                     tune_threshold as _tune_threshold)
        from node_classification import label_propagation as _label_propagation
        from sklearn.linear_model import LogisticRegression as _LR
        from sklearn.preprocessing import normalize as _l1_normalize

        # Refresh both backbones' forward caches so saved Z / H1 align
        # with the saved weights (occlusion in STEP 6 may have left
        # _cache in an intermediate state).
        _ = gcn_model.predict(A_norm, X_dense)
        Z_full = gcn_model._cache['Z']
        gcn_lp_model.forward(A_norm_train, X_dense, training=False)
        emb_lp = gcn_lp_model._cache['H1']

        # Defensive: refuse to write any artifact that contains NaN/Inf.
        # The GCN's per-step _assert_finite already guards intermediates,
        # so this is belt-and-suspenders before persisting.
        for arr, name in [
            (gcn_model.W0,   'gcn_model.W0'),
            (gcn_model.W1,   'gcn_model.W1'),
            (gcn_lp_model.W0, 'gcn_lp_model.W0'),
            (gcn_lp_model.W1, 'gcn_lp_model.W1'),
            (Z_full,         'node_probabilities Z'),
            (emb_lp,         'gcn_embeddings H1'),
        ]:
            if not np.all(np.isfinite(arr)):
                raise FloatingPointError(
                    f"refusing to save '{name}': contains NaN/Inf"
                )

        # 1. GCN weights — both backbones in one .npz
        np.savez(os.path.join(RESULTS_DIR, 'gcn_weights.npz'),
                 W0=gcn_model.W0, W1=gcn_model.W1,
                 W0_lp=gcn_lp_model.W0, W1_lp=gcn_lp_model.W1)

        # 2. GCN embeddings — H1 from LP-only backbone (used by /predict_link)
        np.save(os.path.join(RESULTS_DIR, 'gcn_embeddings.npy'), emb_lp)

        # 3. Node softmax probabilities from the node-classification GCN
        np.save(os.path.join(RESULTS_DIR, 'node_probabilities.npy'), Z_full)

        # 4. Reuse the GCN+LR link classifier already fitted above for
        #    the offline link-explanation step (so the saved demo
        #    artifact is the SAME object the explanation JSON used —
        #    no double-fit, no risk of drift). Tune the decision
        #    threshold on validation pairs only.
        _val_pos = link_splits['val']['pos']
        _val_neg = link_splits['val']['neg']
        _val_pairs = np.vstack([_val_pos, _val_neg])
        _y_val = np.array([1] * len(_val_pos) + [0] * len(_val_neg))
        _val_scores = _predict_link_gcn(_link_clf, emb_lp, _val_pairs)
        _link_threshold = float(_tune_threshold(_y_val, _val_scores))

        with open(os.path.join(RESULTS_DIR, 'link_classifier.pkl'), 'wb') as f:
            _pickle.dump(_link_clf, f)

        # 5. Full-graph Label-Propagation predictions (for /classify)
        _lp_preds_full, _ = _label_propagation(A_full, labels, train_idx, 7)
        np.save(os.path.join(RESULTS_DIR, 'lp_predictions.npy'), _lp_preds_full)

        # 6. Full-graph LR-baseline predictions (for /classify)
        _X_norm_l1 = _l1_normalize(X_dense, norm='l1')
        # Same Accelerate-BLAS-FPE workaround as in node_classification.py
        # — finite-checked inputs, scope-suppressed warning, real numerical
        # issues would still be caught by `_assert_finite`.
        from gcn_model import safe_blas as _safe_blas
        with _safe_blas():
            _clf_lr_baseline = _LR(max_iter=300, C=1.0)
            _clf_lr_baseline.fit(_X_norm_l1[train_idx], labels[train_idx])
            _lr_preds_full = _clf_lr_baseline.predict(_X_norm_l1)
        np.save(os.path.join(RESULTS_DIR, 'lr_predictions.npy'), _lr_preds_full)

        # 7. Splits / indices (NumPy arrays → npz, per the rule)
        np.savez(os.path.join(RESULTS_DIR, 'splits.npz'),
                 train_idx=np.asarray(train_idx),
                 val_idx=np.asarray(val_idx),
                 test_idx=np.asarray(test_idx),
                 train_pos=link_splits['train']['pos'])

        # 8. Small metadata pickle (no NumPy arrays in here)
        _demo_state = {
            'link_threshold': _link_threshold,
            'n_features':     int(X_dense.shape[1]),
            'n_hidden':       int(args.hidden_dim),
            'n_classes':      int(labels.max()) + 1,
            # Raw `cora.cites` has 5,429 lines; load_cora collapses
            # (A→B, B→A) and exact duplicates into 5,278 unique
            # undirected edges. Both numbers are surfaced in the demo.
            'n_edges_raw':    int(n_raw_citations),
            'n_edges_unique': int(len(edges)),
            'class_names': [
                'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                'Probabilistic_Methods', 'Reinforcement_Learning',
                'Rule_Learning', 'Theory',
            ],
        }
        with open(os.path.join(RESULTS_DIR, 'demo_state.pkl'), 'wb') as f:
            _pickle.dump(_demo_state, f)

        for fname in ('gcn_weights.npz', 'gcn_embeddings.npy',
                      'node_probabilities.npy', 'lp_predictions.npy',
                      'lr_predictions.npy', 'splits.npz',
                      'link_classifier.pkl', 'demo_state.pkl'):
            print(f"  Saved → {os.path.join(RESULTS_DIR, fname)}")
        print(f"  Tuned link threshold (val): {_link_threshold:.4f}")

    # ──────────────────────────────────────────────────────────
    # Summary Table
    # ──────────────────────────────────────────────────────────
    banner("RESULTS SUMMARY")

    print("\n── Link Prediction ──────────────────────────────────────")
    print(f"  {'Method':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'Thresh':>9}")
    print(f"  {'-'*72}")
    for method, res in lp_summary.items():
        label = method.replace('_', ' ').title()
        thr   = res.get('chosen_threshold')
        thr_s = f"{thr:>9.4f}" if thr is not None else f"{'—':>9}"
        print(f"  {label:<22} {res['precision']:>10.4f} {res['recall']:>10.4f} "
              f"{res['f1']:>10.4f} {res['auc']:>10.4f} {thr_s}")

    print("\n── Node Classification ──────────────────────────────────")
    print(f"  {'Method':<22} {'Accuracy':>10} {'Macro P':>10} {'Macro R':>10} {'Macro F1':>10}")
    print(f"  {'-'*62}")
    for method, res in nc_summary.items():
        label = method.replace('_', ' ').title()
        print(f"  {label:<22} {res['accuracy']:>10.4f} {res['macro_precision']:>10.4f} {res['macro_recall']:>10.4f} {res['macro_f1']:>10.4f}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"\n  All results saved to: {RESULTS_DIR}/")

    # ──────────────────────────────────────────────────────────
    # Optional STEP 7: Robustness experiments
    # Skipped by default because it retrains 1 + 9 GCN pairs
    # (~40-60s extra). Run with:  python src/main.py --robustness
    # ──────────────────────────────────────────────────────────
    if args.run_robustness:
        banner("STEP 7: Robustness Experiments")
        from robustness import run_robustness_experiments
        robustness_results = run_robustness_experiments(
            labels=labels, edges=edges, features=features,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            rates=(0.05, 0.10, 0.20),
            seed=args.seed, save_dir=RESULTS_DIR, verbose=True,
        )
        # Per-task F1 + robustness-drop plots, alongside the combined
        # robustness_comparison.png produced inside the orchestrator.
        plot_robustness_f1(robustness_results, task='nc')
        plot_robustness_f1(robustness_results, task='lp')

    # ──────────────────────────────────────────────────────────
    # Optional STEP 8: Defense (edge-dropout-trained GCN vs normal)
    # Run with:  python src/main.py --run-defense
    # ──────────────────────────────────────────────────────────
    if args.run_defense:
        banner("STEP 8: Defense Comparison (DropEdge)")
        from robustness import run_defense_comparison
        run_defense_comparison(
            labels=labels, edges=edges, features=features,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            edge_dropout_rate=0.10, attack_rate=0.10,
            seed=args.seed, save_dir=RESULTS_DIR, verbose=True,
        )

    return gcn_model, link_results, nc_results, history


if __name__ == '__main__':
    main()
