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

import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

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
    plot_per_class_f1, plot_graph_statistics, plot_embedding_tsne
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


def main():
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

    # Two adjacency matrices:
    #   A_norm_full : full graph, used for node classification GCN
    #   A_norm_train: training edges only, used for link prediction GCN
    A_train      = link_splits['train_A']
    A_norm_train = normalize_adjacency(A_train)
    A_norm       = normalize_adjacency(A_full)  # full graph for node classification GCN

    print(f"\n  Dataset summary:")
    print(f"    Nodes:          {N}")
    print(f"    Total edges:    {len(edges)}")
    print(f"    Feature dim:    {features.shape[1]}")
    print(f"    Classes:        {int(labels.max())+1}")
    print(f"    Train nodes:    {len(train_idx)}")
    print(f"    Val nodes:      {len(val_idx)}")
    print(f"    Test nodes:     {len(test_idx)}")
    print(f"    Train edges:    {len(link_splits['train']['pos'])}")
    print(f"    Test pos/neg edges: {len(link_splits['test']['pos'])}/{len(link_splits['test']['neg'])}")

    # ──────────────────────────────────────────────────────────
    # STEP 2: Train GCN (shared backbone)
    # ──────────────────────────────────────────────────────────
    banner("STEP 2: Training GCN Backbone")

    print("\n  Hyperparameters:")
    print("    Hidden dim:     64")
    print("    Learning rate:  0.01")
    print("    Weight decay:   5e-4")
    print("    Dropout:        0.5")
    print("    Max epochs:     300")
    print("    Early stop:     30\n")

    gcn_model, history = train_gcn(
        A_norm, X_dense, labels,
        train_idx, val_idx, test_idx,
        n_hidden=64, lr=0.01, epochs=300,
        weight_decay=5e-4, dropout=0.5,
        patience=30, verbose=True
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
    # STEP 3: Link Prediction
    # ──────────────────────────────────────────────────────────
    banner("STEP 3: Link Prediction")
    print()
    link_results = run_link_prediction(
        A_train, link_splits,
        gcn_model=gcn_model, A_norm=A_norm_train, X_dense=X_dense
    )

    print("\n[Plot] Link prediction comparison...")
    plot_link_prediction_comparison(link_results)

    # Save link prediction results
    lp_summary = {}
    for method, res in link_results.items():
        lp_summary[method] = {
            'precision': round(res['precision'], 4),
            'recall':    round(res['recall'],    4),
            'f1':        round(res['f1'],        4),
            'auc':       round(res['auc'],       4),
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
    # Summary Table
    # ──────────────────────────────────────────────────────────
    banner("RESULTS SUMMARY")

    print("\n── Link Prediction ──────────────────────────────────────")
    print(f"  {'Method':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'-'*62}")
    for method, res in lp_summary.items():
        label = method.replace('_', ' ').title()
        print(f"  {label:<22} {res['precision']:>10.4f} {res['recall']:>10.4f} {res['f1']:>10.4f} {res['auc']:>10.4f}")

    print("\n── Node Classification ──────────────────────────────────")
    print(f"  {'Method':<22} {'Accuracy':>10} {'Macro P':>10} {'Macro R':>10} {'Macro F1':>10}")
    print(f"  {'-'*62}")
    for method, res in nc_summary.items():
        label = method.replace('_', ' ').title()
        print(f"  {label:<22} {res['accuracy']:>10.4f} {res['macro_precision']:>10.4f} {res['macro_recall']:>10.4f} {res['macro_f1']:>10.4f}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"\n  All results saved to: {RESULTS_DIR}/")

    return gcn_model, link_results, nc_results, history


if __name__ == '__main__':
    main()
