# Explainable and Robust Graph Learning on the Cora Citation Network
**CS550 Project — Option 2 (Social Networks)**

---

## 1. Overview

This project performs graph learning on the **Cora citation network** with a
**from-scratch NumPy** Graph Convolutional Network and a small set of
classical baselines, then layers two optional trustworthiness analyses on top:

- **Node classification** — predict the research topic of a paper from its
  bag-of-words features and the citation graph.
- **Link prediction** — predict whether a citation should exist between
  two papers, using both topological heuristics and learned GCN embeddings.
- **Custom NumPy GCN** — a two-layer Kipf & Welling GCN implemented in pure
  NumPy + scipy.sparse (no PyTorch / PyG / DGL anywhere in the project),
  trained with Adam, dropout, and early stopping.
- **Explainability** — for any node or link the demo (and the offline
  pipeline) produces an occlusion-based explanation: predicted-class
  probabilities, the most influential neighbors, the most important active
  word features, and a one-sentence natural-language summary.
- **Robustness / anti-attack analysis** — re-evaluation under random edge
  removal, fake edge addition, and feature noise; plus a simple
  **DropEdge** training-time regularizer evaluated head-to-head against the
  normal GCN as a candidate anti-attack strategy (results are mixed —
  see §4b).
- **Flask demo** — a single-page web app at `http://localhost:5050` that
  loads the trained artifacts and exposes live `/classify` and
  `/predict_link` endpoints, both returning the explanation fields.

---

## 2. Dataset

[Cora](https://linqs.org/datasets/#cora) — a benchmark citation graph
distributed under `data/cora/`:

| Property                | Value |
|-------------------------|-------|
| Nodes (papers)          | 2,708 |
| Raw citation links      | **5,429** (one row per line in `cora.cites`) |
| Unique undirected edges | **5,278** (used by every model in this project) |
| Binary BoW features     | 1,433 |
| Classes                 | 7 (Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory) |

> **Why two edge counts?** The raw `cora.cites` file has 5,429 rows because
> some citations are listed in both directions (paper A cites B *and*
> B cites A) and a handful of exact rows appear twice. `data_preprocessing.load_cora`
> canonicalizes every pair to `(min(u, v), max(u, v))`, drops self-loops,
> and deduplicates — leaving **5,278** unique undirected edges. Every
> model (NC GCN, LP GCN, heuristics) operates on the **5,278** version,
> and that is the count printed by the console output, persisted in
> `demo_state.pkl`, and shown in the demo. The raw 5,429 number is
> still surfaced in both the console banner (`Raw citations: 5429`)
> and the demo so the discrepancy is never ambiguous.

The standard transductive split is used: 20 labeled nodes per class for
training (140 total), 500 for validation, 1,000 for test. For link
prediction, edges are split 70 / 10 / 20 into train / val / test, with an
equal number of negative (non-edge) pairs sampled per split.

---

## 3. Methods

### Node classification
- **Logistic Regression** — sklearn `LogisticRegression` on row-normalized
  raw BoW features (no graph signal).
- **Label Propagation** — iterative propagation
  `F^{t+1} = α D⁻¹A F^t + (1−α) Y₀` over the full graph.
- **GCN (ours, NumPy)** — two-layer graph convolution
  `Z = softmax(Â · ReLU(Â · X · W₀) · W₁)` with
  `Â = D^{−½}(A + I)D^{−½}`. Trained with Adam (lr 0.01, weight decay 5e-4),
  dropout 0.5 on the hidden layer, max 300 epochs, early stopping on
  validation accuracy with patience 30.

### Link prediction
- **Common Neighbors**, **Jaccard**, **Adamic-Adar** — classical
  heuristics scored against the **train-only** adjacency.
- **GCN + Logistic Regression (ours)** — a *separate* GCN backbone is
  trained on the train-only adjacency to avoid leaking test edges
  into message passing, and the hidden-layer embeddings `H₁` are
  used to fit an `LR` head on hadamard products of edge endpoints.
- For every method the decision threshold is **tuned on the
  validation split**, not on test (heuristic thresholds default to
  the F1-optimum on val, which on Cora's sparse triadic structure
  turns out to be 0 — see the Results section below).

---

## 4. Optional trustworthiness tasks

### 4a. Transparency / explainability
Per-prediction explanations are computed by **occlusion**:

- **Confidence:** the predicted class index/name, the true class
  (when available), and the full softmax sorted descending.
- **Influential neighbors:** for each neighbor `v` of the target
  node, drop the `(u, v)` edge from a copy of the adjacency,
  re-normalize, re-forward, and record the change in predicted-class
  probability.
- **Important active features:** for each currently-active BoW
  feature `j` on the target node, set `X[u, j] = 0` on a copy and
  record the change in predicted-class probability.
- **Link explanation:** common-neighbor count (against train graph),
  Jaccard, Adamic-Adar, GCN-embedding cosine, and the GCN+LR
  probability — all wrapped in a one-sentence summary plus a
  ground-truth flag indicating whether the edge actually exists in
  the full graph.

### 4b. Robustness / anti-attack
Three perturbation types applied at 5 %, 10 %, 20 %:

- **Random edge removal** — drop a fraction of training edges and
  retrain.
- **Random fake edge addition** — inject the same fraction of
  non-existent edges.
- **Feature noise** — flip the same fraction of the binary feature
  matrix (0 ↔ 1).

A simple **DropEdge** training-time regularizer (Rong et al., 2020) is
also evaluated as a candidate anti-attack strategy: each epoch a fresh
random fraction of edges is masked from the training adjacency, then
re-normalized. Default behavior is unchanged when
`edge_dropout_rate = 0`. The DropEdge-trained GCN is compared
head-to-head against a normal GCN under all three test-time attacks
(`--run-defense`, results in `defense_comparison.json`).

> **Honest defense results (10 % attack rate, macro F1 on the test set,
> from `defense_comparison.json`).**
>
> | Condition           | Normal GCN | DropEdge GCN | Δ (defended − normal) |
> |---------------------|-----------|--------------|-----------------------|
> | Clean               | 0.7855    | 0.7884       | **+0.003** (≈ tie)    |
> | Edge removal 10 %   | 0.7844    | 0.7906       | **+0.006** (small win)|
> | Edge addition 10 %  | 0.7805    | 0.7793       | −0.001 (≈ tie)        |
> | Feature noise 10 %  | 0.6043    | 0.5268       | **−0.077** (worse)    |
>
> The result is **mixed rather than conclusively positive**. DropEdge
> gives a small improvement when training edges are dropped at test
> time and is roughly neutral on clean inputs and on fake-edge
> addition. It does **not** protect against feature noise — and in
> fact performs noticeably worse there. This is expected: DropEdge
> regularizes the model against missing or noisy *graph structure*,
> while feature-noise is a corruption of the *node-feature* input
> space, a different attack surface that edge-level dropout has no
> mechanism to defend. We report DropEdge here as an honest negative /
> mixed result rather than as a successful general defense; a true
> feature-space defense (e.g. feature dropout, denoising auto-encoders,
> adversarial training on `X`) would be needed to address noise on
> `X`.

---

## 5. Install

Requires Python ≥ 3.9. Install dependencies with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` lists `numpy`, `scipy`, `scikit-learn`,
`matplotlib`, `flask`, and `pytest` — no deep-learning framework is
needed.

The Cora dataset is shipped in this repository under `data/cora/`. If
you ever need to refetch it:
```bash
curl -L 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz' -o data/cora.tgz
tar -xzf data/cora.tgz -C data/
```

---

## 6. Run

### Standard pipeline (≈ 6 s)
```bash
python src/main.py
```
Trains both GCN backbones, runs all baselines, generates every
standard figure, runs the explainability smoke check, and saves all
demo artifacts.

### Optional CLI flags
```bash
python src/main.py --run-robustness        # add the perturbation sweep (~50 s)
python src/main.py --run-defense           # add the DropEdge comparison (~10 s)
python src/main.py --no-save-artifacts     # skip writing demo artifacts

# Hyperparameter overrides (defaults match the values listed in §3):
python src/main.py --epochs 200 --hidden-dim 32 --lr 0.005 \
                   --dropout 0.3 --weight-decay 1e-3 --patience 20 --seed 7
```

### Web demo
```bash
python demo/app.py
# Then open http://localhost:5050
```
The demo loads the artifacts produced by `python src/main.py` and
**does not retrain** (~1 s startup). If artifacts are missing it exits
with a clear message telling you to run `main.py` first.

### Tests
```bash
python -m pytest tests/ -v
```
25 tests, ~1 s. They cover data loading, edge / node split disjointness,
GCN forward shape and softmax / NaN sanity, link-prediction heuristic
values on a tiny toy graph, and the explainer contract.

---

## 7. Results

These numbers are from the deterministic default run
(`python src/main.py`, seed 42). They are written into the JSON files
listed below; report any of them directly.

### Node classification (test = 1000 nodes)
| Method                | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-----------------------|----------|-----------------|--------------|----------|
| Logistic Regression   | 0.5670   | 0.5517          | 0.5914       | 0.5596   |
| Label Propagation     | 0.6810   | 0.6683          | 0.7331       | 0.6786   |
| **GCN (ours)**        | **0.7980** | **0.7877**    | **0.8186**   | **0.7955** |

### Link prediction (test = 1055 pos + 1055 neg)
| Method            | Precision | Recall | F1     | AUC    | Threshold |
|-------------------|-----------|--------|--------|--------|-----------|
| Common Neighbors  | 0.9718    | 0.3261 | 0.4883 | 0.6587 | 0.5000    |
| Adamic-Adar       | 0.9718    | 0.3261 | 0.4883 | 0.6596 | 0.1029    |
| Jaccard           | 0.9718    | 0.3261 | 0.4883 | 0.6585 | 0.0038    |
| **GCN + LR (ours)** | 0.6738  | 0.7754 | **0.7210** | **0.7764** | 0.3920 |

> **About the heuristic numbers.** Cora's training graph is very
> sparse, so only ≈ 33 % of held-out positive pairs share *any* common
> neighbor in the train-only adjacency, while almost no negative pair
> does. This makes the neighbor-counting heuristics **high-precision
> but low-recall**: when they fire they are almost always right
> (P ≈ 0.97), but they stay silent on roughly two thirds of the real
> edges (R ≈ 0.33).
>
> **Why all three heuristics report identical Precision / Recall / F1.**
> The strict-`>` threshold tuner (`tune_threshold`) picks each method's
> F1-best operating point on the validation split: 0.5000 for Common
> Neighbors (an integer count), 0.1029 for Adamic-Adar, 0.0038 for
> Jaccard. At each of those thresholds the decision rule collapses to
> the same predicate — *"predict a link iff the two endpoints share at
> least one common neighbor in the train graph"* — because Adamic-Adar
> and Jaccard are both strictly positive whenever the common-neighbor
> count is ≥ 1, and strictly zero otherwise. Same predictions ⇒ same
> P / R / F1. The AUC values differ slightly (0.6585 – 0.6596) only
> because the three methods *rank* the non-zero pairs differently.
>
> **Why GCN + LR wins.** It learns dense node embeddings from the
> graph and the bag-of-words features, then a logistic-regression head
> over the hadamard product of endpoint embeddings — so it can score
> pairs that have **no** shared neighbor in the training graph
> (impossible for the heuristics). The result is a much better
> precision–recall balance (0.674 / 0.775 → **F1 = 0.7210**) and a
> noticeably higher AUC (**0.7764** vs ≈ 0.66 for the heuristics).

---

## 8. Output files (under `results/`)

### Standard pipeline
| File | Contents |
|---|---|
| `node_classification_results.json` | Accuracy + macro P/R/F1 per method |
| `link_prediction_results.json`     | P/R/F1/AUC + tuned threshold per method |
| `gcn_training.png`                 | Loss + val/test acc curves |
| `node_classification_comparison.png` | Bar chart across NC methods |
| `link_prediction_comparison.png`   | Bar chart across LP methods |
| `confusion_matrix_gcn.png`         | Normalized GCN confusion matrix |
| `per_class_f1.png`                 | Per-class F1 across NC methods |
| `embedding_tsne.png`               | t-SNE of GCN hidden embeddings |
| `graph_statistics.png`             | Degree + class distribution |

### Explainability (default run)
| File | Contents |
|---|---|
| `explanation_node_<id>.json`       | Full node explanation dict |
| `explanation_link_<u>_<v>.json`    | Full link explanation dict |
| `explain_node_example.png`         | 3-panel: probs / neighbor influence / feature importance |
| `explain_link_example.png`         | 2-panel: bounded scores + heuristic counts |

### Robustness (`--run-robustness`)
| File | Contents |
|---|---|
| `robustness_results.json`          | Baseline + every (perturbation, rate) result |
| `robustness_comparison.png`        | 3-panel overview (NC + LP F1 vs rate) |
| `robustness_node_f1.png`           | NC F1 + drop, per perturbation type |
| `robustness_link_f1.png`           | LP F1 + drop, per perturbation type |

### Defense (`--run-defense`)
| File | Contents |
|---|---|
| `defense_comparison.json`          | Normal vs DropEdge GCN under three attacks |
| `defense_comparison.png`           | Grouped bars: clean / removal / addition / noise |

### Artifacts persisted for the demo (skip with `--no-save-artifacts`)
| File | Contents |
|---|---|
| `gcn_weights.npz`        | Both GCN backbones' `W₀`, `W₁` |
| `gcn_embeddings.npy`     | Hidden-layer `H₁` from the LP backbone |
| `node_probabilities.npy` | Softmax `Z` from the NC backbone |
| `lp_predictions.npy`     | Full-graph Label Propagation predictions |
| `lr_predictions.npy`     | Full-graph LR-baseline predictions |
| `splits.npz`             | `train_idx`, `val_idx`, `test_idx`, `train_pos` |
| `link_classifier.pkl`    | sklearn LR head for GCN+LR link prediction |
| `demo_state.pkl`         | `link_threshold`, `n_features`, `n_hidden`, `n_classes`, `class_names` |

---

## 9. Demo usage

After `python src/main.py` has produced the artifacts:

```bash
python demo/app.py
# then open http://localhost:5050
```

The page has two interactive cards plus two summary tables:

- **Node Classification** — enter a node ID (0–2707) and click
  *Classify Node*. The card returns the GCN prediction + confidence,
  the full sorted probability distribution, the same prediction from
  Label Propagation and Logistic Regression, the top-5 most
  influential neighbors with their classes and confidence drops, and
  the top-5 most important active BoW features. A one-sentence
  natural-language explanation appears at the top of the explanation
  panel.

- **Link Prediction** — enter source and target node IDs (different,
  both 0–2707) and click *Predict Link*. The card returns the GCN+LR
  link probability against the val-tuned threshold, the heuristic
  scores (CN, Jaccard, AA), the embedding cosine similarity, whether
  the edge actually exists in the original graph, and a one-sentence
  natural-language summary.

Validation messages handled in the UI:
- *Node ID must be between 0 and 2707.*
- *Source and target must be different.*
- *Inputs must be integers.*

---

## 10. Reproducibility

- Single global `--seed` flag (default `42`) controls every random
  choice: NumPy and `random` are reseeded at the start of `main()`,
  every robustness / defense sampler accepts a `seed` argument, and
  the GCN's Glorot init / dropout streams are deterministic at a
  fixed seed.
- All artifacts under `results/` are produced from this deterministic
  pipeline — running `python src/main.py` twice yields identical
  weights, identical metric JSON, and a demo that returns identical
  predictions across restarts.
- The demo never retrains; it loads the saved weights, embeddings,
  and predictions directly. Restarting the demo gives byte-identical
  predictions until you re-run `python src/main.py`.
- Tests (`pytest tests/`) avoid full GCN training entirely — they
  use tiny toy graphs and untrained GCN forward passes, so the suite
  runs in ~1 s and stays insulated from training non-determinism.

---

## Project structure

```
cs550_project/
├── data/cora/                  # Cora content + cites files
├── src/
│   ├── data_preprocessing.py   # loading, splits, normalization
│   ├── gcn_model.py            # NumPy GCN + DropEdge
│   ├── link_prediction.py      # CN / AA / Jaccard / GCN+LR + threshold tuning
│   ├── node_classification.py  # LR baseline + Label Propagation + GCN eval
│   ├── visualization.py        # all matplotlib plots
│   ├── explainability.py       # occlusion-based node + link explainers
│   ├── robustness.py           # perturbations, sweep, DropEdge defense
│   └── main.py                 # orchestrator (argparse CLI)
├── demo/app.py                 # Flask demo (loads saved artifacts)
├── tests/                      # pytest suite (25 tests, ~1 s)
├── results/                    # generated JSON, PNG, npz, pkl
├── requirements.txt
└── README.md
```

---

## References

1. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification
   with Graph Convolutional Networks*. ICLR.
2. Adamic, L. A., & Adar, E. (2003). *Friends and neighbors on the
   web*. Social Networks 25(3).
3. Rong, Y., Huang, W., Xu, T., & Huang, J. (2020). *DropEdge: Towards
   Deep Graph Convolutional Networks on Node Classification*. ICLR.
4. McCallum, A. K., Nigam, K., Rennie, J., & Seymore, K. (2000).
   *Automating the construction of internet portals with machine
   learning*. Information Retrieval 3(2). (Cora dataset.)
