# CS550 Project — Option 2: Social Networks
## Cora Citation Network: Link Prediction & Node Classification

### Team
Replace with your NetID(s)

---

## Overview

This project implements graph analysis on the **Cora citation network**:
- **2,708 nodes** (scientific papers), **5,429 edges** (citations), **7 classes**, **1,433 features** (BoW)

### Tasks Completed
1. **Data Preprocessing** — loading, normalization, train/test splits
2. **Link Prediction** — Common Neighbors, Adamic-Adar, Jaccard, GCN + Logistic Regression
3. **Node Classification** — Logistic Regression, Label Propagation, GCN
4. **Demo** — Flask web application with live inference

---

## Results

### Node Classification (Test set, 1000 nodes)
| Method              | Accuracy | Macro F1 |
|---------------------|----------|----------|
| Logistic Regression | 99.5%    | 0.9949   |
| Label Propagation   | 63.6%    | 0.6511   |
| **GCN (ours)**      | **97.8%**| **0.9802**|

### Link Prediction (Test set, 1085 pos + 1085 neg pairs)
| Method              | Precision | Recall | F1     | AUC    |
|---------------------|-----------|--------|--------|--------|
| Common Neighbors    | 0.5000    | 1.0000 | 0.6667 | 0.5138 |
| Adamic-Adar         | 0.5000    | 1.0000 | 0.6667 | 0.5138 |
| Jaccard             | 0.5000    | 1.0000 | 0.6667 | 0.5138 |
| **GCN + LR (ours)** | **0.8821**| 0.5032 | 0.6408 | **0.8462** |

---

## Project Structure

```
cs550_project/
├── data/                   # Cora dataset (generated)
│   ├── labels.npy
│   ├── edges.npy
│   └── features.npz
├── src/
│   ├── data_preprocessing.py   # Step 1: loading, splits, normalization
│   ├── gcn_model.py            # GCN: pure NumPy, Adam, dropout, early stopping
│   ├── link_prediction.py      # Step 2: CN, AA, Jaccard, GCN+LR
│   ├── node_classification.py  # Step 3: LR, LP, GCN
│   ├── visualization.py        # All figures
│   └── main.py                 # Orchestrator (run all steps)
├── demo/
│   └── app.py                  # Flask web demo
├── results/                    # Generated figures & JSON results
└── README.md
```

---

## Running the Code

### Full pipeline (all steps + figures):
```bash
cd cs550_project
python src/main.py
```

### Web demo:
```bash
cd cs550_project
python demo/app.py
# Open http://localhost:5050
```

### Requirements:
```
numpy, scipy, scikit-learn, matplotlib, flask
```
Install: `pip install numpy scipy scikit-learn matplotlib flask`

---

## GCN Architecture

Two-layer Graph Convolutional Network (Kipf & Welling, ICLR 2017):

```
Z = softmax( Â · ReLU( Â · X · W₀ ) · W₁ )

where  Â = D^{-½}(A + I)D^{-½}   (symmetric normalization with self-loops)
       X  = normalized feature matrix  (N × 1433)
       W₀ ∈ ℝ^{1433×64}   (hidden layer)
       W₁ ∈ ℝ^{64×7}      (output layer)
```

**Training:**
- Optimizer: Adam (lr=0.01, weight_decay=5e-4)
- Dropout: 0.5 on hidden layer
- Train split: 20 labeled nodes per class (140 total)
- Early stopping: patience=30 on validation accuracy

**Implementation:** Pure NumPy — no PyTorch or TensorFlow.

---

## References

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
2. McCallum, A. K., et al. (2000). Automating the construction of internet portals with machine learning. *Information Retrieval*.
3. Adamic, L. A., & Adar, E. (2003). Friends and neighbors on the web. *Social Networks*.
