# t-SNE Dimensionality Reduction - MNIST Digit Visualization
Unsupervised ML - Module 2 - Assignment 2

---

## Dataset
**MNIST in CSV** - Download from Kaggle:
  https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

Place the downloaded file as:
  tSNE_DimensionalityReduction_Tek/data/mnist_train.csv

---

## Project Structure

  tSNE_DimensionalityReduction_Tek/
  ├── app.py                                      <- Streamlit web app
  ├── train_model.py                              <- Train and save model artifacts
  ├── requirements.txt
  ├── README.md
  ├── data/
  │   └── mnist_train.csv                        <- (download from Kaggle)
  ├── models/
  │   ├── tsne_embedding.pkl                     <- Saved 2D t-SNE embedding (generated)
  │   ├── tsne_labels.pkl                        <- Labels for the embedding (generated)
  │   ├── pca_pre.pkl                            <- PCA pre-reducer 784->50 (generated)
  │   ├── pca_2d.pkl                             <- PCA 2D for comparison (generated)
  │   ├── scaler.pkl                             <- StandardScaler (generated)
  │   └── meta.pkl                              <- All stats and metadata (generated)
  ├── notebooks/
  │   └── tSNE_DimensionalityReduction.ipynb   <- Full hands-on notebook
  └── static/
      ├── styles.css
      ├── digit_grid.png                         <- (generated)
      ├── tsne_plot.png                          <- (generated)
      ├── pca_vs_tsne.png                        <- (generated)
      └── tsne_perplexity_comparison.png        <- (generated)

---

## Quick Start

  pip install -r requirements.txt
  python train_model.py      (note: takes 3-5 minutes due to t-SNE computation)
  streamlit run app.py

---

## Algorithm: t-SNE

  Perplexity       : 30 (default; comparison at 5/30/50/100 in notebook)
  Iterations       : 1000
  Learning Rate    : auto
  Init             : pca (more stable than random)
  Pre-reduction    : PCA 784 -> 50 dims before t-SNE
  Samples used     : 5000 (t-SNE is O(n^2) in memory)

---

## Pipeline

  Raw pixels (784)
    -> StandardScaler
    -> PCA pre-reduction (50 dims, retains ~85% variance)
    -> t-SNE (2 dims, KL divergence minimized)

---

## Key Differences vs PCA (Assignment 1)

  PCA                         t-SNE
  Linear transformation       Nonlinear transformation
  Maximises global variance   Preserves local neighbourhood structure
  Deterministic               Stochastic (random seed affects layout)
  Axes = % variance           Axes have no interpretable meaning
  Can reconstruct data        Cannot reconstruct original data
  Good for compression        Good for visualization only
  Fast (O(n))                 Slow (O(n^2))
  Generalizes to new points   Does not generalize to new points

---

## Evaluation Metrics

  KL Divergence   - Optimization objective; lower = better 2D representation
  KNN Accuracy    - Downstream classification on the 2D embedding (diagnostic only)
  PCA Variance    - How much info is retained after PCA pre-reduction

---

## Important Notes

  t-SNE is for visualization only.
  Do not use t-SNE embeddings as features for downstream ML models.
  Use PCA-50 or raw features for classification tasks.
  Each run may produce a different layout due to stochastic initialization.
