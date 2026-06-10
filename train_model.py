"""
train_model.py  -  Run once to generate:
  models/tsne_embedding.pkl
  models/pca_pre.pkl
  models/scaler.pkl
  models/meta.pkl
  static/tsne_plot.png
  static/tsne_perplexity_comparison.png
  static/pca_vs_tsne.png
  static/digit_grid.png

Dataset: MNIST in CSV (Kaggle)
  https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
  Place CSV as: data/mnist_train.csv
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#0f0a1e"
AX_BG  = "#1a0f2e"
ACCENT = "#c084fc"

DIGIT_COLORS = [
    "#f87171","#fb923c","#fbbf24","#a3e635",
    "#34d399","#22d3ee","#60a5fa","#a78bfa",
    "#f472b6","#94a3b8"
]

def style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color(ACCENT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d1f4e")

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/mnist_train.csv")
print("Raw shape:", df.shape)

label_col = df.columns[0]      # usually 'label'
y_all = df[label_col].values
X_all = df.drop(columns=[label_col]).values.astype(np.float32)
print(f"Features: {X_all.shape[1]}  |  Samples: {X_all.shape[0]}")
print(f"Classes: {np.unique(y_all)}  |  Distribution: {np.bincount(y_all)}")

# ── 2. Sample for speed (t-SNE is O(n^2)) ────────────────────────────────────
N_SAMPLE = 5000
np.random.seed(42)
idx = np.random.choice(len(X_all), size=N_SAMPLE, replace=False)
X_sample = X_all[idx]
y_sample = y_all[idx]
print(f"\nUsing {N_SAMPLE} samples for t-SNE")

# ── 3. Scale + PCA Pre-reduction ─────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# Pre-reduce with PCA to 50 dims before t-SNE (standard practice)
PCA_DIMS = 50
pca_pre  = PCA(n_components=PCA_DIMS, random_state=42)
X_pca50  = pca_pre.fit_transform(X_scaled)
pca_var  = pca_pre.explained_variance_ratio_.sum()
print(f"PCA pre-reduction: {X_scaled.shape[1]} -> {PCA_DIMS} dims  ({pca_var*100:.1f}% variance)")

# Also get 2D PCA for comparison
pca_2d   = PCA(n_components=2, random_state=42)
X_pca2d  = pca_2d.fit_transform(X_scaled)

# ── 4. t-SNE with default perplexity ─────────────────────────────────────────
print("\nRunning t-SNE (perplexity=30)...")
tsne_default = TSNE(
    n_components = 2,
    perplexity   = 30,
    max_iter     = 1000,
    learning_rate= "auto",
    init         = "pca",
    random_state = 42,
    n_jobs       = None
)
X_tsne = tsne_default.fit_transform(X_pca50)
print(f"t-SNE KL divergence: {tsne_default.kl_divergence_:.4f}")

# ── 5. Main t-SNE Plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor(BG)
style_ax(ax)

for digit in range(10):
    mask = y_sample == digit
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=DIGIT_COLORS[digit], label=str(digit),
               alpha=0.7, s=18, edgecolors="none")

ax.set_title("t-SNE 2D Projection of MNIST Digits",
             fontsize=14, fontweight="bold")
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.legend(title="Digit", facecolor=AX_BG, labelcolor="white",
          title_fontsize=10, fontsize=10, markerscale=2,
          loc="upper right")
ax.grid(True, linestyle="--", alpha=0.15)

plt.tight_layout()
plt.savefig("static/tsne_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved tsne_plot.png")

# ── 6. PCA vs t-SNE Comparison ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
for ax in axes:
    style_ax(ax)

for digit in range(10):
    mask = y_sample == digit
    axes[0].scatter(X_pca2d[mask, 0], X_pca2d[mask, 1],
                    c=DIGIT_COLORS[digit], label=str(digit),
                    alpha=0.6, s=12, edgecolors="none")
axes[0].set_title("PCA 2D Projection", fontsize=13, fontweight="bold")
axes[0].set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].legend(title="Digit", facecolor=AX_BG, labelcolor="white",
               fontsize=9, markerscale=2, loc="upper right")
axes[0].grid(True, linestyle="--", alpha=0.15)

for digit in range(10):
    mask = y_sample == digit
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    c=DIGIT_COLORS[digit], label=str(digit),
                    alpha=0.7, s=12, edgecolors="none")
axes[1].set_title("t-SNE 2D Projection", fontsize=13, fontweight="bold")
axes[1].set_xlabel("t-SNE Dim 1")
axes[1].set_ylabel("t-SNE Dim 2")
axes[1].legend(title="Digit", facecolor=AX_BG, labelcolor="white",
               fontsize=9, markerscale=2, loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.15)

plt.suptitle("PCA vs t-SNE: MNIST Digit Separation",
             color="white", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("static/pca_vs_tsne.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pca_vs_tsne.png")

# ── 7. Perplexity Comparison ──────────────────────────────────────────────────
# Use smaller subsample for speed
N_PERP   = 2000
idx2     = np.random.choice(N_SAMPLE, size=N_PERP, replace=False)
X_perp   = X_pca50[idx2]
y_perp   = y_sample[idx2]

perplexities = [5, 30, 50, 100]
tsne_results = {}
kl_divs      = {}

print("\nRunning perplexity comparison...")
for p in perplexities:
    print(f"  perplexity={p}...")
    t = TSNE(n_components=2, perplexity=p, max_iter=500,
             learning_rate="auto", init="pca", random_state=42)
    emb = t.fit_transform(X_perp)
    tsne_results[p] = emb
    kl_divs[p]      = round(float(t.kl_divergence_), 4)
    print(f"    KL divergence: {kl_divs[p]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor(BG)
axes = axes.flatten()

for ax, p in zip(axes, perplexities):
    style_ax(ax)
    emb = tsne_results[p]
    for digit in range(10):
        mask = y_perp == digit
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=DIGIT_COLORS[digit], label=str(digit),
                   alpha=0.7, s=14, edgecolors="none")
    ax.set_title(f"Perplexity = {p}  |  KL = {kl_divs[p]:.4f}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    ax.legend(title="Digit", facecolor=AX_BG, labelcolor="white",
              fontsize=8, markerscale=1.5, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.15)

plt.suptitle("t-SNE Perplexity Comparison (MNIST)",
             color="white", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("static/tsne_perplexity_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved tsne_perplexity_comparison.png")

# ── 8. Sample Digit Grid ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 10, figsize=(16, 4))
fig.patch.set_facecolor(BG)
fig.suptitle("Sample Digit Images from MNIST Dataset",
             color="white", fontsize=13, fontweight="bold")

for digit in range(10):
    digit_idx = np.where(y_sample == digit)[0]
    for row in range(2):
        sample_i = digit_idx[row]
        img      = X_sample[sample_i].reshape(28, 28)
        ax       = axes[row, digit]
        ax.imshow(img, cmap="inferno", interpolation="nearest")
        ax.axis("off")
        if row == 0:
            ax.set_title(str(digit), color=DIGIT_COLORS[digit],
                         fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("static/digit_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved digit_grid.png")

# ── 9. KNN on t-SNE Embedding ─────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X_tsne, y_sample, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_tr, y_tr)
knn_acc = accuracy_score(y_te, knn.predict(X_te))
print(f"\nKNN accuracy on t-SNE embedding: {knn_acc*100:.2f}%")

# KNN on PCA 50-dim
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_pca50, y_sample, test_size=0.2, random_state=42)
knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_tr2, y_tr2)
knn_acc_pca = accuracy_score(y_te2, knn2.predict(X_te2))
print(f"KNN accuracy on PCA-50 embedding: {knn_acc_pca*100:.2f}%")

# ── 10. Save Artifacts ────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

meta = {
    "n_samples_used":      N_SAMPLE,
    "n_features_original": X_all.shape[1],
    "pca_pre_dims":        PCA_DIMS,
    "pca_pre_variance":    round(float(pca_var), 4),
    "tsne_perplexity":     30,
    "tsne_n_iter":         1000,  # kept for display
    "tsne_kl_divergence":  round(float(tsne_default.kl_divergence_), 4),
    "knn_acc_tsne":        round(knn_acc, 4),
    "knn_acc_pca50":       round(knn_acc_pca, 4),
    "digit_colors":        DIGIT_COLORS,
    "kl_by_perplexity":    kl_divs,
    "class_distribution":  {int(d): int((y_sample==d).sum()) for d in range(10)},
    "pca_2d_variance":     [round(float(v), 4) for v in pca_2d.explained_variance_ratio_],
}

pickle.dump(X_tsne,   open("models/tsne_embedding.pkl", "wb"))
pickle.dump(y_sample, open("models/tsne_labels.pkl",    "wb"))
pickle.dump(pca_pre,  open("models/pca_pre.pkl",        "wb"))
pickle.dump(pca_2d,   open("models/pca_2d.pkl",         "wb"))
pickle.dump(scaler,   open("models/scaler.pkl",         "wb"))
pickle.dump(meta,     open("models/meta.pkl",           "wb"))

print("\nAll artifacts saved!")
print(f"  t-SNE KL divergence    : {tsne_default.kl_divergence_:.4f}")
print(f"  KNN acc on t-SNE       : {knn_acc*100:.2f}%")
print(f"  KNN acc on PCA-50      : {knn_acc_pca*100:.2f}%")
print(f"  PCA pre-reduction var  : {pca_var*100:.1f}%")
