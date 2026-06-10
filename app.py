import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="t-SNE Dimensionality Reduction",
    page_icon=None,
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
with open("static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    meta = pickle.load(open("models/meta.pkl", "rb"))
    return meta

meta = load_artifacts()

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 class='main-title'>t-SNE - Dimensionality Reduction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class='info-box'>
    Visualize 784-dimensional MNIST digits in 2D using
    <b>t-SNE (t-distributed Stochastic Neighbor Embedding)</b>
    &nbsp;|&nbsp; Dataset: MNIST Digits
    </div>
    """,
    unsafe_allow_html=True
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Algorithm Info")
st.sidebar.success("Algorithm: t-SNE")
st.sidebar.info(f"Original Dimensions: {meta['n_features_original']}")
st.sidebar.info(f"PCA Pre-reduction: {meta['n_features_original']} -> {meta['pca_pre_dims']} dims")
st.sidebar.info(f"t-SNE Output: 2 dims")
st.sidebar.info(f"Perplexity: {meta['tsne_perplexity']}")
st.sidebar.info(f"Iterations: {meta['tsne_n_iter']}")
st.sidebar.success("Dataset: MNIST Digit Recognizer")
st.sidebar.markdown("---")
st.sidebar.subheader("Model Metrics")
st.sidebar.metric("KL Divergence",          f"{meta['tsne_kl_divergence']:.4f}")
st.sidebar.metric("KNN Acc (t-SNE 2D)",     f"{meta['knn_acc_tsne']*100:.2f}%")
st.sidebar.metric("KNN Acc (PCA-50)",        f"{meta['knn_acc_pca50']*100:.2f}%")
st.sidebar.metric("PCA Pre-reduction Var",   f"{meta['pca_pre_variance']*100:.1f}%")
st.sidebar.metric("Samples Used",            f"{meta['n_samples_used']:,}")

# ── Top Metric Cards ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Reduction Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Original Dimensions",  meta["n_features_original"])
c2.metric("After PCA Pre-reduce", meta["pca_pre_dims"])
c3.metric("After t-SNE",          "2")
c4.metric("KL Divergence",        f"{meta['tsne_kl_divergence']:.4f}")

# ── Sample Digit Grid ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Sample Digit Images from MNIST")
if os.path.exists("static/digit_grid.png"):
    st.image("static/digit_grid.png", use_container_width=True)
    st.caption("Two samples per digit class from the dataset used for t-SNE training.")

# ── Main t-SNE Plot ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("t-SNE 2D Embedding")
if os.path.exists("static/tsne_plot.png"):
    st.image("static/tsne_plot.png", use_container_width=True)
    st.caption(
        "Each point is one digit image projected to 2D. "
        "Well-separated clusters indicate t-SNE successfully captured the digit structure."
    )
else:
    st.warning("Run train_model.py first.")

# ── PCA vs t-SNE ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("PCA vs t-SNE Comparison")
if os.path.exists("static/pca_vs_tsne.png"):
    st.image("static/pca_vs_tsne.png", use_container_width=True)
    st.caption(
        "PCA (left) shows a linear projection - digits overlap heavily. "
        "t-SNE (right) reveals nonlinear cluster structure that PCA misses."
    )
else:
    st.warning("Run train_model.py first.")

# ── Perplexity Comparison ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Perplexity Hyperparameter Effect")
if os.path.exists("static/tsne_perplexity_comparison.png"):
    st.image("static/tsne_perplexity_comparison.png", use_container_width=True)
    st.caption(
        "Perplexity controls the effective number of neighbours. "
        "Too low: fragmented clusters. Too high: global structure lost. "
        "5-50 is typical; 30 is the standard default."
    )
else:
    st.warning("Run train_model.py first.")

# ── KL Divergence Table ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("KL Divergence by Perplexity")
st.caption("Lower KL divergence means the 2D embedding better preserves the original high-dimensional structure.")

kl_rows = [
    {
        "Perplexity": p,
        "KL Divergence": f"{v:.4f}",
        "Selected": "<-- Default" if p == meta["tsne_perplexity"] else ""
    }
    for p, v in sorted(meta["kl_by_perplexity"].items())
]
st.dataframe(pd.DataFrame(kl_rows), use_container_width=True)

# ── Digit Distribution ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Digit Distribution in Sample")

dist_rows = [
    {"Digit": d, "Count": c, "Percentage": f"{c/meta['n_samples_used']*100:.1f}%"}
    for d, c in sorted(meta["class_distribution"].items())
]
st.dataframe(pd.DataFrame(dist_rows), use_container_width=True)

# ── KNN Accuracy Comparison ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Downstream Task: KNN Classification Accuracy")
st.write(
    "A KNN classifier (k=5) was trained on the reduced embeddings "
    "to measure how well the dimensionality reduction preserves class-relevant structure."
)

c1, c2 = st.columns(2)
c1.metric("KNN on t-SNE 2D",  f"{meta['knn_acc_tsne']*100:.2f}%",
          delta="2 dimensions")
c2.metric("KNN on PCA-50 dims", f"{meta['knn_acc_pca50']*100:.2f}%",
          delta="50 dimensions")

# ── PCA vs t-SNE Key Differences ─────────────────────────────────────────────
st.markdown("---")
st.subheader("t-SNE vs PCA - Key Differences")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**PCA**")
    st.write("- Linear transformation")
    st.write("- Maximises global variance")
    st.write("- Deterministic and fast")
    st.write("- Axes have meaningful units (% variance)")
    st.write("- Can reconstruct original data")
    st.write("- Good for compression and preprocessing")
with col2:
    st.markdown("**t-SNE**")
    st.write("- Nonlinear transformation")
    st.write("- Preserves local neighbourhood structure")
    st.write("- Stochastic - different runs give different layouts")
    st.write("- Axes have no interpretable units")
    st.write("- Cannot reconstruct original data")
    st.write("- Best for visualization only; not for compression")

# ── t-SNE Pipeline Explanation ────────────────────────────────────────────────
st.markdown("---")
st.subheader("Pipeline Used in This Project")

st.markdown(
    """
    **Step 1 - Scale**: StandardScaler on 784 raw pixel features

    **Step 2 - PCA Pre-reduction**: 784 -> 50 dimensions  
    (captures {pvar}% variance, speeds up t-SNE dramatically)

    **Step 3 - t-SNE**: 50 -> 2 dimensions  
    (perplexity=30, 1000 iterations, PCA init)
    """.format(pvar=round(meta["pca_pre_variance"]*100, 1))
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888;'>"
    "Unsupervised ML - Module 2 - t-SNE Dimensionality Reduction - MNIST Dataset"
    "</p>",
    unsafe_allow_html=True
)
