import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import OPTICS
from sklearn.datasets import make_moons, make_blobs
import pandas as pd
import numpy as np

st.set_page_config(page_title="OPTICS Clustering", layout="centered")
st.title("🔍 Day 24 — OPTICS: Density-Based Clustering")

# Dataset selection
dataset_choice = st.radio("Choose a dataset", ["Moons", "Blobs"])

# Generate synthetic dataset
if dataset_choice == "Moons":
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
else:
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

st.subheader("📂 Sample Data (first 5 rows)")
st.write(pd.DataFrame(X, columns=["Feature1", "Feature2"]).head())

# OPTICS parameters
min_samples = st.slider("Minimum samples", 2, 20, 5)
xi = st.slider("Xi (cluster sensitivity)", 0.01, 0.2, 0.05)
min_cluster_size = st.slider("Minimum cluster size (fraction)", 0.01, 0.5, 0.1)

# Run OPTICS
optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
labels = optics.fit_predict(X)

# Plot results
fig, ax = plt.subplots()
palette = sns.color_palette("Set1", len(set(labels)))
sns.scatterplot(
    x=X[:, 0], y=X[:, 1],
    hue=labels,
    palette=palette,
    legend="full",
    ax=ax,
    s=50
)
ax.set_title("OPTICS Clustering Results")
st.pyplot(fig)

# Cluster summary
st.subheader("📊 Cluster Label Counts")
st.write(pd.Series(labels).value_counts())

st.success("✅ OPTICS clustering complete — density-based patterns detected.")
