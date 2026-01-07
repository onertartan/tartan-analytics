import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

# ------------------------
# Configuration
# ------------------------
DATA_DIR = Path("files/GMMEngine")

scaler_labels = {
    "Share of Top 30 (L1 Norm)": "L1 (Top 30)",
    "Share of Total": "Share (All)",
    "TF-IDF": "TF–IDF",
    "L2 Normalization": "L2"
}

covariance_types = {
    "diag": "Diagonal",
    "tied": "Tied",
    "spherical": "Spherical"
}

linestyles = {
    "diag": "-",
    "tied": "--",
    "spherical": ":"
}

METRICS = [
    ("Silhouette_mean", "Cluster separation", "Separation"),
    ("ARI_mean", "Clustering stability", "Stability"),
    ("Consensus", "Dominant structure", "Dominant structure")
]

# ------------------------
# Load data
# ------------------------
dfs = []

for scaler, cov in product(scaler_labels, covariance_types):
    f = DATA_DIR / f"{scaler}_cov_{cov}.csv"
    if not f.exists():
        continue

    df = pd.read_csv(f)
    df["Number of clusters"] = df["Number of clusters"].astype(int)
    df["scaler"] = scaler
    df["covariance"] = cov
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# ------------------------
# Plot
# ------------------------
fig, axes = plt.subplots(
    1, len(METRICS),
    figsize=(15, 4),
    sharex=True,
    constrained_layout=True
)

for ax, (col, ylabel, title) in zip(axes, METRICS):
    for (scaler, cov), g in df_all.groupby(["scaler", "covariance"]):
        ax.plot(
            g["Number of clusters"],
            g[col],
            linewidth=1,
            marker="o",
            linestyle=linestyles[cov],
            label=f"{scaler_labels[scaler]} | {covariance_types[cov]}"
        )

        if False and col != "Consensus":
            ax.fill_between(
                g["Number of clusters"],
                g[col] - g[col.replace("_mean", "_std")],
                g[col] + g[col.replace("_mean", "_std")],
                alpha=0.10
            )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    if col in ("ARI_mean", "Consensus"):
        ax.set_ylim(0, 1.05)

# ------------------------
# Legends (two-part, clean)
# ------------------------

# Legend for normalizations (colors)
handles, labels = axes[0].get_legend_handles_labels()
unique = dict(zip(labels, handles))

fig.legend(
    unique.values(),
    unique.keys(),
    title="Normalization | Covariance",
    loc="lower center",
    bbox_to_anchor=(0.85, 0.58),
    ncol=2,
    frameon=False
)

plt.show()
