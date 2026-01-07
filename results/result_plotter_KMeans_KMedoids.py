import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------
# Configuration
# ------------------------
engine_name = "KMeansEngine"
#engine_name = "KMedoidsEngine"

DATA_DIR = Path(f"files/{engine_name}")

scaler_labels = {
    "Share of Top 30 (L1 Norm)": "L1 (Top 30)",
    "Share of Total": "Share (All)",
    "TF-IDF": "TF–IDF",
    "L2 Normalization": "L2"
}

METRICS = [
    ("Silhouette_mean", "Cluster separation", "Separation"),
    ("ARI_mean", "Clustering stability", "Stability"),
    ("Consensus", "Dominant structure", "Dominant structure")
]
#if engine_name == "KMeansEngine":
#    METRICS.append(("Inertia_mean", "Inertia", "Inertia"))
# ------------------------
# Load data
# ------------------------
dfs = {}
for scaler in scaler_labels:
    f = DATA_DIR / f"{scaler}.csv"
    df = pd.read_csv(f)
    df["Number of clusters"] = df["Number of clusters"].astype(int)
    dfs[scaler] = df

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
    for scaler, df in dfs.items():
        ax.plot(
            df["Number of clusters"],
            df[col],
            linewidth=2,
            marker="o",
            label=scaler_labels[scaler]
        )
        if col!= "Consensus":
            ax.fill_between(
                df["Number of clusters"],
                df[col] - df[col.replace("_mean", "_std")],
                df[col] + df[col.replace("_mean", "_std")],
                alpha=0.15
            )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    if col in ("ARI_mean", "Consensus"):
        ax.set_ylim(0, 1.05)

# Legend
# Legend (single, below figure)
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    title="Normalization",
    loc="upper right",
    bbox_to_anchor=(0.98, 0.90),
    ncol=1,
    frameon=False
)




#fig.subplots_adjust(wspace=0.2)
plt.show()
