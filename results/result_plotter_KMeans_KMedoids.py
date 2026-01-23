import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

def plot_k_analysis(engines, METRICS, pattern_map, scaler_labels,fill_std=False):
    n_rows = len(METRICS)
    n_cols = len(engines)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, constrained_layout=True)
    # Ensure axes is always 2D
    if n_rows==1:
        axes = axes.reshape(1, -1)

    for col_idx, (engine_name, engine_label) in enumerate(engines):

        DATA_DIR = Path(f"files/{engine_name}")
        # ---- load data ----
        dfs = {}
        for scaler, regex in pattern_map.items():
            matches = [f for f in DATA_DIR.glob("*.csv") if regex.fullmatch(f.name)]
            if not matches:
                raise FileNotFoundError(f"No file found for {scaler} ({engine_name})")
            df = pd.read_csv(matches[0])
            df["Number of clusters"] = df["Number of clusters"].astype(int)
            dfs[scaler] = df
        # ---- plot ----
        for row_idx, (metric, ylabel, title) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            for i,(scaler, df) in enumerate(dfs.items()):
                print(engine_name)
                print(df.head().columns)

                jitter = 0.005 if (engine_name == "KMedoidsEngine" and scaler != "TF-IDF") else 0.0

                ax.plot(df["Number of clusters"], df[metric]+jitter*(i-1), marker="o", linewidth=2, label=scaler_labels[scaler],alpha=0.8)
                if fill_std:
                    ax.fill_between(
                        df["Number of clusters"],
                        df[metric] - df[metric.replace("_mean", "_std")],
                        df[metric] + df[metric.replace("_mean", "_std")],
                        alpha=0.15
                    )

            # Titles only on first row
            ax.set_ylabel(metric, fontsize=12)

            # Engine label only on first column
            ax.set_title(f"{engine_label}\n{ylabel}")


            ax.set_xlabel("Number of clusters (k)")
            ax.grid(alpha=0.3)

            #if metric in ("ARI_mean", "Consensus"):
            ax.set_ylim(0, 1.02)

    # ---- shared legend ----
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Normalization",
        loc="center right" if metric=="ARI_mean" else "upper right",
    bbox_to_anchor=(1, .31)if metric=="ARI_mean"  else (.95, .9),

        frameon=False
    )
    fig.suptitle("Sensitivity of K-Means and K-Medoids clustering stability to normalization schemes", fontsize=14)
    return fig
scaler_labels = {
    "Share of Top 30 (L1 Norm)": "L1 (Top 30)",
    "Share of Total": "Share (All)",
    "TF-IDF": "TF–IDF",
   "L2 Normalization": "L2"
}



pattern_map = {"Share of Top 30 (L1 Norm)": re.compile(r"Share of Top 30 \(L1 Norm\)_(\d{4}_\d{4})\.csv"),
    "Share of Total": re.compile(r"Share of Total_(\d{4}_\d{4})\.csv"),
    "TF-IDF": re.compile(r"TF-IDF_(\d{4}_\d{4})\.csv"),
   "L2 Normalization": re.compile(r"L2 Normalization_(\d{4}_\d{4})\.csv")
                }
engines = [("KMeansEngine", "K-Means"), ("KMedoidsEngine", "K-Medoids")]

METRICS = [  ("ARI_mean", "Clustering stability accross scalers", "Stability")]
fig = plot_k_analysis(engines, METRICS, pattern_map, scaler_labels)
fig.savefig(f"files/{engines[0]}-{engines[1]}-{METRICS[0][0]}.png", dpi=300, bbox_inches="tight")

plt.show()
METRICS = [("Silhouette_mean (cosine)", "Clustering seperability accross scalers", "Stability"),
("Silhouette_mean (euclidean)", "Clustering seperability accross scalers", "Stability")]

fig = plot_k_analysis(engines, METRICS, pattern_map, scaler_labels)

plt.show()
