import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

def parse_filename(fname):
    """
    Expected format:
    <GEOMETRY>_<SCALER NAME>_<N_NEIGHBORS>.csv

    Examples:
    euclidean_Share of Top 30 (L1 Norm)_20.csv
    cosine_Share of Total_7.csv
    """
    m = re.match(r"(euclidean|cosine)_(.+)_(\d+)\.csv", fname.name)
    if m is None:
        raise ValueError(f"Cannot parse filename: {fname.name}")

    geometry = m.group(1)
    scaler = m.group(2)
    n_neighbors = int(m.group(3))

    return geometry, scaler, n_neighbors

DATA_DIR = Path("files/SpectralClusteringEngine")

files = sorted(DATA_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError("No spectral CSV files found")

dfs = []
for f in files:
    geometry, scaler, n_nb = parse_filename(f)

    df = pd.read_csv(f)
    df["Number of clusters"] = df["Number of clusters"].astype(int)

    df["geometry"] = geometry          # euclidean / cosine
    df["scaler"] = scaler
    df["n_neighbors"] = n_nb

    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all.rename(columns={"Number of clusters": "k"}, inplace=True)
scalers = list(dict.fromkeys(df_all["scaler"]))
n_rows = len(scalers) + 1  # +1 for comparison row

metrics = [
    ("Silhouette_mean", "Silhouette (cosine)", "Separation"),
    ("ARI_mean", "Mean ARI", "Stability"),
    ("Consensus", "Consensus index", "Dominant structure"),
]

scaler_abbr = {
    "Share of Top 30 (L1 Norm)": "L1",
    "Share of Total": "S",
    "TF-IDF": "TF",
    "L2 Normalization": "L2",
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # fixed per scaler

fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=len(metrics),
    figsize=(15, 4 * n_rows),
    sharex=True
)

# ------------------------
# Per-metric columns
# ------------------------
for j, (col, ylabel, title) in enumerate(metrics):

    # -------- Rows 1..N: per-scaler details --------
    for i, scaler in enumerate(scalers):
        ax = axes[i, j]
        df_s = df_all[df_all["scaler"] == scaler]

        # plot all n_neighbors (thin lines)
        for n_nb, g in df_s.groupby("n_neighbors"):
            g = g.sort_values("k")
            ax.plot(g["k"], g[col], linewidth=1, alpha=0.3)

        # annotate best n_neighbors per k (within this scaler)
        idx = df_s.groupby("k")[col].idxmax()
        df_best = df_s.loc[idx].sort_values("k")
        for _, row in df_best.iterrows():
            ax.text(
                row["k"], row[col],
                f'{int(row["n_neighbors"])}',
                fontsize=8, fontweight='bold',
                va='bottom', ha='center', color='black'
            )

        # titles / labels
        if i == 0:
            ax.set_title(title, fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{ylabel}\n[{scaler}]")
        else:
            ax.set_ylabel(ylabel)

        if col in ("ARI_mean", "Consensus"):
            ax.set_ylim(0, 1.05)

        ax.grid(True, alpha=0.2)

    # -------- Last row: comparison across scalers (best per k) --------
    ax_comp = axes[-1, j]

    # collect per-k annotation stacking info
    k_vals = np.sort(df_all["k"].unique())
    k_data = {k: {'infos': [], 'max_y': -np.inf} for k in k_vals}

    for i, scaler in enumerate(scalers):
        df_s = df_all[df_all["scaler"] == scaler]
        idx = df_s.groupby("k")[col].idxmax()
        g = df_s.loc[idx].sort_values("k")

        abbr = scaler_abbr.get(scaler, scaler)
        color = colors[i % len(colors)]

        ax_comp.plot(
            g["k"], g[col],
            marker='o', linewidth=2,
            label=abbr, color=color, alpha=0.7
        )

        for _, row in g.iterrows():
            k_val = row["k"]
            val = row[col]
            k_data[k_val]['infos'].append((f"{abbr}\nn:{int(row['n_neighbors'])}", color))
            k_data[k_val]['max_y'] = max(k_data[k_val]['max_y'], val)

    # stack annotations above each k
    for k_val, data in k_data.items():
        peak_y = data['max_y']
        for rank, (text, color) in enumerate(data['infos']):
            y_offset = 15 + rank * 8
            if k_val <= 3 or j == 1:
                y_offset -= 10
            ax_comp.annotate(
                text,
                xy=(k_val, peak_y),
                xytext=(0, y_offset),
                textcoords="offset points",
                fontsize=6, fontweight='bold',
                color=color, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0)
            )

    # expand y-limit to make room for annotations
    ymin, ymax = ax_comp.get_ylim()
    ax_comp.set_ylim(ymin, ymax * (1.01 if j == 1 else 1.06))

    ax_comp.set_ylabel(ylabel)
    ax_comp.grid(True, alpha=0.2)
    ax_comp.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.8, 1.0))

# ------------------------
# Shared x-axis formatting
# ------------------------
for ax in axes.flat:
    ax.set_xticks(k_vals)
    ax.set_xlabel("Number of clusters (k)")

fig.subplots_adjust(wspace=0.2)
plt.show()
"""
SELECTED_K = 6
for row in axes:
    for ax in row:
        ax.axvline(
            SELECTED_K,
            color="black",
            linestyle="--",
            alpha=0.5
        )
"""

