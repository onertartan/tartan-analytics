import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

DEFAULT_METRICS = [
    ("Silhouette_mean", "Cluster separation", "Separation"),
    ("ARI_mean", "Clustering stability", "Stability"),
    ("Consensus", "Dominant structure", "Dominant structure"),
]

DEFAULT_SCALER_ABBR = {
    "Share of Top 30 (L1 Norm)": "L1",
    "Share of Total": "S",
    "TF-IDF": "TF",
    "L2 Normalization": "L2",
}
def _parse_spectral_filename(fname: Path):
    m = re.match(
        r"(.+?)_(euclidean|cosine)_(\d{4}_\d{4})_(\d+)\.csv",
        fname.name
    )
    if m is None:
        raise ValueError(f"Cannot parse filename: {fname.name}")

    scaler = m.group(1)
    geometry = m.group(2)
    year_range = m.group(3)
    n_neighbors = int(m.group(4))

    return geometry, scaler, year_range, n_neighbors



def plot_spectral_k_analysis(
    geometry="both",                      # "cosine" | "euclidean" | "both" | ("cosine","euclidean")
    data_dir="files/SpectralClusteringEngine",
    metrics=DEFAULT_METRICS,
    scaler_abbr=DEFAULT_SCALER_ABBR,
    show_comparison_row=True,
    figsize_per_row=4.0,                  # row height multiplier
    alpha_thin=0.30,
    linewidth_thin=1.0,
    linewidth_best=2.0,
    marker_best="o",
    save_path=None,
    dpi=300,
):
    """
    Spectral clustering k-analysis plotter.

    Rows:
      - One row per scaler (shows all n_neighbors as thin lines + best n_neighbors per k annotations)
      - Optional last row: best-per-k comparison across scalers
    Columns:
      - Metrics (e.g., Silhouette, ARI, Consensus)

    Geometry selection:
      geometry="cosine" or "euclidean" filters files.
      geometry="both" or tuple/list includes both (data combined).
    """

    # ---- normalize geometry argument ----
    if geometry == "both":
        geometries = {"euclidean", "cosine"}
    elif isinstance(geometry, (tuple, list, set)):
        geometries = set(geometry)
    else:
        geometries = {str(geometry).lower()}

    bad = geometries - {"euclidean", "cosine"}
    if bad:
        raise ValueError(f"Invalid geometry values: {bad}. Use 'cosine', 'euclidean', or 'both'.")

    DATA_DIR = Path(data_dir)
    files = sorted(f for f in DATA_DIR.glob("*.csv")  if not f.name.startswith("consensus_labels_all_") )

    if not files:
        raise FileNotFoundError(f"No spectral CSV files found in: {data_dir}")

    # ---- load and filter ----
    dfs = []
    for f in files:
        geom, scaler,year_range, n_nb = _parse_spectral_filename(f)
        if geom not in geometries:
            continue

        df = pd.read_csv(f)
        df["Number of clusters"] = df["Number of clusters"].astype(int)
        df = df.rename(columns={"Number of clusters": "k"})

        df["geometry"] = geom
        df["scaler"] = scaler
        df["n_neighbors"] = n_nb
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No files matched geometry={geometries} in {data_dir}")

    df_all = pd.concat(dfs, ignore_index=True)

    # ---- identify scalers, k values ----
    scalers = list(dict.fromkeys(df_all["scaler"]))
    k_vals = np.sort(df_all["k"].unique())

    # ---- layout ----
    n_rows = len(scalers) + (1 if show_comparison_row else 0)
    n_cols = len(metrics)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, figsize_per_row * n_rows),
        sharex=True,
        constrained_layout=True
    )

    # ensure axes is always 2D
    if n_rows == 1:
        axes = np.array(axes).reshape(1, -1)

    # fixed colors per scaler (stable order)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # ---- per-metric columns ----
    for j, (col, ylabel, title) in enumerate(metrics):

        # rows 0..len(scalers)-1 : per-scaler detail
        for i, scaler in enumerate(scalers):
            ax = axes[i, j]
            df_s = df_all[df_all["scaler"] == scaler]

            # thin lines: all n_neighbors (within this scaler)
            for n_nb, g in df_s.groupby("n_neighbors"):
                g = g.sort_values("k")
                ax.plot(g["k"], g[col], linewidth=linewidth_thin, alpha=alpha_thin)

            # annotate best n_neighbors per k (within this scaler)
            idx = df_s.groupby("k")[col].idxmax()
            df_best = df_s.loc[idx].sort_values("k")
            for _, row in df_best.iterrows():
                ax.text(
                    row["k"], row[col],
                    f"{int(row['n_neighbors'])}",
                    fontsize=8, fontweight="bold",
                    va="bottom", ha="center", color="black"
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

        # last row: comparison across scalers (best per k)
        if show_comparison_row:
            ax_comp = axes[-1, j]

            k_data = {k: {"infos": [], "max_y": -np.inf} for k in k_vals}

            for i, scaler in enumerate(scalers):
                df_s = df_all[df_all["scaler"] == scaler]
                idx = df_s.groupby("k")[col].idxmax()
                g = df_s.loc[idx].sort_values("k")

                abbr = scaler_abbr.get(scaler, scaler)
                color = colors[i % len(colors)]

                ax_comp.plot(
                    g["k"], g[col],
                    marker=marker_best, linewidth=linewidth_best,
                    label=abbr, color=color, alpha=0.7
                )

                for _, row in g.iterrows():
                    k_val = row["k"]
                    val = row[col]
                    k_data[k_val]["infos"].append((f"{abbr}\nn:{int(row['n_neighbors'])}", color))
                    k_data[k_val]["max_y"] = max(k_data[k_val]["max_y"], val)

            # stack annotations above each k
            for k_val, data in k_data.items():
                peak_y = data["max_y"]
                for rank, (text, color) in enumerate(data["infos"]):
                    y_offset = 15 + rank * 8
                    if k_val <= 3 or j == 1:
                        y_offset -= 10
                    ax_comp.annotate(
                        text,
                        xy=(k_val, peak_y),
                        xytext=(0, y_offset),
                        textcoords="offset points",
                        fontsize=6, fontweight="bold",
                        color=color, ha="center", va="bottom",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0)
                    )

            ymin, ymax = ax_comp.get_ylim()
            ax_comp.set_ylim(ymin, ymax * (1.01 if j == 1 else 1.06))
            ax_comp.set_ylabel(ylabel)
            ax_comp.grid(True, alpha=0.2)
            ax_comp.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.8, 1.0))

    # shared x-axis formatting
    for ax in axes.flat:
        ax.set_xticks(k_vals)
        ax.set_xlabel("Number of clusters (k)")

    # optional: annotate which geometries are included
    geom_label = " & ".join(sorted(geometries))
    fig.suptitle(f"Spectral clustering k-analysis ({geom_label})", y=1.01, fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig

plot_spectral_k_analysis(geometry="cosine")
plt.show()
