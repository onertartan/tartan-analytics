import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

DEFAULT_METRICS = [
    #("Silhouette_mean", "Cluster separation", "Separation"),
    ("ARI_mean", "Clustering stability", "Stability"),
]

DEFAULT_SCALER_ABBR = {
    "Share of Top 30 (L1 Norm)": "L1",
    "Share of Total": "S",
    "TF-IDF": "TF",
    "L2 Normalization": "L2",
}
def _parse_spectral_filename(fname: Path,geometry:str):
    m = re.match(rf"(.+?)_({re.escape(geometry)})_(\d{{4}}_\d{{4}})_(\d+)\.csv",fname.name)

    if m is None:
        return None

    scaler = m.group(1)
    geometry = m.group(2)
    year_range = m.group(3)
    n_neighbors = int(m.group(4))

    return geometry, scaler, year_range, n_neighbors

def load_spectral_results(data_dir, geometry):
    DATA_DIR = Path(data_dir)
    files = sorted(
        f for f in DATA_DIR.glob("*.csv")
        if not f.name.startswith("consensus_labels_all_")
    )

    dfs = []
    for f in files:
        parsed = _parse_spectral_filename(f, geometry)
        if parsed is None:
            continue

        geom, scaler, year_range, n_nb = parsed
        df = pd.read_csv(f)
        df["k"] = df["Number of clusters"].astype(int)
        df.drop(columns=["Number of clusters"], inplace=True)

        df["geometry"] = geom
        df["scaler"] = scaler
        df["n_neighbors"] = n_nb
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No files found for geometry={geometry}")

    return pd.concat(dfs, ignore_index=True)
import numpy as np

def plot_spectral_row(
    ax_row,
    df_all,
    metric="ARI_mean",
    ylabel="Clustering stability",
    title="Stability",
    scaler_abbr=DEFAULT_SCALER_ABBR,
    show_comparison_col=True,
    alpha_thin=0.30,
    linewidth_thin=1.0,
    linewidth_best=2.0,
    marker_best="o",
):
    scalers = list(dict.fromkeys(df_all["scaler"]))
    k_vals = np.sort(df_all["k"].unique())

    # Fixed colors per scaler (stable order)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # ---- per-scaler columns ----
    for j, scaler in enumerate(scalers):
        ax = ax_row[j]
        df_s = df_all[df_all["scaler"] == scaler]

        # Thin lines: all n_neighbors
        for n_nb, g in df_s.groupby("n_neighbors"):
            g = g.sort_values("k")
            ax.plot(g["k"], g[metric], linewidth=linewidth_thin, alpha=alpha_thin)

        # Best n_neighbors per k
        # ---- select best per k: max ARI_mean, tie → smallest n_neighbors ----
        df_best = (
            df_s
            .sort_values(
                by=["k", metric, "n_neighbors"],
                ascending=[True, False, True]
            )
            .groupby("k", as_index=False)
            .first()
            .sort_values("k")
        )

        for _, row in df_best.iterrows():
            ax.text(
                row["k"], row[metric],
                f"{int(row['n_neighbors'])}",
                fontsize=8, fontweight="bold",
                va="bottom", ha="center", color="black"
            )

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(f"{ylabel}\n[{scaler}]")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)

    # ---- last column: comparison across scalers + stacked annotations ----
    if show_comparison_col:
        ax_comp = ax_row[-1]

        # Track per-k labels to stack
        k_data = {k: {"infos": [], "max_y": -np.inf, "min_y": np.inf} for k in k_vals}

        df_hit = df_s[df_s["ARI_mean"] ==1]
        result = (df_hit.groupby("k")["n_neighbors"].unique())
        result_sorted = result.apply(lambda x: sorted(x))
        print(result_sorted)

        for sc, scaler in enumerate(scalers):
            df_s = df_all[df_all["scaler"] == scaler]

            # ---- select best per k: max ARI_mean, tie → smallest n_neighbors ----
            df_best = (
                df_s
                .sort_values(
                    by=["k", metric, "n_neighbors"],
                    ascending=[True, False, True]
                )
                .groupby("k", as_index=False)
                .first()
                .sort_values("k")
            )

            abbr = scaler_abbr.get(scaler, scaler)
            color = colors[sc % len(colors)]

            ax_comp.plot(
                df_best["k"], df_best[metric],
                marker=marker_best,
                linewidth=linewidth_best,
                label=abbr,
                color=color,
                alpha=0.7
            )

            for _, row in df_best.iterrows():
                k_val = int(row["k"])
                val = float(row[metric])

                k_data[k_val]["infos"].append(
                    (f"{abbr}-{int(row['n_neighbors'])}", color)
                )
                k_data[k_val]["max_y"] = max(k_data[k_val]["max_y"], val)
                k_data[k_val]["min_y"] = min(k_data[k_val]["min_y"], val)

        # ---- stack annotations (unchanged logic) ----
        for k_val, data in k_data.items():
            peak_y = data["max_y"]
            bottom_y = data["min_y"]

            for rank, (text, color) in enumerate(data["infos"]):
                y_offset = 15 + rank * 8

                if metric == "ARI_mean":
                    y_offset = -15 - abs(1 - bottom_y) * 140 - rank * 10
                elif k_val <= 3:
                    y_offset -= 10

                ax_comp.annotate(
                    text,
                    xy=(k_val, peak_y),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    fontsize=6,
                    fontweight="bold",
                    color=color,
                    ha="center",
                    va="bottom",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0)
                )

        ax_comp.set_title("Best-per-k across scalers", fontsize=11)

        ax_comp.set_ylim(0, 1.05)
        ax_comp.set_ylabel(ylabel)
        ax_comp.set_yticks(np.arange(0, 1.01, 0.1))
        ax_comp.grid(True, alpha=0.2)

        # Legend: show once per row; place inside panel
        ax_comp.legend(fontsize=8, loc="center")



def plot_spectral_k_analysis_two_geometries(
    geometries=("cosine", "euclidean"),
    data_dir="files/SpectralClusteringEngine",
    metric="ARI_mean",
    figsize_per_row=4.0,
    show_comparison_col=True,
    save_path=None,
    dpi=300
):
    dfs = {g: load_spectral_results(data_dir, g) for g in geometries}

    scalers = list(dict.fromkeys(dfs[geometries[0]]["scaler"]))
    n_cols = len(scalers) + (1 if show_comparison_col else 0)
    n_rows = len(geometries)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, figsize_per_row * n_rows),
        sharex=True,
        constrained_layout=True
    )

    if n_rows == 1:
        axes = np.array([axes])

    for i, geom in enumerate(geometries):
        plot_spectral_row(
            ax_row=axes[i],
            df_all=dfs[geom],
            metric=metric,
            ylabel="ARI",
            title=f"{geom.capitalize()} geometry",
            show_comparison_col=show_comparison_col
        )

        axes[i][0].annotate(
            geom.capitalize(),
            xy=(-0.15, 0.5),
            xycoords="axes fraction",
            fontsize=11,
            fontweight="bold",
            rotation=90,
            va="center"
        )

    k_vals = np.sort(dfs[geometries[0]]["k"].unique())
    for ax in axes.flat:
        ax.set_xticks(k_vals)
        ax.set_xlabel("Number of clusters (k)")

    fig.suptitle("Sensitivity of spectral clustering stability to affinity geometry", fontsize=13)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig,dfs




fig, df_all = plot_spectral_k_analysis_two_geometries(
    geometries=("cosine", "euclidean"),
    metric="ARI_mean"
)
plt.show()

def plot_spectral_ari1_regions_two_geometries(
    geometries=("cosine", "euclidean"),
    data_dir="files/SpectralClusteringEngine",
    metric="ARI_mean",
    scaler_abbr=DEFAULT_SCALER_ABBR,
    figsize_per_col=5.5,
    alpha=0.6,
    point_size=35,
    save_path=None,
    dpi=300,
):
    """
    Plot (k, n_neighbors) points where ARI_mean == 1
    for two geometries, side by side (1×2 layout).
    """

    # ---- load data ----
    dfs = {g: load_spectral_results(data_dir, g) for g in geometries}

    # consistent scaler order & colors
    scalers = list(dict.fromkeys(dfs[geometries[0]]["scaler"]))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_map = {
        scaler: colors[i % len(colors)]
        for i, scaler in enumerate(scalers)
    }

    n_cols = len(geometries)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_cols,
        figsize=(figsize_per_col * n_cols, 4.5),
        sharey=True,
        constrained_layout=True
    )

    # ensure iterable
    if n_cols == 1:
        axes = [axes]

    # ---- plotting ----
    for ax, geom in zip(axes, geometries):
        df = dfs[geom]

        # filter perfect-stability points
        df_hit = df[df[metric] ==1]

        for i,scaler in enumerate(scalers):
            df_s = df_hit[df_hit["scaler"] == scaler]
            if df_s.empty:
                continue

            # deterministic jitter per scaler
            jitter = (hash(scaler) % 7 - 3) * 0.15  # small, stable offset
            jitter=(i%4-1.5)*0.1
            ax.scatter(
                df_s["k"] + jitter,
                df_s["n_neighbors"],
                s=point_size,
                label=scaler_abbr.get(scaler, scaler),
                color=color_map[scaler],
                edgecolor="black",
                linewidth=0.4
            )

        ax.set_title(f"{geom.capitalize()} geometry", fontsize=12)
        ax.set_xlabel("Number of clusters (k)")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("n_neighbors")

    # ---- shared legend ----
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Scaler",
        loc="upper center",
        ncol=len(scalers),
        frameon=False
    )

    fig.suptitle(
        "Neighborhood sizes yielding perfect clustering stability (ARI_mean = 1)",
        fontsize=13
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, dfs

fig, dfs = plot_spectral_ari1_regions_two_geometries(
    geometries=("cosine", "euclidean"),
    metric="ARI_mean"
)
plt.show()
