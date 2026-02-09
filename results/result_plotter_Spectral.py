import os

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
def _parse_spectral_filename(fname: Path,geometry):
    """
    Expected filename format:
    {affinity}_{scaler}_{year1}_{year2}_{n}.csv
    Examples:
    nearest_neighbors_Share of Total_2018_2024_5.csv
    rbf_TF-IDF_2018_2024_10.csv
    """
    m = re.match(
        r"^(nearest_neighbors|rbf)_(.+?)_(\d{4}_\d{4})_(\d+)\.csv$",
        fname.name
    )
    if m is None:
        return None

    affinity = m.group(1)          # nearest_neighbors | rbf
    scaler = m.group(2)            # may contain spaces
    year_range = m.group(3)        # 2018_2024
    n_neighbors = int(m.group(4))  # kNN size (or gamma index if rbf)
    return affinity, scaler, year_range, n_neighbors


def load_spectral_results(data_dir, geometry):
    data_dir +="/SpectralClusteringEngine"
    DATA_DIR = Path(data_dir)
    files = sorted(f for f in DATA_DIR.glob("*.csv")  if not f.name.startswith("consensus_labels_all_") )

    dfs = []
    for f in files:
        parsed = _parse_spectral_filename(f, geometry)
        if parsed is None:
            continue

        geom, scaler, year_range, n_nb = parsed
        df = pd.read_csv(f)
        df["Number of clusters"] = df["Number of clusters"].astype(int)
        df.rename(columns={"Number of clusters":"k"}, inplace=True)
        df["geometry"] = geom
        df["scaler"] = scaler
        df["n_neighbors"] = n_nb
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No files found for geometry={geometry}")
    print("shape:",pd.concat(dfs, ignore_index=True).shape,"unique ",pd.concat(dfs, ignore_index=True)["geometry"].unique())
    return pd.concat(dfs, ignore_index=True)

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
    marker_best="o",line_style="solid"
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
            ax.plot(g["k"], g[metric], linewidth=linewidth_thin, alpha=alpha_thin,linestyle=line_style)

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
        ax_comp.legend(fontsize=8, loc="center", title="Scaler")



def plot_spectral_k_analysis_two_geometries(
    geometries=None,
    data_dir=None,
    metrics=["ARI_mean"],
    figsize_per_row=4.0,
    show_comparison_col=True,
    save_path=None,
    dpi=300,
    ylabel=None
):
    dfs = {g: load_spectral_results(data_dir, g) for g in geometries}
    scalers = list(dict.fromkeys(dfs[geometries[0]]["scaler"]))
    n_cols = len(scalers) + (1 if show_comparison_col else 0)
    n_rows = len(geometries)
    fig_size=(5 * n_cols, figsize_per_row * n_rows)
    fig, axes = plt.subplots( nrows=n_rows, ncols=n_cols, figsize=fig_size, sharex=True, constrained_layout=True)

    if n_rows == 1:
        axes = np.array([axes])
    MAX_K_TO_PLOT = 10

    for i, geom in enumerate(geometries):
        for metric,line_style in zip( metrics,["solid","dashed"]):
            df_all = dfs[geom]
            df_all = df_all[df_all["k"] <= MAX_K_TO_PLOT]

            plot_spectral_row(
                ax_row=axes[i],
                df_all=df_all,
                metric=metric,
                ylabel=ylabel,
                title=f"{geom.capitalize()} geometry",
                show_comparison_col=show_comparison_col,line_style=line_style
            )

            #axes[i][0].annotate(geom.capitalize(),xy=(-0.15, 0.5),xycoords="axes fraction", fontsize=11, fontweight="bold", rotation=90,va="center")

        df_first = dfs[geometries[0]] # dfs is a dict, so we take the first geometry's df to get the k values for x-axis ticks
        k_vals = np.sort(df_first[df_first["k"]<= MAX_K_TO_PLOT]["k"].unique())
        for ax in axes.flat:
            ax.set_xticks(k_vals)
            ax.set_xlabel("Number of clusters (k)")

        fig.suptitle("Sensitivity of spectral clustering stability to affinity geometry, normalization scheme and n_neighbors", fontsize=13)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig,dfs

def save_max_silhouette_per_geometry(data_dir=None, geometries=None):

    for geom in geometries:
        df = load_spectral_results(data_dir, geom)
        # choose the correct silhouette column
        if geom == "euclidean":
            sort_by_cols = ["k", "Silhouette_mean (euclidean)", "Silhouette_mean (cosine)", "ARI_mean"]
        else:
            sort_by_cols = ["k", "Silhouette_mean (cosine)", "Silhouette_mean (euclidean)", "ARI_mean"]

        df=df[["k","Silhouette_mean (cosine)", "Silhouette_mean (euclidean)", "ARI_mean","scaler","geometry","n_neighbors"]].round(3)
        # ---- max silhouette per k ----
        df_best=df.sort_values(by=sort_by_cols,ascending=[True, False,False,False]).groupby("k", as_index=False).first()
        df_best.T.to_csv(f"{data_dir}/SpectralClustering_{geom}_best.csv")
        df = df.set_index("k")
        df = df.sort_values(by=sort_by_cols, ascending=[True, False,False,False])
        df.to_csv(f"{data_dir}/SpectralClustering_{geom}_all.csv")


def plot_spectral_ari1_regions_two_geometries(
    geometries=("cosine", "euclidean"),
    data_dir=None,
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
        df_hit = df[df[metric] == 1]

        for i, scaler in enumerate(scalers):
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
        loc="upper right",
        bbox_to_anchor=(0.98, 0.88),
        ncol=1,
       frameon=True,
       fancybox=True,
      framealpha=0.95
      )

    fig.suptitle(
        "Neighborhood sizes yielding perfect clustering stability (ARI_mean = 1)",
        fontsize=13
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, dfs

genders= ["female"]
for gender in genders:
    data_dir = "files/"
    if gender:
        data_dir += gender
    #geometries= ("cosine", "euclidean")
    geometries = ("euclidean",)
    fig, dfs = plot_spectral_k_analysis_two_geometries(data_dir=data_dir, geometries=geometries,  metrics=["ARI_mean"],ylabel="mean ARI",save_path=data_dir+"/stability_spectral_ari_mean.png")
    plt.show()
    fig, dfs = plot_spectral_ari1_regions_two_geometries(data_dir=data_dir, geometries=geometries, metric="ARI_mean",save_path=data_dir+"/stability_spectral_ari1_regions.png")
    plt.show()
    fig, dfs = plot_spectral_k_analysis_two_geometries(data_dir=data_dir, geometries=geometries,
                                                           metrics=["Silhouette_mean (cosine)","Silhouette_mean (euclidean)"],
                                                           ylabel="Mean Silhouette Score",save_path=data_dir+"/stability_spectral_silhouette.png")
    plt.show()
    save_max_silhouette_per_geometry(data_dir,geometries)