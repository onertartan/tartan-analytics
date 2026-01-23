import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


scaler_labels = {
    "Share of Top 30 (L1 Norm)": "L1 (Top 30)",
    "Share of Total": "Share (All)",
    "TF-IDF": "TF–IDF",
    "L2 Normalization": "L2",
}

linestyles = {
    "diag": "-",
    "tied": "--",
    "spherical": ":"
}

METRIC = "ARI_mean"


def _parse_gmm_filename(fname: Path):
    m = re.match(r"(.+?)_(diag|tied|spherical)_(\d{4}_\d{4})\.csv$", fname.name)
    if m is None:
        return None
    return m.group(1), m.group(2), m.group(3)


def load_gmm_results(data_dir):
    data_dir = Path(data_dir)
    dfs = []

    for f in data_dir.glob("*.csv"):
        parsed = _parse_gmm_filename(f)
        if parsed is None:
            continue

        scaler, cov, year_range = parsed
        df = pd.read_csv(f)

        if "Number of clusters" in df.columns:
            df["k"] = df["Number of clusters"].astype(int)
        else:
            df["k"] = df["k"].astype(int)

        df["scaler"] = scaler
        df["covariance"] = cov
        df["year_range"] = year_range
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No matching GMM result files found.")

    return pd.concat(dfs, ignore_index=True)


def plot_gmm_k_analysis_one_row(
    data_dir="files/GMMEngine",
    scaler_labels=scaler_labels,
    covariance_order=("diag", "tied", "spherical"),
    linestyles=linestyles,
    figsize=(18, 4.5),
    ylim=(0, 1.05),
    dpi=300,
    save_path=None
):
    df_all = load_gmm_results(data_dir)

    scalers = list(scaler_labels.keys())
    k_vals = np.sort(df_all["k"].unique())

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(scalers),
        figsize=figsize,
        sharey=True,
        constrained_layout=True
    )

    if len(scalers) == 1:
        axes = [axes]

    for ax, scaler in zip(axes, scalers):
        df_s = df_all[df_all["scaler"] == scaler]

        for cov in covariance_order:
            df_sc = df_s[df_s["covariance"] == cov].sort_values("k")

            if df_sc.empty:
                continue

            ax.plot(
                df_sc["k"],
                df_sc[METRIC],
                linestyle=linestyles.get(cov, "-"),
                marker="o",
                label=cov
            )

        ax.set_title(scaler_labels.get(scaler, scaler), fontsize=12)
        ax.set_xticks(k_vals)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Number of clusters (k)")

    axes[0].set_ylabel("ARI (stability across seeds)")

    # One shared legend (cleaner)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Covariance type",
        loc="upper center",
        ncol=3,
        frameon=False
    )

    fig.suptitle(
        "Sensitivity of GMM clustering stability to covariance structure and  normalization schemes",
        fontsize=14
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig

# Example usage:
fig = plot_gmm_k_analysis_one_row(
     data_dir="files/GMMEngine",
     save_path="gmm_k_analysis_grid.png"
 )
plt.show()
