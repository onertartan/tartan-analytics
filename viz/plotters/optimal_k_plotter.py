import pandas as pd
import numpy as np
from kneed import KneeLocator
from matplotlib import pyplot as plt
import streamlit as st

from clustering.base_clustering import Clustering
from clustering.models.kmeans import KMeansEngine
from clustering.models.gmm import GMMEngine

METRIC_OBJECTIVES = {
    "Silhouette_mean": "max",
    "ARI_mean": "max",
    "Consensus": "max",
    "DaviesBouldin_mean": "min",

    # Only present for GMM
    "BIC_mean": "min",
    "AIC_mean": "min",
}

class OptimalKPlotter:

    @staticmethod
    def plot_optimal_k_analysis(engine_class, num_seeds_to_plot,k_values,random_states,metrics_all,metrics_mean,ari_mean,ari_std,consensus_indices):
        st.header("Running "+engine_class.__name__+" Optimal k Analysis for "+str(len(random_states))+" seeds.")

        TITLE_FONTSIZE = 14
        AXIS_LABEL_FONTSIZE = 12
        TICK_LABEL_FONTSIZE = 11
        LEGEND_FONTSIZE = 11
        # Create subplot grid: (seeds + mean + ARI) rows, num_cols columns
        #  Smaller Figure Size, Higher DPI
        # Width 20 is plenty for 3 columns. Height 25 gives enough vertical room for 5 rows.
        if engine_class is KMeansEngine:
            num_cols=3
        else:
            num_cols=4
        num_seeds_to_plot = min(num_seeds_to_plot, len(random_states))
        fig, axs = plt.subplots(num_seeds_to_plot + 2, num_cols,  figsize=(20, 25), dpi=200)
        # Titles for each column
        column_titles = ['Silhouette Score', 'Davies-Bouldin Index']
        if engine_class is KMeansEngine:
            column_titles += ['Elbow Analysis']
        elif engine_class is GMMEngine:
            column_titles += ['AIC', 'BIC']
        column_titles += ['Consensus Index']

        df_optimal_k = pd.DataFrame(data=0,index=column_titles,columns=k_values)
        for i, random_state in enumerate(random_states):
            optimal_k_sil = k_values[np.argmax( metrics_all["Silhouette Score"][i])]
            df_optimal_k.loc['Silhouette Score', optimal_k_sil] += 1
            optimal_k_db = k_values[np.argmin(metrics_all["Davies-Bouldin Index"][i])]
            df_optimal_k.loc['Davies-Bouldin Index', optimal_k_db] += 1
            if engine_class == 'KMeansEngine':
                elbow = KneeLocator(k_values, metrics_all["Inertia"][i], curve='convex', direction='decreasing')
                elbow = elbow.elbow
                df_optimal_k.loc['Elbow Analysis', elbow] += 1

        # Plot metrics for each seed of the first num_seeds_to_plot random states
        for i, random_state in enumerate(random_states[:num_seeds_to_plot]):
            # Silhouette Score
            axs[i, 0].plot(k_values,  metrics_all["Silhouette Score"][i], 'ro-')
            axs[i, 0].set_title(f'Seed {random_state}: {column_titles[0]}', fontsize=TITLE_FONTSIZE)
            axs[i, 0].set_ylabel("Score", fontsize=AXIS_LABEL_FONTSIZE)
            axs[i, 1].set_ylabel("Index Value", fontsize=AXIS_LABEL_FONTSIZE)

            # Davies-Bouldin Index
            axs[i, 1].plot(k_values, metrics_all["Davies-Bouldin Index"][i], 'co-')
            axs[i, 1].set_title(f'Seed {random_state}: {column_titles[1]}', fontsize=TITLE_FONTSIZE)
            if engine_class is KMeansEngine:
                # Inertia
                axs[i, 2].plot(k_values, metrics_all["Inertia"][i], 'bo-')
                axs[i, 2].set_ylabel("Inertia", fontsize=AXIS_LABEL_FONTSIZE)

                elbow = KneeLocator(k_values, metrics_all["Inertia"][i], curve='convex', direction='decreasing')
                if elbow.elbow:
                    axs[i, 2].axvline(x=elbow.elbow, color='r', linestyle='--')
                axs[i, 2].set_title(f'Seed {random_state}: {column_titles[2]}', fontsize=TITLE_FONTSIZE)

        # Plot mean metrics
        axs[num_seeds_to_plot, 0].plot(k_values, metrics_mean["Silhouette Score"], 'ro-')
        optimal_k_mean_sil = k_values[np.argmax(metrics_mean["Silhouette Score"],)]
        axs[num_seeds_to_plot, 0].axvline(x=optimal_k_mean_sil, color='r', linestyle='--')
        axs[num_seeds_to_plot, 0].set_title(f'Mean: {column_titles[1]}', fontsize=TITLE_FONTSIZE)

        axs[num_seeds_to_plot, 1].plot(k_values, metrics_mean["Davies-Bouldin Index"], 'co-')
        optimal_k_mean_db = k_values[np.argmin( metrics_mean["Davies-Bouldin Index"])]
        axs[num_seeds_to_plot, 1].axvline(x=optimal_k_mean_db, color='r', linestyle='--')
        axs[num_seeds_to_plot, 1].set_title(f'Mean: {column_titles[2]}', fontsize=TITLE_FONTSIZE)

        if engine_class is KMeansEngine:
            axs[num_seeds_to_plot, 2].plot(k_values, metrics_mean["Inertia"], 'bo-')
            mean_elbow = KneeLocator(k_values, metrics_mean["Inertia"], curve='convex', direction='decreasing')
            if mean_elbow.elbow:
                axs[num_seeds_to_plot, 2].axvline(x=mean_elbow.elbow, color='r', linestyle='--')
            axs[num_seeds_to_plot, 2].set_title(f'Mean: {column_titles[2]}', fontsize=TITLE_FONTSIZE)
        elif engine_class is GMMEngine:
            axs[num_seeds_to_plot, 2].plot(k_values, metrics_mean["AIC"], label="AIC", marker="o")
            axs[num_seeds_to_plot, 2].axvline(k_values[np.argmin(metrics_mean["AIC"])], linestyle="--", label="min AIC")
            axs[num_seeds_to_plot, 2].set_title("Mean AIC vs k", fontsize=TITLE_FONTSIZE)
            axs[num_seeds_to_plot, 3].plot(k_values, metrics_mean["BIC"], label="BIC", marker="s")
            axs[num_seeds_to_plot, 3].axvline(k_values[np.argmin(metrics_mean["BIC"])],linestyle=":", label="min BIC")
            axs[num_seeds_to_plot, 3].set_title("Mean BIC vs k", fontsize=TITLE_FONTSIZE)


        # Plot ARI metrics
        axs[num_seeds_to_plot + 1, 0].plot(k_values, ari_mean, 'bo-')
        axs[num_seeds_to_plot + 1, 0].set_title('Mean ARI vs Clusters', fontsize=TITLE_FONTSIZE)
        # Annotate each point with its value
        for k, val in zip(k_values, ari_mean):
            axs[num_seeds_to_plot + 1, 0].text(k, val, f'{val:.2f}',fontsize=12,
                         ha='center', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        axs[num_seeds_to_plot + 1, 1].scatter(k_values, ari_std, color='g')
        axs[num_seeds_to_plot + 1, 1].set_title('ARI Std vs Clusters',  fontsize=TITLE_FONTSIZE)

        # Plot Consensus Index
        axs[num_seeds_to_plot+1, 2].plot(k_values, consensus_indices, 'purple', marker='o', linestyle='-')
        optimal_k_consensus = k_values[np.argmax(consensus_indices)]
        axs[num_seeds_to_plot+1, 2].axvline(x=optimal_k_consensus, color='r', linestyle='--')
        axs[num_seeds_to_plot+1, 2].set_title(f'Mean: {column_titles[-1]}', fontsize=TITLE_FONTSIZE)

        # Hide unused subplots in ARI row
        for j in range(3, num_cols):
            axs[num_seeds_to_plot + 1, j].axis('off')

        # Set labels and layout
        for ax in axs.flat:
            ax.set_xlabel('Number of Clusters (k)', fontsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(True)

        fig.tight_layout(pad=3.0) # Adds padding to prevent labels overlapping
        fig.savefig('kmeans_metrics_analysis.png', dpi=300, bbox_inches='tight')
        st.dataframe(df_optimal_k)
        st.pyplot(fig)

    def style_metrics_dataframe(df: pd.DataFrame):
        display = pd.DataFrame(index=df.index)

        def mean_pm_std(mean_col_, std_col, prec=3):
            return (
                    df[mean_col_].map(lambda x: f"{x:.{prec}f}") +
                    " ± " +
                    df[std_col].map(lambda x: f"{x:.{prec}f}")
            )

        # ---- Always-present metrics ----
        display["Silhouette Score"] = mean_pm_std("Silhouette_mean", "Silhouette_std")
        display["Davies–Bouldin"] = mean_pm_std("DaviesBouldin_mean", "DaviesBouldin_std" )
        display["ARI"] = mean_pm_std("ARI_mean", "ARI_std")
        display["Consensus"] = df["Consensus"].map(lambda x: f"{x:.3f}")

        # ---- GMM-only metrics (guarded) ----
        if "BIC_mean" in df.columns:
            display["BIC"] = mean_pm_std("BIC_mean", "BIC_std", prec=0)

        if "AIC_mean" in df.columns:
            display["AIC"] = mean_pm_std("AIC_mean", "AIC_std", prec=0)

        # ---- Highlighting logic ----
        def highlight_best(_mean_col):
            values = df[_mean_col]
            if METRIC_OBJECTIVES[_mean_col] == "max":
                best = values.idxmax()
            else:
                best = values.idxmin()

            return [
                "background-color: #d4f7d4" if idx == best else ""
                for idx in df.index
            ]

        styler = display.style

        for mean_col in METRIC_OBJECTIVES:

            if mean_col not in df.columns:
                continue
            label = (
                mean_col
                .replace("_mean", "")
                .replace("DaviesBouldin", "Davies–Bouldin")
            )
            if label == "Silhouette":
                label += " Score"

            if label in display.columns:
                styler = styler.apply(
                    lambda _, mc=mean_col: highlight_best(mc),  # ← FIX
                    axis=0,
                    subset=[label]
                )
            else:
                st.header(label+" not found in dataframe columns!"+str(df.columns))

        return styler



