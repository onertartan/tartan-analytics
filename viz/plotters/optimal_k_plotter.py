import pandas as pd
import numpy as np
from kneed import KneeLocator
from matplotlib import pyplot as plt
import streamlit as st
from clustering.models.kmeans import KMeansEngine
from clustering.models.gmm import GMMEngine

class OptimalKPlotter:


    @staticmethod
    def plot_optimal_k_analysis(engine_class, num_seeds_to_plot,k_values,random_states,metrics_all,metrics_mean,ari_mean,ari_std,consensus_indices):
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

        if engine_class is 'KMeansEngine':
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

        df = OptimalKPlotter.metrics_to_dataframe(metrics_mean, k_values)
        st.dataframe( OptimalKPlotter.style_metrics_dataframe(df), use_container_width=True )

        st.header("Gaussian Mixture Model – Optimal k Analysis")
        st.caption("Mean values across random initializations")
        metric_names_minimum = ["Davies-Bouldin Index"]
        if engine_class is GMMEngine:
            metric_names_minimum += ["BIC", "AIC"]
        st.dataframe(
            df.style
            .highlight_min(subset=metric_names_minimum, color="#d4f7d4")
            .highlight_max(subset=["Silhouette Score"], color="#d4f7d4"),
            use_container_width=True
        )
        st.pyplot(fig)

    @staticmethod
    def metrics_to_dataframe(metrics_mean, k_values):
        df = pd.DataFrame(metrics_mean)
        df["k"] = list(k_values)
        df = df.set_index("k")
        return df

    @staticmethod
    def style_metrics_dataframe(df):
        """
        Apply presentation-only formatting (no data mutation).
        """
        format_rules = {
            "Silhouette Score": "{:.3f}",
            "Davies-Bouldin Index": "{:.3f}",
            "Inertia": "{:,.2f}",
            "AIC": "{:,.1f}",
            "BIC": "{:,.1f}",
            "NegLogLikelihood": "{:,.1f}",
        }

        return df.style.format({
            col: fmt for col, fmt in format_rules.items() if col in df.columns
        })
