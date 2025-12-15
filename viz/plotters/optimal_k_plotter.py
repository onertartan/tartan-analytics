import pandas as pd
import numpy as np
from kneed import KneeLocator
from matplotlib import pyplot as plt
import streamlit as st

class OptimalKPlotter:

    @staticmethod
    def plot_optimal_k_analysis(engine_class, num_seeds_to_plot,k_values,random_states,metrics_all,metrics_mean,ari_mean,ari_std,consensus_indices):
        # Create subplot grid: (seeds + mean + ARI) rows, num_cols columns
        #  Smaller Figure Size, Higher DPI
        # Width 20 is plenty for 3 columns. Height 25 gives enough vertical room for 5 rows.
        num_cols=3
        fig, axs = plt.subplots(num_seeds_to_plot + 2, num_cols,  figsize=(20, 25), dpi=200)

        # Titles for each column
        column_titles = ['Silhouette Score', 'Davies-Bouldin Index']
        if engine_class == 'KMeansEngine':
            column_titles += ['Elbow Analysis']
        column_titles += ['Consensus Index']

        df_optimal_k = pd.DataFrame(data=0,index=column_titles,columns=k_values)
        st.dataframe(df_optimal_k)
        for i, random_state in enumerate(random_states):
            optimal_k_sil = k_values[np.argmax( metrics_all["Silhouette"][i])]
            df_optimal_k.loc['Silhouette Score', optimal_k_sil] += 1
            optimal_k_db = k_values[np.argmin(metrics_all["Davies-Bouldin"][i])]
            df_optimal_k.loc['Davies-Bouldin Index', optimal_k_db] += 1
            if engine_class == 'KMeansEngine':
                elbow = KneeLocator(k_values, metrics_all["Inertia"][i], curve='convex', direction='decreasing')
                elbow = elbow.elbow
                df_optimal_k.loc['Elbow Analysis', elbow] += 1

        # Plot metrics for each seed
        for i, random_state in enumerate(random_states[:num_seeds_to_plot]):
            # Silhouette Score
            axs[i, 0].plot(k_values,  metrics_all["Silhouette"][i], 'ro-')
            axs[i, 0].set_title(f'Seed {random_state}: {column_titles[0]}')

            # Davies-Bouldin Index
            axs[i, 1].plot(k_values, metrics_all["Davies-Bouldin"][i], 'co-')
            axs[i, 1].set_title(f'Seed {random_state}: {column_titles[1]}')
            if engine_class == 'KMeansEngine':
                # Inertia
                axs[i, 2].plot(k_values, metrics_all["Inertia"][i], 'bo-')
                elbow = KneeLocator(k_values, metrics_all["Inertia"][i], curve='convex', direction='decreasing')
                if elbow.elbow:
                    axs[i, 2].axvline(x=elbow.elbow, color='r', linestyle='--')
                axs[i, 2].set_title(f'Seed {random_state}: {column_titles[2]}')


        # Plot mean metrics


        axs[num_seeds_to_plot, 0].plot(k_values, metrics_mean["Silhouette"], 'ro-')
        optimal_k_mean_sil = k_values[np.argmax(metrics_mean["Silhouette"],)]
        axs[num_seeds_to_plot, 0].axvline(x=optimal_k_mean_sil, color='r', linestyle='--')
        axs[num_seeds_to_plot, 0].set_title(f'Mean: {column_titles[1]}')

        axs[num_seeds_to_plot, 1].plot(k_values, metrics_mean["Davies-Bouldin"], 'co-')
        optimal_k_mean_db = k_values[np.argmin( metrics_mean["Davies-Bouldin"])]
        axs[num_seeds_to_plot, 1].axvline(x=optimal_k_mean_db, color='r', linestyle='--')
        axs[num_seeds_to_plot, 1].set_title(f'Mean: {column_titles[2]}')

        if engine_class == 'KMeansEngine':
            axs[num_seeds_to_plot, 2].plot(k_values, metrics_mean["Inertia"], 'bo-')
            mean_elbow = KneeLocator(k_values, metrics_mean["Inertia"], curve='convex', direction='decreasing')
            if mean_elbow.elbow:
                axs[num_seeds_to_plot, 2].axvline(x=mean_elbow.elbow, color='r', linestyle='--')
            axs[num_seeds_to_plot, 2].set_title(f'Mean: {column_titles[2]}')

        # Plot ARI metrics
        axs[num_seeds_to_plot + 1, 0].plot(k_values, ari_mean, 'bo-')
        axs[num_seeds_to_plot + 1, 0].set_title('Mean ARI vs Clusters')
        # Annotate each point with its value
        for k, val in zip(k_values, ari_mean):
            axs[num_seeds_to_plot + 1, 0].text(k, val, f'{val:.2f}',
                         ha='center', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        axs[num_seeds_to_plot + 1, 1].scatter(k_values, ari_std, color='g')
        axs[num_seeds_to_plot + 1, 1].set_title('ARI Std vs Clusters')

        # Plot Consensus Index
        axs[num_seeds_to_plot+1, 2].plot(k_values, consensus_indices, 'purple', marker='o', linestyle='-')
        optimal_k_consensus = k_values[np.argmax(consensus_indices)]
        axs[num_seeds_to_plot+1, 2].axvline(x=optimal_k_consensus, color='r', linestyle='--')
        axs[num_seeds_to_plot+1, 2].set_title(f'Mean: {column_titles[-1]}')

        # Hide unused subplots in ARI row
        for j in range(3, num_cols):
            axs[num_seeds_to_plot + 1, j].axis('off')

        # Set labels and layout
        for ax in axs.flat:
            ax.set_xlabel('Number of Clusters (k)')
            ax.grid(True)
        fig.tight_layout(pad=3.0) # Adds padding to prevent labels overlapping
        fig.savefig('kmeans_metrics_analysis.png', dpi=300, bbox_inches='tight')
        st.pyplot(fig)