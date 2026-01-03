from adjustText import adjust_text
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import streamlit as st

class PCAPlotter:

    def plot_pca(self, df_pivot, df_clusters, dense_threshold, mid_threshold, colors, title=""):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(df_pivot)
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df_clusters.apply(lambda x: colors[x-1]), edgecolors='w')
        # Get the explained variance ratios for each component
        explained_variance_ratios = pca.explained_variance_ratio_
        # Calculate cumulative variance for the first two components
        cumulative_variance_two = sum(explained_variance_ratios[:2])
        # --- Add Legend Here ---
        # Get unique cluster IDs and sort them (e.g., [1, 2, 3])
        unique_clusters = sorted(df_clusters.unique())
        # Create legend labels and handles
        legend_labels = [f'Cluster {cluster_id}' for cluster_id in unique_clusters]
        texts = []

        # Map each cluster to its color in the colormap,         # Add legend to plot
        legend_handles = [plt.Line2D([], [], marker='o', linestyle='', color=colors[i], markersize=10, label=label) for i, label in enumerate(legend_labels)]
        ax.legend(handles=legend_handles, title='Clusters', loc='best')

        cluster_counts = df_clusters.value_counts()  # Calculate ONCE

        for i, name in enumerate(df_pivot.index):
            cluster_id = df_clusters[name]
            cluster_size = cluster_counts[cluster_id]
            # Skip dense clusters (>=10% of data)
            if cluster_size >= dense_threshold:
                continue
            # For mid-density (5-10%), annotate every 5th point
            if cluster_size >= mid_threshold and i % 5 != 0:
                continue
            # Annotate sparse clusters (<5%) and selected mid-density points
         #   texts.append(ax.annotate(name, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=8, alpha=0.7))

        adjust_text(texts)
        ax.set_title(f"{title}\nExplained_variance ratios{explained_variance_ratios.round(2)}\nCumulative variance of two components:{cumulative_variance_two.round(2)}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True)
        # from mpl_toolkits.mplot3d import Axes3D
        # pca = PCA(n_components=3)
        # reduced_data = pca.fit_transform(df_pivot)
        # fig = plt.figure(figsize=(10, 8))
        # # Get the explained variance ratios for each component
        # explained_variance_ratios = pca.explained_variance_ratio_
        #
        # # Calculate cumulative variance for the first two components
        # cumulative_variance_two = sum(explained_variance_ratios[:2])
        #
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
        #            c=df_clusters.apply(lambda x: self.COLORS[x - 1]), edgecolors='w')
        # # Annotate in 3D with offset to reduce overlap
        # for i, name in enumerate(df_pivot.index):
        #     # Use ax.text for 3D annotations with a slight offset
        #     ax.text(
        #         reduced_data[i, 0] ,  # Small offset in x
        #         reduced_data[i, 1] ,  # Small offset in y
        #         reduced_data[i, 2] ,  # Small offset in z
        #         name,
        #         fontsize=8,
        #         alpha=0.7
        #     )
        # ax.set_title(f"\nExplained_variance ratios{explained_variance_ratios}\nCumulative variance of two components:{cumulative_variance_two}")
        #
        # # Set labels
        # ax.set_xlabel('PC1')
        # ax.set_ylabel('PC2')
        # ax.set_zlabel('PC3')
        st.pyplot(fig)