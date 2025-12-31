"""
KMeans clustering engine for geographic or tabular data.
Decoupled from UI and Streamlit. Easily testable.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, davies_bouldin_score, silhouette_score, silhouette_samples

from clustering.base_clustering import Clustering
from clustering.evaluation.stability import stability_and_consensus
import streamlit as st
import time

class KMeansEngine(Clustering):
    """
    A clean, UI-independent KMeans clustering module.
    """
    def __init__(self, n_cluster: int,  random_state: int = 1, n_init=-1):
        """
        Parameters
        ----------
        n_cluster : int
            Number of clusters.
        random_state : int
            Random seed for reproducibility
        n_init : int
            Number of random restarts.
        """
        self.model = KMeans(n_clusters=n_cluster, n_init=n_init, init="k-means++", random_state=random_state)
        self.metric_for_silhouette = "euclidean"
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(metrics_all, ari_mean, ari_std, consensus_indices, k_values):
        rows = []

        for k_value in k_values:
            k_idx = list(k_values).index(k_value)

            sil_m, sil_s = Clustering.mean_sd_at_k(metrics_all, "Silhouette Score", k_idx )
            db_m, db_s = Clustering.mean_sd_at_k(metrics_all, "Davies-Bouldin Index", k_idx)

            rows.append({
                "Number of clusters": k_value,
               # Internal validity
                "Silhouette_mean": sil_m, "Silhouette_std": sil_s,
                "DaviesBouldin_mean": db_m, "DaviesBouldin_std": db_s,
                # Stability
                "ARI_mean": ari_mean[k_idx], "ARI_std": ari_std[k_idx],
                "Consensus": consensus_indices[k_idx],
            })

        df = pd.DataFrame(rows).set_index("Number of clusters")
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def relabel_with_priority(labels: pd.Series, priority_list: List[str]) -> List[int]:
        """
        Optional: deterministic label remapping.

        Useful when you want certain provinces to define cluster order.

        Parameters
        ----------
        labels : pd.Series
            Original labels indexed by province name.
        priority_list : List[str]
            Provinces in the desired label order.

        Returns
        -------
        new_labels : List[int]
        """
        mapping = {}
        next_new = 1  # labels will be 1..K

        # priority first
        for prov in priority_list:
            old_lbl = labels.loc[prov]
            if old_lbl not in mapping:
                mapping[old_lbl] = next_new
                next_new += 1

        # remaining clusters
        for old_lbl in sorted(set(labels) - set(mapping)):
            mapping[old_lbl] = next_new
            next_new += 1

        return labels.map(mapping).tolist()

    def temp(self,df):
        X=df.iloc[:, :-1].values
        # Create a subplot with 1 row and 2 columns
        fig, axs = plt.subplots(3, 4)
        fig.set_size_inches(18, 7)
        for n_clusters in range(2,14):


            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            axs = axs.flatten()

            for i in range(len(axs)):
                axs[i].set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
                axs[i].set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            ax_index = n_clusters - 2  # To select the correct subplot
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                axs[ax_index].fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                axs[ax_index].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

           # axs[ax_index].set_title("The silhouette plot for the various clusters.")
            axs[ax_index].set_xlabel("The silhouette coefficient values")
            axs[ax_index].set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            axs[ax_index].axvline(x=silhouette_avg, color="red", linestyle="--")

            axs[ax_index].set_yticks([])  # Clear the yaxis labels / ticks
            axs[ax_index].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        st.pyplot(fig)