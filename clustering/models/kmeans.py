"""
KMeans clustering engine for geographic or tabular data.
Decoupled from UI and Streamlit. Easily testable.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, davies_bouldin_score, silhouette_score

from clustering.base_clustering import Clustering
from clustering.evaluation.stability import stability_and_consensus
import streamlit as st
import time

class KMeansEngine:
    """
    A clean, UI-independent KMeans clustering module.
    """
    def __init__(self, n_cluster: int, n_init: int, random_state: int = 1):
        """
        Parameters
        ----------
        n_cluster : int
            Number of clusters.
        n_init : int
            Number of random restarts.
        random_state : int
            Random seed for reproducibility.
        """
        self.kmeans = KMeans(n_clusters=n_cluster, n_init=n_init, init="k-means++", random_state=random_state)
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        self.kmeans.fit(df)
        df_out = df.copy()
        df_out["clusters"] = self.kmeans.labels_ + 1 # start labels at 1
        return df_out


    @staticmethod
    def optimal_k_analysis(df, random_states, n_init=1, k_values=range(2, 15)):
        X = df.values if hasattr(df, "values") else df
        n_samples = X.shape[0]

        # ---- Model-specific storage ----
        metrics_all = {
            "Inertia": [],
            "Silhouette Score": [],
            "Davies-Bouldin Index": []
        }

        labels_all = {seed: {} for seed in random_states}
        progress_bar = st.progress(0.0)
        status_text = st.empty()  # This will hold the "X / Y completed" message
        total_states = len(random_states)
        start_time = time.time()  # Record overall start time

        # ---- Run K-Means ----
        for idx, random_state in enumerate(random_states):
            inertias = []
            silhouettes = []
            db_scores = []
            seed_start = time.time()

            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(X)

                labels = kmeans.labels_
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(X, labels))
                db_scores.append(davies_bouldin_score(X, labels))

                labels_all[random_state][k] = labels
            # Update progress and status after loop for one random state is completed
            progress_bar.progress((idx + 1) / total_states)
            elapsed_total = time.time() - start_time
            elapsed_minutes, elapsed_seconds = divmod(int(elapsed_total), 60)
            seed_time = int(time.time() - seed_start)
            status_text.text(f"The last seed took {seed_time}s")
            status_text.text(f"Completed {idx + 1}/{total_states} seeds.Elapsed: {elapsed_minutes}m {elapsed_seconds}s")

            metrics_all["Inertia"].append(inertias)
            metrics_all["Silhouette Score"].append(silhouettes)
            metrics_all["Davies-Bouldin Index"].append(db_scores)


        progress_bar.empty()
        status_text.empty()
        # ---- Mean metrics across seeds ----
        metrics_mean = {key: np.mean(metrics_all[key], axis=0) for key in metrics_all}

        # ---- Model-independent evaluation ----
        ari_mean, ari_std, consensus_indices, consensus_labels_all = \
            stability_and_consensus(labels_all=labels_all, k_values=k_values, random_states=random_states, n_samples=n_samples)
        df_summary = KMeansEngine.summarize(metrics_all, ari_mean, ari_std, consensus_indices, k_values)
        return df_summary,metrics_all, metrics_mean, ari_mean, ari_std, consensus_indices, consensus_labels_all

    @staticmethod
    def summarize(metrics_all, ari_mean, ari_std, consensus_indices, k_values):
        rows = []

        for k_value in k_values:
            k_idx = list(k_values).index(k_value)

            sil_m, sil_s = Clustering.mean_sd_at_k(
                metrics_all, "Silhouette Score", k_idx
            )
            db_m, db_s = Clustering.mean_sd_at_k(
                metrics_all, "Davies-Bouldin Index", k_idx
            )

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
