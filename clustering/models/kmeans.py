"""
KMeans clustering engine for geographic or tabular data.
Decoupled from UI and Streamlit. Easily testable.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, davies_bouldin_score, silhouette_score

from clustering.evaluation.stability import stability_and_consensus


class KMeansEngine:
    """
    A clean, UI-independent KMeans clustering module.
    """
    def __init__(self, n_cluster: int, n_init: int = 1, random_state: int = 1):
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
    def fit(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        # 1. K-Means'i çalıştır
        self.kmeans.fit(df)
        # 2. Etiketleri ata ve DataFrame'i hazırla
        df_out = df.copy()
        # Etiketleri 1'den başlatıyoruz
        df_out["clusters"] = self.kmeans.labels_ + 1
        return df_out

    @staticmethod
    def optimal_k_analysis(df, random_states, n_init=1, k_values=range(2, 15)):
        X = df.values if hasattr(df, "values") else df
        n_samples = X.shape[0]

        # ---- Model-specific storage ----
        metrics_all = {
            "Inertia": [],
            "Silhouette": [],
            "Davies-Bouldin": []
        }

        labels_all = {seed: {} for seed in random_states}

        # ---- Run K-Means ----
        for random_state in random_states:
            inertias = []
            silhouettes = []
            db_scores = []

            for k in k_values:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init=n_init
                ).fit(X)

                labels = kmeans.labels_
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(X, labels))
                db_scores.append(davies_bouldin_score(X, labels))

                labels_all[random_state][k] = labels

            metrics_all["Inertia"].append(inertias)
            metrics_all["Silhouette"].append(silhouettes)
            metrics_all["Davies-Bouldin"].append(db_scores)

        # ---- Mean metrics across seeds ----
        metrics_mean = {key: np.mean(metrics_all[key], axis=0) for key in metrics_all}

        # ---- Model-independent evaluation ----
        ari_mean, ari_std, consensus_indices, consensus_labels_all = \
            stability_and_consensus(labels_all=labels_all, k_values=k_values, random_states=random_states, n_samples=n_samples)

        return metrics_all, metrics_mean, ari_mean, ari_std, consensus_indices, consensus_labels_all

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
