"""
Gaussian Mixture Model (GMM) clustering engine.
Independent from UI and Streamlit. Designed for testability and modularity.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture

from clustering.base_clustering import Clustering
from clustering.evaluation.stability import stability_and_consensus
import streamlit as st
import time
class GMMEngine(Clustering):
    """
    A clean Gaussian Mixture clustering engine for tabular data.
    """
    def __init__(self, n_cluster: int, n_init: int,  random_state: int = 1,  covariance_type: str = ""):
        """
        Parameters
        ----------
        n_clusters (n_components) : int
            Number of mixture components.
        n_init : int
            Number of restarts.
        covariance_type : str
            'tied', 'diag', or 'spherical'.
        random_state : int
            Random seed for reproducibility.
        """
        self.model = GaussianMixture(
            n_components=n_cluster,
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=random_state
        )
        self.metric_for_silhouette = "euclidean"
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(metrics_all, ari_mean, ari_std, consensus_indices,  k_values):
        rows=[]
        for k_value in k_values:

            k_idx = list(k_values).index(k_value)
            sil_m, sil_s = Clustering.mean_sd_at_k(metrics_all, "Silhouette Score", k_idx)
            db_m,  db_s = Clustering.mean_sd_at_k(metrics_all, "Davies-Bouldin Index", k_idx)
            bic_m, bic_s = Clustering.mean_sd_at_k(metrics_all, "BIC", k_idx)
            aic_m, aic_s = Clustering.mean_sd_at_k(metrics_all, "AIC", k_idx)
            rows.append({
                "Number of clusters": k_value,
                "Silhouette_mean": sil_m, "Silhouette_std": sil_s,
                "DaviesBouldin_mean": db_m, "DaviesBouldin_std": db_s,
                "ARI_mean": ari_mean[k_idx], "ARI_std": ari_std[k_idx],
                "Consensus": consensus_indices[k_idx],
                "BIC_mean": bic_m, "BIC_std": bic_s,
                "AIC_mean": aic_m, "AIC_std": aic_s,
            })

        df_summary = pd.DataFrame(rows)
        df_summary = df_summary.set_index("Number of clusters")
        return df_summary

    # ------------------------------------------------------------------
    @staticmethod
    def relabel_with_priority(labels: pd.Series, priority_list: List[str]) -> List[int]:
        """
        Optional deterministic label remapping.

        Parameters
        ----------
        labels : pd.Series
            Original labels indexed by province name.
        priority_list : List[str]
            Ordered list of provinces.

        Returns
        -------
        new_labels : List[int]
        """
        mapping = {}
        next_new = 1

        # Priority-based label ordering
        for prov in priority_list:
            old_lbl = labels.loc[prov]
            if old_lbl not in mapping:
                mapping[old_lbl] = next_new
                next_new += 1

        # Remaining clusters in numeric order
        for old_lbl in sorted(set(labels) - set(mapping)):
            mapping[old_lbl] = next_new
            next_new += 1

        return labels.map(mapping).tolist()
