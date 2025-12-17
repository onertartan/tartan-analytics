"""
Gaussian Mixture Model (GMM) clustering engine.
Independent from UI and Streamlit. Designed for testability and modularity.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture

from clustering.evaluation.stability import stability_and_consensus
import streamlit as st
from stqdm import stqdm
import time
class GMMEngine:
    """
    A clean Gaussian Mixture clustering engine for tabular data.
    """
    def __init__(self, n_cluster: int, n_init: int = 1, covariance_type: str = "full", random_state: int = 1):
        """
        Parameters
        ----------
        n_clusters (n_components) : int
            Number of mixture components.
        n_init : int
            Number of restarts.
        covariance_type : str
            'full', 'tied', 'diag', or 'spherical'.
        random_state : int
            Random seed for reproducibility.
        """
        self.gmm = GaussianMixture(
            n_components=n_cluster,
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=random_state
        )
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        self.gmm.fit(df.values)
        labels = self.gmm.predict(df.values) + 1
        df_out = df.copy()
        df_out["clusters"] = labels
        return df_out

    @staticmethod
    def optimal_k_analysis(df, random_states=(0, 1, 2, 3, 4), n_init=1, k_values=range(2, 15), covariance_type="full"):
        X = df.values if hasattr(df, "values") else df
        n_samples = X.shape[0]

        # ---- Model-specific storage ----
        metrics_all = {
            "NegLogLikelihood": [],
            "AIC": [],
            "BIC": [],
            "Silhouette Score": [],
            "Davies-Bouldin Index": []
        }

        labels_all = {seed: {} for seed in random_states}
        progress_bar = st.progress(0.0)
        status_text = st.empty()  # This will hold the "X / Y completed" message
        total_states = len(random_states)
        start_time = time.time()  # Record overall start time
        # ---- Run GMM ----
        for idx,random_state in enumerate(random_states):

            nlls, aics, bics, silhouettes, dbs = [], [], [], [], []
            seed_start = time.time()

            for k in k_values:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    n_init=n_init,
                    random_state=random_state
                ).fit(X)

                labels = gmm.predict(X)
                nlls.append(-gmm.score(X) * n_samples)
                aics.append(gmm.aic(X))
                bics.append(gmm.bic(X))
                silhouettes.append(silhouette_score(X, labels))
                dbs.append(davies_bouldin_score(X, labels))
                labels_all[random_state][k] = labels
            progress_bar.progress((idx + 1) / total_states)
            elapsed_total = time.time() - start_time
            elapsed_minutes, elapsed_seconds = divmod(int(elapsed_total), 60)
            status_text.text(f"Completed {idx + 1}/{total_states} seeds.Elapsed: {elapsed_minutes}m {elapsed_seconds}s")

            metrics_all["NegLogLikelihood"].append(nlls)
            metrics_all["AIC"].append(aics)
            metrics_all["BIC"].append(bics)
            metrics_all["Silhouette Score"].append(silhouettes)
            metrics_all["Davies-Bouldin Index"].append(dbs)
        progress_bar.empty()
        status_text.empty()
        # ---- Mean metrics across seeds ----
        metrics_mean = { key: np.mean(metrics_all[key], axis=0)  for key in metrics_all }

        # ---- Model-independent evaluation ----
        ari_mean, ari_std, consensus_indices, consensus_labels_all = \
            stability_and_consensus(labels_all=labels_all, k_values=k_values, random_states=random_states, n_samples=n_samples)

        return metrics_all, metrics_mean, ari_mean, ari_std, consensus_indices, consensus_labels_all

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
