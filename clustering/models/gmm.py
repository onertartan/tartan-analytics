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
            "Silhouette": [],
            "Davies-Bouldin": []
        }

        labels_all = {seed: {} for seed in random_states}

        # ---- Run GMM ----
        for random_state in random_states:
            nlls, aics, bics, silhouettes, dbs = [], [], [], [], []

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

            metrics_all["NegLogLikelihood"].append(nlls)
            metrics_all["AIC"].append(aics)
            metrics_all["BIC"].append(bics)
            metrics_all["Silhouette"].append(silhouettes)
            metrics_all["Davies-Bouldin"].append(dbs)

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
