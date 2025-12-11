"""
Gaussian Mixture Model (GMM) clustering engine.
Independent from UI and Streamlit. Designed for testability and modularity.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class GMMEngine:
    """
    A clean Gaussian Mixture clustering engine for tabular data.
    """

    def __init__(self, n_clusters: int, n_init: int = 1, covariance_type: str = "full", random_state: int = 1):
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
        self.n_clusters = n_clusters # n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fit GMM clustering to a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix, index = province names.

        Returns
        -------
        df_out : pd.DataFrame
            Copy of df with 'clusters' (1..K) column.
        representatives : List[str]
            One representative province per cluster
            (highest posterior probability).
        """

        gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            random_state=self.random_state
        )

        gmm.fit(df.values)

        # Hard cluster labels (1..K)
        labels = gmm.predict(df.values) + 1

        # Posterior probability matrix: (N x K)
        posterior = gmm.predict_proba(df.values)

        # Representative for each cluster = highest posterior member
        representatives = []
        for k in range(self.n_clusters):
            idx = np.argmax(posterior[:, k])
            representatives.append(df.index[idx])

        # Output dataframe (copy to avoid mutating input)
        df_out = df.copy()
        df_out["clusters"] = labels

        return df_out, representatives

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
