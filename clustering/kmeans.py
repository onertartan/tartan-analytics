"""
KMeans clustering engine for geographic or tabular data.
Decoupled from UI and Streamlit. Easily testable.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


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
        self.n_cluster = n_cluster
        self.n_init = n_init
        self.random_state = random_state
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        # 1. K-Means'i çalıştır
        kmeans = KMeans(n_clusters=self.n_cluster, n_init=self.n_init, init="k-means++",
                        random_state=self.random_state).fit(df)
        # 2. Etiketleri ata ve DataFrame'i hazırla
        df_out = df.copy()
        # Etiketleri 1'den başlatıyoruz
        df_out["clusters"] = kmeans.labels_ + 1
        return df_out

    @staticmethod
    def get_representatives(df: pd.DataFrame) -> List[str]:
        """
        Calculates representatives based on the 'clusters' column in the dataframe.
        No need for class instance (self) parameters.
        IMPORTANT : LAST COLUMN OF df AFTER CALLING fit METHOD ABOVE BECOMES CLUSTER COLUMN
        this is the reason of using df.iloc[:-1]
        """
        calculated_centers = df.groupby("clusters").mean()
        # En yakın noktaları bulma
        # Hesapladığımız merkezleri kullanarak en yakın noktaları (representatives) buluyoruz.
        # calculated_centers DataFrame olduğu için .values ile numpy array'e çevirmemiz gerekebilir
        # (ancak pairwise_distances genelde DF de kabul eder).
        closest_indices, _ = pairwise_distances_argmin_min(calculated_centers.values, df.iloc[:,:-1])
        representatives = df.index[closest_indices].tolist()
        return  representatives

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
