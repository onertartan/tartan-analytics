from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import streamlit as st


class DBSCANEngine:
    def __init__(self, eps=0.25, min_samples=5, metric="euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, df_pivot):
        """
        Adapts DBSCAN to the .fit() interface used by your other engines.
        """
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric).fit(df_pivot)
        labels = db.labels_

        # Handle Noise and Clusters logic...
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters == 0:
            st.warning("DBSCAN found no clusters; falling back to k=2.")
            labels = np.full(df_pivot.shape[0], 1)
            closest = [df_pivot.index[0]]
        else:
            # Logic to find representatives (closest to core)
            core_mask = db.core_sample_indices_
            core_labels = labels[core_mask]
            closest = []
            for cid in range(n_clusters):
                # Safety check if cluster has core points
                cluster_core_indices = core_mask[core_labels == cid]
                if len(cluster_core_indices) > 0:
                    pick = cluster_core_indices[0]
                    closest.append(df_pivot.index[pick])
                else:
                    # Fallback if no core points found (rare but possible in edge cases)
                    closest.append(df_pivot.index[labels == cid][0])

        df_pivot["clusters"] = labels + 1
        return df_pivot, closest