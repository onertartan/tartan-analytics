from typing import List
import numpy as np
import pandas as pd
import time
import streamlit as st
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from clustering.base_clustering import Clustering
from clustering.evaluation.stability import stability_and_consensus


class SpectralClusteringEngine(Clustering):

    def __init__(self, n_cluster: int,
                 n_neighbors: int,
                 random_state: int = 1,
                 affinity: str = "",
                 assign_labels: str = ""):
        self.n_neighbors = n_neighbors
        self.metric_for_silhouette = "euclidean"
        self.affinity=affinity
        self.model = SpectralClustering(n_clusters=n_cluster, affinity=affinity, n_neighbors=n_neighbors, assign_labels=assign_labels, random_state=random_state)

   # Overrides fit_predict method inhertited from Clustering class
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if self.affinity == "precomputed (cosine)":
            self.metric_for_silhouette="cosine"
            # 1. Full Cosine Similarity Matrix (Non-negative)
            # This ensures A_ij >= 0, which is required for a stable Laplacian.
            df_out = cosine_similarity(df_out)#df_out is affinity matrix
            # 2. Numerical Stability: Ensure diagonal is exactly 1.0
            np.fill_diagonal(df_out, 1.0)
            df_out = np.clip(df_out, 0, 1)

        labels = self.model.fit_predict(df_out) + 1
        return labels
