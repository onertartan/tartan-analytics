from typing import List
import numpy as np
import pandas as pd
import time
import streamlit as st
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

from clustering.base_clustering import Clustering
from clustering.evaluation.stability import stability_and_consensus


class SpectralClusteringEngine(Clustering):

    def __init__(self, n_cluster: int,
                 n_neighbors: int,
                 random_state: int = 1,
                 affinity: str = "",
                 assign_labels: str = "",
                 spectral_geometry=""):
        self.n_neighbors = n_neighbors
        self.spectral_geometry = spectral_geometry
        self.metric_for_silhouette = spectral_geometry
        if spectral_geometry=="euclidean":
            affinity="nearest_neighbors"
        else: #cosine
            affinity="precomputed"
        self.model = SpectralClustering(n_clusters=n_cluster, affinity=affinity, n_neighbors=n_neighbors, assign_labels=assign_labels, random_state=random_state)

   # Overrides fit_predict method inhertited from Clustering class
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if self.spectral_geometry == "cosine":
            nn = NearestNeighbors(n_neighbors=self.n_neighbors,  metric="cosine")
            df_out = nn.fit(df_out).kneighbors_graph(df_out, mode="connectivity")

        labels = self.model.fit_predict(df_out) + 1
        df_out = df.copy()
        df_out["clusters"] = labels
        return df_out
