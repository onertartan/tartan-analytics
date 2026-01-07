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