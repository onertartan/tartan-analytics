import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score
import streamlit as st
from clustering.base_clustering import Clustering
from clustering.evaluation.stability import stability_and_consensus
import time


class KMedoidsEngine(Clustering):
    def __init__(self, n_cluster: int, random_state: int = 1,  metric = "", max_iter=-1, method: str = "pam"):

        self.model = KMedoids(
            n_clusters=n_cluster,
            metric=metric,
            method=method,
            max_iter=max_iter,
            init="k-medoids++",
            random_state=random_state,
        )
        self.metric_for_silhouette = metric #"cosine"
