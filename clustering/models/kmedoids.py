import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score
import streamlit as st
from clustering.base_clustering import Clustering
from clustering.evaluation.stability import stability_and_consensus
import time

class KMedoidsEngine:
    def __init__(
        self, n_cluster: int, random_state: int = 1, metric: str = "cosine", method: str = "pam", max_iter: int = 300):
        self.metric = metric
        self.model = KMedoids(
            n_clusters=n_cluster,
            metric=metric,
            method=method,
            max_iter=max_iter,
            init="k-medoids++",
            random_state=random_state,
        )

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        labels = self.model.fit_predict(df.values)
        df_out = df.copy()
        df_out["clusters"] = labels + 1
        return df_out

    @staticmethod
    def optimal_k_analysis(df, random_states, metric, max_iter, k_values=range(2, 15)):
        X = df.values if hasattr(df, "values") else df
        n_samples = X.shape[0]

        # ---- Model-specific storage ----
        metrics_all = {"Silhouette Score": [], "Davies-Bouldin Index": []}

        labels_all = {seed: {} for seed in random_states}
        progress_bar = st.progress(0.0)
        status_text = st.empty()  # This will hold the "X / Y completed" message
        total_states = len(random_states)
        start_time = time.time()  # Record overall start time

        # ---- Run K-Medoids (PAM) ----
        for idx, random_state in enumerate(random_states):
            silhouettes = []
            db_scores = []
            seed_start = time.time()

            for k in k_values:
                model = KMedoids(n_clusters=k, metric=metric, method="pam", init="k-medoids++", max_iter=max_iter, random_state=random_state)
                labels = model.fit_predict(X)
                silhouettes.append(silhouette_score(X, labels))
                db_scores.append(davies_bouldin_score(X, labels))
                labels_all[random_state][k] = labels
            # Update progress and status after loop for one random state is completed
            progress_bar.progress((idx + 1) / total_states)
            elapsed_total = time.time() - start_time
            elapsed_minutes, elapsed_seconds = divmod(int(elapsed_total), 60)
            seed_time = int(time.time() - seed_start)
            status_text.text(f"The last seed took {seed_time}s")
            status_text.text(f"Completed {idx + 1}/{total_states} seeds.Elapsed: {elapsed_minutes}m {elapsed_seconds}s")

            metrics_all["Silhouette Score"].append(silhouettes)
            metrics_all["Davies-Bouldin Index"].append(db_scores)

        progress_bar.empty()
        status_text.empty()
        # ---- Mean metrics across seeds ----
        metrics_mean = {key: np.mean(metrics_all[key], axis=0) for key in metrics_all}

        # ---- Model-independent evaluation ----
        ari_mean, ari_std, consensus_indices, consensus_labels_all = \
            stability_and_consensus(labels_all=labels_all, k_values=k_values, random_states=random_states,
                                    n_samples=n_samples)
        df_summary = KMedoidsEngine.summarize(metrics_all, ari_mean, ari_std, consensus_indices, k_values)
        return df_summary, metrics_all, metrics_mean, ari_mean, ari_std, consensus_indices, consensus_labels_all

    @staticmethod
    def summarize(metrics_all, ari_mean, ari_std, consensus_indices, k_values):
        rows = []
        for k in k_values:
            idx = list(k_values).index(k)
            sil_m, sil_s = Clustering.mean_sd_at_k(metrics_all, "Silhouette Score", idx)
            db_m, db_s = Clustering.mean_sd_at_k(metrics_all, "Davies-Bouldin Index", idx)
            rows.append({
                "Number of clusters": k,
                "Silhouette_mean": sil_m,
                "Silhouette_std": sil_s,
                "DaviesBouldin_mean": db_m,
                "DaviesBouldin_std": db_s,
                "ARI_mean": ari_mean[idx],
                "ARI_std": ari_std[idx],
                "Consensus": consensus_indices[idx],
            })
        return pd.DataFrame(rows).set_index("Number of clusters")

