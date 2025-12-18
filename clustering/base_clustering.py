from typing import List
import pandas as pd

import numpy as np
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    davies_bouldin_score, pairwise_distances_argmin_min
)


class Clustering:
    """
    Unified clustering factory.
    Creates the correct engine and executes fit().
    """

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
        return representatives

    @staticmethod
    def recompute_centroid_provinces(df_pivot):
        """Recompute centroid provinces after clusters changed due to consensus relabel.
        The last column of df_pivot must be 'clusters', first len(df_pivot.columns-1) columns are features."""
        features = df_pivot.columns[:-1]  # exclude 'clusters'
        centroids = df_pivot.groupby('clusters')[features].mean()
        closest_indices, _ = pairwise_distances_argmin_min(centroids, df_pivot[features])
        return closest_indices

    @staticmethod
    def update_geo_cluster_centers(gdf_dict, geo_scale, df_pivot, closest_indices):
        """
        Attach cluster labels to geodata and compute centroids for display.
        Args:
            gdf_dict: Dictionary containing geodataframes (e.g., {'province': gdf...})
            geo_scale: Geographical scale key (province etc. ) to select the appropriate GeoDataFrame
            df_pivot: DataFrame containing the 'clusters' column
            closest_indices: List of indices representing cluster centers

        Returns:
            gdf_clusters: GeoDataFrame merged with clusters
            gdf_centroids: GeoDataFrame of the cluster representatives
        """
        # Prepare the base GeoDataFrame
        # Note: We assume gdf_dict has keys matching the values in st.session_state["geo_scale"]
        # or that mapping logic handles it. Based on BasePage logic:
        gdf_clusters = gdf_dict[geo_scale].set_index(geo_scale)
        # Merge with clusters
        gdf_clusters = gdf_clusters.merge(df_pivot["clusters"], left_index=True, right_index=True)
        # Compute centroids for representatives
        gdf_centroids = gdf_clusters[gdf_clusters.index.isin(closest_indices)].copy()
        gdf_centroids["centroid"] = gdf_centroids.geometry.centroid
        return gdf_clusters, gdf_centroids

    @staticmethod
    def optimal_k_analysis(df, random_states=1, n_init=1, k_values=range(2, 15)):
        pass
    @staticmethod
    def mean_sd_at_k(metrics_all, metric_name, k_index):
        """
        metrics_all[metric_name] = list of lists
        outer list: seeds
        inner list: k_values
        """
        values = [seed_vals[k_index] for seed_vals in metrics_all[metric_name]]
        return np.mean(values), np.std(values)

    # Not used
    @staticmethod
    def remap_clusters(labels: pd.Series, priority: List[str]) -> pd.Series:
        """
        labels:   pd.Series indexed by province name, values are original kmeans.labels_
        priority: list of province names in the order you want new labels assigned.

        Returns a new pd.Series of same index with relabeled cluster IDs 0,1,2…
        """
        new_label_map = {}
        next_new = 0
        for prov in priority:
            old_lbl = labels.loc[prov]
            if old_lbl not in new_label_map:
                new_label_map[old_lbl] = next_new
                next_new += 1

        # If you have provinces outside your priority list and want to
        # give them labels too, you could continue:
        for old_lbl in sorted(set(labels) - set(new_label_map)):
            new_label_map[old_lbl] = next_new
            next_new += 1

        return labels.map(new_label_map).to_list()