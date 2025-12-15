from typing import List
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from clustering.models.kmeans import KMeansEngine
from clustering.models.gmm import GMMEngine
from sklearn.cluster import KMeans
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
    def get_engine_class(algo: str):
        mapping = {
            "kmeans": KMeansEngine,
            "gmm": GMMEngine,
          #  "dbscan": DBSCANEngine
        }
        if algo not in mapping:
            raise ValueError(f"Unsupported algorithm: {algo}")
        return mapping[algo]

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
   # @staticmethod
   # def run_clustering(df: pd.DataFrame, algorithm: str, **kwargs) -> pd.DataFrame:
        """
        Run clustering and return labelled data
        Parameters
        ----------
        df : features only (no 'clusters' column)
        algorithm : 'kmeans' | 'gmm' | 'dbscan'
        kwargs : engine-specific hyper-parameters
            kmeans : n_clusters, n_init, random_state
            gmm    : n_clusters, n_init, random_state, covariance_type
            dbscan : eps, min_samples, metric
        """
      #  engine = Clustering.get_engine(algorithm)(**kwargs)
      #  df_out = engine.fit(df)
      #  return df_out

   # @staticmethod
   # def get_representatives(df: pd,algorithm: str) -> List[str]:
   #     engine_class = Clustering.get_engine(algorithm)
   #     return engine_class.get_representatives(df)


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
        # Convert df to numpy array if it's a DataFrame
        n_samples = df.shape[0]
        # Initialize lists to store metrics for all seeds
        metrics_all = {"Inertia":[],"Silhouette":[],"Davies-Bouldin":[]}
        labels_all = {seed: {} for seed in random_states}
        # Compute metrics for each seed
        for random_state in random_states:
            inertias = []
            silhouette_scores = []
            davies_bouldin_scores = []

            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(df)
                labels = kmeans.labels_
                inertia = kmeans.inertia_
                inertias.append(inertia)
                #priority list to force to use same labels for centroids in different random states
                # it is tried but results didn't change,so same cities are already assigned to same clusters
                # priority_list = [
                #     "İzmir", "Van", "Afyonkarahisar", "Samsun", "Mardin",
                #     "Şanlıurfa", "Erzurum", "Tunceli", "Hatay"
                # ]
                # labels = self.remap_clusters(pd.Series(labels,index=df.index), priority_list)
                silhouette_scores.append(silhouette_score(df, labels))
                davies_bouldin_scores.append( davies_bouldin_score(df, labels))
                labels_all[random_state][k] = labels

            metrics_all["Inertia"].append(inertias)
            metrics_all["Silhouette"].append(silhouette_scores)
            metrics_all["Davies-Bouldin"].append(davies_bouldin_scores)

        # Compute mean metrics across seeds
        metrics_mean={"Inertia":np.mean(metrics_all["Inertia"], axis=0),
                      "Silhouette": np.mean(metrics_all["Silhouette"], axis=0),
                      "Davies-Bouldin": np.mean(metrics_all["Davies-Bouldin"], axis=0)}

        # Compute ARI scores
        ari_scores = {k: [] for k in k_values}
        for k in k_values:
            for i in range(len(random_states)):
                for j in range(i + 1, len(random_states)):
                    ari = adjusted_rand_score(labels_all[random_states[i]][k], labels_all[random_states[j]][k])
                    ari_scores[k].append(ari)
        ari_mean = [np.mean(ari_scores[k]) for k in k_values]
        ari_std = [np.std(ari_scores[k]) for k in k_values]
        # Compute Consensus Clustering Stability Metric (Average Consensus Index)
        # Initialize storage for consensus labels

        consensus_labels_all = {k: None for k in k_values}  # Dictionary to store consensus labels for each k
        consensus_indices = []
        for k in k_values:
            # Build consensus matrix for this k
            consensus_matrix = np.zeros((n_samples, n_samples))
            for seed in random_states:
                labels = labels_all[seed][k]
                for i in range(n_samples):
                    for j in range(i + 1, n_samples):
                        if labels[i] == labels[j]:
                            consensus_matrix[i, j] += 1
                            consensus_matrix[j, i] += 1
            print("53755 k=",k,"\nLABB:",labels)
            # Normalize by number of runs
            consensus_matrix /= len(random_states)
            # Compute average consensus index (mean of non-diagonal elements)
            mask = ~np.eye(n_samples, dtype=bool)  # Exclude diagonal
            avg_consensus = np.mean(consensus_matrix[mask])
            consensus_indices.append(avg_consensus)
            # Compute consensus labels using hierarchical clustering
            dissimilarity = 1 - consensus_matrix
            Z = linkage(dissimilarity[np.triu_indices(n_samples, k=1)], method='average')
            consensus_labels = fcluster(Z, t=k, criterion='maxclust')
            consensus_labels_all[k] = consensus_labels  # S
        return metrics_all,metrics_mean,ari_mean,ari_std,consensus_indices,consensus_labels_all

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