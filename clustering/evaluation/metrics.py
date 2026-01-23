"""
# Helper function for Dunn Index
def dunn_index(self, df, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    intra_dists = []
    for label in unique_labels:
        cluster_points = df[labels == label]
        if len(cluster_points) > 1:
            intra_dist = np.max(pdist(cluster_points))
        else:
            intra_dist = 0
        intra_dists.append(intra_dist)
    max_intra_dist = max(intra_dists)
    inter_dists = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = df[labels == unique_labels[i]]
            cluster_j = df[labels == unique_labels[j]]
            inter_dist = np.min(cdist(cluster_i, cluster_j))
            inter_dists.append(inter_dist)
    min_inter_dist = min(inter_dists)
    return min_inter_dist / max_intra_dist if max_intra_dist > 0 else np.inf


# Helper function for approximate BIC
def approximate_bic(self, df, k, inertia):
    n, d = df.shape
    if inertia == 0:
        return np.inf
    return n * np.log(inertia / n) + k * d * np.log(n)


def compute_gap_statistic(self, df, k, random_state, n_refs=5):
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=100).fit_predict(df)
    observed_inertia = kmeans.inertia_
    ref_inertias = []
    for _ in range(n_refs):
        ref_X = np.random.uniform(low=df.min(axis=0), high=df.max(axis=0), size=df.shape)
        ref_kmeans = KMeans(n_clusters=k, n_init=100).fit_predict(ref_X)
        ref_inertias.append(ref_kmeans.inertia_)
    gap = np.mean(np.log(ref_inertias)) - np.log(observed_inertia)
    return gap
"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)  # For reproducibility

range_n_clusters = [2, 3, 4, 5, 6]
# Create a subplot with 1 row and 2 columns
fig, axs = plt.subplots(1, len(range_n_clusters))
fig.set_size_inches(18, 7)
for n_clusters,ax1, in zip(range_n_clusters,axs.flatten()):


    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])



plt.show()