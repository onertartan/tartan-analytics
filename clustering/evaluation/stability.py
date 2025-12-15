from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

def stability_and_consensus(labels_all, k_values, random_states, n_samples):
    """
    Compute ARI-based stability and consensus clustering metrics.

    Parameters
    ----------
    labels_all : dict
        labels_all[seed][k] -> array-like of shape (n_samples,)
    k_values : iterable
        Candidate numbers of clusters.
    random_states : iterable
        Random seeds used in clustering.
    n_samples : int
        Number of samples.

    Returns
    -------
    ari_mean : list
        Mean ARI across random seeds for each k.
    ari_std : list
        Standard deviation of ARI for each k.
    consensus_indices : list
        Average consensus index for each k.
    consensus_labels_all : dict
        consensus_labels_all[k] -> consensus labels
    """

    # ---- ARI stability ----
    ari_scores = {k: [] for k in k_values}

    for k in k_values:
        for i in range(len(random_states)):
            for j in range(i + 1, len(random_states)):
                ari = adjusted_rand_score(
                    labels_all[random_states[i]][k],
                    labels_all[random_states[j]][k]
                )
                ari_scores[k].append(ari)

    ari_mean = [np.mean(ari_scores[k]) for k in k_values]
    ari_std = [np.std(ari_scores[k]) for k in k_values]

    # ---- Consensus clustering ----
    consensus_indices = []
    consensus_labels_all = {}

    for k in k_values:
        consensus_matrix = np.zeros((n_samples, n_samples))

        for seed in random_states:
            labels = labels_all[seed][k]
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] == labels[j]:
                        consensus_matrix[i, j] += 1
                        consensus_matrix[j, i] += 1

        consensus_matrix /= len(random_states)

        mask = ~np.eye(n_samples, dtype=bool)
        consensus_indices.append(np.mean(consensus_matrix[mask]))

        dissimilarity = 1 - consensus_matrix
        Z = linkage(
            dissimilarity[np.triu_indices(n_samples, k=1)],
            method="average"
        )
        consensus_labels_all[k] = fcluster(Z, t=k, criterion="maxclust")

    return ari_mean, ari_std, consensus_indices, consensus_labels_all
