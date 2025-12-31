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
