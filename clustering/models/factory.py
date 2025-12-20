from clustering.models.kmeans import KMeansEngine
from clustering.models.gmm import GMMEngine
from clustering.models.kmedoids import KMedoidsEngine

def get_engine_class(algo: str):
    mapping = {
        "kmeans": KMeansEngine,
        "gmm": GMMEngine,
        "kmedoids": KMedoidsEngine

    }
    if algo not in mapping:
        raise ValueError(f"Unsupported algorithm: {algo}")
    return mapping[algo]
