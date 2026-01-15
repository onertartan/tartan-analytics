from typing import Dict
import pandas as pd


def create_cluster_color_mapping(
        gdf: pd.DataFrame,  # must contain a 'clusters' column
        cluster_color_defaults: Dict[str, str],  # JSON content from config
        ) -> Dict[int, str]:
    """
    Return {cluster_id: matplotlib_colour}
    Guarantees every cluster (1..n_clusters) has a colour.
    Cyclic fallback is used if there are more clusters than available colors.
    """
    color_map: Dict[int, str] = {-1: "gray"}
    used_colors = set()
    clusters = set(gdf["clusters"].unique())

    # 1st pass – honour defaults where possible
    for idx, color in cluster_color_defaults.items():
        if idx in gdf.index:
            cluster = gdf.loc[idx, "clusters"]
            if isinstance(cluster, pd.Series):
                cluster = cluster.iloc[0]

            if cluster not in color_map and color not in used_colors:
                color_map[cluster] = color
                used_colors.add(color)

    # 2nd pass – fill leftovers
    remaining_colors = [c for c in cluster_color_defaults.values()
                        if c not in used_colors]
    remaining_clusters = clusters - set(color_map)

    # If there are more clusters than colors, use a cyclic fallback
   # if len(remaining_clusters) > len(remaining_colors):
     #   remaining_colors = (remaining_colors * (len(remaining_clusters) // len(remaining_colors)) +
      #                      remaining_colors[:len(remaining_clusters) % len(remaining_colors)])

    # Assign remaining colors to remaining clusters
    for i, cluster in enumerate(remaining_clusters):
        color_map[cluster] = remaining_colors[i]

    return color_map
