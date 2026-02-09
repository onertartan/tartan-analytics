from adjustText import adjust_text
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import streamlit as st
import numpy as np
import pandas as pd
class PCAPlotter:
    import numpy as np
    import pandas as pd

    def apply_clr(self,df: pd.DataFrame, pseudocount: float = 1e-10) -> pd.DataFrame:
        """
        Apply centered log-ratio (CLR) transformation row-wise.

        Assumptions (important for your study):
        - Rows = names
        - Columns = provinces
        - Values are non-negative frequencies or proportions
        - Each row represents a compositional profile over provinces

        Parameters
        ----------
        df : pd.DataFrame
            Input data (rows = compositions).
        pseudocount : float
            Small constant added to avoid log(0).

        Returns
        -------
        pd.DataFrame
            CLR-transformed data with same shape and labels.
        """

        # --- safety checks ---
        if (df < 0).any().any():
            raise ValueError("CLR requires non-negative values.")

        # add pseudocount to avoid log(0)
        X = df + pseudocount

        # log-transform
        log_X = np.log(X)

        # row-wise geometric mean (mean of logs)
        row_means = log_X.mean(axis=1)

        # CLR transform: log(x_ij) - mean_j(log(x_i·))
        clr_values = log_X.sub(row_means, axis=0)

        return clr_values
    def plot_pca(
        self,
        df_pivot: pd.DataFrame,
        df_clusters: pd.Series,
        dense_threshold: int,
        mid_threshold: int,
        colors,
        title: str = "",
        raw_label: str = "Share-normalized",
        clr_label: str = "CLR-transformed",
        annotate: bool = True,
        show_loadings: bool = True,
        n_loadings: int = 5,
        point_alpha: float = 0.90,
        same_axis_limits: bool = True,
    ):
        """
        Produce a 1x2 side-by-side PCA figure contrasting raw/share-normalized vs CLR-transformed inputs.

        Notes (matches reviewer request):
        - Axis labels include explained variance (%).
        - Optional printing of example loadings.
        - Loadings correspond to FEATURES (df_pivot.columns). In your setup (rows=names, cols=provinces),
          these are PROVINCE loadings, not name loadings.
        """

        # ---- alignment checks (avoid silent mismatch) ----
        if not df_clusters.index.equals(df_pivot.index):
            df_clusters = df_clusters.reindex(df_pivot.index)

        # ---- build two inputs ----
        X_raw = df_pivot
        X_clr = self.apply_clr(df_pivot)

        # ---- robust cluster->color mapping ----
        unique_clusters = sorted(df_clusters.dropna().unique())
        cluster_to_color = {cid: colors[i] for i, cid in enumerate(unique_clusters)}
        point_colors = df_clusters.map(cluster_to_color)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)


        def _panel(ax, X, panel_title: str):
            pca = PCA(n_components=2)
            scores = pca.fit_transform(X)
            pc1, pc2 = pca.explained_variance_ratio_[:2]
            ax.scatter(
                scores[:, 0],
                scores[:, 1],
                c=point_colors,
                edgecolors="w",
                linewidths=0.5,
                alpha=point_alpha,
            )
            # axis variance (%), as requested by reviewer
            ax.set_xlabel(f"PC1 ({pc1:.2%})")
            ax.set_ylabel(f"PC2 ({pc2:.2%})")

            ax.set_title(f"{panel_title} | PC1+PC2 = {(pc1+pc2):.2%}", fontsize=11)
            ax.grid(True, alpha=0.3)

            # optional sparse annotation
            if annotate:
                texts = []
                cluster_counts = df_clusters.value_counts()

                for i, name in enumerate(df_pivot.index):
                    cid = df_clusters.loc[name]
                    size = cluster_counts.get(cid, 0)

                    if size >= dense_threshold:
                        continue
                    if size >= mid_threshold and i % 5 != 0:
                        continue

                    texts.append(
                        ax.annotate(
                            name,
                            (scores[i, 0], scores[i, 1]),
                            fontsize=8,
                            alpha=0.7,
                        )
                    )
                if texts:
                    adjust_text(texts, ax=ax)

            # optional example loadings (features = provinces)
            if show_loadings:
                load = pd.DataFrame(
                    pca.components_.T,
                    columns=["PC1", "PC2"],
                    index=X.columns
                )

                pc1_idx = load["PC1"].abs().sort_values(ascending=False).head(n_loadings)
                pc2_idx = load["PC2"].abs().sort_values(ascending=False).head(n_loadings)

                print(f"\n[{panel_title}] Top |PC1| feature loadings (provinces):")
                for feat, val in pc1_idx.items():
                    print(f"  {feat}: {val:.3f}")

                print(f"[{panel_title}] Top |PC2| feature loadings (provinces):")
                for feat, val in pc2_idx.items():
                    print(f"  {feat}: {val:.3f}")

            return scores, pca

        scores_raw, pca_raw = _panel(axes[0], X_raw, raw_label)
        scores_clr, pca_clr = _panel(axes[1], X_clr, clr_label)

        # ---- shared legend (once) ----
        legend_handles = [
            plt.Line2D([], [], marker="o", linestyle="",
                       color=cluster_to_color[cid], markersize=8,
                       label=f"Cluster {cid}")
            for cid in unique_clusters
        ]
        axes[1].legend(handles=legend_handles, title="Clusters", loc="best")
        # ---- match axis limits across panels (important for fair visual comparison) ----
        same_axis_limits = False
        if same_axis_limits:
            all_x = np.concatenate([scores_raw[:, 0], scores_clr[:, 0]])
            all_y = np.concatenate([scores_raw[:, 1], scores_clr[:, 1]])
            xpad = 0.05 * (all_x.max() - all_x.min() + 1e-12)
            ypad = 0.05 * (all_y.max() - all_y.min() + 1e-12)

            xlim = (all_x.min() - xpad, all_x.max() + xpad)
            ylim = (all_y.min() - ypad, all_y.max() + ypad)

            for ax in axes:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        # ---- figure super title ----
        if title:
            fig.suptitle(title, fontsize=13)
        st.pyplot(fig)
        # return PCA objects too (useful for reporting PC variance + loadings in text/tables)
        pca = PCA()
        pca.fit(df_pivot)
        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
        fig, ax = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)

        ax.plot(range(1, len(per_var) + 1), per_var.cumsum(), marker="o", linestyle="--")
        ax.grid(True)
        ax.set_ylabel("Percentage Cumulative of Explained Variance")
        ax.set_xlabel("Number of Components")
        ax.set_title("Explained Variance by Component")
        st.pyplot(fig)
        return fig, (pca_raw, pca_clr)
    def plot_pca_old(self, df_pivot, df_clusters, dense_threshold, mid_threshold, colors, title=""):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(df_pivot)
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df_clusters.apply(lambda x: colors[x-1]), edgecolors='w')
        # Get the explained variance ratios for each component
        explained_variance_ratios = pca.explained_variance_ratio_
        # Calculate cumulative variance for the first two components
        cumulative_variance_two = sum(explained_variance_ratios[:2])
        # --- Add Legend Here ---
        # Get unique cluster IDs and sort them (e.g., [1, 2, 3])
        unique_clusters = sorted(df_clusters.unique())
        # Create legend labels and handles
        legend_labels = [f'Cluster {cluster_id}' for cluster_id in unique_clusters]
        texts = []

        # Map each cluster to its color in the colormap,         # Add legend to plot
        legend_handles = [plt.Line2D([], [], marker='o', linestyle='', color=colors[i], markersize=10, label=label) for i, label in enumerate(legend_labels)]
        ax.legend(handles=legend_handles, title='Clusters', loc='best')

        cluster_counts = df_clusters.value_counts()  # Calculate ONCE

        for i, name in enumerate(df_pivot.index):
            cluster_id = df_clusters[name]
            cluster_size = cluster_counts[cluster_id]
            # Skip dense clusters (>=10% of data)
            if cluster_size >= dense_threshold:
                continue
            # For mid-density (5-10%), annotate every 5th point
            if cluster_size >= mid_threshold and i % 5 != 0:
                continue
            # Annotate sparse clusters (<5%) and selected mid-density points
         #   texts.append(ax.annotate(name, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=8, alpha=0.7))

        adjust_text(texts)
        ax.set_title(f"{title}\nExplained_variance ratios{explained_variance_ratios.round(2)}\nCumulative variance of two components:{cumulative_variance_two.round(2)}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True)
        # from mpl_toolkits.mplot3d import Axes3D
        # pca = PCA(n_components=3)
        # reduced_data = pca.fit_transform(df_pivot)
        # fig = plt.figure(figsize=(10, 8))
        # # Get the explained variance ratios for each component
        # explained_variance_ratios = pca.explained_variance_ratio_
        #
        # # Calculate cumulative variance for the first two components
        # cumulative_variance_two = sum(explained_variance_ratios[:2])
        #
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
        #            c=df_clusters.apply(lambda x: self.COLORS[x - 1]), edgecolors='w')
        # # Annotate in 3D with offset to reduce overlap
        # for i, name in enumerate(df_pivot.index):
        #     # Use ax.text for 3D annotations with a slight offset
        #     ax.text(
        #         reduced_data[i, 0] ,  # Small offset in x
        #         reduced_data[i, 1] ,  # Small offset in y
        #         reduced_data[i, 2] ,  # Small offset in z
        #         name,
        #         fontsize=8,
        #         alpha=0.7
        #     )
        # ax.set_title(f"\nExplained_variance ratios{explained_variance_ratios}\nCumulative variance of two components:{cumulative_variance_two}")
        #
        # # Set labels
        # ax.set_xlabel('PC1')
        # ax.set_ylabel('PC2')
        # ax.set_zlabel('PC3')
        st.pyplot(fig)