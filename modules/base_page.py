# Standard Packages
import json
from typing import Dict, List
from abc import ABC, abstractmethod

# Data Processing and Analysis
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster

# Visualization
import matplotlib.pyplot as plt
from matplotlib import colormaps
from adjustText import adjust_text
from viz import PCAPlotter, OptimalKPlotter
# Machine Learning (Scikit-Learn)
from sklearn.cluster import KMeans, DBSCAN
from clustering.kmeans import KMeansEngine
from clustering.gmm import GMMEngine
from clustering.clustering import Clustering

# Streamlit & Tools
import streamlit as st
import extra_streamlit_components as stx

class BasePage(ABC):
    features = None
    page_name = None
    correlation_analysis = False
    animation_available = True
    geo_scales = ["province (ibbs3)", "sub-region (ibbs2)", "region (ibbs1)", "district"]
    top_row_cols = []
    checkbox_group = {}
    data = None  # Class-level data storage
    gdf_centroids = None  # closest provinces to centroids

    @property
    def COLORS(self):
        if "COLORS" not in st.session_state:
            st.session_state["COLORS"] = ["red", "purple", "orange", "green", "dodgerblue", "magenta", "gold", "darkorange", "darkolivegreen",
              "cyan",  "lightblue", "lightgreen", "darkkhaki", "brown", "lime", "orangered", "blue", "mediumpurple", "turquoise"] +list(colormaps["Dark2"].colors)+ list(colormaps["Set2"].colors)+list(colormaps["Pastel1"].colors)+["yellow","silver"]
        return st.session_state["COLORS"]

    @property
    def CLUSTER_COLOR_MAPPING(self):
        if "CLUSTER_COLOR_MAPPING" not in st.session_state:
            with open('json_files/cluster_colors.json', 'r', encoding='utf-8') as f:
                st.session_state["CLUSTER_COLOR_MAPPING"] = json.load(f)
        return st.session_state["CLUSTER_COLOR_MAPPING"]

    @property
    def VA_POSITIONS(self):
        if "VA_POSITIONS" not in st.session_state:
            with open('json_files/va_positions.json', 'r', encoding='utf-8') as f:
                st.session_state["VA_POSITIONS"] = json.load(f)
        return st.session_state["VA_POSITIONS"]

    @property
    def HA_POSITIONS(self):
        if "HA_POSITIONS" not in st.session_state:
            st.session_state["HA_POSITIONS"] = {"Zonguldak": "right", "Adana": "right","Yalova":"right"}
        return st.session_state["HA_POSITIONS"]

    @property
    def gdf(self):
        if "gdf" not in st.session_state:
            st.session_state["gdf"] = BasePage.load_geo_data()
        return st.session_state["gdf"]

    @property
    def gdf_clusters(self):
        return st.session_state.get("gdf_clusters")

    @gdf_clusters.setter
    def gdf_clusters(self, value):
        st.session_state["gdf_clusters"] = value


    def load_css(self,file_path: str) -> None:
        """
        Load CSS from a file and inject it into the Streamlit app using st.markdown.

        Args:
            file_path: Path to the CSS file.
        """
        try:
            with open(file_path, "r") as f:
                css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        except FileNotFoundError:
            print(f"CSS file not found: {file_path}")
        except Exception as e:
            print(f"CSS file not found: {file_path}")

    def create_color_mapping(self, gdf: gpd.GeoDataFrame, n_clusters: int) -> Dict[int, str]:
        """Generate cluster color mapping with province-based defaults."""
        color_map = {}
        used_colors = set()  # Track which colors have been assigned
        clusters = set(range(1, n_clusters + 1))

        for idx, color in self.CLUSTER_COLOR_MAPPING.items():
            if idx in gdf.index:
                cluster = gdf.loc[idx, "clusters"][0] if isinstance(gdf.loc[idx, "clusters"], pd.Series) else gdf.loc[idx, "clusters"]# it returns multiple values(series) for the same name

                # Only assign if both cluster is new AND color hasn't been used
                if cluster not in color_map and color not in used_colors:
                    color_map[cluster] = color
                    used_colors.add(color)

        # Assign remaining colors to any clusters without colors
        remaining_colors = [c for c in self.CLUSTER_COLOR_MAPPING.values() if c not in used_colors]
        remaining_clusters = clusters - set(color_map.keys())

        for i, cluster in enumerate(remaining_clusters):
            color_map[cluster] = remaining_colors[i]
        return color_map

    def load_data(self):
        if self.data is None:
            self.data = self.get_data()
        return self.data


    @st.cache_data
    def get_data(self, geo_scale=None):
        """Base implementation (override in subclasses)"""
        if self.data is None:
            raise NotImplementedError("Subclasses must implement get_data")
        return self.data

    @staticmethod
    def convert_year_index_data_type(df):
        """Not all data years are integers.In particular elections data of 2015 June and 2015 November are not string"""
        return pd.MultiIndex.from_arrays([df.index.get_level_values(0).astype(str),  df.index.get_level_values(1)], names=df.index.names)

    def fun_extras(self, *args):
        pass

    # @staticmethod
    # @st.cache_data
    # def get_data(geo_scale=None):
    #     pass


    @staticmethod
    @st.cache_data
    def load_geo_data():
        gdf = {}
        gdf["district"] = gpd.read_file("data/preprocessed/gdf_borders_district.geojson")
        gdf["province"] = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
        return gdf

    def render(self):
        pass

    def run(self):
        st.session_state["page_name"] = self.page_name
        self.load_data()
        self.render()

    def get_selected_features(self, cols_nom_denom):
        selected_features = {}
        for nom_denom in cols_nom_denom.keys():  # cols_nom_denom is a dict whose keys are "nominator" or "denominator" and values are st.columns
            selected_features[nom_denom] = ()  # tuple type is needed for multiindex columns
            for i, feature in enumerate(self.features[nom_denom]):
                selected_features[nom_denom] = selected_features[nom_denom] + ( self.get_selected_feature_options(cols_nom_denom[nom_denom][i], feature, nom_denom),)
        #    Checkbox_Group.age_group_quick_select()
        return selected_features

    def get_selected_feature_options(self, col, feature_name, nom_denom_key_suffix):
        disabled = not st.session_state["display_percentage"] if nom_denom_key_suffix == "denominator" else False
        if feature_name in ["marital_status", "education", "age", "Party/Alliance", "sex", "month"]:
            self.quick_selection(feature_name, nom_denom_key_suffix)
            self.checkbox_group[feature_name].place_checkboxes(col, nom_denom_key_suffix, disabled, feature_name)
            selected_feature = self.checkbox_group[feature_name].get_checked_keys(nom_denom_key_suffix, feature_name)
        return selected_feature

    @staticmethod
    def update_selected_slider_and_years(slider_index):
        st.session_state.selected_slider = slider_index
        if slider_index == 1:
            st.session_state["year_1"]= st.session_state["year_2"] = int(st.session_state.slider_year_1)
        else:
            st.session_state["year_1"], st.session_state["year_2"] = int(st.session_state.slider_year_2[0]), int( st.session_state.slider_year_2[1])


    def ui_basic_setup(self):
        cols_title = st.columns(2)
        cols_title[0].markdown("<h3 style='color: red;'>Select primary parameters.</h3>", unsafe_allow_html=True)
        cols_title[0].markdown("<br><br><br>", unsafe_allow_html=True)

        # Checkbox to switch between population and percentage display
        cols_title[1].markdown("<h3 style='color: blue;'>Select secondary parameters.</h3>", unsafe_allow_html=True)
        cols_title[1].checkbox("Check to get ratio: primary parameters/secondary parameters.", key="display_percentage")
        cols_title[1].write("Uncheck to show counts of primary parameters.")

        cols_all = st.columns(self.col_weights)  # There are 2*n columns(for example: 3 for nominator,3 for denominator)
        with cols_all[len(cols_all) // 2]:
            st.html(
                '''
                    <div class="divider-vertical-line"></div>
                    <style>
                        .divider-vertical-line {
                            border-left: 1px solid rgba(49, 51, 63, 0.2);
                            height: 180px;
                            margin: auto;
                        }
                    </style>
                '''
            )
        cols_nom_denom = {"nominator": cols_all[0:len(self.col_weights)//2], "denominator": cols_all[len(self.col_weights)//2 + 1:]}
        return cols_nom_denom


    def animation_slider_changed(self):
        st.session_state["animate"] = True

    @staticmethod
    def sidebar_controls_basic_setup(*args):
        """
           Renders the sidebar controls
           Parameters: starting year, ending year
        """
        # Inject custom CSS to set the width of the sidebar
        #   st.markdown("""<style>section[data-testid="stSidebar"] {width: 300px; !important;} </style> """,  unsafe_allow_html=True)
        if "visualization_option" not in st.session_state:
            st.session_state["visualization_option"] = "matplotlib"
        start_year = args[0]
        end_year = args[1]
        with (st.sidebar):
            st.header('Visualization options')
            st.write(start_year,end_year)
            # if ifadesine gerek olmadığı düşünülerek (hata olursa bu if kalktığı için olabilir) metot classmethod'dan static'e dönüştü. Böylelikle higher-education kullanabildi.
            # options= list(range(start_year, end_year + 1)) if cls.page_name != "sex_age_edu_elections" else [2018,2023]
            options = list(range(start_year, end_year + 1))
            # Create a slider to select a single year
            st.select_slider("Select a year", options, 2023, on_change=BasePage.update_selected_slider_and_years, args=[1],  key="slider_year_1")
            # Create sliders to select start and end years
            st.select_slider("Or select start and end years",options, [options[0],options[-1]],on_change=BasePage.update_selected_slider_and_years, args=[2], key="slider_year_2")

            # Main content
            if "animation_images_generated" not in st.session_state:
                st.session_state["animation_images_generated"] = False

            if "selected_slider" not in st.session_state:
                st.session_state["selected_slider"] = 1
            BasePage.update_selected_slider_and_years(st.session_state.selected_slider)

            if st.session_state.selected_slider == 1:
                st.write("You have selected a single year from the first slider.")
                st.write("Selected year:", st.session_state["year_1"])
            else:
                st.write("You have selected start and end years from the second slider.")
                st.write("Selected start year:", st.session_state["year_1"], "\nSelected end year:", st.session_state["year_2"])


    def sidebar_controls(self, *args):  # start_year=2007,end_year=2023
        self.sidebar_controls_basic_setup(*args)
        self.sidebar_controls_plot_options_setup(*args)


    def sidebar_controls_plot_options_setup(self):
        pass

    def quick_selection(self, feature_name, nom_denom_key_suffix):
        pass

    def set_checkbox_values_for_quick_selection(self, keys_to_check, nom_denom_key_suffix, feature_name):
         for key in self.checkbox_group[feature_name].basic_keys:
            if key in keys_to_check:
                val = True
            else:
                val = False
            st.session_state[self.page_name+"_"+nom_denom_key_suffix+"_"+feature_name+"_"+key] = val
            #cls.checkbox_group[feature_name].checked_dict[nom_denom_key_suffix][key] = val
       #1 print("$$$",nom_denom_key_suffix,"$$$",cls.checkbox_group[feature_name].checked_dict[nom_denom_key_suffix])

    def figure_setup(self,display_change=False):
        if st.session_state["visualization_option"] != "matplotlib":
            return None, None
        if st.session_state["year_1"] == st.session_state["year_2"] or st.session_state["selected_slider"] == 1 or \
                st.session_state["animate"]:
            n_rows = 1
        elif display_change:
            n_rows = 3
        else:
            n_rows = 2
        fig, axs = plt.subplots(n_rows, 1, squeeze=False, figsize=(10, 4 * n_rows),
                                gridspec_kw={'wspace': 0, 'hspace': 0.1})  # axs has size (3,1)
        # fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=0.5, hspace=0.5)
        return fig, axs

    @abstractmethod
    def preprocess_clustering(self, df, *args):
        # Overriden by sub-classes Base_Page_Names & Base_Page_Common
        pass


    def tab_clustering(self, df, *args):
        # 0. Render UI
        clustering_algorithm = self.gui_clustering()
        if not clustering_algorithm:
            return
        col_plot, col_df = st.columns([5, 1])
        # 1. Run clustering: Preprocess
        df_pivot = self.preprocess_clustering(df, *args)

        random_states = range(st.session_state["number_of_seeds"])  # 50 random seeds
        n_init = st.session_state["n_init_" + self.page_name]
        k_values = list(range(2, 15))
        n_cluster = st.session_state["n_clusters_" + self.page_name]
        num_seeds_to_plot = 3

        with col_plot:
            """ If optimal_k_analysis is selected or use_consensus_labels is checked but it is not present(optimal_k_analysis has not previously run) """
            if st.session_state.get("optimal_k_analysis", False) or (st.session_state.get("use_consensus_labels_" + self.page_name, False) and "consensus_labels_" + self.page_name not in st.session_state):
                metrics_all, metrics_mean, ari_mean, ari_std, consensus_indices, consensus_labels_all = Clustering.optimal_k_analysis(df_pivot, random_states, n_init, k_values)
                st.session_state["consensus_labels_" + self.page_name] = consensus_labels_all
                df_pivot["clusters"] = consensus_labels_all[n_cluster]
                OptimalKPlotter.plot_optimal_k_analysis(num_seeds_to_plot, k_values, random_states, metrics_all, metrics_mean, ari_mean, ari_std, consensus_indices)
            elif st.session_state.get("use_consensus_labels_" + self.page_name, False):
                df_pivot["clusters"] = st.session_state["consensus_labels_" + self.page_name][n_cluster]
                st.header("Using consensus labels")
            else:
                kwargs = {"n_init": n_init, "n_cluster": n_cluster}
                df_pivot = Clustering.run_clustering(df_pivot, clustering_algorithm, **kwargs)
            # Step: Update geodata
            representatives = Clustering.get_representatives(df_pivot, clustering_algorithm)
            if st.session_state.get("selected_tab_" + self.page_name, "") == "tab_geo_clustering":
                self.update_geo_cluster_centers(df_pivot, representatives)

        if st.session_state.get("selected_tab_" + self.page_name, "") == "tab_geo_clustering":
            # Step-6: Render geo-cluster plots
            self.render_geo_clustering_plots(df_pivot, col_plot, col_df, df)
        #Step-7: PCA
        # PCA plot
        df_clusters = df_pivot["clusters"]
        df_features = df_pivot.drop(columns=["clusters"])
        with col_plot:
            total_points = len(df_clusters)
            factor = .1 if self.page_name == "names_surnames" else 1
            dense_threshold = total_points / (10 * factor) if st.session_state["selected_tab_" + self.page_name] != "tab_map" else 100  # Define thresholds
            mid_threshold = total_points / (20 * factor) if st.session_state["selected_tab_" + self.page_name] != "tab_map" else 100  #
            PCAPlotter().plot_pca(df_features, df_clusters,dense_threshold, mid_threshold, self.COLORS)

    def render_geo_clustering_plots(self, df_pivot, col_plot, col_df, df_original):
        """Tab-1 Step-6:   plot clusters and show clusters dataframe."""
        df_clusters = df_pivot["clusters"]
        # Determine year or year range
        start_year = df_original.index.get_level_values(0).min()
        end_year = df_original.index.get_level_values(0).max()
        if start_year == end_year:
            year_label = f"in {start_year}"
        else:
            year_label = f"between {start_year}-{end_year}"
        # Plot geographic clusters
        self.plot_geo_clusters(year_label, col_plot)
        # Show raw cluster assignments
        col_df.dataframe(df_clusters)

    def plot_geo_clusters(self, year_in_title, col_plot):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        n_clusters = st.session_state["n_clusters_" + self.page_name]
        # Define a color map for the categories
        color_map = self.create_color_mapping(self.gdf_clusters, n_clusters)
        # Map the colors to the GeoDataFrame
        #SEÇİM - BİRİNCİ PARTİ
        secim = False
        if secim:
            file_name = "elections2023.csv"
            self.gdf_clusters["clusters"] = pd.read_csv(file_name, index_col=0)["cluster"].tolist() # elections1-->1.figure
            color_map = {1: "darkorange", 2: "red", 3: "purple", 4: "gold"}
        #
        # #SEÇİM1-SON

        self.gdf_clusters["color"] = self.gdf_clusters["clusters"].map(color_map)
        nan_rows = self.gdf_clusters[self.gdf_clusters.isna().any(axis=1)]
        print("nan rows:",nan_rows,"n_clusters",n_clusters)
        print(self.gdf_clusters.index)
        self.gdf_clusters.plot(ax=ax, color=self.gdf_clusters['color'], legend=True, edgecolor="black", linewidth=.2)
        ax.axis("off")
        ax.margins(x=0)

        # Add province names (from index) at centroids
        bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6)
        ha_positions, va_positions = self.HA_POSITIONS, self.VA_POSITIONS
        for province in self.gdf_clusters.index:
            ax.annotate(text=province,  # Use index (province name) directly
                        xy=(self.gdf_clusters.loc[province, "geometry"].centroid.x,
                            self.gdf_clusters.loc[province, "geometry"].centroid.y),
                        ha=ha_positions.get(province, "center"), va=va_positions.get(province, "center"), fontsize=5, color="black", bbox=bbox)

        if secim:
            from matplotlib.patches import Patch
            if file_name == "elections2022.csv":
                legend_handles = [
                    Patch(facecolor='darkorange', label='People’s Alliance'),
                    Patch(facecolor='red', label="Nation's Alliance"),
                    Patch(facecolor='purple', label="Labour's Alliance")
                ]
                title = "2023 Turkish Parliamentary Elections: Provincial Wins by Alliance Blocs"
            else:
                legend_handles = [
                    Patch(facecolor='darkorange', label='People’s Alliance (AKP + MHP)'),
                    Patch(facecolor='red', label='CHP'),
                    Patch(facecolor='purple', label='DEM Party')
                ]
                title = "2024 Turkish Municipal Council Elections: Provincial Wins by Alliance Blocs and Competing Parties"
            ax.legend(
                handles=legend_handles,
                loc=[.55, .87],
                fontsize=6,
                title_fontsize=6,
                frameon=False  # Remove if you want a background
            )
            ax.set_title(title)
        else:
            ax.set_title(f"{n_clusters} Clusters Identified {year_in_title} (K-means)")
            ax.legend(loc="upper right", fontsize=6)
            # Compute centroids of the closest provinces and plot them as markers
            closest_provinces_centroids = self.gdf_centroids.to_crs("EPSG:4326").copy()
            closest_provinces_centroids["centroid_geometry"] = closest_provinces_centroids.geometry.centroid
            # Create a temporary GeoDataFrame with centroid geometries (points)
            closest_provinces_points = gpd.GeoDataFrame(closest_provinces_centroids, geometry="centroid_geometry",
                                                        crs=self.gdf_clusters.crs)
            # Add markers using the centroid points (no fill color change   # Transparent fill)
            closest_provinces_points.plot(ax=ax, facecolor="none", markersize=120, edgecolor="black", linewidth=1.5,
                                          label=f"Closest provinces\nto  cluster centers")
        col_plot.pyplot(fig)


    def run_clustering(self, df_pivot, clustering_algorithm):
        # 1. Define the mapping (Dispatch Dictionary)
        # Maps algorithm name (string) to the corresponding class method (function)
        n_init = st.session_state["n_init_" + self.page_name]
        n_clusters = st.session_state["n_clusters_" + self.page_name]
        algorithm_map = {"kmeans": KMeansEngine(n_clusters, n_init),
                         "gmm": GMMEngine(n_clusters, n_init), "dbscan": self.dbscan}
        # 2. Look up the method
        # Use .get() for safe access and provide a default error if the key isn't found
        clustering_engine = algorithm_map[clustering_algorithm]
        # 3. Call the selected method
        df_pivot, closest_indices = clustering_engine.fit(df_pivot)
        return df_pivot, closest_indices

    @abstractmethod
    def scale(self, df, *args):
        pass


    def compute_gap_statistic(self, df, k, random_state, n_refs=5):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=100).fit(df)
        observed_inertia = kmeans.inertia_
        ref_inertias = []
        for _ in range(n_refs):
            ref_X = np.random.uniform(low=df.min(axis=0), high=df.max(axis=0), size=df.shape)
            ref_kmeans = KMeans(n_clusters=k, n_init=100).fit(ref_X)
            ref_inertias.append(ref_kmeans.inertia_)
        gap = np.mean(np.log(ref_inertias)) - np.log(observed_inertia)
        return gap

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

    def remap_clusters(self, labels: pd.Series,
                       priority: List[str]) -> pd.Series:
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



    # methods used in base_page_names

    def update_geo_cluster_centers(self, df_pivot, closest_indices):
        """Attach cluster labels to geodata and compute centroids for display."""
        self.gdf_clusters = self.gdf[st.session_state["geo_scale"] ].set_index(st.session_state["geo_scale"] )
        self.gdf_clusters = self.gdf_clusters.merge(df_pivot["clusters"], left_index=True, right_index=True)
        # centroid provinces
        self.gdf_centroids = self.gdf_clusters[self.gdf_clusters.index.isin(closest_indices)]
        self.gdf_centroids["centroid"] = self.gdf_centroids.geometry.centroid
    # CLUSTERING GUIs and METHODS


    # ---------- 1. GUI - geo_clustering -----------------
    def gui_clustering(self):
        col1, col2, _ = st.columns([2, 2, 6])
        with col1:
            self.gui_clustering_up_col1()
        with col2:
            self.gui_clustering_up_col2()
        selected_algo = self.gui_clustering_bottom()
        return selected_algo

    def gui_clustering_up_col1(self):
        pass

    def gui_clustering_up_col2(self):
        # scaler
        st.session_state["optimal_k_analysis"] = st.checkbox("Run cluster analysis")
        st.session_state["number_of_seeds"] = st.number_input("Number of seeds", min_value=1, max_value=100, value=1)
        st.checkbox("Use consensus labels", False, key="use_consensus_labels_" + self.page_name)


    def gui_clustering_bottom(self):
        # Algorithm Selection
        algos = {
            "kmeans": {"label": "K-means", "gui_func": self.gui_clustering_kmeans},
            "gmm": {"label": "GMM", "gui_func": self.gui_clustering_gmm},
            "dbscan": {"label": "DBSCAN", "gui_func": self.dbscan_gui_options},
        }
        cols = st.columns(len(algos))
        selected_algo = None

        for col, (key, config) in zip(cols, algos.items()):
            with col:
                with st.form("submit_form_" + key + self.page_name):
                    submitted = st.form_submit_button(config["label"], use_container_width=True)
                    config["gui_func"]()

                    if submitted:
                        selected_algo = key
                        # NEW: Explicitly sync the input value to the logic variable here
                        if key in ["kmeans", "gmm"]:
                            st.session_state["n_clusters_" + self.page_name] = st.session_state["n_clusters_" + key]
                            st.session_state["n_init_" + self.page_name] = st.session_state["n_init_" + key]

        return selected_algo

    def gui_clustering_kmeans_gmm_common(self, key):
        # CHANGED: Removed direct assignment to st.session_state["n_clusters_" + self.page_name]
        # We just render the widget. Streamlit stores the value in st.session_state["n_clusters_" + key] automatically.
        st.number_input("Number of clusters / components", 2, 15, 6, key="n_clusters_" + key)
        st.number_input("Random restarts (n_init)", 1, 100, 10, key="n_init_" + key)
    def gui_clustering_kmeans(self):
        self.gui_clustering_kmeans_gmm_common("kmeans")

    def gui_clustering_gmm(self):
            self.gui_clustering_kmeans_gmm_common("gmm")
            self.gmm_k = st.number_input("Components", min_value=2, max_value=15, value=6)
            self.gmm_cov = st.selectbox("Covariance", options=["diag", "full", "tied", "spherical"])
            self.gmm_n_init =st.number_input("Restarts", min_value=1, max_value=50, value=10)


    def dbscan_gui_options(self):
        """
        exp1, exp2, exp3 are the three columns you already pass in.
        """
        self.db_eps = st.number_input("ε (eps)",
                                        min_value=0.05, max_value=2.0,
                                        value=0.25, step=0.05,
                                        help="Max distance for neighbourhood")
        self.db_min = st.number_input("minPts",
                                        min_value=2, max_value=20,
                                        value=5, step=1,
                                        help="Min points to form a core region")
        # optional metric (keep Euclidean unless you need something else)
        self.db_metric = st.selectbox("Metric", options=["euclidean", "cosine"])

    # ---------- 2. CLUSTERING ---------------------------------------------------
    def dbscan(self, df_pivot):
        pass
        from sklearn.cluster import DBSCAN
        #
        # 2. fit DBSCAN
        db = DBSCAN(eps=self.db_eps,
                    min_samples=self.db_min,
                    metric=self.db_metric).fit(df_pivot)

        labels = db.labels_  # -1 means noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # 3. safety: if everything is noise, fall back
        if n_clusters == 0:
            st.warning("DBSCAN found no clusters; falling back to k=2.")
            n_clusters = 2
            labels = np.full(df_pivot.shape[0], 1)  # dummy single cluster
            closest = [df_pivot.index[0]]  # arbitrary rep
        else:
            # 4. representative province for each cluster (highest core-sample score)
            core_mask = db.core_sample_indices_  # indices of core points
            core_labels = labels[core_mask]  # their cluster ids
            closest = []
            for cid in range(n_clusters):
                # pick first core point of that cluster
                pick = core_mask[core_labels == cid][0]
                closest.append(df_pivot.index[pick])

        # 5. store in dataframe (noise → cluster 0 so we can still colour it)
        df_pivot["clusters"] = labels + 1  # -1→0, 0→1, …, k→k+1
        return df_pivot, closest