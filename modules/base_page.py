# Standard Packages
from typing import Dict, List
from abc import ABC, abstractmethod
# Data Processing and Analysis
import pandas as pd
import geopandas as gpd
# Visualization
import matplotlib.pyplot as plt
import streamlit
from adjustText import adjust_text
from sklearn.decomposition import PCA

from clustering.models.factory import get_engine_class
from clustering.models.hierarchical import HierarchicalClusteringEngine
from clustering.models.kmeans import KMeansEngine
from clustering.models.spectral import SpectralClusteringEngine
from viz import PCAPlotter, OptimalKPlotter
# Machine Learning
from clustering.base_clustering import Clustering
# Streamlit & Tools
from viz.gui_helpers.clustering_helpers import *
from viz.plotters.geo_cluster_plotter import GeoClusterPlotter
from viz.config import COLORS, CLUSTER_COLOR_MAPPING, VA_POSITIONS, HA_POSITIONS
from clustering.models.gmm import GMMEngine
from clustering.models.kmedoids import KMedoidsEngine

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

    @staticmethod
    @st.cache_data  # This is the best way!
    def load_geo_data():
        gdf = {}
        gdf["district"] = gpd.read_file("data/preprocessed/gdf_borders_district.geojson")
        gdf["province"] = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
        return gdf

    @property
    def gdf(self):
        # Just call the cached function directly.
        # Streamlit handles the speed and memory optimization for you.
        return BasePage.load_geo_data()
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

    @property
    def gdf_clusters(self):
        return st.session_state.get("gdf_clusters")

    @gdf_clusters.setter
    def gdf_clusters(self, value):
        st.session_state["gdf_clusters"] = value

    def load_css(self, file_path: str) -> None:
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

    @staticmethod
    def convert_year_index_data_type(df):
        """Not all data years are integers.In particular elections data of 2015 June and 2015 November are not string"""
        return pd.MultiIndex.from_arrays([df.index.get_level_values(0).astype(str),  df.index.get_level_values(1)], names=df.index.names)

    def fun_extras(self, *args):
        pass

    @staticmethod
    @st.cache_data
    def load_geo_data():
        gdf = {"district": gpd.read_file("data/preprocessed/gdf_borders_district.geojson"),
               "province": gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")}
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
        st.session_state["selected_slider"] = slider_index
        if slider_index == 1:
            st.session_state["year_1"] = st.session_state["year_2"] = int(st.session_state.slider_year_1)
        else:
            st.session_state["year_1"], st.session_state["year_2"] = int(st.session_state.slider_year_2[0]), int(st.session_state.slider_year_2[1])


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
             # if ifadesine gerek olmadığı düşünülerek (hata olursa bu if kalktığı için olabilir) metot classmethod'dan static'e dönüştü. Böylelikle higher-education kullanabildi.
            # options= list(range(start_year, end_year + 1)) if cls.page_name != "sex_age_edu_elections" else [2018,2023]
            options = list(range(start_year, end_year + 1))
            # Create a slider to select a single year
            st.select_slider("Select a year", options, 2023, on_change=BasePage.update_selected_slider_and_years, args=[1],  key="slider_year_1")
            # Create sliders to select start and end years
            st.select_slider("Or select start and end years",options, [options[0],options[-1]],on_change=BasePage.update_selected_slider_and_years, args=[2], key="slider_year_2")

            if "selected_slider" not in st.session_state:
                st.session_state["selected_slider"] = 1
            BasePage.update_selected_slider_and_years(st.session_state["selected_slider"])

            if st.session_state["selected_slider"] == 1:
                st.write("You have selected a single year from the first slider.")
                st.write("Selected year:", st.session_state["year_1"])
            else:
                st.write("You have selected start and end years from the second slider.")
                st.write("Selected start year:", st.session_state["year_1"], "\nSelected end year:", st.session_state["year_2"])

            # Main content
            if "animation_images_generated" not in st.session_state:
                st.session_state["animation_images_generated"] = False

    def sidebar_controls(self, *args):  # start_year=2007,end_year=2023
        self.sidebar_controls_basic_setup(*args)
        self.sidebar_controls_plot_options_setup(*args)

    def sidebar_controls_plot_options_setup(self,*args):
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
            #cls.checkb ox_group[feature_name].checked_dict[nom_denom_key_suffix][key] = val
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
        pass    # Overriden by sub-classes Base_Page_Names & Base_Page_Common

    def tab_clustering(self, df, *args):
        # 0. Render UI
        clustering_algorithm = gui_clustering_main()

        if not clustering_algorithm:
            return
        engine_class = get_engine_class(clustering_algorithm)
        n_cluster = st.session_state["n_cluster"] = st.session_state.get("n_cluster_" + clustering_algorithm, -1)
        def prepare_kwargs():
            kwargs = {}
            if engine_class is GMMEngine or engine_class is KMeansEngine:
                    st.session_state["n_init"] = st.session_state["n_init_" + clustering_algorithm]
                    kwargs["n_init"]= st.session_state["n_init"]
            if engine_class is GMMEngine:
                kwargs["covariance_type"] = st.session_state["gmm_covariance_type"]
            elif engine_class is KMedoidsEngine:
                kwargs["metric"] = st.session_state["distance_metric_pam"]
                kwargs["max_iter"] = st.session_state["max_iter_kmedoids"]
                kwargs["method"]="pam"
            elif engine_class is SpectralClusteringEngine:
                kwargs["affinity"] = st.session_state["affinity_spectral"]
                kwargs["n_neighbors"] = st.session_state["n_neighbors_spectral"]
                kwargs["assign_labels"] = "kmeans"
                kwargs["spectral_geometry"] = st.session_state["spectral_geometry"]
            elif engine_class is HierarchicalClusteringEngine:
                kwargs["metric"] = st.session_state["distance_metric_hierarchical"]
                kwargs["linkage_method"] = st.session_state["linkage_hierarchical"]
            return  kwargs
        kwargs = prepare_kwargs()
        # 1. Run clustering: Preprocess
        df_pivot = self.preprocess_clustering(df, *args)
     #   pca = PCA(n_components=50)
      #  temp= pca.fit_transform(df_pivot.iloc[:, :-1])
       # df_pivot = pd.DataFrame(temp,index=df_pivot.index )

        # If optimal_k_analysis is selected or use_consensus_labels is checked but it is not present(optimal_k_analysis has not previously run)
        if st.session_state.get("optimal_k_analysis", False) or (st.session_state.get("use_consensus_labels_" + self.page_name, False) and "consensus_labels_" + self.page_name not in st.session_state):
            k_values = list(range(2, 15)) if not (engine_class is  HierarchicalClusteringEngine) else range(n_cluster, n_cluster + 1)
            random_states = range(st.session_state["number_of_seeds"]) if engine_class.__name__ != "HierarchicalClusteringEngine" else range(1)
            num_seeds_to_plot = 3  if engine_class.__name__ != "HierarchicalClusteringEngine" else 1
            try_all_neighbors=True
            if engine_class is SpectralClusteringEngine and try_all_neighbors:
                scaler, spectral_geometry, year1, year2 = st.session_state['scaler'], st.session_state['spectral_geometry'], st.session_state["year_1"], st.session_state["year_2"]
                st.write(scaler, spectral_geometry, year1, year2 )
                for n in range(5, 21):
                    kwargs["n_neighbors"] = n
                    st.write(f"Running optimal k analysis for Spectral Clustering with n_neighbors={n} and scaler={scaler}")
                    df_summary, metrics_all, metrics_mean, ari_mean, ari_std, consensus_labels_all = engine_class.optimal_k_analysis(df_pivot, random_states, k_values, kwargs)
                    st.write(f"Completed optimal k analysis for n_neighbors={n}")
                    df_summary.to_csv(f"results/files/{engine_class.__name__}/{scaler}_{spectral_geometry}_{year1}_{year2}_{n}.csv")
                    pd.DataFrame(consensus_labels_all).to_csv(f"results/files/{engine_class.__name__}/consensus_labels_all_{scaler}_{spectral_geometry}_{year1}_{year2}_{n}.csv")

                return
            else:
                df_summary, metrics_all, metrics_mean, ari_mean, ari_std, consensus_labels_all = engine_class.optimal_k_analysis(df_pivot, random_states, k_values, kwargs)
                st.session_state["consensus_labels_"+engine_class.__name__] = consensus_labels_all
                df_pivot["clusters"] = consensus_labels_all[n_cluster]
                OptimalKPlotter.plot_optimal_k_analysis(engine_class, num_seeds_to_plot, k_values, random_states, metrics_all, metrics_mean, ari_mean, ari_std, kwargs)
                col1, col2 = st.columns(2)
                col1.write("Formatted results")
                col1.dataframe(OptimalKPlotter.style_metrics_dataframe(df_summary))
                col2.write("Raw results")
                col2.dataframe(df_summary)
                scaler, gmm_cov, year1, year2 = st.session_state['scaler'], st.session_state['gmm_covariance_type'], st.session_state["year_1"], st.session_state["year_2"]
                if engine_class is KMedoidsEngine or engine_class is KMeansEngine:
                    df_summary.to_csv(f"results/files/{engine_class.__name__}/{scaler}_{year1}_{year2}.csv")
                    pd.DataFrame(consensus_labels_all).to_csv(f"results/files/{engine_class.__name__}/{scaler}_{year1}_{year2}_consensus_labels_all.csv")
                elif engine_class is GMMEngine:
                    df_summary.to_csv(f"results/files/{engine_class.__name__}/{scaler}_{gmm_cov}_{year1}_{year2}.csv")
                    pd.DataFrame(consensus_labels_all).to_csv(f"results/files/{engine_class.__name__}/{scaler}_{gmm_cov}_{year1}_{year2}_consensus_labels_all.csv")

        elif st.session_state.get("use_consensus_labels_"+engine_class.__name__, False):
            df_pivot["clusters"] = st.session_state["consensus_labels_" + engine_class.__name__][n_cluster]
            st.header("Using previously saved consensus labels")
        else:
            silhouette_analysis=True
            if silhouette_analysis:
                engine_class.silhouette_analysis(df_pivot, kwargs=kwargs)
                return
            kwargs["n_cluster"] = n_cluster
            engine = get_engine_class(clustering_algorithm)(**kwargs)
            labels = engine.fit_predict(df_pivot)
            df_pivot["clusters"] = labels
           # st.dataframe(engine.probabilities(df_pivot.drop(columns=["clusters"])))
            #st.dataframe(df_pivot)
        # Step: Update geodata
        representatives = Clustering.get_representatives(df_pivot)
        if st.session_state.get("selected_tab_" + self.page_name, "") == "tab_geo_clustering":
            # self.update_geo_cluster_centers(df_pivot, representatives)
            self.gdf_clusters, self.gdf_centroids = Clustering.update_geo_cluster_centers(self.gdf, st.session_state["geo_scale"], df_pivot, representatives)

        col_plot, col_df = st.columns([5, 1])

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
            dense_threshold = total_points / (10 * factor) if st.session_state.get("selected_tab_" + self.page_name,"no_tab") != "tab_map" else 100  # Define thresholds
            mid_threshold = total_points / (20 * factor) if st.session_state.get("selected_tab_" + self.page_name,"no_tab") != "tab_map" else 100  #
            PCAPlotter().plot_pca(df_features, df_clusters,dense_threshold, mid_threshold, COLORS)

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
        with col_plot:
            GeoClusterPlotter(CLUSTER_COLOR_MAPPING, HA_POSITIONS, VA_POSITIONS).plot(self.gdf_clusters, self.gdf_centroids, st.session_state["n_cluster"], year_label)
            #GeoClusterPlotter(self.CLUSTER_COLOR_MAPPING, self.HA_POSITIONS, self.VA_POSITIONS).plot_elections(self.gdf_clusters)
        col_df.dataframe(df_clusters)

    def run_clustering(self, df_pivot, clustering_algorithm):
        """
        Delegates clustering execution to the unified Clustering factory.
        """
        # 1. Gather parameters from session state
        n_init = st.session_state["n_init_" + self.page_name]
        n_clusters = st.session_state["n_clusters_" + self.page_name]
        # 2. Prepare kwargs for the engine
        # Only pass n_clusters/n_init for algorithms that support it (KMeans, GMM)
        # DBSCAN parameters are typically handled internally by the DBSCANEngine or session state
        kwargs = {}
        if clustering_algorithm in ["kmeans", "gmm"]:
            kwargs = {"n_clusters": n_clusters, "n_init": n_init}
        # 3. Call the Factory Method
        # The Clustering class handles engine selection and .fit() execution
        # fit() returns (df_pivot, closest_indices)
        engine = Clustering.get_engine_class(clustering_algorithm)(**kwargs)
        labels = engine.fit_predict(df_pivot)
        df_out=df_pivot.copy()
        df_out["clusters"] = labels["clusters"]
        return df_out