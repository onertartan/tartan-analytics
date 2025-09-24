import json
from collections import Counter
from typing import Dict, List
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score, davies_bouldin_score
from kneed import KneeLocator
import streamlit as st
import extra_streamlit_components as stx
from PIL import Image
from sklearn import preprocessing
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
import geopandas as gpd
import pandas as pd
from matplotlib import colormaps, pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
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
            st.session_state["COLORS"]=["red", "purple", "orange", "green", "dodgerblue", "magenta", "gold", "darkorange", "darkolivegreen",
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
        st.markdown("""<style> .main > div {padding-left:1rem;padding-right:1rem;padding-top:4rem;}</style>""", unsafe_allow_html=True)
        st.session_state["geo_scale"] = self.top_row_cols[0].radio("Choose geographic scale",self.geo_scales).split()[0]
        print("ÇÇÇ:",st.session_state["geo_scale"])
        st.markdown("""<style> [role=radiogroup]{ gap: 0rem; } </style>""", unsafe_allow_html=True)
        self.fun_extras() # for optional columns at the top row
        cols_nom_denom = self.ui_basic_setup()
        # get cached data
        df_data = self.get_data()
        print("0. çekpoint",df_data["denominator"]["district"])
        geo_scale = "province" if st.session_state["geo_scale"]!="district" else "district"
        gdf_borders = self.gdf[geo_scale]
        start_year = df_data["nominator"][geo_scale].index.get_level_values(0).min()
        end_year = df_data["nominator"][geo_scale].index.get_level_values(0).max()
        self.sidebar_controls(start_year, end_year)
        st.write("""<style>[data-testid="stHorizontalBlock"]{align-items: top;}</style>""", unsafe_allow_html=True)

        with st.form("submit_form"):
            (col_show_results, col_animation) = st.columns(2)
            show_results = col_show_results.form_submit_button("Show results")
            if self.animation_available:
                play_animation = col_animation.form_submit_button("Play animation")
                col_animation.write("Animation Controls")
                col_animation.slider("Animation Speed (seconds)", min_value=0.5, max_value=5., value=1., step=1., key="animation_speed")
                col_animation.checkbox("Auto-play", value=True,key="auto_play")
                if play_animation:
                    st.session_state["animate"] = True
                else:
                    st.session_state["animate"] = False

            # Run on first load OR when form is submitted
            selected_features = self.get_selected_features(cols_nom_denom)
            if show_results or (self.animation_available and play_animation):
            #  st.session_state["animation_images_generated"] = False

              # delete_temp_files()
                col_plot, col_df = st.columns((4, 1), gap="small")
                print("df_data : LOL",df_data["denominator"]["district"])
                self.plot_main(col_plot, col_df, df_data, gdf_borders, selected_features, [st.session_state["geo_scale"]])



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
            st.session_state["year_1"], st.session_state["year_2"] = int(st.session_state.slider_year_2[0]), int(
                st.session_state.slider_year_2[1])


    def ui_basic_setup(self):

       # st.markdown(""" <style>[role=checkbox]{ gap: 1rem; }</style>""", unsafe_allow_html=True)

    #    st.markdown("""
    #            <div class="top-align">
     #               <style>
     #                  .top-align [data-testid="stHorizontalBlock"] {
      #                      align-items: flex-start;
       #                 }
      #              </style>
        #        </div>
       #     """, unsafe_allow_html=True)

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

        st.divider()
        cols_clustering = st.columns([2, 1, 1, 5])
        cols_clustering[0].checkbox("Apply K-means clustering", key="clustering_cb_" + self.page_name)
        cols_clustering[0].write("In K-means clustering primary parameters are not aggregated, instead they are used as individual features.")
        cols_clustering[1].selectbox("Select K: number of clusters", range(2, 15), key="n_clusters_" + self.page_name)

       # scaler
        options = ["MaxAbsScaler", "MinMaxScaler", "StandardScaler", "No scaling"]
        stored_value = st.session_state.get("scaler" + self.page_name, options[0])
        default_index = options.index(stored_value) if stored_value in options else 0
        st.session_state["scaler"]=cols_clustering[2].radio("Select scaling option",options=options, index=default_index)
        cols_clustering[3].checkbox("Elbow Method", key="elbow")
        return cols_nom_denom


    def animation_slider_changed(self):
        st.session_state["animate"]=True

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
                st.write("Single year is selected from the first slider.")
                st.write("Selected year:", st.session_state["year_1"])
            else:
                st.write("Start and end years are selected from the second slider.")
                st.write("Selected start year:", st.session_state["year_1"], "\nSelected end year:",
                         st.session_state["year_2"])


    def sidebar_controls(self, *args):  # start_year=2007,end_year=2023
        self.sidebar_controls_basic_setup(*args)
        self.sidebar_controls_plot_options_setup(*args)


    def sidebar_controls_plot_options_setup(self):
        pass

    def quick_selection(self, feature_name, nom_denom_key_suffix):
        pass

    def set_checkbox_values_for_quick_selection(self, keys_to_check, nom_denom_key_suffix, feature_name):
        print("###", keys_to_check)
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

    def scale(self,df):
        scaler_name = st.session_state.get("scaler", "MaxAbsScaler")
        if scaler_name != "No scaling":
            scaler_class = getattr(preprocessing, scaler_name)
            scaler = scaler_class()
            print("BEFORE SCALING",scaler,df)
            df_scaled = scaler.fit_transform(df)
            print("AFTER SCALING",df)

            df = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)
        return df


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



    # Main function to analyze optimal k with new metrics
    def optimal_k_analysis(self, df):

        if "consensus_labels_"+self.page_name not in st.session_state:
            st.session_state["consensus_labels_"+self.page_name] = None
        random_states = range(100)  # 50 random seeds
        k_values = list(range(2, 15))
        num_seeds_to_plot = 3
        # Convert df to numpy array if it's a DataFrame
        n_samples = df.shape[0]
        # Initialize lists to store metrics for all seeds
        inertias_all = []
        silhouette_scores_all = []
        calinski_scores_all = []
        gap_scores_all = []
        davies_bouldin_scores_all = []
        dunn_scores_all = []
        bic_scores_all = []
        labels_all = {seed: {} for seed in random_states}

        # Compute metrics for each seed
        for random_state in random_states:
            inertias = []
            silhouette_scores = []
            calinski_scores = []
            gap_scores = []
            davies_bouldin_scores = []
            dunn_scores = []
            bic_scores = []


            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=100).fit(df)
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
                calinski_scores.append(calinski_harabasz_score(df, labels))
                gap_scores.append(self.compute_gap_statistic(df, k, random_state))
                davies_bouldin_scores.append( davies_bouldin_score(df, labels))
                dunn_scores.append(self.dunn_index(df, labels))
                bic_scores.append(self.approximate_bic(df, k, inertia))
                labels_all[random_state][k] = labels

            inertias_all.append(inertias)
            silhouette_scores_all.append(silhouette_scores)
            calinski_scores_all.append(calinski_scores)
            gap_scores_all.append(gap_scores)
            davies_bouldin_scores_all.append(davies_bouldin_scores)
            dunn_scores_all.append(dunn_scores)
            bic_scores_all.append(bic_scores)

        # Compute mean metrics across seeds
        mean_inertias = np.mean(inertias_all, axis=0)
        mean_silhouette = np.mean(silhouette_scores_all, axis=0)
        mean_calinski = np.mean(calinski_scores_all, axis=0)
        mean_gap = np.mean(gap_scores_all, axis=0)
        mean_davies_bouldin = np.mean(davies_bouldin_scores_all, axis=0)
        mean_dunn = np.mean(dunn_scores_all, axis=0)
        mean_bic = np.mean(bic_scores_all, axis=0)


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

        # Create subplot grid: (seeds + mean + ARI) rows, 7 columns
        fig, axs = plt.subplots(num_seeds_to_plot + 2, 7, figsize=(40, 20))

        # Titles for each column
        column_titles = [
            'Elbow Analysis',
            'Silhouette Score',
            'Calinski-Harabasz Score',
            'Gap Statistic',
            'Davies-Bouldin Index',
            'Dunn Index',
            'Approximate BIC',
            'Consensus Index'
        ]
        df_optimal_k = pd.DataFrame(data=0,index=column_titles,columns=k_values)

        for i, random_state in enumerate(random_states):
            elbow = KneeLocator(k_values, inertias_all[i], curve='convex', direction='decreasing')
            elbow=elbow.elbow
            df_optimal_k.loc['Elbow Analysis',elbow] +=1
            optimal_k_sil = k_values[np.argmax(silhouette_scores_all[i])]
            df_optimal_k.loc['Silhouette Score', optimal_k_sil] += 1
            optimal_k_cal = k_values[np.argmax(calinski_scores_all[i])]
            df_optimal_k.loc['Calinski-Harabasz Score', optimal_k_cal] += 1
            optimal_k_db = k_values[np.argmin(davies_bouldin_scores_all[i])]
            df_optimal_k.loc['Davies-Bouldin Index', optimal_k_db] += 1
            optimal_k_dunn = k_values[np.argmax(dunn_scores_all[i])]
            df_optimal_k.loc['Dunn Index', optimal_k_dunn] += 1
            optimal_k_bic = k_values[np.argmin(bic_scores_all[i])]
            df_optimal_k.loc['Approximate BIC', optimal_k_bic] += 1


        # Plot metrics for each seed
        for i, random_state in enumerate(random_states[:num_seeds_to_plot]):
            # Inertia
            axs[i, 0].plot(k_values, inertias_all[i], 'bo-')
            elbow = KneeLocator(k_values, inertias_all[i], curve='convex', direction='decreasing')
            if elbow.elbow:
                axs[i, 0].axvline(x=elbow.elbow, color='r', linestyle='--')
            axs[i, 0].set_title(f'Seed {random_state}: {column_titles[0]}')

            # Silhouette Score
            axs[i, 1].plot(k_values, silhouette_scores_all[i], 'ro-')
            optimal_k_sil = k_values[np.argmax(silhouette_scores_all[i])]
            axs[i, 1].axvline(x=optimal_k_sil, color='r', linestyle='--')
            axs[i, 1].set_title(f'Seed {random_state}: {column_titles[1]}')

            # Calinski-Harabasz Score
            axs[i, 2].plot(k_values, calinski_scores_all[i], 'go-')
            optimal_k_cal = k_values[np.argmax(calinski_scores_all[i])]
            axs[i, 2].axvline(x=optimal_k_cal, color='r', linestyle='--')
            axs[i, 2].set_title(f'Seed {random_state}: {column_titles[2]}')

            # Gap Statistic
            axs[i, 3].plot(k_values, gap_scores_all[i], 'mo-')
            optimal_k_gap = k_values[np.argmax(gap_scores_all[i])]
            axs[i, 3].axvline(x=optimal_k_gap, color='r', linestyle='--')
            axs[i, 3].set_title(f'Seed {random_state}: {column_titles[3]}')

            # Davies-Bouldin Index
            axs[i, 4].plot(k_values, davies_bouldin_scores_all[i], 'co-')
            optimal_k_db = k_values[np.argmin(davies_bouldin_scores_all[i])]
            axs[i, 4].axvline(x=optimal_k_db, color='r', linestyle='--')
            axs[i, 4].set_title(f'Seed {random_state}: {column_titles[4]}')

            # Dunn Index
            axs[i, 5].plot(k_values, dunn_scores_all[i], 'yo-')
            optimal_k_dunn = k_values[np.argmax(dunn_scores_all[i])]
            axs[i, 5].axvline(x=optimal_k_dunn, color='r', linestyle='--')
            axs[i, 5].set_title(f'Seed {random_state}: {column_titles[5]}')

            # Approximate BIC
            axs[i, 6].plot(k_values, bic_scores_all[i], 'ko-')
            optimal_k_bic = k_values[np.argmin(bic_scores_all[i])]
            axs[i, 6].axvline(x=optimal_k_bic, color='r', linestyle='--')
            axs[i, 6].set_title(f'Seed {random_state}: {column_titles[6]}')


        # Plot mean metrics
        axs[num_seeds_to_plot, 0].plot(k_values, mean_inertias, 'bo-')
        mean_elbow = KneeLocator(k_values, mean_inertias, curve='convex', direction='decreasing')
        if mean_elbow.elbow:
            axs[num_seeds_to_plot, 0].axvline(x=mean_elbow.elbow, color='r', linestyle='--')
        axs[num_seeds_to_plot, 0].set_title(f'Mean: {column_titles[0]}')

        axs[num_seeds_to_plot, 1].plot(k_values, mean_silhouette, 'ro-')
        optimal_k_mean_sil = k_values[np.argmax(mean_silhouette)]
        axs[num_seeds_to_plot, 1].axvline(x=optimal_k_mean_sil, color='r', linestyle='--')
        axs[num_seeds_to_plot, 1].set_title(f'Mean: {column_titles[1]}')

        axs[num_seeds_to_plot, 2].plot(k_values, mean_calinski, 'go-')
        optimal_k_mean_cal = k_values[np.argmax(mean_calinski)]
        axs[num_seeds_to_plot, 2].axvline(x=optimal_k_mean_cal, color='r', linestyle='--')
        axs[num_seeds_to_plot, 2].set_title(f'Mean: {column_titles[2]}')

        axs[num_seeds_to_plot, 3].plot(k_values, mean_gap, 'mo-')
        optimal_k_mean_gap = k_values[np.argmax(mean_gap)]
        axs[num_seeds_to_plot, 3].axvline(x=optimal_k_mean_gap, color='r', linestyle='--')
        axs[num_seeds_to_plot, 3].set_title(f'Mean: {column_titles[3]}')

        axs[num_seeds_to_plot, 4].plot(k_values, mean_davies_bouldin, 'co-')
        optimal_k_mean_db = k_values[np.argmin(mean_davies_bouldin)]
        axs[num_seeds_to_plot, 4].axvline(x=optimal_k_mean_db, color='r', linestyle='--')
        axs[num_seeds_to_plot, 4].set_title(f'Mean: {column_titles[4]}')

        axs[num_seeds_to_plot, 5].plot(k_values, mean_dunn, 'yo-')
        optimal_k_mean_dunn = k_values[np.argmax(mean_dunn)]
        axs[num_seeds_to_plot, 5].axvline(x=optimal_k_mean_dunn, color='r', linestyle='--')
        axs[num_seeds_to_plot, 5].set_title(f'Mean: {column_titles[5]}')

        axs[num_seeds_to_plot, 6].plot(k_values, mean_bic, 'ko-')
        optimal_k_mean_bic = k_values[np.argmin(mean_bic)]
        axs[num_seeds_to_plot, 6].axvline(x=optimal_k_mean_bic, color='r', linestyle='--')
        axs[num_seeds_to_plot, 6].set_title(f'Mean: {column_titles[6]}')


        # Plot ARI metrics
        axs[num_seeds_to_plot + 1, 0].plot(k_values, ari_mean, 'bo-')
        axs[num_seeds_to_plot + 1, 0].set_title('Mean ARI vs Clusters')
        # Annotate each point with its value
        for k, val in zip(k_values, ari_mean):
            axs[num_seeds_to_plot + 1, 0].text(k, val, f'{val:.2f}',
                         ha='center', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        axs[num_seeds_to_plot + 1, 1].scatter(k_values, ari_std, color='g')
        axs[num_seeds_to_plot + 1, 1].set_title('ARI Std vs Clusters')

        # Plot Consensus Index
        axs[num_seeds_to_plot+1, 2].plot(k_values, consensus_indices, 'purple', marker='o', linestyle='-')
        optimal_k_consensus = k_values[np.argmax(consensus_indices)]
        axs[num_seeds_to_plot+1, 2].axvline(x=optimal_k_consensus, color='r', linestyle='--')
        axs[num_seeds_to_plot+1, 2].set_title(f'Mean: {column_titles[7]}')

        # Hide unused subplots in ARI row
        for j in range(3, 7):
            axs[num_seeds_to_plot + 1, j].axis('off')

        # Set labels and layout
        for ax in axs.flat:
            ax.set_xlabel('Number of Clusters (k)')
            ax.grid(True)
        fig.tight_layout()
        fig.savefig('kmeans_metrics_analysis.png')
        st.write(df_optimal_k)
        st.session_state["consensus_labels_"+self.page_name] = consensus_labels_all
        return fig


