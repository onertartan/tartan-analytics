from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from modules.base_page import BasePage
import pandas as pd
import geopandas as gpd
import streamlit as st
from matplotlib import cm
import numpy as np
import locale

from utils.plot_map_common import figure_setup

locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')

class PageNames(BasePage):
    @classmethod
    def k_means_clustering(cls, col_plot, df_data, gdf_borders):
        gdf_borders.set_index("province",inplace=True)
        page_name = cls.page_name
        if page_name == "names_surnames" and st.session_state["name_surname_rb"] == "Surname":
            df_year = df_data["surname"].loc[st.session_state["year_1"]]
        elif st.session_state["select_both_sexes_" + page_name]:
            df_year_male, df_year_female = df_data["male"].loc[st.session_state["year_1"]], df_data["female"].loc[st.session_state["year_1"]]
            overlapping_names = set(df_year_male["name"]) & set(df_year_female["name"])
            df_year_male['name'] = df_year_male.apply( lambda x: f"{x['name']}_female" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year_female['name'] = df_year_female.apply(
                lambda x: f"{x['name']}_male" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year = pd.concat([df_year_male, df_year_female])
        else:
            sex = st.session_state["sex_" + page_name].lower()
            df_year = df_data[sex].loc[st.session_state["year_1"]]
        scaler = MaxAbsScaler()
        data_scaled = scaler.fit_transform( df_year.loc["Adana", ["count"]] )


        # Get unique provinces
        provinces = df_year.index.get_level_values(0).unique()
        # Scale counts for each province
        for province in provinces:
            # Get counts for current province
            province_count_sum_first_30 = df_year.loc[province, 'count'].sum()

            province_counts = df_year.loc[province, 'count'].values.reshape(-1, 1)
            scaled_counts = scaler.fit_transform(province_counts)            # Fit and transform the counts
            df_year.loc[province, 'scaled_count_sklearn'] = scaled_counts.flatten()             # Update the DataFrame with scaled values


            # Fit and transform the counts
            # Update the DataFrame with scaled values
            df_year.loc[province, 'scaled_count_top_30'] =df_year.loc[province, "count"] /province_count_sum_first_30

        #print("SCAL:",data_scaled)
        df_year.loc[:, "ratio"] = df_year.loc[:, "count"] / df_year.loc[:, "total_count"]
        print("XCV:",df_year.loc["Adana"])

        df_pivot = pd.pivot_table(df_year, values='ratio', index=df_year.index, columns=['name'],  aggfunc=lambda x: x, dropna=False, fill_value=0)
        n_clusters = st.session_state["n_clusters_" + page_name]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42,  init='k-means++',n_init=10).fit(df_pivot)
        # Get cluster centroids (shape: n_clusters × n_features)
        centroids = kmeans.cluster_centers_
        # After fitting KMeans
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, df_pivot)
        closest_cities = df_pivot.index[closest_indices].tolist()  # Get city/province names
        # Get the subset of provinces that are closest to centroids
        closest_provinces = gdf_borders[gdf_borders.index.isin(closest_cities)]
        # Compute centroids of their geometries (for marker placement)
        closest_provinces["centroid"] = closest_provinces.geometry.centroid


        print("şşl",gdf_borders.index        )
        df_pivot["clusters"] = kmeans.labels_
        gdf_borders = gdf_borders.merge(df_pivot["clusters"],left_index=True,right_index=True)
        print("ççöö",gdf_borders)

        # Define a color map for the categories
        # ORIGINAL
        color_map = {0: "red", 1: "purple", 2: "orange", 3: "green", 4: "blue", 5: "magenta", 6: "cyan", 7: "yellow",
                   8: "gray",9:"dark blue",10:"white",11:"black"}  # original
        clusters = list(range(n_clusters))
        color_map = {gdf_borders.loc["İzmir","clusters"]: "red", gdf_borders.loc["Van","clusters"]: "purple"}
        if n_clusters > 2 and gdf_borders.loc["Konya", "clusters"] not in color_map.keys():
            color_map[ gdf_borders.loc["Konya","clusters"] ] = "orange"
        if n_clusters==4 and gdf_borders.loc["Samsun", "clusters"]  not in  color_map.keys():
            color_map[ gdf_borders.loc["Samsun", "clusters"] ] = "green"

        remaining_clusters = set(clusters- color_map.keys())
        remaining_colors = ["blue", "magenta", "cyan","yellow", "gray", "dark blue", "white",  "black"]  # original
        for i,remaining_cluster in enumerate(remaining_clusters):
            color_map[remaining_cluster] =  remaining_colors[i]


        # Map the colors to the GeoDataFrame
        gdf_borders["color"] = gdf_borders["clusters"].map(color_map)
        fig, axs = figure_setup()
        gdf_borders.plot(ax=axs[0, 0], color=gdf_borders['color'], legend=True, edgecolor="black", linewidth=.2)
        axs[0, 0].axis("off")
        axs[0, 0].margins(x=0)

        # Compute centroids of the closest provinces and plot them as markers
        closest_provinces_centroids = closest_provinces.copy()
        closest_provinces_centroids["centroid_geometry"] = closest_provinces_centroids.geometry.centroid

        # Create a temporary GeoDataFrame with centroid geometries (points)
        closest_provinces_points = gpd.GeoDataFrame(
            closest_provinces_centroids,
            geometry="centroid_geometry",
            crs=gdf_borders.crs
        )

        # Add markers using the centroid points (no fill color change)
        closest_provinces_points.plot(
            ax=axs[0, 0],

            facecolor="none",  # Transparent fill
            markersize=200,
            edgecolor="black",  # Marker edge color
            linewidth=1.5,
            label="Cluster Centers"
        )
        # Add province names (from index) at centroids
        for province in gdf_borders.index:
            axs[0, 0].annotate(
                text=province,  # Use index (province name) directly
                xy=(gdf_borders.loc[province,"geometry"].centroid.x, gdf_borders.loc[province,"geometry"].centroid.y),
                ha="center",
                va="center",
                fontsize=5,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.6
                )
            )
        # Optional: Add legend
        axs[0, 0].legend(loc="upper right")



        col_plot.pyplot(fig)

    @classmethod
    def fun_extras(cls, *args):
        pass

    @classmethod
    def render(cls):
        page_name = cls.page_name
        cls.set_session_state_variables()
        df_data, gdf_borders = cls.get_data()
        start_year, end_year = df_data["male"].index.get_level_values(0).min(), df_data["male"].index.get_level_values(0).max()
        BasePage.sidebar_controls_basic_setup(start_year, end_year )
        # basic_sidebar_controls(2018, 2023)
        col_1, col_2, col_3 = st.columns([1, 1, 3])

        if not st.session_state["clustering_cb_" + page_name]:
            st.session_state["select_both_sexes_" + page_name] = False

        if page_name == "names_surnames":
             name_surname = col_1.radio("Select name or surname", ["Name", "Surname"], key="name_surname_rb").lower()
             if name_surname == "surname":
                 st.session_state["select_both_sexes_" + page_name] = True
             sex = col_2.radio("Choose sex", ["Male", "Female"],  disabled=st.session_state["select_both_sexes_" + page_name], key="sex_" + page_name).lower()
        else:
             sex = col_1.radio("Choose sex", ["Male", "Female"],  disabled=st.session_state["select_both_sexes_" + page_name],key="sex_" + page_name).lower()


        st.header(f"Clustering or Displaying {page_name}")
        col_1, col_2, col_3 = st.columns([1, 1, 3])

        clustering_cb = col_1.checkbox("Apply K-means clustering", key="clustering_cb_" + page_name)
        col_1.checkbox("Select both sexes", disabled=not st.session_state["clustering_cb_" + page_name], key="select_both_sexes_" + page_name)
        col_1.selectbox("Select K: number of clusters", range(2, 11), key="n_clusters_" + page_name)
        disable_when_clustering = True if clustering_cb else False
        expression_in_sentence =  "names or surnames" if page_name == "names_surnames" else "baby names"
        choice = col_3.radio("Or select an displaying option", [f"Show the most common {expression_in_sentence}", f"Select {expression_in_sentence} and top-n number to filter"],
                             horizontal=True, disabled=disable_when_clustering, key="secondary_option_" + page_name)

        disable_top_n_selection = not disable_when_clustering and choice == f"Show the most common {page_name}"

        col_3.selectbox("Select a number N to check if the entered name is among the top N names", options=list(range(1, 31)), disabled=disable_top_n_selection,
                        key="top_n_selection_" + page_name)

        sex_or_surname = "surname" if page_name == "names_surnames" and st.session_state["name_surname_rb"] == "Surname" else sex
        col_3.multiselect("Select name(s). By default most popular name is shown.",
                          sorted(df_data[sex_or_surname]["name"].unique(), key=locale.strxfrm),
                          disabled=disable_when_clustering, key="names_" + page_name)
        col_plot, colf_df = st.columns([5, 1])

        if st.session_state["clustering_cb_" + page_name]:
            cls.k_means_clustering(col_plot, df_data, gdf_borders)
        else:
            PageNames.plot_geopandas(col_plot, df_data, gdf_borders)

    @staticmethod
    def plot_result(df_result, ax, names):
        # Create a color map
        cmap = cm.get_cmap('tab20', len(names))
        colors = {name: cmap(i) for i, name in enumerate(names)}
        colors[np.nan] = "gray"

        # Assign colors to each row in the GeoDataFrame
        df_result['color'] = df_result['name'].map(colors).fillna("gray")
        # After groupby df_result becomdes Pandas dataframe, we have to convert it to GeoPandas dataframe
        df_result = gpd.GeoDataFrame(df_result, geometry='geometry')
        # Plotting
        df_result.plot(ax=ax, color=df_result['color'], legend=True, legend_kwds={"shrink": .6}, edgecolor="black",
                       linewidth=.2)

        df_result.apply(lambda x: ax.annotate(
            text=x["province"].upper() + "\n" + x['name'].title() if isinstance(x['name'], str) else x["province"],
            size=4, xy=x.geometry.centroid.coords[0], ha='center', va="center"), axis=1)
        ax.axis("off")
        ax.margins(x=0)
        # Add a legend
        for name, color in set(zip(df_result['name'], df_result['color'])):
            ax.plot([], [], color=color, label=name, linestyle='None', marker='o')

    #  ax.legend(title='Names', fontsize=4, bbox_to_anchor=(0.01, 0.01), loc='lower right', fancybox=True, shadow=True)

    @staticmethod
    def plot_geopandas(col_plot, df_data, gdf_borders):
        page_name = st.session_state["page_name"]
        names_or_surnames = "names"
        if page_name == "names_surnames" and st.session_state["name_surname_rb"] == "Surname":
            names_or_surnames = "surnames"
            df = df_data["surname"]
            title_prefix = "The most common surnames "
        else:
            sex = st.session_state["sex_" + page_name].lower()
            df = df_data[sex]
            title_prefix = "The most common "+sex+" names "

        display_option = st.session_state["secondary_option_" + page_name]
        names_from_multi_select = st.session_state["names_" + page_name]
        top_n = int(st.session_state["top_n_selection_" + page_name])

        fig, axs = figure_setup()
        df_results = []
        year_1, year_2 = st.session_state["year_1"], st.session_state["year_2"]
        most_popular_names_5_year = sorted(df[df["rank"] == 1]["name"].unique())

        for i, year in enumerate(sorted({year_1, year_2})):
            if "most common" in display_option:
                df_year = df.loc[year].reset_index()
                df_result = df_year[df_year["rank"] == 1]
                df_result = gdf_borders.merge(df_result, left_on="province", right_on="province")
                df_result = df_result.groupby(["geometry", "province"])["name"].apply(
                    lambda x: "%s" % '\n '.join(x)).to_frame().reset_index()
                df_results.append(df_result)
                PageNames.plot_result(df_results[i], axs[i, 0], names=most_popular_names_5_year)
                axs[i, 0].set_title(title_prefix+f'in {year}')
            else:  # top_n option: Select single year, name(s) and top-n number to filter
                df_year = df.loc[year].reset_index()
                for name in names_from_multi_select:
                    df_result = df_year[(df_year["name"] == name) & (df_year["rank"] <= top_n)]
                    if df_result.empty:
                        st.write(f"{name} is not in the top {top_n} for the year {year_1}")
                if names_from_multi_select:
                    df_result = df_year[(df_year["name"].isin(names_from_multi_select)) & (df_year["rank"] <= top_n)]
                    df_result_not_null = gdf_borders.merge(df_result, left_on="province", right_on="province")

                    df_result_not_null = df_result_not_null.groupby(["geometry", "province"])["name"].apply(
                        lambda x: "%s" % '\n '.join(x)).to_frame().reset_index()
                    df_results.append(df_result_not_null)
                    df_result_with_nulls = gdf_borders.merge(df_result_not_null[["province", "name"]],
                                                             left_on="province", right_on="province", how="left")
                    PageNames.plot_result(df_result_with_nulls, axs[i, 0], names=sorted(df_result_not_null['name'].unique()))

                    axs[i, 0].set_title(f"Provinces where  selected {names_or_surnames} in the top {top_n} for {year}")

            if not df_results:
                st.write("No results found.")
        if df_results:
            col_plot.pyplot(fig)
    @classmethod
    def set_session_state_variables(cls):
        page_name = cls.page_name
        if "clustering_cb_" + page_name not in st.session_state:
            st.session_state["clustering_cb_" + page_name] = False
        if "select_both_sexes_" + page_name not in st.session_state:
            st.session_state["select_both_sexes_" + page_name] = False
        if "top_n_checkbox_" + page_name not in st.session_state:
            st.session_state["top_n_checkbox_" + page_name] = False
        if "top_n_selection_" + page_name not in st.session_state:
            st.session_state["top_n_selection_" + page_name] = 2
        if "name_surname_rb" not in st.session_state:
            st.session_state["name_surname_rb"] = "Name"