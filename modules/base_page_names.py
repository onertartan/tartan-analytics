import itertools
from typing import Dict
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
    COLORS = ["red", "purple", "orange", "green", "blue", "magenta",
              "cyan", "yellow", "gray", "darkblue", "white", "black"]
    PROVINCE_COLOR_MAPPING = {"İzmir": "red", "Van": "purple", "Konya": "orange", "Samsun": "green"}
    # Province annotation positioning settings
    VA_POSITIONS = {
        "Kocaeli": "bottom", "Zonguldak": "bottom", "Sakarya": "top",
        "Bolu": "top", "Giresun": "bottom", "Gümüşhane": "bottom",
        "Trabzon": "bottom", "Gaziantep": "bottom", "Bartın": "bottom"
    }
    HA_POSITIONS = {"Zonguldak": "right", "Adana": "right"}
    @classmethod
    def _create_color_mapping(cls, gdf: gpd.GeoDataFrame, n_clusters: int) -> Dict[int, str]:
        """Generate cluster color mapping with province-based defaults."""
        color_map = {}
        clusters = set(range(n_clusters))
        # Assign predefined province colors
        for province, color in cls.PROVINCE_COLOR_MAPPING.items():
            if province in gdf.index:
                cluster = gdf.loc[province, "clusters"]
                if cluster not in color_map:
                    color_map[cluster] = color
        # Assign remaining colors
        remaining_clusters = clusters - set(color_map.keys())
        for i, cluster in enumerate(remaining_clusters):
            color_map[cluster] = cls.COLORS[i % len(cls.COLORS)]
        return color_map

    @classmethod
    def k_means_clustering(cls, df, gdf_borders, year):
        gdf_borders = gdf_borders.set_index("province")
        page_name = cls.page_name
        top_n = st.session_state["n_" + page_name]
        if page_name == "names_surnames" and st.session_state["name_surname_rb"] == "Surname":
            df_year = df["surname"].loc[year]
            df_year = df_year[df_year["rank"] <= top_n]  # use top-n for clustering (n=30 for all)
        elif len(st.session_state["sex_" + page_name]) != 1:  # if both sexes are selected
            df_year_male, df_year_female = df[df["sex"] == "male"].loc[year], df[df["sex"] == "female"].loc[year]
            df_year_male = df_year_male[df_year_male["rank"] <= top_n]
            df_year_female = df_year_female[df_year_female["rank"] <= top_n]
            overlapping_names = set(df_year_male["name"]) & set(df_year_female["name"])
            df_year_male['name'] = df_year_male.apply(lambda x: f"{x['name']}_female" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year_female['name'] = df_year_female.apply(lambda x: f"{x['name']}_male" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year = pd.concat([df_year_male, df_year_female])
        else:  # single gender selected for names
            sex = st.session_state["sex_" + page_name]
            df_year = df[df["sex"].isin(sex)].loc[year]
        scaler = MaxAbsScaler()
        data_scaled = scaler.fit_transform(df_year.loc["Adana", ["count"]])

        # Get unique provinces
        provinces = df_year.index.get_level_values(0).unique()
        # Scale counts for each province
        for province in provinces:
            # Get counts for current province
            province_count_sum_first_30 = df_year.loc[province, 'count'].sum()
            province_counts = df_year.loc[province, 'count'].values.reshape(-1, 1)
            scaled_counts = scaler.fit_transform(province_counts)                    # Fit and transform the counts
            df_year.loc[province, 'scaled_count_sklearn'] = scaled_counts.flatten()  # Update the DataFrame with scaled values
            # Fit and transform the counts
            # Update the DataFrame with scaled values
            df_year.loc[province, 'scaled_count_top_30'] = df_year.loc[province, "count"] / province_count_sum_first_30
        print("SCAL:",data_scaled)
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
        df_pivot["clusters"] = kmeans.labels_
        gdf_borders = gdf_borders.merge(df_pivot["clusters"], left_index=True, right_index=True)
        return gdf_borders,closest_provinces

    @classmethod
    def plot_clusters(cls,gdf_borders,closest_provinces,ax):
        n_clusters = st.session_state["n_clusters_" + cls.page_name]
        # Define a color map for the categories
        clusters = list(range(n_clusters))
        color_map = cls._create_color_mapping(gdf_borders, n_clusters)

        # Map the colors to the GeoDataFrame
        gdf_borders["color"] = gdf_borders["clusters"].map(color_map)
        gdf_borders.plot(ax=ax, color=gdf_borders['color'], legend=True, edgecolor="black", linewidth=.2)
        ax.axis("off")
        ax.margins(x=0)
        # Compute centroids of the closest provinces and plot them as markers
        closest_provinces_centroids = closest_provinces.copy()
        closest_provinces_centroids["centroid_geometry"] = closest_provinces_centroids.geometry.centroid
        # Create a temporary GeoDataFrame with centroid geometries (points)
        closest_provinces_points = gpd.GeoDataFrame( closest_provinces_centroids, geometry="centroid_geometry", crs=gdf_borders.crs)
        # Add markers using the centroid points (no fill color change   # Transparent fill)
        closest_provinces_points.plot(ax=ax,  facecolor="none", markersize=200, edgecolor="black", linewidth=1.5, label="Cluster Centers")
        # Add province names (from index) at centroids
        bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6)
        for province in gdf_borders.index:
            ax.annotate(text=province,  # Use index (province name) directly
                xy=(gdf_borders.loc[province, "geometry"].centroid.x, gdf_borders.loc[province,"geometry"].centroid.y),
                ha="center",  va="center", fontsize=5,  color="black", bbox=bbox)
        # Optional: Add legend
        ax.legend(loc="upper right")

    @classmethod
    def fun_extras(cls, *args):
        pass

    @staticmethod
    def get_ordinal(n):
        if 11 <= (n % 100) <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    @staticmethod
    def get_df_year_rank(df, year, target_rank, n_to_top_inclusive):
        df_year = df.loc[year]
        if n_to_top_inclusive:  # select the names above the target rank(inclusive)
            df_year_rank = df_year[df_year["rank"] <= target_rank]
        else:  # select the names with target rank
            df_year_rank = df_year[df_year["rank"] == target_rank]
        # if surname is selected or only single gender selected, we do not need combinatiobs
        if "sex" not in df_year_rank.columns or len(df_year_rank["sex"].unique()) == 1:  # if st.session_state["name_surname_rb"]=="surname":
            return df_year_rank
        # generate combinations if two genders are selected and surname is not selected
        results = {}
        for province in df_year_rank.index.unique():
                province_data = df_year_rank.loc[province]
                male_names = province_data[province_data['sex'] == 'male']['name'].tolist()
                female_names = province_data[province_data['sex'] == 'female']['name'].tolist()
                combinations = []
                for male in male_names:
                    for female in female_names:
                        combinations.append(f"{male}\n{female}")
                results[province] = ', '.join(combinations)
        # Create the final dataframe
        final_df = pd.DataFrame(results.items(), columns=['province', 'name'])
        final_df.set_index('province', inplace=True)
        return final_df

    @classmethod
    def render(cls):
        page_name = cls.page_name
        cls.set_session_state_variables()
        df_data, gdf_borders = cls.get_data()
        header = "Names & Surnames Analysis" if page_name == "names_surnames" else "Baby Names Analysis"
        st.header(header)
        #start_year, end_year = df_data["male"].index.get_level_values(0).min(), df_data["male"].index.get_level_values(0).max()
        start_year, end_year = 2018, 2023
        BasePage.sidebar_controls_basic_setup(start_year, end_year)
        col_1, col_2, col_3 = st.columns([1, 1, 6])
        name_surname = "name"  # single option for baby names dataset
        if page_name == "names_surnames":
            name_surname = col_2.radio("Select name or surname", ["Name", "Surname"], key="name_surname_rb").lower()
            df = df_data[name_surname.lower()]
        else:
            df = df_data
        disable = name_surname == "surname"
        col_1.checkbox("Male", key="male_" + page_name, disabled=disable, value=1)
        col_1.checkbox("Female", key="female_" + page_name, disabled=disable)  # if "surnames" option is selected disable gender options

        st.session_state["sex_" + page_name] = []
        if st.session_state["male_" + page_name]:
            st.session_state["sex_" + page_name].append("male")
        if st.session_state["female_" + page_name]:
            st.session_state["sex_" + page_name].append("female")
        if disable or not st.session_state["sex_" + page_name]:  # if "surnames" option is selected or not any genders are selected, select both
            st.session_state["sex_" + page_name] = ["male", "female"]
        if name_surname != "surname":
            df = df[df['sex'].isin(st.session_state["sex_" + page_name])]
        col, _ = st.columns([2, 6])
        col.selectbox("Select n ", options=list(range(1, 31)), key="n_" + page_name)
        col_1, col_2, _ = st.columns([2, 1, 5])
        expression_in_sentence = "names or surnames" if page_name == "names_surnames" else "baby names"
        col_1.radio("Select an  option",options=["Apply K-means clustering using top-n", f"Show the nth most common {expression_in_sentence}", f"Select {expression_in_sentence} and top-n number to filter"],
                          index = 1, key="display_option_" + page_name)
        col_2.selectbox("Select K: number of clusters", range(2, 11), key="n_clusters_" + page_name)
        col_1.multiselect(f"Select {expression_in_sentence} names",  sorted(df["name"].unique(), key=locale.strxfrm), key="names_" + page_name)
        col_plot, colf_df = st.columns([5, 1])
        cls.plot_geopandas(col_plot, df, gdf_borders)

    @staticmethod
    def plot_result(df_result, ax, names_unique):
        # Create a color map
        cmap = cm.get_cmap('tab20', len(names_unique))
        colors = {name: cmap(i) for i, name in enumerate(names_unique)}
        colors[np.nan] = "gray"
        # Assign colors to each row in the GeoDataFrame
        df_result['color'] = df_result['name'].map(colors).fillna("gray")
        # After groupby df_result becomdes Pandas dataframe, we have to convert it to GeoPandas dataframe
        df_result = gpd.GeoDataFrame(df_result, geometry='geometry')
        # Plotting
        df_result.plot(ax=ax, color=df_result['color'], legend=True, legend_kwds={"shrink": .6}, edgecolor="black", linewidth=.2)
        bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6)
        df_result.apply(lambda x: ax.annotate(
            text=x["province"].upper() + "\n" + x['name'].title() if isinstance(x['name'], str) else x["province"],
            size=4, xy=x.geometry.centroid.coords[0], ha=PageNames.HA_POSITIONS.get(x["province"], "center"), va=PageNames.VA_POSITIONS.get(x["province"], "center"), bbox=bbox), axis=1)
        ax.axis("off")
        ax.margins(x=0)
        # Add a legend
       # for name, color in set(zip(df_result['name'], df_result['color'])):
       #     ax.plot([], [], color=color, label=name, linestyle='None', marker='o')
       #     ax.legend(title='Names', fontsize=4, bbox_to_anchor=(0.01, 0.01), loc='lower right', fancybox=True, shadow=True)

    @classmethod
    def get_title(cls):
        page_name = st.session_state["page_name"]
        rank = st.session_state["n_"+page_name]
        names_or_surnames = "names"
        selected_gender = "male and female" if len(st.session_state["sex_" + cls.page_name])!=1 else st.session_state["sex_" + cls.page_name][0]
        # Adjust phrasing based on rank
        if rank == 1:
            title_prefix = "The most common "
        else:
            title_prefix = f"The {PageNames.get_ordinal(rank)} most common "

        if page_name == "names_surnames":
            if st.session_state["name_surname_rb"] == "Surname":
                title = title_prefix + "surnames"
            else:
                title = title_prefix + selected_gender+" names"
        else:
            title = title_prefix + selected_gender + " baby names"
        return title, names_or_surnames

    @classmethod
    def plot_geopandas(cls, col_plot, df, gdf_borders):
        title, names_or_surnames = cls.get_title()
        display_option = st.session_state["display_option_" + cls.page_name]
        top_n = int(st.session_state["n_" + cls.page_name])

        fig, axs = figure_setup()
        df_results = []
        year_1, year_2 = st.session_state["year_1"], st.session_state["year_2"]

        for i, year in enumerate(sorted({year_1, year_2})):
            # Display option 1: Show the nth most common baby names
            if "nth most common" in display_option:
                df_year_rank = PageNames.get_df_year_rank(df, year, target_rank=top_n, n_to_top_inclusive=False)
                df_result = gdf_borders.merge(df_year_rank, left_on="province", right_index=True)
                df_result = df_result.groupby(["geometry", "province"])["name"].apply(
                    lambda x: "%s" % '\n '.join(x)).to_frame().reset_index()
                df_results.append(df_result)
                PageNames.plot_result(df_results[i], axs[i, 0], sorted(df_result["name"].unique()))
                axs[i, 0].set_title(title + f' in {year}')
            elif "top-n number to filter" in display_option:  # Display option 2: Select single year, name(s) and top-n number to filter
                names_from_multi_select = st.session_state["names_" + cls.page_name]
                df_year = df.loc[year].reset_index()
                if names_from_multi_select:
                    df_result = df_year[(df_year["name"].isin(names_from_multi_select)) & (df_year["rank"] <= top_n)]
                    names_or_surnames_statement = names_or_surnames[:-1] + " is" if len(
                        names_from_multi_select) == 1 else names_or_surnames + " are"  # drop "s" if single name or surname selected
                    if df_result.empty:
                        st.write(
                            f"Selected {names_or_surnames_statement} are not in the top {top_n} for the year {year}")
                    df_result_not_null = gdf_borders.merge(df_result, left_on="province", right_on="province")
                    df_result_not_null = df_result_not_null.groupby(["geometry", "province"])["name"].apply(
                        lambda x: "%s" % '\n '.join(x)).to_frame().reset_index()
                    df_results.append(df_result_not_null)
                    df_result_with_nulls = gdf_borders.merge(df_result_not_null[["province", "name"]],
                                                             left_on="province", right_on="province", how="left")
                    PageNames.plot_result(df_result_with_nulls, axs[i, 0], sorted(df_result_not_null['name'].unique()))
                    axs[i, 0].set_title(f"Provinces where selected {names_or_surnames_statement} in the top {top_n} for {year}")
            else:  # K-means
                df_result, closest_provinces = cls.k_means_clustering(df, gdf_borders.copy(), year)
                cls.plot_clusters(df_result, closest_provinces,  axs[i, 0])
                df_results.append(df_result)

            if not df_results:
                st.write("No results found.")
        if df_results:
            col_plot.pyplot(fig)

    @classmethod
    def set_session_state_variables(cls):
        page_name = cls.page_name
        if "sex_" + page_name not in st.session_state:
            st.session_state["sex_" + page_name] = []
        if "n_" + page_name not in st.session_state:
            st.session_state["n_" + page_name] = 1
        if "name_surname_rb" not in st.session_state:
            st.session_state["name_surname_rb"] = "Name"