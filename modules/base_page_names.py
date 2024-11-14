from sklearn.cluster import KMeans
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
    def k_means_clustering(cls,col_plot, df_data, gdf_borders):
        page_name = cls.page_name
        if page_name == "names or surnames" and st.session_state["name_surname_rb"] == "Surname":
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
        df_year.loc[:, "ratio"] = df_year.loc[:, "count"] / df_year.loc[:, "total_count"]
        df_pivot = pd.pivot_table(df_year, values='ratio', index=df_year.index, columns=['name'],  aggfunc=lambda x: x, dropna=False, fill_value=0)
        kmeans = KMeans(n_clusters=st.session_state["n_clusters_" + page_name], random_state=0).fit(df_pivot)

        df_pivot["clusters"] = kmeans.labels_
        gdf_borders = gdf_borders.merge(df_pivot["clusters"], left_on="province", right_on=df_pivot.index)
        # Define a color map for the categories
        # color_map = {0: "purple", 1: "orange", 2: "green",3:"cyan",4:"red",5:"blue",6:"magenta",7:"gray",8:"yellow"}#female

        # color_map = {0: "orange", 1: "orange", 2: "red",3:"red",4:"orange",5:"magenta",6:"red",7:"orange",8:"orange"}#male-8
        #  color_map = {0: "purple", 1: "red", 2: "orange",3:"red",4:"orange",5:"magenta",6:"cyan",7:"yellow",8:"gray"}#female-5
        # color_map = {0: "purple", 1: "orange", 2: "orange",3:"red",4:"orange",5:"red",6:"cyan",7:"yellow",8:"gray"}#female-6
        # color_map = {0: "orange", 1: "red", 2: "red",3:"red",4:"orange",5:"magenta",6:"magenta",7:"orange",8:"gray"}#total-8
        # color_map = {0: "red", 1: "purple", 2: "orange",3:"green",4:"blue",5:"magenta",6:"cyan",7:"yellow",8:"gray"}#original
        # color_map = {0: "orange", 1: "orange", 2: "purple",3:"red",4:"red",5:"red",6:"purple",7:"yellow",8:"gray"}#total-7
        # color_map = {0: "orange", 1: "orange", 2: "purple",3:"red",4:"red",5:"red",6:"cyan",7:"yellow",8:"gray"} #total-6
        # color_map = {0: "orange", 1: "red", 2: "orange",3:"red",4:"orange",5:"magenta",6:"red",7:"orange",8:"orange"}#female-9

        color_map = {0: "red", 1: "purple", 2: "orange", 3: "green", 4: "blue", 5: "magenta", 6: "cyan", 7: "yellow",
                     8: "gray"}  # original
        #  color_map = {0: "orange", 1: "orange", 2: "red",3:"red",4:"orange",5:"magenta",6:"red",7:"yellow",8:"gray"}

        #   color_map = {0: "red", 1: "orange", 2: "red",3:"red",4:"orange",5:"red",6:"red",7:"yellow",8:"gray"}

        # Map the colors to the GeoDataFrame
        gdf_borders["color"] = gdf_borders["clusters"].map(color_map)
        fig, axs = figure_setup()
        gdf_borders.plot(ax=axs[0, 0], color=gdf_borders['color'], legend=True, edgecolor="black", linewidth=.2)
        axs[0, 0].axis("off")
        axs[0, 0].margins(x=0)
        col_plot.pyplot(fig)

    @classmethod
    def render(cls):
        page_name =cls.page_name
        cls.set_session_state_variables()
        df_data, gdf_borders = cls.get_data()

        BasePage.basic_sidebar_controls(min(df_data["male"].index.get_level_values(0)),max(df_data["male"].index.get_level_values(0)))
        # basic_sidebar_controls(2018, 2023)
        col_1, col_2, col_3 = st.columns([1, 1, 3])

        if not st.session_state["clustering_cb_" + page_name]:
            st.session_state["select_both_sexes_" + page_name] = False

        if page_name == "names or surnames":
            name_surname = col_1.radio("Select name or surname", ["Name", "Surname"], key="name_surname_rb").lower()
            if name_surname == "surname":
                st.session_state["select_both_sexes_" + page_name] = True
            sex = col_2.radio("Choose sex", ["Male", "Female"],   disabled=st.session_state["select_both_sexes_" + page_name], key="sex_" + page_name).lower()
        else:
            sex = col_1.radio("Choose sex", ["Male", "Female"],  disabled=st.session_state["select_both_sexes_" + page_name],key="sex_" + page_name).lower()

        st.header(f"Clustering or Displaying {page_name}")
        col_1, col_2, col_3 = st.columns([1, 1, 3])

        clustering_cb = col_1.checkbox("Apply K-means clustering", key="clustering_cb_" + page_name)
        col_1.checkbox("Select both sexes", disabled=not st.session_state["clustering_cb_" + page_name], key="select_both_sexes_" + page_name)
        col_1.selectbox("Select K: number of clusters", range(2, 11), key="n_clusters_" + page_name)
        disable_when_clustering = True if clustering_cb else False

        choice = col_3.radio("Or select an displaying option", [f"Show the most common {page_name}", f"Select {page_name} and top-n number to filter"],
                             horizontal=True, disabled=disable_when_clustering, key="secondary_option_" + page_name)

        disable_top_n_selection = not disable_when_clustering and choice == f"Show the most common {page_name}"

        col_3.selectbox("Select a number N to check if the entered name is among the top N names", options=list(range(1, 31)), disabled=disable_top_n_selection,
                        key="top_n_selection_" + page_name)

        sex_or_surname = "surname" if page_name == "Surname" and st.session_state["name_surname_rb"] == "Surname" else sex
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
        if page_name == "names or surnames" and st.session_state["name_surname_rb"] == "Surname":
            df = df_data["surname"]
        else:
            sex = st.session_state["sex_" + page_name].lower()
            df = df_data[sex]
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
                axs[i, 0].set_title(f'Results for {year}')
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
                    axs[i, 0].set_title(f'Results for {year}\n top_n = {top_n}')

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