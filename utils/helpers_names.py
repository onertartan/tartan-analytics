import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from utils.helpers_plot_names import plot_geopandas
from utils.helpers_ui import figure_setup, ui_basic_setup_names


def set_session_state_variables(page_name):
    if "clustering_cb_" + page_name not in st.session_state:
        st.session_state["clustering_cb_" + page_name] = False
    if "select_both_sexes_" + page_name not in st.session_state:
        st.session_state["select_both_sexes_" + page_name] = False
    if "top_n_checkbox_" + page_name not in st.session_state:
        st.session_state["top_n_checkbox_" + page_name] = False
    if "top_n_selection_" + page_name not in st.session_state:
        st.session_state["top_n_selection_" + page_name] = 2
    if "name_surname_rb"  not in st.session_state:
        st.session_state["name_surname_rb"] = "Name"

def main(page_name, get_data_func):
    set_session_state_variables(page_name)
    df_data, gdf_borders = get_data_func()
    ui_basic_setup_names(page_name, df_data)
    col_plot, colf_df = st.columns([5, 1])

    if st.session_state["clustering_cb_" + page_name]:
        k_means_clustering(col_plot, df_data, gdf_borders, page_name)
    else:
        plot_geopandas(col_plot, df_data, gdf_borders, page_name)


def k_means_clustering(col_plot, df_data, gdf_borders, page_name):
    if page_name == "names or surnames" and st.session_state["name_surname_rb"] == "Surname":
        df_year = df_data["surname"].loc[st.session_state["year_1"]]
    elif st.session_state["select_both_sexes_" + page_name]:
        df_year_male, df_year_female = df_data["male"].loc[st.session_state["year_1"]], df_data["female"].loc[
            st.session_state["year_1"]]
        overlapping_names = set(df_year_male["name"]) & set(df_year_female["name"])
        print("CB:2")
        df_year_male['name'] = df_year_male.apply(
            lambda x: f"{x['name']}_female" if x['name'] in overlapping_names else x['name'], axis=1)
        df_year_female['name'] = df_year_female.apply(
            lambda x: f"{x['name']}_male" if x['name'] in overlapping_names else x['name'], axis=1)
        df_year = pd.concat([df_year_male, df_year_female])
    else:
        sex = st.session_state["sex_" + page_name].lower()
        df_year = df_data[sex].loc[st.session_state["year_1"]]
    df_year.loc[:, "ratio"] = df_year.loc[:, "count"] / df_year.loc[:, "total_count"]
    df_pivot = pd.pivot_table(df_year, values='ratio', index=df_year.index, columns=['name'],
                              aggfunc=lambda x: x, dropna=False, fill_value=0)
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