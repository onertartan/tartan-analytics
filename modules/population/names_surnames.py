from modules.base_page_names import PageNames
import pandas as pd
import geopandas as gpd
import streamlit as st


class PageNamesSurnames(PageNames):
    page_name = "names_surnames"

    @staticmethod
    @st.cache_data
    def get_data():
        df_data = {"name": pd.read_csv("data/preprocessed/population/names.csv", index_col=[0, 1]),
                   "surname": pd.read_csv("data/preprocessed/population/most_common_surnames.csv", index_col=[0, 1])
                   }

        gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
        return df_data, gdf_borders



PageNamesSurnames().run();