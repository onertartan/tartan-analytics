from modules.base_page_names import PageNames
import pandas as pd
import geopandas as gpd
import streamlit as st


class PageBabyNames(PageNames):
    page_name = "baby_names"

    @staticmethod
    @st.cache_data
    def get_data():
        df_data = {}
        file_path = "data/preprocessed/population/most_common_baby_names_male.csv"
        df_data["male"] = pd.read_csv(file_path, index_col=[0, 1])
        file_path = "data/preprocessed/population/most_common_baby_names_female.csv"
        df_data["female"] = pd.read_csv(file_path, index_col=[0, 1])
        gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
        return df_data, gdf_borders



PageBabyNames().run();