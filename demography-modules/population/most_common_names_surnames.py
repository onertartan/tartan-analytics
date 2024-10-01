from utils.helpers_names import names_main
import streamlit as st
import pandas as pd
import geopandas as gpd


@st.cache_data
def get_data():
    df_data = {"male": pd.read_csv("data/preprocessed/population/most_common_names_male.csv", index_col=[0, 1]),
               "female": pd.read_csv("data/preprocessed/population/most_common_names_female.csv", index_col=[0, 1]),
               "surname": pd.read_csv("data/preprocessed/population/most_common_surnames.csv", index_col=[0, 1])
               }

    gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
    return df_data, gdf_borders


page_name = "names or surnames"
names_main(page_name, get_data)
