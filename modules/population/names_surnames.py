from modules.base_page_names import PageNames
import pandas as pd
import geopandas as gpd
import streamlit as st


class PageNamesSurnames(PageNames):
    page_name = "names_surnames"

    @classmethod
    @st.cache_data
    def get_data(cls, geo_scale=None):
        df_data = {"name": pd.read_csv("data/preprocessed/population/names.csv", index_col=[0, 1]),
                   "surname": pd.read_csv("data/preprocessed/population/most_common_surnames.csv", index_col=[0, 1])
                   }
        return df_data


PageNamesSurnames().run()