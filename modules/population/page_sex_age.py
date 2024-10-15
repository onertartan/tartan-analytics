from modules.population.Basic_Page import BasePage
import pandas as pd
import streamlit as st
import geopandas as gpd

class Page_Sex_Age(BasePage):
    @staticmethod
    def fun_extras(cols):
        cols[1].radio("Age group selection", ["Custom selection", "Quick selection for total age dependency ratio",
                                          "Quick selection for child age dependency ratio",
                                          "Quick selection for old-age dependency ratio"],
                  key="sex_age_age_group_selection")
        cols[2].image("images/age-dependency.jpg")

    @st.cache_data
    def get_data(geo_scale):
        page_name = st.session_state["page_name"]
        if geo_scale != "district":
            gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
            df = pd.read_csv("data/preprocessed/population/age-sex-ibbs3-2007-2023.csv", index_col=[0, 1],
                             header=[0, 1])
        else:
            gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_district.geojson")
            df = pd.read_csv("data/preprocessed/population/age-sex-district-2018-2023.csv", index_col=[0, 1, 2],
                             header=[0, 1])

        df = df.sort_index()
        df_data = {"nominator": df.loc[:, ((slice(None, None), st.session_state["age_group_keys"][page_name][1:]))]}  ##sort age groups
        df_data["denominator"] = df_data["nominator"]

        return df_data, gdf_borders

