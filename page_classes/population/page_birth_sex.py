from page_classes.base_page import BasePage
import pandas as pd
import streamlit as st
import geopandas as gpd

class PageBirthSex(BasePage):
    page_name = "birth"
    features = {"nominator":["sex"], "denominator":["sex", "age"]}

    @staticmethod
    def fun_extras(cols):
        cols[1].radio("Age group selection", ["Custom selection", "Quick selection for general fertility rate"],
                      key="birth_age_group_selection")
        cols[2].image("images/fertility-rate.jpg")

    @st.cache_data
    def get_data(geo_scale_code):
        if geo_scale_code is not None:  # not district
            file_path = "data/preprocessed/population/birth-sex-ibbs3-2009-2023.csv"
            df = pd.read_csv(file_path, index_col=[0, 1], header=[0])
        else:
            file_path = "data/preprocessed/population/birth-district-2014-2023.csv"
            df = pd.read_csv(file_path, index_col=[0, 1, 2], header=[0])

        df_data_denom = pd.read_csv("data/preprocessed/population/age-sex-ibbs3-2007-2023.csv", index_col=[0, 1],
                                    header=[0, 1])
        df_data = {"nominator": df, "denominator": df_data_denom}
        return df_data



