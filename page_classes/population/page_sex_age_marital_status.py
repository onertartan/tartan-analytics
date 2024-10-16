from page_classes.base_page import BasePage
import pandas as pd
import streamlit as st
import geopandas as gpd

class PageSexAgeMaritalStatus(BasePage):
    page_name = "marital_status"
    features = {"nominator":  ["sex", "marital_status", "age"], "denominator": ["sex","marital_status","age"]}
    @staticmethod
    def fun_extras(cols):
        pass

    @st.cache_data
    def get_data(geo_scale):
        page_name = st.session_state["page_name"]
        if geo_scale != "district":
            file_path = "data/preprocessed/population/age-sex-maritial-status-ibbs3-2008-2023.csv"
            df = pd.read_csv(file_path, index_col=[0, 1], header=[0, 1, 2])
        # df.index.set_names('year', level=0, inplace=True)
        else:
            df = pd.read_csv("data/preprocessed/population/age-sex-maritial-status-district-2018-2023.csv",
                             index_col=[0, 1, 2], header=[0, 1, 2])

        df_data = {"nominator": df.loc[:, ((
        slice(None, None), st.session_state["age_group_keys"][page_name][1:]))]}  # #sort age groups
        df_data["denominator"] = df_data["nominator"]
        return df_data


