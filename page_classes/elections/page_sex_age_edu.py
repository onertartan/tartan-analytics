from page_classes.base_page import BasePage
import pandas as pd
import streamlit as st
import geopandas as gpd

class PageSexAgeEdu(BasePage):
    page_name = "sex_age_edu_elections"
    features = {"nominator":  ["sex", "education", "age"], "denominator": ["sex", "education", "age"]}
    @staticmethod
    def fun_extras(cols):
        pass

    @st.cache_data
    def get_data(geo_scale):
        page_name = st.session_state["page_name"]
        df = pd.read_csv("data/preprocessed/elections/df_edu.csv", index_col=[0, 1, 2], header=[0, 1, 2])


        #df_data = {"nominator": df.loc[:, (( slice(None, None), st.session_state["age_group_keys"][page_name][1:]))]}  # #sort age groups
        # this data is sorted
        df_data={"denominator": df, "nominator": df}
        return df_data


