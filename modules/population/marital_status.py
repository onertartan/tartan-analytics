from modules.base_page import BasePage
from utils.checkbox_group import Checkbox_Group
import pandas as pd
import streamlit as st


class PageSexAgeMaritalStatus(BasePage):
    page_name = "marital_status"
    features = {"nominator":  ["sex", "marital_status", "age"], "denominator": ["sex","marital_status","age"]}

    checkbox_group = {"age": Checkbox_Group(page_name, "age", 3, ["all"] + [f"{i}-{i + 4}" for i in range(15, 90, 5)] + ["90+"]),
                      "marital_status": Checkbox_Group(page_name, "marital_status",  1,
                                                       ["all", "never married", "married", "divorced", "widowed"],message="Select marital status."),
                      "sex": Checkbox_Group(page_name, "sex", 1, ["all", "male", "female"])}
    top_row_cols = st.columns([1], gap="small")#first col is for geo_scale, others are optional(extras)
    col_weights = [1, 2, 4, .1, 1, 2, 4]

    @classmethod
    def fun_extras(cls):
        pass

    @staticmethod
    @st.cache_data
    def get_data(geo_scale):
        if geo_scale != "district":
            file_path = "data/preprocessed/population/age-sex-marital-status-ibbs3-2008-2023.csv"
            df = pd.read_csv(file_path, index_col=[0, 1], header=[0, 1, 2])
        else:
            df = pd.read_csv("data/preprocessed/population/age-sex-marital-status-district-2018-2023.csv",  index_col=[0, 1, 2], header=[0, 1, 2])

        df_data = {"nominator": df}#.loc[:, ((  slice(None, None), st.session_state["age_group_keys"][page_name][1:]))]}  # #sort age groups
        df_data["denominator"] = df_data["nominator"]
        return df_data

PageSexAgeMaritalStatus.run()