from modules.base_page import BasePage
from utils.checkbox_group import Checkbox_Group
import streamlit as st
import pandas as pd


class PageBirthSex(BasePage):
    page_name = "birth"
    features = {"nominator":["sex"], "denominator":["sex", "age"]}
    checkbox_group = {"age": Checkbox_Group(page_name, "age", 3,
                                            ["all"] + [f"{i}-{i + 4}" for i in range(0, 90, 5)] + ["90+"]),
                      "sex": Checkbox_Group(page_name, "sex", 1, ["all", "male", "female"])}
    top_row_cols = st.columns([1, 2.2, 2.5, 1.2], gap="small") # first col is for geo_scale, others are optional(extras)
    col_weights = [1, 2, .1, 1, 2]

    @classmethod
    def fun_extras(cls):
        cls.top_row_cols[1].radio("Age group selection", ["Custom selection", "Quick selection for general fertility rate"],
                      key="birth_age_group_selection")
        cls.top_row_cols[2].image("images/fertility-rate.jpg")

    @staticmethod
    @st.cache_data
    def get_data(geo_scale):
        if geo_scale != "district": # not district
            file_path = "data/preprocessed/population/birth-sex-ibbs3-2009-2023.csv"
            df = pd.read_csv(file_path, index_col=[0, 1], header=[0])
        else:
            file_path = "data/preprocessed/population/birth-district-2014-2023.csv"
            df = pd.read_csv(file_path, index_col=[0, 1, 2], header=[0])

        df_data_denom = pd.read_csv("data/preprocessed/population/age-sex-ibbs3-2007-2023.csv", index_col=[0, 1], header=[0, 1])
        df_data = {"nominator": df, "denominator": df_data_denom}
        return df_data

PageBirthSex().run();