#from page_classes.population.page_sex_age import PageSexAge
from modules.base_page import BasePage
import pandas as pd
import streamlit as st
from utils.checkbox_group import Checkbox_Group


class PageSexAge(BasePage):
    page_name = "sex_age"
    features = {"nominator":["sex","age"],"denominator":["sex","age"]}
    checkbox_group = {"age":Checkbox_Group(page_name, "age", 3, ["all"] + [f"{i}-{i + 4}" for i in range(0, 90, 5)] + ["90+"]),
                      "sex": Checkbox_Group(page_name, "sex", 1, ["all", "male", "female"]) }
    top_row_cols = st.columns([1, 2.2, 2.5, 1.2], gap="small")#first col is for geo_scale, others are optional(extras)
    col_weights = [1, 2, .1, 1, 2]

    @classmethod
    def fun_extras(cls):
        cls.top_row_cols[1].radio("Age group selection", ["Custom selection", "Quick selection for total age dependency ratio",
                                          "Quick selection for child age dependency ratio",
                                          "Quick selection for old-age dependency ratio"],
                  key="sex_age_age_group_selection")
        cls.top_row_cols[2].image("images/age-dependency.jpg")

    @staticmethod
    @st.cache_data
    def get_data(geo_scale):
        if geo_scale != "district":
            df = pd.read_csv("data/preprocessed/population/age-sex-ibbs3-2007-2023.csv", index_col=[0, 1], header=[0, 1])
        else:
            df = pd.read_csv("data/preprocessed/population/age-sex-district-2018-2023.csv", index_col=[0, 1, 2], header=[0, 1])

        df = df.sort_index()
        df_data = {"nominator": df}#.loc[:, ((slice(None, None), slice(None, None)))]}  ##sort age groups
        df_data["denominator"] = df_data["nominator"]

        return df_data

PageSexAge.run()
