from modules.base_page_common import PageCommon
from utils.checkbox_group import Checkbox_Group
import streamlit as st
import pandas as pd


class PageBirthSex(PageCommon):
    page_name = "birth"
    features = {"nominator":["sex"], "denominator":["sex", "age"]}
    checkbox_group = {"age": Checkbox_Group(page_name, "age", 3,
                                            ["all"] + [f"{i}-{i + 4}" for i in range(0, 90, 5)] + ["90+"]),
                      "sex": Checkbox_Group(page_name, "sex", 1, ["all", "male", "female"])}
    top_row_cols = st.columns([1, 2.2, 2.5, 1.2], gap="small") # first col is for geo_scale, others are optional(extras)
    col_weights = [1, 2, .1, 1, 2]

    @classmethod
    def fun_extras(cls):
        # for shortcut selection of age groups for general fertility rate we add an option using radio button
        cls.top_row_cols[1].button("Quick selection for general fertility rate",
                                  key="quick_selection_fertility_age_group")
        cls.top_row_cols[2].image("images/fertility-rate.jpg")

    @classmethod
    def quick_selection(cls,feature_name,nom_denom_key_suffix):

        #if st.session_state["birth_age_group_selection"] != "Custom selection" and feature_name=="age" and nom_denom_key_suffix=="denominator":# then select checkboxes for general fertility rate
        if st.session_state["quick_selection_fertility_age_group"] and feature_name=="age" and nom_denom_key_suffix=="denominator":
            keys_to_check  = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"] #mid_age_groups
            cls.set_checkbox_values_for_quick_selection( keys_to_check, nom_denom_key_suffix, feature_name)
        #  st.session_state[page_name + "_age_checkbox_group"].set_checkbox_values_for_quick_selection(page_name, basic_keys, mid_age_groups, "denominator")


    @staticmethod
    @st.cache_data
    def get_data(geoscale=False):
        df = {}
        file_path = "data/preprocessed/population/birth-sex-ibbs3-2009-2024.csv"
        df["province"] = pd.read_csv(file_path, index_col=[0, 1], header=[0])
        file_path = "data/preprocessed/population/birth-district-2014-2023.csv"
        df["district"] = pd.read_csv(file_path, index_col=[0, 1, 2], header=[0])

        df_data_denom ={}
        file_path = "data/preprocessed/population/age-sex-ibbs3-2007-2023.csv"
        df_data_denom["province"] = pd.read_csv(file_path, index_col=[0, 1], header=[0, 1])
        file_path = "data/preprocessed/population/age-sex-district-2018-2023.csv"
        df_data_denom["district"] = pd.read_csv(file_path, index_col=[0, 1, 2], header=[0, 1])
        df_data_denom["district"] = df_data_denom["district"].sort_index()

        df_data = {"nominator": df, "denominator": df_data_denom}
        return df_data

PageBirthSex().run();