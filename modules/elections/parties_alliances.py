from modules.base_page_common import PageCommon
import pandas as pd
import streamlit as st
from utils.checkbox_group import Checkbox_Group


class PagePartiesAlliances(PageCommon):
    page_name = "parties_alliances"
    features = {"nominator":  ["sex", "education", "age"], "denominator": ["sex", "education", "age"]}
    checkbox_group = {"age": Checkbox_Group(page_name, "age", 2,
                      ["all"] + ["18-24"] + [f"{i}-{i + 4}" for i in range(25, 75, 5)] + ["75+"]),
                      "education": Checkbox_Group(page_name,"education",2,
            ["all",'illiterate', 'literate but did not complete any school', 'primary school', 'elementary school', 'secondary school or equivalent',
             'high school or equivalent', 'pre-license or bachelor degree', 'master degree', 'phd', 'unknown']),
                      "sex": Checkbox_Group(page_name, "sex", 1, ["all", "male", "female"])}

    top_row_cols = st.columns([1])  # first col is for geo_scale, others are optional(extras)
    col_weights = [1, 4, 2, .1,1, 4, 2,]

    @staticmethod
    @st.cache_data
    def get_data(geo_scale):
        df={}
        df["district"] = pd.read_csv("data/preprocessed/elections/df_election.csv", index_col=[0, 1, 2])
        #   if geo_scale != "district":
        df["province"] = pd.read_csv("data/preprocessed/elections/df_edu.csv", index_col=[0, 1, 2], header=[0, 1, 2]).droplevel(2).groupby(["year", "province"]).sum()

        df_data = {"denominator": df, "nominator": df}
        return df_data

    @classmethod
    def sidebar_controls_basic_setup(cls):
        with (st.sidebar):
            st.header('Select options')
            if "selected_election_year" not in st.session_state:
                st.session_state["selected_election_year"] = 2018
            st.session_state["year_1"] = st.session_state["selected_election_year"] = st.radio("Select year", options=[2018, 2023], key="election", index=0)
        basic_keys = ["all"] + df.loc[st.session_state["year_1"]].dropna(axis=1).columns[5:].tolist()  # Parties start from index 5
        cls.checkbox_group["Party/Alliance"] = Checkbox_Group(cls.page_name, "Party/Alliance", 4, basic_keys)

    @classmethod
    def render(cls):
        df_data = cls.get_data()
        year = cls.sidebar_controls_basic_setup()




PagePartiesAlliances.run();
