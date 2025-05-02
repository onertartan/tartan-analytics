from modules.base_page_common import PageCommon
import pandas as pd
import streamlit as st
from utils.checkbox_group import Checkbox_Group


class PageSexAge(PageCommon):
    page_name = "sex_age"
    features = {"nominator":["sex","age"],"denominator":["sex","age"]}
    checkbox_group = {"age":Checkbox_Group(page_name, "age", 3, ["all"] + [f"{i}-{i + 4}" for i in range(0, 90, 5)] + ["90+"]),
                      "sex": Checkbox_Group(page_name, "sex", 1, ["all", "male", "female"]) }
    top_row_cols = st.columns([1, 2.2, 2.5, 1.2], gap="small")#first col is for geo_scale, others are optional(extras)
    col_weights = [1, 2, .1, 1, 2]

    def fun_extras(self):
        # cls.top_row_cols[1].radio("Age group selection", ["Custom selection", "Quick selection for total age dependency ratio",
        #                                   "Quick selection for child age dependency ratio",
        #                                   "Quick selection for old-age dependency ratio"],
        #           key="sex_age_age_group_selection")

        self.top_row_cols[1].markdown("<div style='margin-top:60x;'></div>", unsafe_allow_html=True)
        self.top_row_cols[1].markdown(""" <style> div[data-testid="stButton"] { padding-bottom: 5px; } </style> """, unsafe_allow_html=True)

        self.top_row_cols[1].button("Quick selection for  total age dependency ratio",  key="quick_selection_total_age_group")
        self.top_row_cols[1].button("Quick selection for child age dependency ratio",  key="quick_selection_child_age_group")
        self.top_row_cols[1].button("Quick selection for old-age dependency ratio",  key="quick_selection_old_age_group")
        self.top_row_cols[2].image("images/age-dependency.jpg")


    def quick_selection(self,feature_name,nom_denom_key_suffix):
        age_groups= [f"{i}-{i + 4}" for i in range(0, 90, 5)] + ["90+"]
        child_age_groups, elderly_age_groups = (["0-4", "5-9", "10-14"],   ["65-69", "70-74", "75-79", "80-84","85-89","90+"])
        working_age_groups = list(set(age_groups) - set(child_age_groups + elderly_age_groups))
        if feature_name == "age" and (st.session_state["quick_selection_total_age_group"] or st.session_state["quick_selection_child_age_group"] or  st.session_state["quick_selection_old_age_group"]):
            if nom_denom_key_suffix == "denominator":
                self.set_checkbox_values_for_quick_selection(working_age_groups, nom_denom_key_suffix,  feature_name)
            else:
                if st.session_state["quick_selection_total_age_group"]:
                    self.set_checkbox_values_for_quick_selection(child_age_groups+elderly_age_groups , nom_denom_key_suffix, feature_name)
                elif st.session_state["quick_selection_child_age_group"]:
                    self.set_checkbox_values_for_quick_selection(child_age_groups, nom_denom_key_suffix,  feature_name)
                elif st.session_state["quick_selection_old_age_group"]:
                    self.set_checkbox_values_for_quick_selection(elderly_age_groups, nom_denom_key_suffix,  feature_name)


    @staticmethod
    @st.cache_data
    def get_data():
        df = {}
        file_path = "data/preprocessed/population/age-sex-ibbs3-2007-2023.csv"
        df["province"] = pd.read_csv(file_path, index_col=[0, 1], header=[0, 1])
        file_path = "data/preprocessed/population/age-sex-district-2018-2023.csv"
        df["district"] = pd.read_csv(file_path, index_col=[0, 1, 2], header=[0, 1])
        df["district"] = df["district"].sort_index()
        df_data = {"nominator": df}#.loc[:, ((slice(None, None), slice(None, None)))]}  ##sort age groups
        df_data["denominator"] = df_data["nominator"]
        return df_data

PageSexAge().run()
