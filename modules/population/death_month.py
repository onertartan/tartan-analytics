import locale
from modules.base_page_common import PageCommon
import pandas as pd
import streamlit as st
import calendar
from utils.checkbox_group import Checkbox_Group


class PageDeathMonth(PageCommon):
    locale.setlocale(locale.LC_TIME, "en_US.UTF-8")

    page_name = "death_month"
    features = {"nominator":["sex","month"],"denominator":["sex","month"]}
    checkbox_group = {"sex": Checkbox_Group(page_name, "sex", 1, ["all", "male", "female"]),
                      "month": Checkbox_Group(page_name, "month",2,
                                              basic_keys=["all"]+list(calendar.month_name)[1:],message="Select month(s)")}
    geo_scales = ["province (ibbs3)", "sub-region (ibbs2)", "region (ibbs1)"]
    top_row_cols = st.columns([1, 2.2, 2.5, 1.2], gap="small")#first col is for geo_scale, others are optional(extras)
    col_weights = [1, 2, .1, 1, 2]
    print("KEYSSS:",list(calendar.month_name)[1:])

    @staticmethod
    @st.cache_data
    def get_data():
        df = {}
        file_path = "data/preprocessed/population/death_sex_month_ibbs3-2009-2023.csv"
        df["province"] = pd.read_csv(file_path, index_col=[0, 1], header=[0, 1])
        df["province"] = df["province"].sort_index()
        file_path = "data/preprocessed/population/death_sex_month_ibbs3-2009-2023.csv"
        df["district"] = pd.read_csv(file_path, index_col=[0, 1, 2], header=[0, 1])
        df["district"] = df["district"].sort_index()

        df_data = {"nominator": df} #.loc[:, ((slice(None, None), slice(None, None)))]}  ##sort age groups
        df_data["denominator"] = df_data["nominator"]
        return df_data


PageDeathMonth().run()
