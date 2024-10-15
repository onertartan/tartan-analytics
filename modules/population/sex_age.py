from modules.population.page_sex_age import Page_Sex_Age
from utils.dashboard_components import sidebar_controls
from utils.plot_map_common import plot
import geopandas as gpd
import pandas as pd
import streamlit as st
from utils.checkbox_group import Checkbox_Group
from utils.helpers_common import feature_choice
from utils.helpers_ui import ui_basic_setup_common
Page_Sex_Age.set_page_name("sex_age")
Page_Sex_Age.set_features(["age","sex"])
Page_Sex_Age.run()


@st.cache_data
def get_data(geo_scale):
    page_name = st.session_state["page_name"]
    if geo_scale != "district":
        gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
        df = pd.read_csv("data/preprocessed/population/age-sex-ibbs3-2007-2023.csv", index_col=[0, 1], header=[0, 1])
    else:
        gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_district.geojson")
        df = pd.read_csv("data/preprocessed/population/age-sex-district-2018-2023.csv", index_col=[0, 1, 2],
                         header=[0, 1])

    df = df.sort_index()
    df_data = {"nominator": df.loc[:, ((
    slice(None, None), st.session_state["age_group_keys"][page_name][1:]))]}  ##sort age groups
    df_data["denominator"] = df_data["nominator"]

    return df_data, gdf_borders


st.markdown("""
    <style>
        .main > div {
            padding-top: 1rem;
        }

    </style>
    """, unsafe_allow_html=True)
cols = st.columns([1, 2.2, 2.5, 1.2], gap="small")
geo_scale = cols[0].radio("Choose geographic scale",
                          ["province (ibbs3)", "sub-region (ibbs2)", "region (ibbs1)", "district"])
if geo_scale != "district":
    geo_scale = geo_scale.split()[0]

st.markdown("""<style> [role=radiogroup]{ gap: .8rem; } </style>""", unsafe_allow_html=True)
cols[1].radio("Age group selection", ["Custom selection", "Quick selection for total age dependency ratio",
                                      "Quick selection for child age dependency ratio",
                                      "Quick selection for old-age dependency ratio"],
              key="sex_age_age_group_selection")
cols[2].image("images/age-dependency.jpg")

st.session_state["page_name"] = "sex_age"
cols_nom_denom = ui_basic_setup_common(num_sub_cols=2)
df_data, gdf_borders = get_data(geo_scale)
sidebar_controls(df_data["nominator"].index.get_level_values(0).min(),
                 df_data["nominator"].index.get_level_values(0).max())
Checkbox_Group.age_group_quick_select()

selected_features = {}
for nom_denom in cols_nom_denom.keys():
    selected_features[nom_denom] = (feature_choice(cols_nom_denom[nom_denom][0], "sex", nom_denom),
                                    feature_choice(cols_nom_denom[nom_denom][1], "age", nom_denom))  # Page

st.write("""<style>[data-testid="stHorizontalBlock"]{align-items: top;}</style>""", unsafe_allow_html=True)
col_plot, col_df = st.columns((4, 1), gap="small")
plot(col_plot, col_df, df_data, gdf_borders, selected_features, [geo_scale])
