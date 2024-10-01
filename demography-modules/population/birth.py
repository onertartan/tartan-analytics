import pandas as pd
import geopandas as gpd
import streamlit as st
from utils.checkbox_group import Checkbox_Group
from utils.dashboard_components import sidebar_controls
from utils.plot_map_common import plot
from utils.helpers_ui import ui_basic_setup_common
from utils.helpers_common import feature_choice

@st.cache_data
def get_data(geo_scale_code):
    if geo_scale_code is not None:  # not district
        file_path = "data/preprocessed/population/birth-sex-ibbs3-2009-2023.csv"
        df = pd.read_csv(file_path, index_col=[0, 1], header=[0])
        gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
    else:
        file_path = "data/preprocessed/population/birth-district-2014-2023.csv"
        df = pd.read_csv(file_path, index_col=[0, 1, 2], header=[0])
        gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_district.geojson")


    df_data_denom = pd.read_csv("data/preprocessed/population/age-sex-ibbs3-2007-2023.csv", index_col=[0, 1], header=[0, 1])
    df_data = {"nominator": df, "denominator": df_data_denom}
    age_group_keys =["all"]+[f"{i}-{i+4}" for i in range(0, 90, 5)]+["90+"]
    return df_data, gdf_borders, age_group_keys


cols = st.columns([1, 1.4, 2.6, 1.2], gap="small")

style = """<style>div.row-widget.stRadio > div {
 flex-direction: column;
 align-items: stretch;
}

div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]  {
 background-color: #9AC5F4;
 padding-right: 10px;
 padding-left: 4px;
 padding-bottom: 3px;
 margin: 4px;
}</style>"""
#st.markdown(f'<style> {style}</style>', unsafe_allow_html=True)

geo_scale = cols[0].radio("Choose geographic scale", ["province (ibbs3)", "sub-region (ibbs2)", "region (ibbs1)", "district"])

if geo_scale == "district":
    geo_scale = ["district"]
    geo_scale_code = None
else:
    geo_scale_code = geo_scale[geo_scale.index("(")+1:geo_scale.index(")")]
    geo_scale = [geo_scale[:geo_scale.index("(")].strip()]

cols[1].radio("Age group selection", ["Custom selection", "Quick selection for general fertility rate"], key = "birth_age_group_selection")
cols[2].image("images/fertility-rate.jpg")

cols_nom_denom = ui_basic_setup_common(num_sub_cols=2)
#start_year, end_year= (2009,2023) if geo_scale == "ibbs3" else (2018,2023)
df_data, gdf_borders, age_group_keys = get_data(geo_scale_code)
sidebar_controls(df_data["nominator"].index.get_level_values(0).min(),df_data["nominator"].index.get_level_values(0).max(),"birth_module" )


page_name = "birth"
if 'birth_age_checkbox_group' not in st.session_state:
    st.session_state[page_name+"_age_checkbox_group"] = None
Checkbox_Group.age_group_quick_select(page_name, age_group_keys)

selected_features = {}
for nom_denom in cols_nom_denom.keys():
     if nom_denom =="nominator": # from left panel only sex can be chosen
         selected_features[nom_denom] = feature_choice(cols_nom_denom[nom_denom][0], "sex", nom_denom)
     else:                      # from right panel general selection is possible
         selected_features[nom_denom] = (feature_choice(cols_nom_denom[nom_denom][0], "sex", nom_denom),
                                  feature_choice(cols_nom_denom[nom_denom][1], "age", nom_denom, 3, age_group_keys, page_name) )

col_plot, col_df = st.columns([5, 1])
plot(col_plot, col_df, df_data, gdf_borders,selected_features, geo_scale)