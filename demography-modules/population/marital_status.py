from matplotlib import cm
from utils.checkbox_group import Checkbox_Group
from utils.dashboard_components import sidebar_controls
from utils.plot_map_common import plot
import streamlit as st
import pandas as pd
import geopandas as gpd
from utils.helpers_ui import ui_basic_setup_common
from utils.helpers_common import feature_choice


if 'maritial_status_age_checkbox_group' not in st.session_state:
    st.session_state["maritial_status_age_checkbox_group"] = None

@st.cache_data
def get_data(geo_scale_code,page_name):
    if geo_scale_code is not None: # not district
        file_path = "data/preprocessed/population/age-sex-maritial-status-ibbs3-2008-2023.csv"
        df = pd.read_csv(file_path, index_col=[0, 1], header=[0,1,2])
        gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
   # df.index.set_names('year', level=0, inplace=True)
    else:
        gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_district.geojson")
        df = pd.read_csv("data/preprocessed/population/age-sex-maritial-status-district-2018-2023.csv", index_col=[0, 1, 2], header=[0, 1, 2])

    df_data = {"nominator": df.loc[:, ((slice(None, None), st.session_state["age_group_keys"][page_name][1:]))]}  # #sort age groups
    df_data["denominator"]= df_data["nominator"]
    return df_data, gdf_borders


geo_scale = st.radio("Choose geographic scale", ["province (ibbs3)", "sub-region (ibbs2)", "region (ibbs1)", "district"])
if geo_scale == "district":
    geo_scale = ["district"]
    geo_scale_code = None
else:
    geo_scale_code = geo_scale[geo_scale.index("(")+1:geo_scale.index(")")]
    geo_scale = [geo_scale[:geo_scale.index("(")].strip()]
print("geo_scale123",geo_scale,"geo_scale123",geo_scale_code)

page_name = "marital_status"
cols_nom_denom = ui_basic_setup_common(num_sub_cols=3)
df_data, gdf_borders = get_data(geo_scale_code,page_name)
sidebar_controls(df_data["nominator"].index.get_level_values(0).min(),df_data["nominator"].index.get_level_values(0).max() )


selected_features_dict = {}
for nom_denom in cols_nom_denom.keys():
    selected_features_dict[nom_denom] = (feature_choice(cols_nom_denom[nom_denom][0], "sex", nom_denom),
                                    feature_choice(cols_nom_denom[nom_denom][2], "age", nom_denom, 3, page_name),
                                    feature_choice(cols_nom_denom[nom_denom][1], "maritial_status", nom_denom) )

st.write("""<style>[data-testid="stHorizontalBlock"]{align-items: center;}</style>""", unsafe_allow_html=True)
col_plot, col_df = st.columns((4, 1), gap="small")
plot(col_plot, col_df, df_data, gdf_borders,  selected_features_dict, geo_scale,page_name)