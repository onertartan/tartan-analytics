from utils.dashboard_components import sidebar_controls
from utils.helpers_plot_common import plot
import geopandas as gpd
import pandas as pd
import streamlit as st
from utils.checkbox_group import Checkbox_Group
from utils.helpers_common import feature_choice
from utils.helpers_ui import ui_basic_setup_common

@st.cache_data
def get_data(geo_scale_code):
    if geo_scale_code is not None: # district
        gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
        df = pd.read_csv("data/preprocessed/population/age-sex-ibbs3-2007-2023.csv", index_col=[0, 1],  header=[0, 1])
    else:
        gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_district.geojson")
        df = pd.read_csv("data/preprocessed/population/age-sex-district-2018-2023.csv", index_col=[0, 1, 2], header=[0, 1])

    df = df.sort_index()
    df_data = {"nominator": df, "denominator": df}
    age_group_keys =["all"]+[f"{i}-{i+4}" for i in range(0, 90, 5)]+["90+"]
    return df_data, gdf_borders, age_group_keys


cols = st.columns([1, 2.2, 2.5,1.2],gap="small")
geo_scale = cols[0].radio("Choose geographic scale", ["province (ibbs3)", "sub-region (ibbs2)", "region (ibbs1)", "district"])
if geo_scale == "district":
    geo_scale = ["district"]
    geo_scale_code = None
else:
    geo_scale_code = geo_scale[geo_scale.index("(")+1:geo_scale.index(")")]
    geo_scale = [geo_scale[:geo_scale.index("(")].strip()]


st.markdown("""<style> [role=radiogroup]{ gap: .8rem; } </style>""", unsafe_allow_html=True)
cols[1].radio("Age group selection", ["Custom selection","Quick selection for total age dependency ratio",
                                     "Quick selection for child age dependency ratio","Quick selection for old-age dependency ratio"],
                                    key="sex_age_age_group_selection")
cols[2].image("images/age-dependency.jpg")

page_name = "sex_age"
cols_nom_denom = ui_basic_setup_common(num_sub_cols=2)
df_data, gdf_borders, age_group_keys = get_data(geo_scale_code)
sidebar_controls(df_data["nominator"].index.get_level_values(0).min(),df_data["nominator"].index.get_level_values(0).max() )

if page_name+'_age_checkbox_group' not in st.session_state:
    st.session_state[page_name+"_age_checkbox_group"] = None
Checkbox_Group.age_group_quick_select(page_name, age_group_keys)


selected_features={}
for nom_denom in cols_nom_denom.keys():
    selected_features[nom_denom] = (feature_choice(cols_nom_denom[nom_denom][0], "sex", nom_denom),
                                    feature_choice(cols_nom_denom[nom_denom][1], "age", nom_denom, 4, age_group_keys, page_name) )# Page

st.write("""<style>[data-testid="stHorizontalBlock"]{align-items: center;}</style>""",unsafe_allow_html=True)
col_plot, col_df = st.columns((4 ,1), gap="small")
plot(col_plot, col_df, df_data, gdf_borders, selected_features, geo_scale)

# Create a Folium map
# m = folium.Map(location=[10, 0], zoom_start=2)
# Add the GeoDataFrame to the map
# folium.GeoJson( df[["geometry","out-migration"]]  ).add_to(m)
# Add layer control
#  folium.LayerControl().add_to(m)
# Display the map in Streamlit
# folium_static(make_choropleth(df_data, selected_feature = "out-migration"))