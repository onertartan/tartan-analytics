import streamlit as st
from utils.checkbox_group import Checkbox_Group
from utils.dashboard_components import sidebar_controls
from utils.helpers_ui import ui_basic_setup_common
from utils.plot_map_common import plot
from abc import ABC
from utils.helpers_common import feature_choice
import geopandas as gpd

class BasePage(ABC):
    features = None
    page_name = None
    @classmethod
    def set_features(cls, features):
        cls.features = features

    @classmethod
    def set_page_name(cls, page_name):
        cls.page_name = page_name
    @staticmethod
    def get_data(geo_scale):
        pass

    @staticmethod
    def fun_extras(cols):
        pass

    @staticmethod
    def get_geo_data(geo_scale):
        if "district" in geo_scale:
            return gpd.read_file("data/preprocessed/gdf_borders_district.geojson")
        else:
            return gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")

    @classmethod
    def render(cls):
        st.markdown("""<style> .main > div { padding-top: 1rem; }</style>""", unsafe_allow_html=True)
        cols = st.columns([1, 2.2, 2.5, 1.2], gap="small")
        geo_scale = cols[0].radio("Choose geographic scale",
                                  ["province (ibbs3)", "sub-region (ibbs2)", "region (ibbs1)", "district"])
        if geo_scale != "district":
            geo_scale = geo_scale.split()[0]

        st.markdown("""<style> [role=radiogroup]{ gap: .8rem; } </style>""", unsafe_allow_html=True)
        cls.fun_extras(cols)


        st.session_state["page_name"] = cls.page_name
        cols_nom_denom = ui_basic_setup_common(num_sub_cols=len(cls.features["denominator"]))

        df_data = cls.get_data(geo_scale)
        gdf_borders = cls.get_geo_data(geo_scale)

        sidebar_controls(df_data["nominator"].index.get_level_values(0).min(), df_data["nominator"].index.get_level_values(0).max())
        Checkbox_Group.age_group_quick_select()

        selected_features = {}
        for nom_denom in cols_nom_denom.keys():
            selected_features[nom_denom] = ()# tuple type is needed for multiindex columns
            for i, feature in enumerate(cls.features[nom_denom]):
                selected_features[nom_denom] =   selected_features[nom_denom] +(feature_choice(cols_nom_denom[nom_denom][i], feature, nom_denom),)

        st.write("""<style>[data-testid="stHorizontalBlock"]{align-items: top;}</style>""", unsafe_allow_html=True)
        col_plot, col_df = st.columns((4, 1), gap="small")
        plot(col_plot, col_df, df_data, gdf_borders, selected_features, [geo_scale])

    @classmethod
    def run(cls):
        cls.render()
