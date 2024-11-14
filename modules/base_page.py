import streamlit as st
from matplotlib import pyplot as plt

import extra_streamlit_components as stx
from utils.plot_map_common import plot
from abc import ABC
import geopandas as gpd
import pandas as pd

class BasePage(ABC):
    features = None
    page_name = None
    geo_scales = ["province (ibbs3)", "sub-region (ibbs2)", "region (ibbs1)", "district"]
    top_row_cols=[]
    checkbox_group = {}

    @staticmethod
    def convert_year_index_data_type(df):
        """Not all data years are integers.In particular elections data of 2015 June and 2015 November are not string"""
        return pd.MultiIndex.from_arrays([ df.index.get_level_values(0).astype(str),  df.index.get_level_values(1)], names=df.index.names)

    @classmethod
    def fun_extras(cls):
        pass

    @staticmethod
    def get_data(geo_scale=None):
        pass
    @staticmethod
    def get_geo_data(geo_scale=None):
        if "district" in geo_scale:
            return gpd.read_file("data/preprocessed/gdf_borders_district.geojson")
        else:
            return gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")

    @classmethod
    def render(cls):
        st.markdown("""<style> .main > div { padding-left:1rem;padding-right:1rem;padding-top: 2rem; }</style>""", unsafe_allow_html=True)

        geo_scale = cls.top_row_cols[0].radio("Choose geographic scale",cls.geo_scales)

        if geo_scale != "district":
            geo_scale = geo_scale.split()[0]

        st.markdown("""<style> [role=radiogroup]{ gap: .8rem; } </style>""", unsafe_allow_html=True)
        cls.fun_extras( )# for optional columns at the top row

        cols_nom_denom = cls.ui_basic_setup_common()
        df_data = cls.get_data(geo_scale)
        gdf_borders = cls.get_geo_data(geo_scale)

        cls.sidebar_controls(start_year=df_data["nominator"].index.get_level_values(0).min(),end_year= df_data["nominator"].index.get_level_values(0).max())

        selected_features = cls.get_selected_features(cols_nom_denom)

        st.write("""<style>[data-testid="stHorizontalBlock"]{align-items: top;}</style>""", unsafe_allow_html=True)
        col_plot, col_df = st.columns((4, 1), gap="small")
        plot(col_plot, col_df, df_data, gdf_borders, selected_features, [geo_scale])

    @classmethod
    def run(cls):
        st.session_state["page_name"] = cls.page_name
        cls.render()

    @classmethod
    def get_selected_features(cls,cols_nom_denom):
        selected_features = {}
        for nom_denom in cols_nom_denom.keys():  # cols_nom_denom is a dict where keys nominator or denominator,values are st.columns
            selected_features[nom_denom] = ()  # tuple type is needed for multiindex columns
            for i, feature in enumerate(cls.features[nom_denom]):
                selected_features[nom_denom] = selected_features[nom_denom] + (
                cls.get_selected_feature_options(cols_nom_denom[nom_denom][i], feature, nom_denom),)
        #    Checkbox_Group.age_group_quick_select()

        return  selected_features

    @classmethod
    def get_selected_feature_options(cls, col, feature_name, nom_denom_key_suffix):
        disabled = not st.session_state["display_percentage"] if nom_denom_key_suffix == "denominator" else False
        if feature_name in ["marital_status","education","age","Party/Alliance","sex"]:
            cls.checkbox_group[feature_name].place_checkboxes(col, nom_denom_key_suffix, disabled, feature_name )
            selected_feature = cls.checkbox_group[feature_name].get_checked_keys(nom_denom_key_suffix, feature_name)
        return selected_feature

    @staticmethod
    def update_selected_slider_and_years(slider_index):
        st.session_state.selected_slider = slider_index
        if slider_index == 1:
            st.session_state["year_1"]= st.session_state["year_2"] = int(st.session_state.slider_year_1)
        else:
            st.session_state["year_1"], st.session_state["year_2"] = int(st.session_state.slider_year_2[0]), int(
                st.session_state.slider_year_2[1])



    @classmethod
    def ui_basic_setup_common(cls):

        st.markdown(""" <style>[role=checkbox]{ gap: 1rem; }</style>""", unsafe_allow_html=True)

        st.markdown("""
                <div class="top-align">
                    <style>              
                        .top-align [data-testid="stHorizontalBlock"] {
                            align-items: flex-start;
                        }

                    </style>
                </div>
            """, unsafe_allow_html=True)

        cols_title = st.columns(2)
        cols_title[0].markdown("<h3 style='color: red;'>Select primary parameters.</h3>", unsafe_allow_html=True)
        cols_title[0].markdown("<br><br><br>", unsafe_allow_html=True)

        # Checkbox to switch between population and percentage display
        cols_title[1].checkbox(":blue[Select secondary parameters](check to get proportion).", key="display_percentage")
        cols_title[1].write("Ratio: primary parameters/secondary parameters.")
        cols_title[1].write("Uncheck to show counts of primary parameters.")


        cols_all = st.columns(cls.col_weights)  # There are 2*n columns(for example: 3 for nominator,3 for denominator)
        with cols_all[len(cols_all) // 2]:
            st.html(
                '''
                    <div class="divider-vertical-line"></div>
                    <style>
                        .divider-vertical-line {
                            border-left: 1px solid rgba(49, 51, 63, 0.2);
                            height: 320px;
                            margin: auto;
                        }
                    </style>
                '''
            )
        cols_nom_denom = {"nominator": cols_all[0:len(cls.col_weights)//2], "denominator": cols_all[len(cls.col_weights)//2 + 1:]}
        return cols_nom_denom

    @classmethod
    def basic_sidebar_controls(cls,start_year, end_year):
        """
           Renders the sidebar controls
           Parameters: starting year, ending year
        """
        # Inject custom CSS to set the width of the sidebar
        #   st.markdown("""<style>section[data-testid="stSidebar"] {width: 300px; !important;} </style> """,  unsafe_allow_html=True)
        with (st.sidebar):
            st.header('Visualization options')
            print("nmme:",cls.page_name)
            options= list(range(start_year, end_year + 1)) if cls.page_name!="sex_age_edu_elections" else [2018,2023]

            # Create a slider to select a single year
            st.select_slider("Select a year", options, 2023,
                             on_change=BasePage.update_selected_slider_and_years, args=[1],
                             key="slider_year_1")
            # Create sliders to select start and end years
            st.select_slider("Or select start and end years",options, [options[0],options[-1]],
                             on_change=BasePage.update_selected_slider_and_years, args=[2], key="slider_year_2")

            if "selected_slider" not in st.session_state:
                st.session_state["selected_slider"] = 1
            BasePage.update_selected_slider_and_years(st.session_state.selected_slider)

            if st.session_state.selected_slider == 1:
                st.write("Single year is selected from the first slider.")
                st.write("Selected year:", st.session_state["year_1"])
            else:
                st.write("Start and end years are selected from the second slider.")
                st.write("Selected start year:", st.session_state["year_1"], "\nSelected end year:",
                         st.session_state["year_2"])

            if "selected_tab" not in st.session_state:
                st.session_state["selected_tab"] = "tab_map"
            tabs = [stx.TabBarItemData(id="tab_map", title="Map/Race plot", description="")]
            if st.session_state["page_name"] in ["sex_age", "marital_status"]:
                st.session_state["selected_tab"] = tabs.append(
                    stx.TabBarItemData(id="tab_pyramid", title="Pop. Pyramid", description=""))

            #   tab_map, tab_pyramid= st.tabs(["Map", "Population pyramid"])
            st.session_state["selected_tab"] = stx.tab_bar(data=tabs, default="tab_map")

            if st.session_state["selected_tab"] == "tab_map":
                if "visualization_option" not in st.session_state:
                    st.session_state["visualization_option"] = "Matplotlib (static)"
                st.session_state["visualization_option"] = st.radio("Choose visualization option",
                                                                    ["Matplotlib (static)", "Folium (static)",
                                                                     "Folium interactive", "Plotly race chart"])
            else:
                st.session_state["visualization_option"] = st.radio("Choose visualization option",
                                                                    ["Matplotlib", "Plotly"])
    @classmethod
    def sidebar_controls(cls,start_year, end_year):  # start_year=2007,end_year=2023

        cls.basic_sidebar_controls(start_year, end_year)
        with st.sidebar:
            # Dropdown menu for colormap selection
            st.selectbox("Select a colormap\n (colormaps in the red boxes are available only for matplotlib) ",  st.session_state["colormap"][st.session_state["visualization_option"]], key="selected_cmap")
            # Display a static image
            st.image("images/colormaps.jpg")