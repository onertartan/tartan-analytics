import os
import time
import streamlit as st
import extra_streamlit_components as stx
from PIL import Image
from sklearn.cluster import KMeans

from utils.plot_map_common import plot, figure_setup, delete_temp_files, analyse
from abc import ABC
import geopandas as gpd
import pandas as pd
from utils.query import get_df_result

class BasePage(ABC):
    features = None
    page_name = None
    correlation_analysis = False
    animation_available = True
    geo_scales = ["province (ibbs3)", "sub-region (ibbs2)", "region (ibbs1)", "district"]
    top_row_cols = []
    checkbox_group = {}

    @staticmethod
    def convert_year_index_data_type(df):
        """Not all data years are integers.In particular elections data of 2015 June and 2015 November are not string"""
        return pd.MultiIndex.from_arrays([df.index.get_level_values(0).astype(str),  df.index.get_level_values(1)], names=df.index.names)

    @classmethod
    def fun_extras(cls, *args):
        pass

    @staticmethod
    @st.cache_data
    def get_data(geo_scale=None):
        pass


    @staticmethod
    @st.cache_data
    def get_geo_data():
        geo_data_dict={}
        print("ÇŞÇ",st.session_state["geo_scale"])
        geo_data_dict["district"] =  gpd.read_file("data/preprocessed/gdf_borders_district.geojson")
        print("ZXCVB")
        geo_data_dict["province"] = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
        return geo_data_dict

    @classmethod
    def render(cls):
        st.markdown("""<style> .main > div {padding-left:1rem;padding-right:1rem;padding-top:4rem;}</style>""", unsafe_allow_html=True)
        st.session_state["geo_scale"] = cls.top_row_cols[0].radio("Choose geographic scale",cls.geo_scales).split()[0]
        print("ÇÇÇ:",st.session_state["geo_scale"])
        st.markdown("""<style> [role=radiogroup]{ gap: 0rem; } </style>""", unsafe_allow_html=True)
        cls.fun_extras() # for optional columns at the top row
        cols_nom_denom = cls.ui_basic_setup()
        # get cached data
        df_data = cls.get_data()
        print("0. çekpoint",df_data["denominator"]["district"])
        geo_scale = "province" if st.session_state["geo_scale"]!="district" else "district"
        gdf_borders = cls.get_geo_data()[geo_scale]


        start_year = df_data["nominator"][geo_scale].index.get_level_values(0).min()
        end_year = df_data["nominator"][geo_scale].index.get_level_values(0).max()
        cls.sidebar_controls(start_year, end_year)
        st.write("""<style>[data-testid="stHorizontalBlock"]{align-items: top;}</style>""", unsafe_allow_html=True)

        with st.form("submit_form"):
            (col_show_results, col_animation) = st.columns(2)
            show_results = col_show_results.form_submit_button("Show results")
            if cls.animation_available:
                play_animation = col_animation.form_submit_button("Play animation")
                col_animation.write("Animation Controls")
                col_animation.slider("Animation Speed (seconds)", min_value=0.5, max_value=5., value=1., step=1., key="animation_speed")
                col_animation.checkbox("Auto-play", value=True,key="auto_play")
                if play_animation:
                    st.session_state["animate"] = True
                else:
                    st.session_state["animate"] = False

            # Run on first load OR when form is submitted
            selected_features = cls.get_selected_features(cols_nom_denom)
            if show_results or (cls.animation_available and play_animation):
            #    st.session_state["animation_images_generated"] = False
              #  delete_temp_files()
                col_plot, col_df = st.columns((4, 1), gap="small")
                print("df_data : LOL",df_data["denominator"]["district"])
                plot(col_plot, col_df, df_data, gdf_borders, selected_features, [st.session_state["geo_scale"]])


    @classmethod
    def run(cls):
        st.session_state["page_name"] = cls.page_name
        cls.render()

    @classmethod
    def get_selected_features(cls, cols_nom_denom):
        selected_features = {}
        for nom_denom in cols_nom_denom.keys():  # cols_nom_denom is a dict whose keys are "nominator" or "denominator" and values are st.columns
            selected_features[nom_denom] = ()  # tuple type is needed for multiindex columns
            for i, feature in enumerate(cls.features[nom_denom]):
                selected_features[nom_denom] = selected_features[nom_denom] + ( cls.get_selected_feature_options(cols_nom_denom[nom_denom][i], feature, nom_denom),)
        #    Checkbox_Group.age_group_quick_select()
        return selected_features

    @classmethod
    def get_selected_feature_options(cls, col, feature_name, nom_denom_key_suffix):
        disabled = not st.session_state["display_percentage"] if nom_denom_key_suffix == "denominator" else False
        if feature_name in ["marital_status", "education", "age", "Party/Alliance", "sex", "month"]:
            cls.quick_selection(feature_name,nom_denom_key_suffix)
            cls.checkbox_group[feature_name].place_checkboxes(col, nom_denom_key_suffix, disabled, feature_name)

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
    def ui_basic_setup(cls):

       # st.markdown(""" <style>[role=checkbox]{ gap: 1rem; }</style>""", unsafe_allow_html=True)

    #    st.markdown("""
    #            <div class="top-align">
     #               <style>
     #                  .top-align [data-testid="stHorizontalBlock"] {
      #                      align-items: flex-start;
       #                 }
      #              </style>
        #        </div>
       #     """, unsafe_allow_html=True)

        cols_title = st.columns(2)
        cols_title[0].markdown("<h3 style='color: red;'>Select primary parameters.</h3>", unsafe_allow_html=True)
        cols_title[0].markdown("<br><br><br>", unsafe_allow_html=True)

        # Checkbox to switch between population and percentage display

        cols_title[1].markdown("<h3 style='color: blue;'>Select secondary parameters.</h3>", unsafe_allow_html=True)
        cols_title[1].checkbox("Check to get ratio: primary parameters/secondary parameters.", key="display_percentage")
        cols_title[1].write("Uncheck to show counts of primary parameters.")

        cols_all = st.columns(cls.col_weights)  # There are 2*n columns(for example: 3 for nominator,3 for denominator)
        with cols_all[len(cols_all) // 2]:
            st.html(
                '''
                    <div class="divider-vertical-line"></div>
                    <style>
                        .divider-vertical-line {
                            border-left: 1px solid rgba(49, 51, 63, 0.2);
                            height: 180px;
                            margin: auto;
                        }
                    </style>
                '''
            )
        cols_nom_denom = {"nominator": cols_all[0:len(cls.col_weights)//2], "denominator": cols_all[len(cls.col_weights)//2 + 1:]}

        st.divider()
        cols_clustering = st.columns([2, 1, 5])
        cols_clustering[0].checkbox("Apply K-means clustering", key="clustering_cb_" + cls.page_name)
        cols_clustering[0].write("In K-means clustering primary parameters are not aggregated, instead they are used as individual features.")
        cols_clustering[1].selectbox("Select K: number of clusters", range(2, 15), key="n_clusters_" + cls.page_name)
        cols_clustering[2].checkbox("Elbow Method", key="elbow")
        return cols_nom_denom

    @classmethod
    def animation_slider_changed(cls):
        st.session_state["animate"]=True

    @staticmethod
    def sidebar_controls_basic_setup( *args):
        """
           Renders the sidebar controls
           Parameters: starting year, ending year
        """
        # Inject custom CSS to set the width of the sidebar
        #   st.markdown("""<style>section[data-testid="stSidebar"] {width: 300px; !important;} </style> """,  unsafe_allow_html=True)
        print("ARGS::",args)
        start_year = args[0]
        end_year = args[1]
        with (st.sidebar):
            st.header('Visualization options')
            st.write(start_year,end_year)
            # if ifadesine gerek olmadığı düşünülerek (hata olursa bu if kalktığı için olabilir) metot classmethod'dan static'e dönüştü. Böylelikle higher-education kullanabildi.
            # options= list(range(start_year, end_year + 1)) if cls.page_name != "sex_age_edu_elections" else [2018,2023]
            options = list(range(start_year, end_year + 1))
            # Create a slider to select a single year
            st.select_slider("Select a year", options, 2023, on_change=BasePage.update_selected_slider_and_years, args=[1],  key="slider_year_1")
            # Create sliders to select start and end years
            st.select_slider("Or select start and end years",options, [options[0],options[-1]],on_change=BasePage.update_selected_slider_and_years, args=[2], key="slider_year_2")

            # Main content
            if "animation_images_generated" not in st.session_state:
                st.session_state["animation_images_generated"] = False

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

    @classmethod
    def sidebar_controls_plot_options_setup(cls, *args):
        with (st.sidebar):
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
                    st.session_state["visualization_option"] = "Matplotlib"
                st.session_state["visualization_option"] = st.radio("Choose visualization option",
                                                                    ["Matplotlib (static)", "Folium (static)",
                                                                     "Folium-interactive", "Raceplotly"]).split(" ")[0]
            else:
                st.session_state["visualization_option"] = st.radio("Choose visualization option",
                                                                    ["Matplotlib", "Plotly"]).split(" ")[0]

    @classmethod
    def sidebar_controls(cls,*args):  # start_year=2007,end_year=2023
        cls.sidebar_controls_basic_setup(*args)
        if cls.page_name != "names_surnames" and cls.page_name != "baby_names":
            cls.sidebar_controls_plot_options_setup(*args)

        if st.session_state["visualization_option"] != "Raceplotly":
            with st.sidebar:
                # Dropdown menu for colormap selection
                st.selectbox("Select a colormap\n (colormaps in the red boxes are available only for matplotlib) ",  st.session_state["colormap"][st.session_state["visualization_option"]], key="selected_cmap")
                # Display a static image
                st.image("images/colormaps.jpg")

    @classmethod
    def quick_selection(cls,feature_name,nom_denom_key_suffix):
        pass

    @classmethod
    def set_checkbox_values_for_quick_selection(cls, keys_to_check, nom_denom_key_suffix, feature_name):
        print("###",keys_to_check)
        for key in cls.checkbox_group[feature_name].basic_keys:
            if key in keys_to_check:
                val = True
            else:
                val = False
            st.session_state[cls.page_name+"_"+nom_denom_key_suffix+"_"+feature_name+"_"+key]=val
            #cls.checkbox_group[feature_name].checked_dict[nom_denom_key_suffix][key] = val
       #1 print("$$$",nom_denom_key_suffix,"$$$",cls.checkbox_group[feature_name].checked_dict[nom_denom_key_suffix])

    @classmethod
    def k_means_clustering(cls,col_plot, col_df, df_result, gdf_borders, geo_scale):
        df_result.to_excel("temp.xlsx")
        kmeans = KMeans(n_clusters=st.session_state["n_clusters_" + cls.page_name], random_state=0).fit(df_result.iloc[:,5:])#cluster feature columns(cols 0-4 are descriptive cols)
        df_result["clusters"] = kmeans.labels_
        gdf_borders = gdf_borders.merge(df_result["clusters"],left_on=geo_scale,right_on=geo_scale)#, left_on="province", right_on=gdf_borders.index)
        # Define a color map for the categories
        # color_map = {0: "purple", 1: "orange", 2: "green",3:"cyan",4:"red",5:"blue",6:"magenta",7:"gray",8:"yellow"}#female
        # color_map = {0: "orange", 1: "orange", 2: "red",3:"red",4:"orange",5:"magenta",6:"red",7:"orange",8:"orange"}#male-8
        #  color_map = {0: "purple", 1: "red", 2: "orange",3:"red",4:"orange",5:"magenta",6:"cyan",7:"yellow",8:"gray"}#female-5
        # color_map = {0: "purple", 1: "orange", 2: "orange",3:"red",4:"orange",5:"red",6:"cyan",7:"yellow",8:"gray"}#female-6
        # color_map = {0: "orange", 1: "red", 2: "red",3:"red",4:"orange",5:"magenta",6:"magenta",7:"orange",8:"gray"}#total-8
        # color_map = {0: "red", 1: "purple", 2: "orange",3:"green",4:"blue",5:"magenta",6:"cyan",7:"yellow",8:"gray"}#original
        # color_map = {0: "orange", 1: "orange", 2: "purple",3:"red",4:"red",5:"red",6:"purple",7:"yellow",8:"gray"}#total-7
        # color_map = {0: "orange", 1: "orange", 2: "purple",3:"red",4:"red",5:"red",6:"cyan",7:"yellow",8:"gray"} #total-6
        # color_map = {0: "orange", 1: "red", 2: "orange",3:"red",4:"orange",5:"magenta",6:"red",7:"orange",8:"orange"}#female-9
        color_map = {0: "red", 1: "purple", 2: "orange", 3: "green", 4: "blue", 5: "magenta", 6: "cyan", 7: "yellow", 8: "gray"}  # original
        color_map = {0: "red", 1: "purple", 2: "orange", 3: "green", 4: "blue", 5: "magenta", 6: "cyan", 7: "yellow",
                     8: "gray",9:"white",10:"black",11:"pink",12:"brown",13:"lightgreen"}  # original
        #  color_map = {0: "orange", 1: "orange", 2: "red",3:"red",4:"orange",5:"magenta",6:"red",7:"yellow",8:"gray"}
        #   color_map = {0: "red", 1: "orange", 2: "red",3:"red",4:"orange",5:"red",6:"red",7:"yellow",8:"gray"}
        # Map the colors to the GeoDataFrame
        gdf_borders["color"] = gdf_borders["clusters"].map(color_map)
        return  gdf_borders
       # print("csq:", df_result["clusters"] ,gdf_borders)
        #fig, axs = figure_setup()
        #gdf_borders.plot(ax=axs[0, 0], color=gdf_borders['color'], legend=True, edgecolor="black", linewidth=.2)
       # axs[0, 0].axis("off")
        #axs[0, 0].margins(x=0)
        #col_plot.pyplot(fig)

