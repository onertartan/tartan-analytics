# from pycirclize import Circos
# from pycirclize.parser import Matrix
import branca.colormap as cm
import folium
from matplotlib.colors import to_hex
from streamlit_folium import folium_static

from modules.base_page import BasePage
import locale
import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import extra_streamlit_components as stx
import numpy as np
from streamlit.components.v1 import html
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from viz.bar_plotter import get_plotter

locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')


class QuotaOccupancy:
    page_name = "edu_migration"

    @staticmethod
    @st.cache_data
    def load_process_data(incoming_or_outgoing="outgoing"):
        df = pd.read_pickle('data/preprocessed/high_edu.pkl')
        df = df["General"]
        df = df[df["quota"] != 0]  # Drop departments with quotas 0(due to the existence in 2025 OSYM catalog)
        #  exclude distant education
        df = df[(df["scholarship"] != "AÖ-Ücretli") & (df["scholarship"] != "UÖ-Ücretli")]  # exclude distant education
        df.loc[:, "uni_type"] = df.loc[:, "uni_type"].map({'Devlet': 'State', 'Vakıf': 'Foundation'})
        scholarship_map = {'Ücretsiz': 'Tuition-Free', '%50 İndirimli': '50% Discounted', 'Burslu': 'Full Scholarship',
                           'Ücretli': 'Paid', '%25 İndirimli': '25% Discounted'}
        df.rename(columns={"entrance_score_type": "Score Type", "city": "province"}, inplace=True)
        df.loc[:, 'scholarship'] = df.loc[:, 'scholarship'].map(scholarship_map)

        df = df.loc[:, ["uni_type", "Score Type", "province", "uni_name", "dep_name", "scholarship", "quota", "placements",  "not_registered"]]
        df_quota = df.groupby(["uni_type", "Score Type", "province", "uni_name", "dep_name", "scholarship"]).sum()
        df_quota = df_quota.sort_index(key=lambda x: pd.Index([locale.strxfrm(e) for e in x]))

        df_years_universities = pd.read_csv("data/preprocessed/higher-education/df_years_universities.csv",
                                            usecols=["city", "uni_name", "foundation_year", "type", "region", "year"])
        df_years_universities = df_years_universities.rename(columns={"city": "province"})
        df_years_universities = df_years_universities[df_years_universities["year"] == 2024]
        df_years_universities.drop("year", axis=1, inplace=True)
        all_provinces = df_years_universities["province"].unique()
        foundation_provinces = df_years_universities.loc[df_years_universities["type"] == "foundation", "province"].unique()

        print("ALL PROVINCES ARE IN:",all_provinces)
        shapefile_path = "data/turkey_province_centers.geojson"
        gdf_centers = gpd.read_file(shapefile_path, encoding='utf-8')
        gdf = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")[["province", "geometry"]]
        gdf = gdf.merge(gdf_centers[["province", "lat", "lon"]], on="province", validate="1:1")

        return df_quota, df_years_universities, all_provinces, foundation_provinces, gdf
    def filter_data(self, df, df_years):
        # df --> dataframe of quotas with columns "uni_type", "Score Type", "province", "uni_name", "dep_name", "scholarship"

        uni_type = st.radio("Select University Type:", options=["State", "Foundation", "Both"])
       # if uni_type != "Both":
       #     df_years = df_years.loc[df_years["type"] == uni_type.lower(), :]
       # provinces_shown = df_years["province"].unique()
       #  universities = df_years["uni_name"].unique()


        col_score_type, col_provinces, col_universities, col_departments, col_scholarships = st.columns([1, 1, 2, 1, 1])
        score_type = st.radio("Select score type:", options=["SAY", "SÖZ", "EA", "DİL", "All"])
        st.dataframe(df.loc[("Foundation", slice(None), slice(None), slice(None), slice(None), slice(None)),:])

        if uni_type == "Both":
            uni_type = slice(None)
        df = df.loc[(uni_type, slice(None), slice(None), slice(None), slice(None), slice(None)),:]
        if score_type == "All":
            score_type = slice(None)
        df = df.loc[(slice(None), score_type, slice(None), slice(None), slice(None), slice(None)),:]

        provinces_shown = df.index._get_level_values(2).unique()
        provinces = col_provinces.multiselect("Select provinces to filter results", options=provinces_shown)
        if not provinces:
            provinces = provinces_shown
        df = df.loc[(slice(None), slice(None), provinces, slice(None), slice(None), slice(None)),:]

        universities_shown = list(set(df.index.get_level_values(3)).intersection(set(df_years["uni_name"].unique())))  # öğrenci almayan üniversiteler var.
      #  universities_shown= df.index._get_level_values(3).unique()
        universities = col_provinces.multiselect("Select universities to filter results", options=universities_shown)
        if not universities:
            universities = universities_shown
        df = df.loc[(slice(None), slice(None), slice(None), universities, slice(None), slice(None)),:]

        departments_shown = df.index.get_level_values(4).unique()
        departments = col_departments.multiselect("Select departments to filter results", options=departments_shown)
        if not departments:
            departments = departments_shown
        df = df.loc[(slice(None), slice(None), slice(None), slice(None), departments, slice(None)),:]

        scholarships_shown = df.index._get_level_values(5).unique()
        scholarships = col_scholarships.multiselect("Select scholarship type",options = scholarships_shown)
        if not scholarships:
            scholarships = scholarships_shown
        df = df.loc[(slice(None), slice(None), slice(None), slice(None), slice(None),scholarships ) ,:]
        return df
    def render(self):
        df_quota, df_years, all_provinces, foundation_provinces,gdf = QuotaOccupancy.load_process_data()

        show_only_new_provinces = st.checkbox("Only new provinces")
        message_slider_1 = "Select provinces where universities founded first time between the years  (inclusive)"
        message_slider_2 = "Select percentage of quota filled (inclusive)"
        year_min, year_max = df_years["foundation_year"].min(), df_years["foundation_year"].max()
        year_1, year_2 = st.slider(message_slider_1, year_min, year_max + 1 , (year_min, year_max) )
        min_percentage, max_percentage = st.slider(message_slider_2, 0, 100, (0, 100))
        if show_only_new_provinces:
            provinces_to_exclude = df_years.loc[df_years["foundation_year"] < year_1, "city"].unique()
            df_years = df_years[~df_years["city"].isin(provinces_to_exclude)]
            df_years = df_years[(year_1 <= df_years["foundation_year"]) & (df_years["foundation_year"] <= year_2)]
        df_years = df_years[(year_1 <= df_years["foundation_year"]) & (df_years["foundation_year"] <= year_2)]

        df_quota = self.filter_data(df_quota, df_years)
        # FILTERING ACCORDING TO PERCENTAGE INTERVAL
        df_quota_city = df_quota.groupby("province").sum()

        df_quota["percentage"] = 100*(df_quota["placements"]-df_quota["not_registered"])/df_quota["quota"]
        df_quota["percentage"] = df_quota["percentage"].round(decimals=2)
        df_quota = df_quota[(df_quota["percentage"] >= min_percentage) & (df_quota["percentage"] <= max_percentage)]
        df_quota_city["percentage"] = 100*(df_quota_city["placements"]-df_quota_city["not_registered"])/df_quota_city["quota"]
        df_quota_city["percentage"] = df_quota_city["percentage"].round(decimals=2)
        df_quota_city = df_quota_city[(df_quota_city["percentage"] >= min_percentage) & (df_quota_city["percentage"] <= max_percentage)]
        df_quota_city.index.name="province"
        df_quota_city = df_quota_city.reset_index().sort_values(by="percentage",ascending=False)
        print("df_quota_city columns after reset_index():")
        print(df_quota_city.columns.tolist(),gdf.columns)
        print("First few rows:")
        print(df_quota_city.head())
        st.dataframe(df_quota_city.head())
        gdf_quota_city = gdf.merge(df_quota_city, on="province", how="right")

        print("HEYT:",gdf_quota_city)
        self.plot_map_folium(gdf_quota_city)

        # ---- pick engine ----
        engine = st.selectbox("Chart library",  ["plotly", "matplotlib", "seaborn", "pandas", "altair"],)

        # ---- plot ----
        col_plot, _ = st.columns([10, 1])
        plotter = get_plotter(engine)
        df_quota.index.name = "province"
        df_quota = df_quota.sort_values("percentage", ascending=False)
        plotter.plot(df_quota_city, col_plot)
        st.dataframe(df_quota)


    def plot_map_folium(self, gdf_result, *args):
        # Create and display the map
        linear = cm.LinearColormap(["green", "yellow", "red"], vmin=gdf_result["percentage"].min(), vmax=100)
        explore_kwargs = {
            "tooltip": ["province", "percentage", "quota","placements","not_registered"],  # Specify exact columns
            "popup":  ["province", "percentage", "quota","placements","not_registered"],
            "tiles": "CartoDB positron",
            "column": "percentage",
            "style_kwds": dict(color="black", line_width=.01),
            "cmap":"viridis"
        }
        m = gdf_result.explore(**explore_kwargs)
        # Add province center markers
        # Add circle markers at province centers
        for idx, row in gdf_result.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=2,
                popup=f"{row['province']}: {row['percentage']}%",
                tooltip=row['province'],
                color='blue',
                fillColor='blue',
                fillOpacity=1,
                weight=1
            ).add_to(m)

        folium_static(m, width=1100, height=450)

    def plot_map(self, df):
        pass
    def run(self):
        self.render()


QuotaOccupancy().run()