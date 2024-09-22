import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import altair as alt
import plotly.express as px

st.set_page_config(page_title="Tartan Analytics", layout="wide")


current_page = st.navigation({
    "Population": [
        st.Page("demography-modules/population/sex-age.py", title="Sex-Age ", icon=":material/public:"),
        st.Page("demography-modules/population/maritial_status.py", title="Sex-Age-Marital status(over age 15) ", icon=":material/wc:"),
        st.Page("demography-modules/population/most_common_baby_names.py", title="Most Common Baby Names ", icon=":material/public:"),
        st.Page("demography-modules/population/most_common_names_surnames.py", title="Most Common Names and Surnames", icon=":material/public:"),
        st.Page("demography-modules/population/birth.py", title="Birth",icon=":material/public:")],
   # "Marriage": [       ],
})

# current_page is also a Page object you can .run()
current_page.run()
