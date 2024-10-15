import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import altair as alt
import plotly.express as px

from modules.population.page_sex_age import Page_Sex_Age

# set features KALDI!!!!! Page_Sex_Age.set_features(....)
st.set_page_config(page_title="Tartan Analytics", layout="wide")
if "page_name" not in st.session_state:
    st.session_state["page_name"]="sex_age"
if "age_group_keys" not in st.session_state:
    st.session_state["age_group_keys"] = {"marital_status":["all"] + [f"{i}-{i + 4}" for i in range(15, 90, 5)] + ["90+"],
                                        "sex_age":["all"]+[f"{i}-{i+4}" for i in range(0, 90, 5)]+["90+"] }
    st.session_state["age_group_keys"]["birth"]=st.session_state["age_group_keys"]["marital_status"]

current_page = st.navigation({
    "Population": [
        st.Page("modules/population/sex_age.py",title="Sex-Age ", icon=":material/public:"),
        st.Page("modules/population/marital_status.py", title="Sex-Age-Marital status(over age 15) ", icon=":material/wc:"),
        st.Page("modules/population/most_common_baby_names.py", title="Most Common Baby Names ", icon=":material/public:"),
        st.Page("modules/population/most_common_names_surnames.py", title="Most Common Names and Surnames", icon=":material/public:"),
        st.Page("modules/population/birth.py", title="Birth", icon=":material/public:")],
   # "Marriage": [       ],
})

# current_page is also a Page object you can .run()
current_page.run()
