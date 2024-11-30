import inspect

import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Tartan Analytics", layout="wide")


current_page = st.navigation({
    "Population": [
        st.Page("modules/population/baby_names.py", title="Baby Names ", icon=":material/public:"),
        st.Page("modules/population/birth.py", title="Birth", icon=":material/public:"),
        st.Page("modules/population/death_month.py", title="Death-Month", icon=":material/public:"),
        st.Page("modules/population/names_surnames.py", title="Names and Surnames", icon=":material/public:"),
        st.Page("modules/population/sex_age.py", title="Sex-Age ", icon=":material/public:"),
        st.Page("modules/population/marital_status.py", title="Sex-Age-Marital status(over age 15) ", icon=":material/wc:")],
    "Elections": [ st.Page("modules/elections/sex_age_edu.py", title="Sex-Age-Edu ", icon=":material/public:"),
                   st.Page("modules/elections/election_correlation.py", title="CORR ", icon=":material/public:")]})

if "colormap_list" not in st.session_state:
    st.session_state["colormap"] = {}
    # folium (static)
    st.session_state["colormap"]["Folium"] =   st.session_state["colormap"]["Folium-interactive"] = ["PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral", "Reds",
                                          "Purples", "Blues", "Greens", "Oranges", "Greys", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn",
                                          "BuGn", "YlGn", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3"]
    # matplotlib
    st.session_state["colormap"]["Matplotlib"] = st.session_state["colormap"]["Folium"] + ["bwr", "coolwarm", "seismic", "tab10", "tab20", "tab20b", "tab20c"]
   # st.session_state["colormap"]["Folium interactive"] = st.session_state["colormap"]["Folium (static)"]
    # Get all colormap names
    colorscale_names = []
    colors_modules = ['carto', 'colorbrewer', 'cmocean', 'cyclical',
                      'diverging', 'plotlyjs', 'qualitative', 'sequential']
    for color_module in colors_modules:
        colorscale_names.extend([name for name, body
                                 in inspect.getmembers(getattr(px.colors, color_module))
                                 if isinstance(body, list)])
    st.session_state["colormap"]["Plotly"] = colorscale_names

if "animate" not in st.session_state:
    st.session_state["animate"] = False

if "elbow" not in st.session_state:
    st.session_state["elbow"] = False
    # Combine them into a single list
# current_page is also a Page object you can .run()
current_page.run()


