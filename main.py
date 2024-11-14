import streamlit as st

st.set_page_config(page_title="Tartan Analytics", layout="wide")

# if "age_group_keys" not in st.session_state:
#     st.session_state["age_group_keys"] = {"marital_status":["all"] + [f"{i}-{i + 4}" for i in range(15, 90, 5)] + ["90+"],
#                                         "sex_age":["all"]+[f"{i}-{i+4}" for i in range(0, 90, 5)]+["90+"] ,
#                                           "sex_age_edu_elections":["all"]+["18-24"]+[f"{i}-{i+4}" for i in range(25, 75, 5)]+["75+"] }
#     st.session_state["age_group_keys"]["birth"]=st.session_state["age_group_keys"]["marital_status"]

current_page = st.navigation({
    "Population": [
        st.Page("modules/population/sex_age.py", title="Sex-Age ", icon=":material/public:"),
        st.Page("modules/population/marital_status.py", title="Sex-Age-Marital status(over age 15) ", icon=":material/wc:"),
        st.Page("modules/population/baby_names.py", title="Most Common Baby Names ", icon=":material/public:"),
        st.Page("modules/population/names_surnames.py", title="Most Common Names and Surnames", icon=":material/public:"),
        st.Page("modules/population/birth.py", title="Birth", icon=":material/public:")],
    "Elections": [ st.Page("modules/elections/sex_age_edu.py", title="Sex-Age-Edu ", icon=":material/public:"),
                   st.Page("modules/elections/election_correlation.py", title="CORR ", icon=":material/public:")]})

if "colormap_list" not in st.session_state:
    st.session_state["colormap"] = {}
    # folium (static)
    st.session_state["colormap"]["Folium (static)"] = ["PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral", "Reds",
                                          "Purples", "Blues", "Greens", "Oranges", "Greys", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn",
                                          "BuGn", "YlGn", "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3"]
    # matplotlib
    st.session_state["colormap"]["Matplotlib (static)"] = st.session_state["colormap"]["Folium (static)"] + ["bwr", "coolwarm", "seismic", "tab10", "tab20", "tab20b", "tab20c"]
    st.session_state["colormap"]["Folium interactive"] = st.session_state["colormap"]["Folium (static)"]
# current_page is also a Page object you can .run()
current_page.run()


