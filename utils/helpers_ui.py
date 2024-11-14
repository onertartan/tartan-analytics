import locale
import streamlit as st
from matplotlib import pyplot as plt
locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')



def ui_basic_setup_names(df_data):
    page_name=st.session_state["page_name"]

    #basic_sidebar_controls(2018, 2023)
    col_1, col_2, col_3 = st.columns([1, 1, 3])

    if not st.session_state["clustering_cb_" + page_name]:
        st.session_state["select_both_sexes_" + page_name] = False

    if page_name == "names or surnames":
        name_surname = col_1.radio("Select name or surname", ["Name", "Surname"], key="name_surname_rb").lower()
        if name_surname == "surname":
            st.session_state["select_both_sexes_" + page_name] = True
        sex = col_2.radio("Choose sex", ["Male", "Female"], disabled=st.session_state["select_both_sexes_"+page_name],key="sex_"+page_name).lower()
    else:
        sex = col_1.radio("Choose sex", ["Male", "Female"], disabled=st.session_state["select_both_sexes_"+page_name],key="sex_"+page_name).lower()

    st.header(f"Clustering or Displaying {page_name}")
    col_1, col_2, col_3 = st.columns([1, 1, 3])

    clustering_cb = col_1.checkbox("Apply K-means clustering", key="clustering_cb_" + page_name)
    col_1.checkbox("Select both sexes", disabled=not st.session_state["clustering_cb_" + page_name], key="select_both_sexes_"+page_name)
    col_1.selectbox("Select K: number of clusters", range(2, 11),key="n_clusters_"+page_name)
    disable_when_clustering = True if clustering_cb else False

    choice = col_3.radio("Or select an displaying option",
                            [f"Show the most common {page_name}", f"Select {page_name} and top-n number to filter"],
                            horizontal=True, disabled=disable_when_clustering, key="secondary_option_" + page_name)

    disable_top_n_selection = not disable_when_clustering and choice == f"Show the most common {page_name}"

    col_3.selectbox("Select a number N to check if the entered name is among the top N names",
                                         options=list(range(1, 31)), disabled=disable_top_n_selection,
                                         key="top_n_selection_" + page_name)

    sex_or_surname = "surname" if page_name == "Surname" and st.session_state["name_surname_rb"] == "Surname" else sex
    col_3.multiselect("Select name(s). By default most popular name is shown.",
                                      sorted(df_data[sex_or_surname]["name"].unique(), key=locale.strxfrm), disabled=disable_when_clustering,key="names_" + page_name)



