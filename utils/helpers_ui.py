import locale
import streamlit as st
from matplotlib import pyplot as plt
from utils.dashboard_components import basic_sidebar_controls
locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')



def ui_basic_setup_common(num_sub_cols):

    st.markdown( """<style>.block-container{padding-top: 2rem; padding-left: 1rem; padding-right: 2rem; padding-bottom: 0rem; margin-top: 2rem;}</style>""",
        unsafe_allow_html=True)
    st.markdown(""" <style>[role=checkbox]{ gap: 3rem; }</style>""", unsafe_allow_html=True)

    cols_title = st.columns(2)
    cols_title[0].write(":red[Select primary parameters.]")

    # Checkbox to switch between population and percentage display
    cols_title[1].checkbox("Check to get proportion.", key="display_percentage")
    cols_title[1].write("Ratio: primary parameters/secondary parameters.")
    cols_title[1].write("Uncheck to show counts of primary parameters.")

    cols_title[1].write(":blue[Select secondary parameters.]")
    if num_sub_cols == 3:
        col_weights = [1, 1, 3, 1, 1, 3]
    elif num_sub_cols == 2:
        col_weights = [1, 2, 1, 2]

    cols_all = st.columns(col_weights)  # There are 6 columns(3 for nominator,3 for denominator)
    cols_nom_denom = {"nominator": cols_all[0:num_sub_cols], "denominator":cols_all [num_sub_cols:]}
    return cols_nom_denom


def ui_basic_setup_names(page_name, df_data):
    basic_sidebar_controls(2018, 2023)
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


def figure_setup(display_change=False):
    if st.session_state["year_1"] == st.session_state["year_2"] or st.session_state["selected_slider"] == 1:
        n_rows = 1
    elif display_change:
        n_rows = 3
    else:
        n_rows = 2
    fig, axs = plt.subplots(n_rows, 1, squeeze=False, figsize=(10, 4 * n_rows), gridspec_kw={'wspace': 0, 'hspace':0.1})  # axs has size (3,1)
    #fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=0.5, hspace=0.5)

    fig.tight_layout()
    return fig, axs
