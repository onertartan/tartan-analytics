import streamlit as st
import extra_streamlit_components as stx


def update_selected_slider_and_years(slider_index):
    st.session_state.selected_slider = slider_index
    if slider_index == 1:
        st.session_state.year_1 = st.session_state.year_2 = int(st.session_state.slider_year_1)
    else:
        st.session_state.year_1, st.session_state.year_2 = int(st.session_state.slider_year_2[0]), int(
            st.session_state.slider_year_2[1])


def basic_sidebar_controls(start_year, end_year):
    """
       Renders the sidebar controls
       Parameters: starting year, ending year
    """
    # Inject custom CSS to set the width of the sidebar
    st.markdown("""<style>section[data-testid="stSidebar"] {width: 300px; !important;} </style> """,  unsafe_allow_html=True)
    with (st.sidebar):
        st.header('Visualization options')
        # Create a slider to select a single year
        st.slider("Select a year", start_year, end_year, 2023, on_change=update_selected_slider_and_years, args=[1],
                  key="slider_year_1")
        # Create sliders to select start and end years
        st.slider("Or select start and end years", start_year, end_year, (start_year, start_year+(end_year-start_year)//2),
                  on_change=update_selected_slider_and_years, args=[2], key="slider_year_2")
        if "selected_slider" not in st.session_state:
            st.session_state["selected_slider"] = 1
        update_selected_slider_and_years(st.session_state.selected_slider)

        if st.session_state.selected_slider == 1:
            st.write("Single year is selected from the first slider.")
            st.write("Selected year:", st.session_state.year_1)
        else:
            st.write("Start and end years are selected from the second slider.")
            st.write("Selected start year:", st.session_state.year_1, "\nSelected end year:", st.session_state.year_2)

        if "selected_tab" not in st.session_state:
            st.session_state["selected_tab"] = "tab_map"
        tabs = [stx.TabBarItemData(id="tab_map", title="Map/Race plot",description="")]
        if st.session_state["page_name"] in ["sex_age","marital_status"]:
            st.session_state["selected_tab"] = tabs.append(stx.TabBarItemData(id="tab_pyramid", title="Pop. Pyramid",description="") )

     #   tab_map, tab_pyramid= st.tabs(["Map", "Population pyramid"])
        st.session_state["selected_tab"] = stx.tab_bar(data=tabs, default="tab_map")


        if st.session_state["selected_tab"] == "tab_map":
            if "visualization_option" not in st.session_state:
                st.session_state["visualization_option"] = "Matplotlib (static)"
            st.session_state["visualization_option"] = st.radio("Choose visualization option", ["Matplotlib (static)","Folium (static)", "Folium interactive", "Plotly race chart"])
        else:
            st.session_state["visualization_option"] = st.radio("Choose visualization option", ["Matplotlib", "Plotly"] )


def sidebar_controls(start_year, end_year):  # start_year=2007,end_year=2023

    basic_sidebar_controls(start_year, end_year)
    if "colormap_list" not in st.session_state:
        st.session_state["colormap_list"]= []

    if st.session_state["visualization_option"] == "Matplotlib (static)" :
        st.session_state["colormap_list"] = ["bwr", "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral",
         "coolwarm", "seismic", "Reds", "Purples", "Blues", "Greens", "Oranges", "Greys", "YlOrRd",
         "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
                "Pastel1","Pastel2","Paired","Accent","Dark2","Set1","Set2","Set3",
                "tab10","tab20","tab20b","tab20c"]
    elif st.session_state["visualization_option"] == "Folium (static)":
        remove_items = ["bwr","coolwarm", "seismic", "tab10","tab20","tab20b","tab20c"]
        # Remove colormap those are available for matplotlib but not folium
        st.session_state["colormap_list"] = [item for item in st.session_state["colormap_list"] if item not in remove_items]

    with st.sidebar:
        # Dropdown menu for colormap selection
        st.selectbox("Select a colormap\n (colormaps in the red boxes are available only for matplotlib) ", st.session_state["colormap_list"], key="selected_cmap")
        # Display a static image
        st.image("images/colormaps.jpg")

