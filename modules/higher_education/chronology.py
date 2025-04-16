import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import branca.colormap as cm
from matplotlib import pyplot as plt
from streamlit_folium import folium_static
import extra_streamlit_components as stx
from modules.base_page import BasePage


class Chronology:
    page_name = "high_education_chronology"
    @st.cache_data
    @staticmethod
    def get_data():
        df_years_universities = pd.read_csv("data/preprocessed/higher-education/df_years_universities.csv",

                                            usecols=["city", "uni_name", "foundation_year", "type", "region", "year"])
        gdf_city_locations = gpd.read_file("data/preprocessed/higher-education/gdf_city_locations.json").set_index(
            "city")  # we did not include city column above, since we have here in gdf_city_locations(we will out merge two dfs)
        return df_years_universities, gdf_city_locations

    def join_unis_of_city(x):
        return "-" + x

    @classmethod
    def get_df_plot(self, df_result,  gdf_city_locations):
        # get city counts
        df_city_counts = df_result.index.value_counts().to_frame("count")
        # join city counts to df_year
        df_plot = df_result.join(df_city_counts)
        df_plot_agg = df_plot.groupby(["city", "count"]).agg(
            {"uni_name": lambda x:  (x+",").str.cat(sep="\n")}).reset_index().set_index("city")
        # join city locations to df_year
        df_plot_agg = df_plot_agg[["count", "uni_name"]].merge(gdf_city_locations, how="outer", left_index=True,
                                                               right_index=True)
        df_plot_agg = df_plot_agg.fillna({"count": 0})
        df_plot_agg["count"] = df_plot_agg["count"].astype("int")
        return  df_plot_agg.set_geometry("geometry")

    @classmethod
    def render(cls):
        #page_name = cls.page_name
        #cls.set_session_state_variables()
        df_data, gdf_city_locations = cls.get_data()
        BasePage.sidebar_controls_basic_setup(df_data["year"].min(), df_data["year"].max())
        st.write("Select university type")
        state, foundation = st.checkbox("State universities", True), st.checkbox("Foundation universities", True)

        uni_types = []
        if state:
            uni_types.append("state")
        if foundation:
            uni_types.append("foundation")

        if "selected_tab" not in st.session_state:
            st.session_state["selected_tab"+cls.page_name] = "tab_map"
        tabs = [stx.TabBarItemData(id = "tab_map", title="Map plot", description=""),
                stx.TabBarItemData(id = "tab_bar", title="Bar plot", description="")]
        #   tab_map, tab_pyramid= st.tabs(["Map", "Population pyramid"])
        st.session_state["selected_tab"+cls.page_name] = stx.tab_bar(data=tabs, default="tab_map")
        df_data = df_data[df_data["type"].isin(uni_types)]  # filter according to uni-type
        if st.session_state["selected_tab"+cls.page_name] == "tab_map":
            cls.tab_map(df_data, gdf_city_locations)
        elif st.session_state["selected_tab"+cls.page_name] == "tab_bar":
            cls.tab_bar(df_data)


    @classmethod
    def plot_map(cls, map_column, df):
        colormap = cm.StepColormap(
            colors=['white', (123, 211, 234), (138, 255, 139), (0, 163, 0),
                    (254, 246, 92), (255, 141, 15), (253, 112, 107), (226, 6, 43)],  # renkler
            index=[-0, 1, 2, 3, 5, 10, 20, 30, 100], vmax=60)
        turkey_map = df.explore(
            legend="count",
            column="count",  # make choropleth based on "count" column
            tooltip="uni_name",  # show "quota" value in tooltip (on hover)
            popup=True,  # show all values in popup (on click)
            cmap=colormap,  # use  matplotlib colormap
            control_scale=False,
            style_kwds=dict(color="black"),  # use black outline
            tiles="CartoDB positron",  # use "CartoDB positron" tiles
            zoom_start=6
        )
        with map_column:
            folium_static(turkey_map)
        st.markdown(f"""<style>
                           iframe {{
                               width: inherit;
                           }}
                           </style>
                         """, unsafe_allow_html=True)

    @classmethod
    def tab_map(cls, df_data, gdf_city_locations):

        st.write("Check to show new the cities")
        show_only_new_cities = st.checkbox("Only new cities")
        if st.session_state.selected_slider == 1:
            mask = (st.session_state["year_1"] == df_data["year"])
            title = f"Universities present in the year {st.session_state['year_1']}"
            df_data.drop("year", axis=1, inplace=True)
        else:  # elif st.session_state.selected_slider == 2:
            title = f"Universities founded between the years {st.session_state['year_1']} and {st.session_state['year_2']} (inclusive)"
            df_data.drop("year", axis=1, inplace=True)
            df_data = df_data.drop_duplicates()
            mask = (st.session_state["year_1"] <= df_data["foundation_year"]) & (df_data["foundation_year"] <= st.session_state["year_2"])
        df_result = df_data.loc[mask].sort_values(by="foundation_year").set_index("city")
        cities_to_exclude = df_data[(df_data["foundation_year"] < st.session_state["year_1"] )]["city"].unique()
        if show_only_new_cities:
            df_result = df_result[~df_result.index.isin(cities_to_exclude)]


        st.title(title)
        map_column, df_column = st.columns([3, 2], gap="small")
        df_result["foundation_year"] = df_result["foundation_year"].astype(str)
        df_plot = cls.get_df_plot(df_result, gdf_city_locations)
        cls.plot_map(map_column, df_plot)
        df_column.dataframe(df_result)


    @classmethod
    def tab_bar(cls, df_data):
        # SLIDER'DAN SEÇİLEN YILLARA GÖRE FİLTRELENECEK
        st.write("**Bar tab requires an period of time and, therefore uses only the second slider.**" )
        st.write("Check to show new the cities")
        show_only_new_cities = st.checkbox("Only new cities")
        selected_index = "foundation_year" if show_only_new_cities else "year"
        if show_only_new_cities:
            df_data.drop("year", axis=1, inplace=True)
            df_data = df_data.drop_duplicates()
        df_year_counts = df_data[[selected_index, "type"]].value_counts().to_frame()
        df_year_counts = df_year_counts.reset_index().pivot_table(index=selected_index, columns='type', values='count', fill_value=0)
        start_year, end_year = int(st.session_state.slider_year_2[0]), int(st.session_state.slider_year_2[1])
        df_result = pd.DataFrame(0, index=range(start_year,end_year+1), columns=["foundation","state"])
        temp = df_year_counts[(start_year <= df_year_counts.index) & (df_year_counts.index <= end_year)]
        df_result.loc[temp.index] = temp

        cls.plot_bar(df_result, start_year, end_year)


    @classmethod
    def plot_bar(cls,df_year_counts, start_year, end_year):
        col1, col2, col3 = st.columns( [1,8,1])
        with col2:
            fig, ax = plt.subplots(1, figsize=(9, 6))
            x_vals = df_year_counts.index
            plt.xticks(x_vals, rotation="vertical")
            font_size = 6
            ax.tick_params(axis='both', which='major', labelsize=font_size)
            bottom = np.zeros(len(df_year_counts))

            for uni_type, count in df_year_counts[["state", "foundation"]].items():
                p = ax.bar(x=x_vals, height=count, bottom=bottom)
                bottom += count

            plt.ylabel("Number of universities", size=8)
            plt.xlabel("Year", size=8)
            for x, (n_state, n_foundation) in zip(range(start_year, end_year + 1),
                                                  df_year_counts[["state", "foundation"]].to_numpy()):
                if n_state != 0 and n_foundation != 0:
                    ax.text(x, n_state / 2, str(int(n_state)), ha="center", va="center", fontsize=font_size, c="white", rotation=45)
                    ax.text(x, n_state + n_foundation // 3 + .5, str(int(n_foundation)), ha="center", va="center", fontsize=font_size, rotation=45)
                    ax.text(x, n_state + n_foundation + 2, str(int(n_state + n_foundation)), ha="center", va="bottom", fontsize=font_size)

                elif n_state != 0 or n_foundation != 0:
                    ax.text(x, n_state + n_foundation , str(int(n_state + n_foundation)), ha="center", va="bottom", fontsize=font_size)
            # for x, height in zip(range(1984, 2024),  df_year_counts[["DEVLET","VAKIF"]].sum(axis=1)):
            #    ax.text(x, height+5, str(height), ha="center", va="bottom", fontsize=10)
            plt.title(f"Number of universities in Türkiye between  {start_year} and {end_year}")

            plt.tight_layout()
            ax.margins(x=0.01)
            ax.legend(labels=["Number of state universities", "Number of foundation universities"], fontsize=font_size)
            st.pyplot(fig)
        #plt.savefig('number_of_universities_by_year.eps', format='eps')
           # fig.savefig('myimage.jpeg', format='jpeg', dpi=300)


    @classmethod
    def run(cls):
        cls.render()


Chronology.run()