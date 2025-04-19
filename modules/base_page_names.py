import json
from typing import Dict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from modules.base_page import BasePage
import pandas as pd
import geopandas as gpd
import streamlit as st
from matplotlib import colormaps, pyplot as plt
import locale
import plotly.express as px
from utils.plot_map_common import figure_setup
from matplotlib.patches import Patch
import extra_streamlit_components as stx
import altair as alt

locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')


class PageNames(BasePage):
    COLORS = ["red", "purple", "orange", "green", "dodgerblue", "magenta","gold","darkorange","darkolivegreen",
              "cyan",  "gray","lightblue","lightgreen", "darkkhaki", "brown","lime","orangered","blue","mediumpurple","turquoise"] + list(colormaps["Set2"].colors)+list(colormaps["Dark2"].colors)+list(colormaps["Pastel1"].colors)+["yellow","silver"]
    with open('json_files/cluster_colors.json', 'r', encoding='utf-8') as f:
        CLUSTER_COLOR_MAPPING = json.load(f)
    # Province annotation positioning settings
    with open('json_files/va_positions.json', 'r', encoding='utf-8') as f:
        VA_POSITIONS = json.load(f)
    HA_POSITIONS = {"Zonguldak": "right", "Adana": "right"}

    @classmethod
    def create_color_mapping(cls, gdf: gpd.GeoDataFrame, n_clusters: int) -> Dict[int, str]:
        """Generate cluster color mapping with province-based defaults."""
        color_map = {}
        clusters = set(range(n_clusters))
        # Assign predefined province colors
        for idx, color in cls.CLUSTER_COLOR_MAPPING.items():
            if idx in gdf.index:
                cluster = gdf.loc[idx, "clusters"][0] if isinstance(gdf.loc[idx, "clusters"], pd.Series) else gdf.loc[idx, "clusters"]# it returns multiple values(series) for the same name
                if cluster not in color_map:
                    color_map[cluster] = color
        # Assign remaining colors
        remaining_clusters = clusters - set(color_map.keys())
        remaining_colors = list(set(cls.COLORS)-set(color_map.values()))
        for i, cluster in enumerate(remaining_clusters):
            color_map[cluster] = remaining_colors[i]
        return color_map

    @classmethod
    def initialize_multiindex_gdf_clusters(cls,df, year):
        new_index = pd.MultiIndex.from_product([[year], df.index], names=["year", "city"])
        cls.gdf_clusters = df.copy()
        cls.gdf_clusters.index = new_index

    @classmethod
    def k_means_clustering(cls, df,  year):
        cls.gdf_clusters = cls.gdf["province"].set_index("province")

        page_name = cls.page_name
        # top_n = st.session_state["n_" + page_name]  # CANCELED : NOW ALL 30 FEATURES ARE USE IN K-MEANS
        if page_name == "names_surnames" and st.session_state["name_surname_rb"] == "Surname":
            df_year = df["surname"].loc[year]
       #     df_year = df_year[df_year["rank"] <= top_n]  # use top-n for clustering (n=30 for all)
        elif len(st.session_state["sex_" + page_name]) != 1:  # if both sexes are selected
            df_year_male, df_year_female = df[df["sex"] == "male"].loc[year], df[df["sex"] == "female"].loc[year]
       #     df_year_male = df_year_male[df_year_male["rank"] <= top_n]
        #    df_year_female = df_year_female[df_year_female["rank"] <= top_n]
            overlapping_names = set(df_year_male["name"]) & set(df_year_female["name"])
            df_year_male['name'] = df_year_male.apply(lambda x: f"{x['name']}_female" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year_female['name'] = df_year_female.apply(lambda x: f"{x['name']}_male" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year = pd.concat([df_year_male, df_year_female])
        else:  # single gender selected for names
            sex = st.session_state["sex_" + page_name]
            df_year = df[df["sex"].isin(sex)].loc[year]
       #     df_year = df_year[df_year["rank"] <= top_n]
        scaler = MaxAbsScaler()
        #data_scaled = scaler.fit_transform(df_year.loc["Adana", ["count"]])
        #print("SCAL:",data_scaled)

        # Get unique provinces
        provinces = df_year.index.get_level_values(0).unique()
        # Scale counts for each province
        for province in provinces:
            # Get counts for current province
            province_count_sum_first_30 = df_year.loc[province, 'count'].sum()
            province_counts = (df_year.loc[province, 'count']).values.reshape(-1, 1)
            scaled_counts = scaler.fit_transform(province_counts)                    # Fit and transform the counts
            df_year.loc[province, 'scaled_count_sklearn'] = scaled_counts.flatten()  # Update the DataFrame with scaled values
            # Fit and transform the counts
            # Update the DataFrame with scaled values
            df_year.loc[province, 'scaled_count_top_30'] = df_year.loc[province, "count"] / province_count_sum_first_30
        df_year.loc[:, "ratio"] = df_year.loc[:, "count"] / df_year.loc[:, "total_count"]
        print("XCV:",df_year.loc["Adana"])

        df_pivot = pd.pivot_table(df_year, values='ratio', index=df_year.index, columns=['name'],  aggfunc=lambda x: x, dropna=False, fill_value=0)
        print("ÜĞP",df_pivot)
        n_clusters = st.session_state["n_clusters_" + page_name]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42,  init='k-means++', n_init=10).fit(df_pivot)

        # Get cluster centroids (shape: n_clusters × n_features)
        centroids = kmeans.cluster_centers_
        # After fitting KMeans
        closest_indices, _ = pairwise_distances_argmin_min(centroids, df_pivot)
        closest_cities = df_pivot.index[closest_indices].tolist()  # Get city/province names
        # Get the subset of provinces that are closest to centroids
        cls.gdf_centroids = cls.gdf_clusters [cls.gdf_clusters.index.isin(closest_cities)]
        # Compute centroids of their geometries (for marker placement)
        cls.gdf_centroids["centroid"] = cls.gdf_centroids.geometry.centroid
        df_pivot["clusters"] = kmeans.labels_
        cls.gdf_clusters = cls.gdf_clusters.merge(df_pivot["clusters"], left_index=True, right_index=True)


    @classmethod
    def fun_extras(cls, *args):
        pass

    @staticmethod
    def get_ordinal(n):
        if 11 <= (n % 100) <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    @staticmethod
    def preprocess_for_map(df, year, target_rank, n_to_top_inclusive):
        df_year = df.loc[year]
        if n_to_top_inclusive:  # select the names above the target rank(inclusive)
            df_year_rank = df_year[df_year["rank"] <= target_rank]
        else:  # select the names with target rank
            df_year_rank = df_year[df_year["rank"] == target_rank]
        # if surname is selected or only single gender selected, we do not need combinatiobs
        if "sex" not in df_year_rank.columns or len(df_year_rank["sex"].unique()) == 1:  # if st.session_state["name_surname_rb"]=="surname":
            return df_year_rank
        # generate combinations if both genders are selected and surname is not selected
        results = {}
        for province in df_year_rank.index.unique():
                province_data = df_year_rank.loc[province]
                male_names = province_data[province_data['sex'] == 'male']['name'].tolist()
                female_names = province_data[province_data['sex'] == 'female']['name'].tolist()
                combinations = []
                for male in male_names:
                    for female in female_names:
                        combinations.append(f"{male}-{female}")
                results[province] = '\n'.join(combinations)
        # Create the final dataframe
        final_df = pd.DataFrame(results.items(), columns=['province', 'name'])
        final_df.set_index('province', inplace=True)
        return final_df

    @classmethod
    def preprocess_for_rank_bar_tabs(cls, df):
        df.index = df.index.droplevel(1)  # drop provinces from multiindex
        df = df.groupby([df.index, "name"]).aggregate({"count": "sum"}).reset_index(level="name")
        df = df.sort_values(by=["year", "count"], ascending=False)
        if "rank" in st.session_state["selected_tab"+cls.page_name]:  # rank tabs work for cumulative results
            # Create rank column
            for year in df.index:
                df.loc[year, 'rank'] = df.loc[year, 'count'].rank(axis=0, method='min', ascending=False)

        if st.session_state["selected_tab" + cls.page_name] == "rank_bar":
            # preprocess rank
            df = df[df['rank'] <= st.session_state["rank_" + cls.page_name]]
        elif st.session_state["selected_tab" + cls.page_name] == "custom_bar":
            selected_names = st.session_state["names_" + cls.page_name]
            df = df[df["name"].isin(selected_names)]
        print("ÇP13",df)

        return df

    @classmethod
    def common_tab_ui(cls):
        # Common helper function for rendering
        col_1, col_2, col_3, col_4, _ = st.columns([2, 2, 3, 2, 4])
        tab_selected = st.session_state["selected_tab" + cls.page_name]
        name_surname = "name"  # single option for baby names dataset
        if cls.page_name == "names_surnames":
            col_2.write()
            name_surname = col_2.radio("Select name or surname", ["Name", "Surname"], key="name_surname_rb").lower()
            df = cls.data[name_surname.lower()]
        else:
            df = cls.data
        disable = name_surname == "surname"
        if tab_selected == "map":
            col_1.write("Select gender(s)")
            stored_value = st.session_state.get("male_" + cls.page_name, True)
            st.session_state["male_" + cls.page_name] = col_1.checkbox("Male",  disabled=disable, value=stored_value)
            # if "surnames" option is selected disable gender options
            stored_value = st.session_state.get("female_" + cls.page_name, False)
            st.session_state["female_" + cls.page_name] = col_1.checkbox("Female", disabled=disable, value=stored_value)

            st.session_state["sex_" + cls.page_name] = []
            if st.session_state["male_" + cls.page_name]:
                st.session_state["sex_" + cls.page_name].append("male")
            if st.session_state["female_" + cls.page_name]:
                st.session_state["sex_" + cls.page_name].append("female")
        else: # Use radio buttons for gender in other tabs
            st.session_state["sex_" + cls.page_name] = [col_1.radio("Select gender", options=["Male", "Female"]).lower()]

        # if "surnames" option is selected or not any genders are selected, select both
        if disable or not st.session_state["sex_" + cls.page_name]:
            st.session_state["sex_" + cls.page_name] = ["male", "female"]
        if name_surname != "surname":
            df = df[df['sex'].isin(st.session_state["sex_" + cls.page_name])]
        if "rank" in tab_selected:  # add rank selectbox if tab_selected == "tab_rank_bump" or tab_selected == "tab_rank_bar":
            st.session_state["rank_"+cls.page_name] = col_2.selectbox(f"Select rank", range(1, 11), index=9)
        elif tab_selected == "custom_bar":  # add name selector for custom name selection instead of top-rank names
            expression_in_sentence = "names or surnames" if cls.page_name == "names_surnames" else "baby names"
            # names_surnames has extra name-surname radio group overlapping with name selector,if so move the selector to right col
            empty_col = col_4 if cls.page_name == "names_surnames" else col_2
            empty_col.multiselect(f"Select {expression_in_sentence}", sorted(df["name"].unique(), key=locale.strxfrm),
                              key="names_" + cls.page_name)
        if tab_selected != "map":
            col_3.radio("Select an option", options=["Use clusters", "Use provinces"]).lower()
            provinces = sorted(df.index.get_level_values(1).unique(), key=locale.strxfrm)
          #  if cls.gdf_clusters is None:

            col_3.multiselect(f"Select clusters (default all)", provinces, key="clusters_" + cls.page_name)
            col_3.multiselect(f"Select provinces (default all)", provinces, key="provinces_" + cls.page_name)
            col_3.checkbox(f"Show aggregated totals (sum counts for selected provinces)", provinces, key="aggregate_provinces_" + cls.page_name)

        return df

    @classmethod
    def render(cls):
        # Apply CSS to all radio groups except the first
        header = "Names & Surnames Analysis" if cls.page_name == "names_surnames" else "Baby Names Analysis"
        st.header(header)
        start_year, end_year = cls.data["name"].index.get_level_values(0).min(), cls.data["name"].index.get_level_values(0).max()
        BasePage.sidebar_controls_basic_setup(start_year, end_year)
        if "selected_tab" not in st.session_state:
            st.session_state["selected_tab"+cls.page_name] = "map"
        tabs = [stx.TabBarItemData(id="map", title="Map Plot", description=""),
                stx.TabBarItemData(id="rank_bump", title="Rank Bump Plot", description=""),
                stx.TabBarItemData(id="rank_bar", title="Rank Bar Graph", description=""),
                stx.TabBarItemData(id="custom_bar", title="Custom Name Bar Graph", description="")
                ]

        st.session_state["selected_tab"+cls.page_name] = stx.tab_bar(data=tabs, default="map")
        tab_selected = st.session_state["selected_tab"+cls.page_name]
        df = cls.common_tab_ui()

        if "display_option_"+cls.page_name not in st.session_state:
            st.session_state["display_option_"+cls.page_name] = 0

        col_plot, col_df = st.columns([5, 1])
        if tab_selected == "map":
            cls.tab_map(df)
        elif tab_selected == "rank_bump":
            cls.tab_rank_bump(df, col_plot)
        elif tab_selected == "rank_bar":
            cls.tab_common(df, col_plot)
        elif tab_selected == "custom_bar":
            cls.tab_common(df, col_plot)

        # print("YUYU",df_data)
        # cls.bump_chart(df, col_plot)

    @classmethod
    def tab_map(cls, df):
        col_1, col_2, _ = st.columns([2.4, 2, 6])
        st.markdown("""
            <style>
            div[data-testid="stRadio"] > div[role="radiogroup"]:nth-child(n+2) {
                gap: 50px;
            }
            </style>
            """, unsafe_allow_html=True)
        expression_in_sentence = "names or surnames" if cls.page_name == "names_surnames" else "baby names"
        options = [
            "Apply K-means clustering",
            f"Select {expression_in_sentence} and top-n to filter",
            f"Show the nth most common {expression_in_sentence}"
        ]

        stored_value = st.session_state.get("display_option_" + cls.page_name, options[1])
        # Convert stored value to index
        default_index = options.index(stored_value) if stored_value in options else 1
        st.markdown('<div class="radio-with-gap">', unsafe_allow_html=True)
        st.session_state["display_option_" + cls.page_name] = col_1.radio("Select an option", options=options, index=default_index)
        st.markdown('</div>', unsafe_allow_html=True)
        col_2.selectbox("Select K: number of clusters", range(2, 11), key="n_clusters_" + cls.page_name)
        col_2.multiselect(f"Select {expression_in_sentence}", sorted(df["name"].unique(), key=locale.strxfrm),
                          key="names_" + cls.page_name)
        col, _ = st.columns([2, 5])

        #options = list(range(2, 31)) if "K-means" in choice else list(range(1, 31))
        options = list(range(1, 31))
        # Ensure current selection is valid for the new options
        key = "n_" + cls.page_name
        if key not in st.session_state or st.session_state[key] not in options:
            st.session_state[key] = options[0]  # Reset to first valid option
        col.selectbox('Choose a number n for the "nth most common" or "top-n" options', options=options, key="n_" + cls.page_name)
        col_plot, col_df = st.columns([5, 1])
        cls.plot_geopandas(col_plot, col_df, df)

    @classmethod
    def tab_rank_bump(cls, df, col_plot):
        df = cls.preprocess_for_rank_bar_tabs(df)
        df = df[df['rank'] <= st.session_state["rank_" + cls.page_name]]
        cls.bump_chart_pyplot(df, col_plot)
        cls.bump_chart_plotly(df, col_plot)

    @classmethod
    def tab_rank_bar(cls, df, col_plot):
        df = cls.preprocess_for_rank_bar_tabs(df)
        names_top_n = df[df['rank'] <= st.session_state["rank_" + cls.page_name]]["name"].to_list()
        df = df[df["name"].isin(names_top_n)]
        col_plot.title("Counts by Name and Year")
        cls.plot_rank_bar(df, col_plot)

    @classmethod
    def tab_common(cls, df, col_plot,tab_name="rank_bar"):
        selected_provinces = st.session_state["provinces_" + cls.page_name]
        plot_method = getattr(cls,"plot_" + tab_name)
        idx = pd.IndexSlice
        if selected_provinces:
            if st.session_state["aggregate_provinces_" + cls.page_name]:
                idx = pd.IndexSlice
                df = df.loc[idx[:, st.session_state["provinces_" + cls.page_name]], :]
                plot_method(cls.preprocess_for_rank_bar_tabs(df), col_plot)
            else:
                for province in selected_provinces:
                    df_province = df.loc[idx[:, province], :]
                    col_plot.subheader(province)
                    plot_method(cls.preprocess_for_rank_bar_tabs(df_province), col_plot)
        else:  # if not any selected, select all provinces
            plot_method(cls.preprocess_for_rank_bar_tabs(df), col_plot)

    @classmethod
    def plot_names(cls, df_result, ax):
        # Create a color map
        df_result["clusters"] = df_result["name"].factorize()[0]
        color_map = cls.create_color_mapping(df_result.set_index("name"), df_result["name"].nunique())

        # Assign colors to each row in the GeoDataFrame
        df_result['color'] = df_result['clusters'].map(color_map).fillna("gray") #-->GEREK YOK mu

        # After groupby df_result becomdes Pandas dataframe, we have to convert it to GeoPandas dataframe
        df_result = gpd.GeoDataFrame(df_result, geometry='geometry')
        # Plotting
        df_result.plot(ax=ax, color=df_result['color'], legend=True,  edgecolor="black", linewidth=.2)
        bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6)
        df_result.apply(lambda x: ax.annotate(
            text=x["province"].upper() + "\n" + x['name'].title() if isinstance(x['name'], str) else x["province"],
            size=4, xy=x.geometry.centroid.coords[0], ha=PageNames.HA_POSITIONS.get(x["province"], "center"), va=PageNames.VA_POSITIONS.get(x["province"], "center"), bbox=bbox), axis=1)
        ax.axis("off")
        ax.margins(x=0)
        # # Add a table (positioned like a legend)
        df_count = (df_result['name'].str.split('\n')  # Split names by newline
                    .explode()        # Create separate row for each name
                    .value_counts()   # Count occurrences
                    .rename_axis('name')
                    .reset_index(name='count')
                   )
        # Format the DataFrame as a string for the legend
        legend_text = "\n".join(f"{row['name']}: {row['count']}" for _, row in df_count.iterrows())
        # Create list of text entries instead of single multi-line string
        legend_entries = [f"{row['name']}: {row['count']}" for _, row in df_count.iterrows()]
        # Create invisible handles for each entry
        handles = [Patch(color='none', label=entry, linewidth=0) for entry in legend_entries]

        ax.plot([], [], label=legend_text)  # Invisible plot
        # Add custom legend entry with text only (no line)
        custom_legend = Patch(color='none', label=legend_text, linewidth=0)
        # Show legend
        ncols = 1+len(legend_entries)//9
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 0.165 if ncols>2 else 0.21), fontsize=4,ncols=ncols,  # Two columns
                  # Reduce left margin parameters:
                  handlelength=0,  # Remove space for legend handle (invisible line)
                  handletextpad=0,  # Remove padding between handle and text
                  alignment='left'  # Force left-aligned text
                  )

       # Add a legend
       # for name, color in set(zip(df_result['name'], df_result['color'])):
       #     ax.plot([], [], color=color, label=name, linestyle='None', marker='o')
       #     ax.legend(title='Names', fontsize=4, bbox_to_anchor=(0.01, 0.01), loc='lower right', fancybox=True, shadow=True)

    @classmethod
    def plot_clusters(cls, ax, year):
        n_clusters = st.session_state["n_clusters_" + cls.page_name]
        # Define a color map for the categories
        clusters = list(range(n_clusters))
        color_map = cls.create_color_mapping(cls.gdf_clusters, n_clusters)
        # Map the colors to the GeoDataFrame
        cls.gdf_clusters["color"] = cls.gdf_clusters["clusters"].map(color_map)
        cls.gdf_clusters.plot(ax=ax, color=cls.gdf_clusters['color'], legend=True, edgecolor="black", linewidth=.2)
        ax.axis("off")
        ax.margins(x=0)
        # Compute centroids of the closest provinces and plot them as markers
        closest_provinces_centroids = cls.gdf_centroids.copy()
        closest_provinces_centroids["centroid_geometry"] = closest_provinces_centroids.geometry.centroid
        # Create a temporary GeoDataFrame with centroid geometries (points)
        closest_provinces_points = gpd.GeoDataFrame(closest_provinces_centroids, geometry="centroid_geometry",
                                                    crs=cls.gdf_clusters.crs)
        # Add markers using the centroid points (no fill color change   # Transparent fill)
        closest_provinces_points.plot(ax=ax, facecolor="none", markersize=120, edgecolor="black", linewidth=1.5,
                                      label=f"Closest provinces\nto  cluster centers")
        # Add province names (from index) at centroids
        bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6)
        for province in cls.gdf_clusters.index:
            ax.annotate(text=province,  # Use index (province name) directly
                        xy=(cls.gdf_clusters.loc[province, "geometry"].centroid.x,
                            cls.gdf_clusters.loc[province, "geometry"].centroid.y),
                        ha=PageNames.HA_POSITIONS.get(province, "center"), va=PageNames.VA_POSITIONS.get(province, "center"), fontsize=5, color="black", bbox=bbox)

        # Optional: Add legend
        ax.set_title(f"{n_clusters} Clusters Identified in {year} (K-means)")
        ax.legend(loc="upper right", fontsize=6)

    @classmethod
    def create_title_for_plot(cls):
        page_name = st.session_state["page_name"]
        rank = st.session_state["n_"+page_name]
        names_or_surnames = "names"
        selected_gender = "male and female" if len(st.session_state["sex_" + cls.page_name]) != 1 else st.session_state["sex_" + cls.page_name][0]
        # Adjust phrasing based on rank
        if rank == 1:
            title_prefix = "The most common "
        else:
            title_prefix = f"The {PageNames.get_ordinal(rank)} most common "

        if page_name == "names_surnames":
            if st.session_state["name_surname_rb"] == "Surname":
                title = title_prefix + "surnames"
            else:
                title = title_prefix + selected_gender+" names"
        else:
            title = title_prefix + selected_gender + " baby names"
        return title, names_or_surnames

    @classmethod
    def plot_geopandas(cls, col_plot, col_df, df):
        gdf_borders = BasePage.gdf["province"]
        title, names_or_surnames = cls.create_title_for_plot()
        display_option = st.session_state["display_option_" + cls.page_name]
        top_n = int(st.session_state["n_" + cls.page_name])
        fig, axs = figure_setup()
        df_results = []
        year_1, year_2 = st.session_state["year_1"], st.session_state["year_2"]

        for i, year in enumerate(sorted({year_1, year_2})):
            # Display option 1: Show the nth most common baby names
            if "nth most common" in display_option:
                df_year_rank = PageNames.preprocess_for_map(df, year, target_rank=top_n, n_to_top_inclusive=False)
                df_result = gdf_borders.merge(df_year_rank, left_on="province", right_index=True)
                df_result = df_result.sort_values(by=['province', 'name'], ascending=[True, True]) # to prevent different orders like "Asel, Defne"  and "Defne, Asel"
                df_result = df_result.groupby(["geometry", "province"])["name"].apply(
                    lambda x: "%s" % '\n'.join(x)).to_frame().reset_index()
                df_results.append(df_result)
                cls.plot_names(df_results[i], axs[i, 0])
                axs[i, 0].set_title(title + f' in {year}')
            elif "top-n to filter" in display_option:  # Display option 2: Select single year, name(s) and top-n number to filter
                names_from_multi_select = st.session_state["names_" + cls.page_name]
                df_year = df.loc[year].reset_index()
                if names_from_multi_select:
                    df_result = df_year[(df_year["name"].isin(names_from_multi_select)) & (df_year["rank"] <= top_n)]
                    # drop "s" if single name or surname selected
                    names_or_surnames_statement = names_or_surnames[:-1] + " is" if len(names_from_multi_select) == 1 else names_or_surnames + " are"
                    if df_result.empty:
                        st.write(
                            f"Selected {names_or_surnames_statement} are not in the top {top_n} for the year {year}")
                    df_result_not_null = gdf_borders.merge(df_result, left_on="province", right_on="province")
                    df_result_not_null = df_result_not_null.groupby(["geometry", "province"])["name"].apply(
                        lambda x: "%s" % '\n '.join(x)).to_frame().reset_index()
                    df_results.append(df_result_not_null)
                    df_result_with_nulls = gdf_borders.merge(df_result_not_null[["province", "name"]],
                                                             left_on="province", right_on="province", how="left")
                    cls.plot_names(df_result_with_nulls, axs[i, 0])#, sorted(df_result_not_null['name'].unique()))  -->GEREKLİ Mİ, fonksiyondan parametre kalkmıştı, buradan yollamaya gerek var mı?
                    axs[i, 0].set_title(f"Provinces where selected {names_or_surnames_statement} in the top {top_n} for {year}")
            else:  # K-means
                cls.k_means_clustering(df,  year)
                cls.plot_clusters( axs[i, 0], year)
                df_results.append(cls.gdf_clusters)

        if not df_results:
            st.write("No results found.")
        if df_results:
            col_plot.pyplot(fig)

    @classmethod
    def bump_chart_pyplot(cls, df, col_plot):
        df_pivot = pd.pivot_table(df, values='rank', index=df.index, columns=['name'],
                            aggfunc=lambda x: x, dropna=False)
        fig, ax = plt.subplots(figsize=(10, 6))  # <-- Explicit figure creation

        # Plot lines for each name
        for i, name in enumerate(df_pivot.columns):
            color = PageNames.COLORS[i]
            series = df_pivot[name]

            # Identify continuous non-NaN segments
            not_nan = series.notna()
            segments = []
            current_segment = []

            for year, valid in zip(series.index, not_nan):
                if valid:
                    current_segment.append((year, series.loc[year]))
                else:
                    if len(current_segment) > 1:
                        segments.append(current_segment)
                    current_segment = []
            if len(current_segment) > 1:
                segments.append(current_segment)

            # Plot each continuous segment
            for segment in segments:
                seg_years, seg_ranks = zip(*segment)
                ax.plot(seg_years, seg_ranks,
                        marker='o',
                        markersize=8,
                        linewidth=2,
                        alpha=0.7,
                        color=color,
                        label=name if segment == segments[0] else None)  # Avoid duplicate labels

        max_rank = int(df["rank"].max())
        # Invert y-axis to show #1 at top
        ax.invert_yaxis()
        ax.set_yticks(range(1, max_rank+1))  # Force show all ticks 1-10
        ax.set_ylim(max_rank+1, 0)  # Add padding to top/bottom
        ax.set_xticks(df_pivot.index)
        # Add padding around plot content
        ax.margins(x=0.15, y=0.1)  # 10% padding on both axes
        fig.tight_layout(pad=2.0)  # Add padding around figure
        # Customize plot
        ax.set_title("Rank Evolution Over Years", fontsize=14, pad=20)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Rank", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # Add data labels for final year
        # Add labels for FIRST and LAST year positions
        from collections import defaultdict

        # Compute offsets to reduce overlapping texts
        y_offset_tracker_first = defaultdict(int)
        y_offset_tracker_last = defaultdict(int)

        for name in df_pivot.columns:
            first_rank = df_pivot[name].iloc[0]
            final_rank = df_pivot[name].iloc[-1]
            first_year = df_pivot.index[0]
            final_year = df_pivot.index[-1]

            # Add small vertical offsets if multiple names share the same rank
            offset_first = y_offset_tracker_first[first_rank] * 0.2
            offset_last = y_offset_tracker_last[final_rank] * 0.2

            ax.text(first_year - 0.1, first_rank + offset_first, name,
                    va='center', ha='right', alpha=0.7, fontsize=9)
            ax.text(final_year + 0.1, final_rank + offset_last, name,
                    va='center', ha='left', alpha=0.7, fontsize=9)

            y_offset_tracker_first[first_rank] += 1
            y_offset_tracker_last[final_rank] += 1

        col_plot.pyplot(fig)

    @classmethod
    def bump_chart_plotly(cls, df, col_plot):
        # Prepare data in long format (from your pivot table)
        max_rank = int(df["rank"].max())

        # Create figure with custom size
        fig = px.line(
            df.reset_index(),
            x='year',
            y='rank',
            color='name',
            markers=True,
        ).update_layout(
            width=1500,  # Figure width in pixels
            height=800,  # Figure height in pixels
            xaxis_title="Year", yaxis_title="Rank",
            margin=dict(l=50, r=50, t=80, b=50),  # Adjust margins for labels
            legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="right", x=1.15),
            title=dict(text="Rank Evolution Over Years", font=dict(size=50), automargin=True, yref='paper'),
            yaxis_title_font=dict(size=32), xaxis_title_font=dict(size=32)
        ).update_yaxes(
            tickvals=list(range(1, max_rank + 1)),
            range=[max_rank + 0.5, 0.5],
            autorange="reversed",
            dtick=1,
            tickfont=dict(size=32)
        ).update_xaxes(
            showline=True,
            tickfont=dict(size=32)
        ).update_traces(
            line=dict(width=5.5),
            marker=dict(size=30)
        )

        # Add annotations for names at the beginning and ending years
        df_reset = df.reset_index()
        years = df_reset['year'].unique()
        first_year = years.min()
        last_year = years.max()

        for name in df_reset['name'].unique():
            # Data for the first year
            first_year_data = df_reset[(df_reset['name'] == name) & (df_reset['year'] == first_year)]
            if not first_year_data.empty:
                fig.add_annotation(
                    x=first_year,
                    y=first_year_data['rank'].iloc[0],
                    text=name,
                    showarrow=False,
                    xshift=-20,  # Shift left for better positioning
                    font=dict(size=20),
                    xanchor="right"
                )

            # Data for the last year
            last_year_data = df_reset[(df_reset['name'] == name) & (df_reset['year'] == last_year)]
            if not last_year_data.empty:
                fig.add_annotation(
                    x=last_year,
                    y=last_year_data['rank'].iloc[0],
                    text=name,
                    showarrow=False,
                    xshift=20,  # Shift right for better positioning
                    font=dict(size=20),
                    xanchor="left"
                )

        col_plot.plotly_chart(fig)



    @classmethod
    def plot_rank_bar(cls, df, col_plot):
        print("5555",df)
        df = df.reset_index()
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('year:O', title='Year'),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color('name:N', legend=None),  # Remove legend
            column=alt.Column('name:N', title=None)
        ).properties(
            width=150
        ).configure_header(
            titleFontSize=20,  # Increase font size for column titles (names)
            labelFontSize=14
        )

        col_plot.altair_chart(chart)

    @classmethod
    def bar_plot_custom(cls, df, col_plot):
        pass
        #chart = cls.bar_plot_rank(df, title=f"Counts by Name and Year - {title}")
        #st.altair_chart(chart, use_container_width=True)
