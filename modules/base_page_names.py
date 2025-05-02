import numpy as np
from matplotlib.colors import BoundaryNorm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from modules.base_page import BasePage
import pandas as pd
import geopandas as gpd
import streamlit as st
from matplotlib import colormaps, pyplot as plt, cm
import locale
import plotly.express as px
from matplotlib.patches import Patch
import extra_streamlit_components as stx
import altair as alt
from adjustText import adjust_text

locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')


class PageNames(BasePage):

    def sidebar_controls_plot_options_setup(self, *args):
        with (st.sidebar):
            if st.session_state.get("selected_tab_"+self.page_name, "map") == "map":
                st.session_state["visualization_option"] = st.radio("Choose visualization option",
                                                                ["Matplotlib", "Folium"]).lower()

    def initialize_multiindex_gdf_clusters(self, df, year):
        new_index = pd.MultiIndex.from_product([[year], df.index], names=["year", "city"])
        self.gdf_clusters = df.copy()
        self.gdf_clusters.index = new_index

    def preprocess_k_means(self, df, year,  transpose_for_name_clustering=False):
        """"
        returns:df_pivot
        """
        page_name = self.page_name
        # top_n = st.session_state["n_" + page_name]  # CANCELED : NOW ALL 30 FEATURES ARE USE IN K-MEANS
        if page_name == "names_surnames" and st.session_state["name_surname_rb"] == "Surname":
            df_year = df.loc[year]
        #     df_year = df_year[df_year["rank"] <= top_n]  # use top-n for clustering (n=30 for all)
        elif len(st.session_state["sex_" + page_name]) != 1:  # if both sexes are selected
            df_year_male, df_year_female = df[df["sex"] == "male"].loc[year], df[df["sex"] == "female"].loc[year]
            #     df_year_male = df_year_male[df_year_male["rank"] <= top_n]
            #    df_year_female = df_year_female[df_year_female["rank"] <= top_n]
            overlapping_names = set(df_year_male["name"]) & set(df_year_female["name"])
            df_year_male['name'] = df_year_male.apply(
                lambda x: f"{x['name']}_female" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year_female['name'] = df_year_female.apply(
                lambda x: f"{x['name']}_male" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year = pd.concat([df_year_male, df_year_female])
        else:  # single gender selected for names
            sex = st.session_state["sex_" + page_name]
            df_year = df[df["sex"].isin(sex)].loc[year]
        #     df_year = df_year[df_year["rank"] <= top_n]
        # data_scaled = scaler.fit_transform(df_year.loc["Adana", ["count"]])
        # print("SCAL:",data_scaled)
        print("QQmhtr", df_year.index)

        if isinstance(df_year.index, pd.MultiIndex):  # If applying temporal clustering (multiple years given as year)
            print("ÖNCEÖNCE", df_year)
            df_year = df_year.droplevel(0)  # Drop the first level year(position 0)
            print("!!==")
            print("SORNA", df_year)

        print("axcas", df_year.index)
        print("bxcas", df_year)

        # Get unique cumulative total counts over years(for each province)
        total_counts = df[["total_count"]].groupby(level=["year", "province"]).first()
        total_counts = total_counts.groupby("province").sum()
        df_year = df_year.groupby([df_year.index, 'name']).agg({'count': 'sum'})
        # Merge
        df_year = df_year.merge(total_counts, left_index=True, right_index=True, how="outer")
        df_year = df_year.reset_index().set_index("province")
        print("yenni SONUÇÇÇ:", df_year)
        # # Scale counts for each province
        # for province in provinces:
        #     # Get counts for current province
        #     province_count_sum_first_30 = df_year.loc[province, 'count'].sum()
        #     province_counts = (df_year.loc[province, 'count']).values.reshape(-1, 1)
        #     scaled_counts = scaler.fit_transform(province_counts)                    # Fit and transform the counts
        #     df_year.loc[province, 'scaled_count_sklearn'] = scaled_counts.flatten()  # Update the DataFrame with scaled values
        #     # Fit and transform the counts
        #     # Update the DataFrame with scaled values
        #   #  df_year.loc[province, 'scaled_count_top_30'] = df_year.loc[province, "count"] / province_count_sum_first_30

        print("yenni:", df_year)
        df_year.loc[:, "ratio"] = df_year.loc[:, "count"] / df_year.loc[:, "total_count"]
        print("XCV:", df_year)

        df_pivot = pd.pivot_table(df_year, values='ratio', index=df_year.index, columns=['name'], aggfunc=lambda x: x,
                                  dropna=False, fill_value=0)

        if transpose_for_name_clustering:
            df_pivot = df_pivot.T
        df_pivot = self.scale(df_pivot)
        return df_pivot

    def k_means_names(self, df, year, obtain_pivot_only=False, transpose_for_name_clustering=False):
        df_pivot = self.preprocess_k_means(df,year,transpose_for_name_clustering)
        n_clusters=st.session_state["n_clusters_" + st.session_state["page_name"]]
        k_means = KMeans(n_clusters=n_clusters, random_state=0,  init='k-means++', n_init=100).fit(df_pivot)
        # After fitting KMeans
        closest_indices, _ = pairwise_distances_argmin_min(k_means.cluster_centers_, df_pivot)
        closest_cities = df_pivot.index[closest_indices].tolist()  # Get city/province names
        df_pivot["clusters"] = k_means.labels_ + 1  # +1 for displaying clusters to use from 1 not 0
        print("666888", df_pivot[["clusters"]])
        if obtain_pivot_only:
            return df_pivot
        self.gdf_clusters = self.gdf["province"].set_index("province")
        # Get the subset of provinces that are closest to centroids
        self.gdf_centroids = self.gdf_clusters[self.gdf_clusters.index.isin(closest_cities)]
        # Compute centroids of their geometries (for marker placement)
        self.gdf_centroids["centroid"] = self.gdf_centroids.geometry.centroid
        self.gdf_clusters = self.gdf_clusters.merge(df_pivot["clusters"], left_index=True, right_index=True)
        print("7000", self.gdf_clusters.clusters)
        return df_pivot

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

    def preprocess_for_rank_bar_tabs(self, df):
        df.index = df.index.droplevel(1)  # drop provinces from multiindex
        df = df.groupby([df.index, "name"]).aggregate({"count": "sum"}).reset_index(level="name")
        df = df.sort_values(by=["year", "count"], ascending=False)
        if "rank" in st.session_state["selected_tab_"+self.page_name]:  # rank tabs work for cumulative results
            # Create rank column
            for year in df.index:
                df.loc[year, 'rank'] = df.loc[year, 'count'].rank(axis=0, method='min', ascending=False)

        if "rank" in st.session_state["selected_tab_" + self.page_name]:
            print("74112",st.session_state["include_all_years"])
            print("887766",st.session_state["include_all_years"] == "Include All Years for Names Ever in Top-n")
            # preprocess rank
            if st.session_state["include_all_years"] == "Include All Years for Names Ever in Top-n":
                selected_names = df[df['rank'] <= st.session_state["rank_" + self.page_name]]["name"]
                df = df[df["name"].isin(selected_names)]
            else:
                df = df[df['rank'] <= st.session_state["rank_" + self.page_name]]

        else:  # st.session_state["selected_tab" + cls.page_name] == "custom_bar":
            selected_names = st.session_state["names_" + self.page_name]
            df = df[df["name"].isin(selected_names)]
        return df

    def common_tab_ui(self,col_1, col_2):
        # Common helper function for rendering
        tab_selected = st.session_state["selected_tab_" + self.page_name]
        name_surname = "name"  # single option for baby names dataset
        # data is a dictionary whose keys are names and surnames, values are corresponding dataframes
        if self.page_name == "names_surnames":
            col_2.write()
            name_surname = col_2.radio("Select name or surname", ["Name", "Surname"], key="name_surname_rb").lower()
            df = self.data[name_surname.lower()]
        else:
            df = self.data

        selected_years = range(st.session_state["year_1"], st.session_state["year_2"] + 1)
        idx = pd.IndexSlice
        df = df.loc[idx[selected_years, :], :]

        disable = name_surname == "surname"
        if tab_selected == "map" or tab_selected == "name_clustering": #TAB1 or TAB5
            col_1.write("Select gender(s)")
            stored_value = st.session_state.get("male_" + self.page_name, True)
            st.session_state["male_" + self.page_name] = col_1.checkbox("Male",  disabled=disable, value=stored_value)
            # if "surnames" option is selected disable gender options
            stored_value = st.session_state.get("female_" + self.page_name, False)
            st.session_state["female_" + self.page_name] = col_1.checkbox("Female", disabled=disable, value=stored_value)

            st.session_state["sex_" + self.page_name] = []
            if st.session_state["male_" + self.page_name]:
                st.session_state["sex_" + self.page_name].append("male")
            if st.session_state["female_" + self.page_name]:
                st.session_state["sex_" + self.page_name].append("female")
        else: # Use radio buttons for gender in other tabs 2-3-4
            st.session_state["sex_" + self.page_name] = [col_1.radio("Select gender", options=["Male", "Female"]).lower()]

        # if "surnames" option is selected or not any genders are selected, select both
        if disable or not st.session_state["sex_" + self.page_name]:
            st.session_state["sex_" + self.page_name] = ["male", "female"]
        if name_surname != "surname":
            df = df[df['sex'].isin(st.session_state["sex_" + self.page_name])]

        return df

    def render(self):

        import streamlit as st
        from PIL import Image
        import base64
        import io
        # Inject html2canvas JavaScript library
        st.components.v1.html(
            """
            <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
            """
        )
        # JavaScript to capture the page and download the image
        js_code = """
           <script>
           html2canvas(document.body).then(canvas => {
               // Adjust scale for resolution (e.g., scale=2 for 2x resolution)
               const scale = 2; // 2x scaling ≈ 300 DPI (adjust as needed)
               const imgData = canvas.toDataURL('image/png');

               // Create a download link
               const link = document.createElement('a');
               link.download = 'streamlit_page.png';
               link.href = imgData;
               link.click();
           });
           </script>
           """
        st.components.v1.html(js_code, height=0, width=0)


        # Apply CSS to all radio groups except the first
        header = "Names & Surnames Analysis" if self.page_name == "names_surnames" else "Baby Names Analysis"
        st.header(header)
        start_year, end_year = self.data["name"].index.get_level_values(0).min(), self.data["name"].index.get_level_values(0).max()
        self.sidebar_controls(start_year, end_year)
        if "selected_tab" not in st.session_state:
            st.session_state["selected_tab_"+self.page_name] = "map"
        tabs = [stx.TabBarItemData(id="map", title="Map Plot and Geographical Clustering", description=""),
                stx.TabBarItemData(id="rank_bump", title="Rank Bump Plot", description=""),
                stx.TabBarItemData(id="rank_bar", title="Rank Bar Plot", description=""),
                stx.TabBarItemData(id="custom_bar", title="Custom Name Bar Plot", description=""),
                stx.TabBarItemData(id="name_clustering", title="Name Clustering", description="")
                ]

        st.session_state["selected_tab_"+self.page_name] = stx.tab_bar(data=tabs, default="map")
        tab_selected = st.session_state["selected_tab_"+self.page_name]
        col_1, col_2, col_3, col_4, _ = st.columns([2, 2, 1.5, 2, 4])

        df = self.common_tab_ui( col_1, col_2)
        # if "display_option_"+self.page_name not in st.session_state:
        #     st.session_state["display_option_"+self.page_name] = 0

        col_plot, col_df = st.columns([5, 1])
        if tab_selected == "map":  # Tab-1
            self.tab_map(df)
        elif tab_selected in ["rank_bump", "rank_bar", "custom_bar"]:  # Tab 2-3-4
            self.tab_2_3_4(df, col_plot, col_df, col_2, col_3, col_4)
        else:  # Tab-5
            self.tab_name_clustering(df, col_plot,col_df,col_2, col_3,col_4)

    def tab_map(self, df, *args):
        # Add this CSS before your radio buttons code

        col_1, col_2, col_3, col4 = st.columns([2.4, 2, 1.3, 5])
        expression_in_sentence = "names or surnames" if self.page_name == "names_surnames" else "baby names"
        self.load_css("assets/styles.css")
        options = [
            "Apply K-means clustering",
            f"Select {expression_in_sentence} and top-n to filter",
            f"Show the nth most common {expression_in_sentence}"
        ]
        stored_value = st.session_state.get("display_option_" + self.page_name, options[0])
        # Display options
        default_index = options.index(stored_value) if stored_value in options else 1
        st.session_state["display_option_" + self.page_name] = col_1.radio("Select an option", options=options, index=default_index)
        self.k_means_gui_options(col_2, col_3, col4)

        col_2.multiselect(f"Select {expression_in_sentence}", sorted(df["name"].unique(), key=locale.strxfrm),
                          key="names_" + self.page_name)
        options = list(range(1, 31))
        # Ensure current selection is valid for the new options
        key = "n_" + self.page_name
        stored_value = st.session_state.get("n_" + self.page_name, options[0])
        default_index = options.index(stored_value) if stored_value in options else 1

        if key not in st.session_state or st.session_state[key] not in options:
            st.session_state[key] = options[0]  # Reset to first valid option
        st.session_state["n_" + self.page_name] = col_1.selectbox('Choose a number n for the "nth most common" or "top-n" options', options=options, index=default_index)
        col_plot, col_df = st.columns([5, 1])
        self.plot_map(col_plot, col_df, df)


    def tab_2_3_4(self, df, col_plot, col_df,col_2,col_3,col_4):
        tab_selected = st.session_state["selected_tab_" + self.page_name]
        if "rank" in tab_selected:  # add rank selectbox if tab_selected == "tab_rank_bump" or tab_selected == "tab_rank_bar": Tabs 2-3
            col_2.selectbox(f"Select rank", range(1, 6), index=4, key="rank_" + self.page_name)
            col_2.radio("Select an option",
                        ["Show Only Years When Names Are in Top-n", "Include All Years for Names Ever in Top-n"],
                        key="include_all_years")
        elif tab_selected == "custom_bar":  # add name selector for custom name selection instead of top-rank names
            expression_in_sentence = "names or surnames" if self.page_name == "names_surnames" else "baby names"
            # names_surnames has extra name-surname radio group overlapping with name selector,if so move the selector to right col
            empty_col = col_4 if self.page_name == "names_surnames" else col_2
            empty_col.multiselect(f"Select {expression_in_sentence}", sorted(df["name"].unique(), key=locale.strxfrm),
                                  key="names_" + self.page_name)
        if tab_selected in ["rank_bump", "rank_bar", "custom_bar"]:  # if tab is one of tab-2,3 or 4
            col_3.radio("Select an option", options=["Use provinces", "Use clusters"],
                        key="province_or_cluster").lower()
            provinces = sorted(df.index.get_level_values(1).unique(), key=locale.strxfrm)
            clusters = range(1, st.session_state["n_clusters_" + self.page_name] + 1)
            col_4.multiselect(f"Select provinces (default all)", provinces, key="provinces_" + self.page_name)
            col_4.multiselect(f"Select clusters (default all)", clusters, key="clusters_" + self.page_name)
            col_4.checkbox(f"Show aggregated totals (sum counts for selected provinces)", provinces,
                           key="aggregate_totals_" + self.page_name)

        # works for two tabs: Rank Bar Graph and Custom Name Bar Graph
        selected_provinces = st.session_state["provinces_" + self.page_name]
        selected_clusters = st.session_state["clusters_" + self.page_name]

        if "bar" in st.session_state["selected_tab_"+self.page_name]:
            plot_method = getattr(self, "plot_rank_bar")
        else:
            plot_method = getattr(self, "plot_rank_bump") #getattr(cls, "plot_" + st.session_state["selected_tab"+cls.page_name])
        col_plot.title("Counts by Name and Year")
        idx = pd.IndexSlice
        if st.session_state["province_or_cluster"] == "Use provinces" and selected_provinces:
            if st.session_state["aggregate_totals_" + self.page_name]:
                idx = pd.IndexSlice
                df = df.loc[idx[:, selected_provinces], :]
                plot_method(self.preprocess_for_rank_bar_tabs(df), col_plot)
            else:
                for province in selected_provinces:
                    df_province = df.loc[idx[:, province], :]
                    col_plot.subheader(province)
                    plot_method(self.preprocess_for_rank_bar_tabs(df_province), col_plot)
            col_df.dataframe(df)
        elif st.session_state["province_or_cluster"] == "Use clusters" and selected_clusters:
            df_pivot = self.k_means_names(df, df.index.get_level_values(0).unique(), obtain_pivot_only=True)
            df_clusters = df_pivot["clusters"]
            df['clusters'] = df.index.get_level_values("province").map(df_clusters)
            for cluster in selected_clusters:
                df_cluster = df[df["clusters"] == cluster]
                col_plot.subheader(f"Cluster {cluster}")
                plot_method(self.preprocess_for_rank_bar_tabs(df_cluster), col_plot)
            col_df.dataframe(df)
        else:  # if not any selected, select all provinces
            plot_method(self.preprocess_for_rank_bar_tabs(df), col_plot)

    def tab_name_clustering(self,df, col_plot,col_df, col_2, col_3, col_4):
        # BU COL2 olacak. col_2 parametre olarak eklenebilir veya bu sütunlar sınıfın özelliği yapılabilir.
        self.k_means_gui_options(col_2, col_3, col_4)
        df_pivot = self.k_means_names(df, df.index.get_level_values(0).unique(), obtain_pivot_only=True,
                                      transpose_for_name_clustering=True)

        df_clusters = df_pivot["clusters"]
        df_pivot = df_pivot.drop(columns=["clusters"])
        if st.session_state["optimal_k_analysis"]:
            col_df.pyplot(self.optimal_k_analysis(df_pivot))
            df_pivot["clusters"] = consensus_labels[st.session_state["n_"]]

        self.plot_pca(df_pivot, df_clusters, col_plot)

    def create_title_for_plot(self):
        page_name = st.session_state["page_name"]
        rank = st.session_state["n_"+page_name]
        names_or_surnames = "names"
        selected_gender = "male and female" if len(st.session_state["sex_" + self.page_name]) != 1 else st.session_state["sex_" + self.page_name][0]
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
    # Plot methods of three tabs
    # Tab-1 plot methods

    def plot_map(self, col_plot, col_df, df):
        gdf_borders = self.gdf["province"]
        title, names_or_surnames = self.create_title_for_plot()
        display_option = st.session_state["display_option_" + self.page_name]
        top_n = int(st.session_state["n_" + self.page_name])

        df_results = []
        year_1, year_2 = st.session_state["year_1"], st.session_state["year_2"]
        if "clustering" in display_option:
            df_pivot = self.k_means_names(df, df.index.get_level_values(0).unique())
            if st.session_state["optimal_k_analysis"]:
                col_df.pyplot(self.optimal_k_analysis(df_pivot.drop(columns=["clusters"])))
                consensus_labels= st.session_state["consensus_labels"]
                print("333777",consensus_labels.shape,"555222,",df_pivot.shape)
                df_pivot["clusters"] = consensus_labels[st.session_state["n_clusters_" + self.page_name]]


            year_in_title = df.index.get_level_values(0).min()
            if df.index.get_level_values(0).max() > year_in_title:
                 year_in_title = f"between {year_in_title}-{df.index.get_level_values(0).max()}"
            else:
                year_in_title = f"in {year_in_title}"
            self.plot_clusters(year_in_title, col_plot)  # ilk ve son yıl için kullanılıyordu
            df_clusters = df_pivot["clusters"]
            df_pivot = df_pivot.drop(columns=["clusters"])
            self.plot_pca(df_pivot, df_clusters, col_plot)
            col_df.dataframe(df_clusters)

        else:
            fig, axs = self.figure_setup()
            for i, year in enumerate(sorted({year_1, year_2})):
                # Display option 1: Show the nth most common baby names
                if "nth most common" in display_option:
                    df_year_rank = PageNames.preprocess_for_map(df, year, target_rank=top_n, n_to_top_inclusive=False)
                    df_result = gdf_borders.merge(df_year_rank, left_on="province", right_index=True)
                    df_result = df_result.sort_values(by=['province', 'name'], ascending=[True, True]) # to prevent different orders like "Asel, Defne"  and "Defne, Asel"
                    df_result = df_result.groupby(["geometry", "province"])["name"].apply(
                        lambda x: "%s" % '\n'.join(x)).to_frame().reset_index()
                    df_results.append(df_result)
                    self.plot_names(df_results[i], axs[i, 0])
                    axs[i, 0].set_title(title + f' in {year}')
                elif "top-n to filter" in display_option:  # Display option 2: Select single year, name(s) and top-n number to filter
                    names_from_multi_select = st.session_state["names_" + self.page_name]
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
                        self.plot_names(df_result_with_nulls, axs[i, 0])#, sorted(df_result_not_null['name'].unique()))  -->GEREKLİ Mİ, fonksiyondan parametre kalkmıştı, buradan yollamaya gerek var mı?
                        axs[i, 0].set_title(f"Provinces where selected {names_or_surnames_statement} in the top {top_n} for {year}")
                # else:  # K-means
                #     self.k_means_clustering(df,  year)   # ilk ve son yıl için kullanılıyordu artık
                #     #self.plot_clusters(axs[i, 0], year)  # artık for üstündeki ilk if çalıştığı için kullanılmıyor
                #     df_results.append(self.gdf_clusters)
            if axs[0, 0].has_data():
                col_plot.pyplot(fig)
            else:
                st.write("No results found.")


    def plot_names(self, df_result, ax):
        # Create a color map
        df_result["clusters"] = df_result["name"].factorize()[0]
        color_map = self.create_color_mapping(df_result.set_index("name"), df_result["name"].nunique())

        # Assign colors to each row in the GeoDataFrame
        df_result['color'] = df_result['clusters'].map(color_map).fillna("gray") #-->GEREK YOK mu

        # After groupby df_result becomdes Pandas dataframe, we have to convert it to GeoPandas dataframe
        df_result = gpd.GeoDataFrame(df_result, geometry='geometry')
        # Plotting
        df_result.plot(ax=ax, color=df_result['color'], legend=True,  edgecolor="black", linewidth=.2)
        bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6)
        ha_positions, va_positions = self.HA_POSITIONS, self.VA_POSITIONS
        df_result.apply(lambda x: ax.annotate(
            text=x["province"].upper() + "\n" + x['name'].title() if isinstance(x['name'], str) else x["province"],
            size=4, xy=x.geometry.centroid.coords[0], ha=ha_positions.get(x["province"], "center"), va=va_positions.get(x["province"], "center"), bbox=bbox), axis=1)
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

        # Add custom legend entry with text only (no line): Create invisible handles for each entry
        handles = [Patch(color='none', label=entry, linewidth=0) for entry in legend_entries]
        ax.plot([], [], label=legend_text)  # Invisible plot

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

    def plot_clusters(self,  year_in_title, col_plot):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        n_clusters = st.session_state["n_clusters_" + self.page_name]
        # Define a color map for the categories
        color_map = self.create_color_mapping(self.gdf_clusters, n_clusters)
        # Map the colors to the GeoDataFrame
        self.gdf_clusters["color"] = self.gdf_clusters["clusters"].map(color_map)
        nan_rows = self.gdf_clusters[self.gdf_clusters.isna().any(axis=1)]
        print("nan rows:",nan_rows,"n_clusters",n_clusters)
        self.gdf_clusters.plot(ax=ax, color=self.gdf_clusters['color'], legend=True, edgecolor="black", linewidth=.2)
        ax.axis("off")
        ax.margins(x=0)
        # Compute centroids of the closest provinces and plot them as markers
        closest_provinces_centroids = self.gdf_centroids.to_crs("EPSG:4326").copy()
        closest_provinces_centroids["centroid_geometry"] = closest_provinces_centroids.geometry.centroid
        # Create a temporary GeoDataFrame with centroid geometries (points)
        closest_provinces_points = gpd.GeoDataFrame(closest_provinces_centroids, geometry="centroid_geometry",
                                                    crs=self.gdf_clusters.crs)
        # Add markers using the centroid points (no fill color change   # Transparent fill)
        closest_provinces_points.plot(ax=ax, facecolor="none", markersize=120, edgecolor="black", linewidth=1.5,
                                      label=f"Closest provinces\nto  cluster centers")
        # Add province names (from index) at centroids
        bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6)
        ha_positions,va_positions = self.HA_POSITIONS, self.VA_POSITIONS
        print("typee",type(ha_positions))
        for province in self.gdf_clusters.index:
            ax.annotate(text=province,  # Use index (province name) directly
                        xy=(self.gdf_clusters.loc[province, "geometry"].centroid.x,
                            self.gdf_clusters.loc[province, "geometry"].centroid.y),
                        ha=ha_positions.get(province, "center"), va=va_positions.get(province, "center"), fontsize=5, color="black", bbox=bbox)

        # Optional: Add legend
        ax.set_title(f"{n_clusters} Clusters Identified {year_in_title} (K-means)")
        ax.legend(loc="upper right", fontsize=6)
        col_plot.pyplot(fig)
    # Tab-2 plot methods

    def plot_rank_bump(self, df, col_plot):  # pyplot
        df_pivot = pd.pivot_table(df, values='rank', index=df.index, columns=['name'],
                                  aggfunc=lambda x: x, dropna=False)
        fig, ax = plt.subplots(figsize=(10, 6))  # Explicit figure creation

        # Plot lines and dots for each name
        for i, name in enumerate(df_pivot.columns):
            color = self.COLORS[i]
            series = df_pivot[name]

            # Identify all non-NaN segments, including single points
            not_nan = series.notna()
            segments = []
            current_segment = []

            for year, valid in zip(series.index, not_nan):
                if valid:
                    current_segment.append((year, series.loc[year]))
                else:
                    if current_segment:  # Changed from len(current_segment) > 1
                        segments.append(current_segment)
                    current_segment = []
            if current_segment:  # Changed from len(current_segment) > 1
                segments.append(current_segment)

            # Plot each segment
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
        ax.set_yticks(range(1, max_rank + 1))  # Force show all ticks 1-10
        ax.set_ylim(max_rank + 1, 0)  # Add padding to top/bottom
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
        # Add data labels for first and last year
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


    def bump_chart_plotly(self, df, col_plot):
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

    # Tab-3-4 plot methods

    def plot_rank_bar(self, df, col_plot):
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

    def plot_pca(self, df_pivot, df_clusters, col_plot):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(df_pivot)
        ax.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=df_clusters.apply(lambda x:self.COLORS[x-1]),
            edgecolors='w'
        )
        # Get the explained variance ratios for each component
        explained_variance_ratios = pca.explained_variance_ratio_

        # Calculate cumulative variance for the first two components
        cumulative_variance_two = sum(explained_variance_ratios[:2])
        # --- Add Legend Here ---
        # Get unique cluster IDs and sort them (e.g., [1, 2, 3])
        unique_clusters = sorted(df_clusters.unique())
        # Create legend labels and handles
        legend_labels = [f'Cluster {cluster_id}' for cluster_id in unique_clusters]
        # Map each cluster to its color in the colormap,         # Add legend to plot
        legend_handles = [
            plt.Line2D([], [], marker='o', linestyle='',
                       color=self.COLORS[i],  # Directly access color from self.COLORS
                       markersize=10, label=label)
            for i, label in enumerate(legend_labels)
        ]
        ax.legend(handles=legend_handles, title='Clusters', loc='best')

        cluster_counts = df_clusters.value_counts()  # Calculate ONCE
        total_points = len(df_clusters)
        texts=[]
        factor = .1 if self.page_name=="names_surnames" else 1
        dense_threshold = total_points / (10*factor) if st.session_state["selected_tab_"+self.page_name] != "map" else 100# Define thresholds
        mid_threshold = total_points / (20*factor) if st.session_state["selected_tab_"+self.page_name] != "map" else 100#
        for i, name in enumerate(df_pivot.index):
            cluster_id = df_clusters[name]
            cluster_size = cluster_counts[cluster_id]
            # Skip dense clusters (>=10% of data)
            if cluster_size >= dense_threshold:
                continue
            # For mid-density (5-10%), annotate every 5th point
            if cluster_size >= mid_threshold and i % 5 != 0:
                continue
            # Annotate sparse clusters (<5%) and selected mid-density points
            texts.append( ax.annotate(name, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=8, alpha=0.7))


        adjust_text(texts)
        if  st.session_state["selected_tab_"+self.page_name]=="name_clustering":
            title = "Names Clustering Based on Provincial Ratios"
        else:
            title = "Province Clustering Based on Name Distributions"
        ax.set_title(f"{title}\nExplained_variance ratios{explained_variance_ratios}\nCumulative variance of two components:{cumulative_variance_two}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True)
        # from mpl_toolkits.mplot3d import Axes3D
        # pca = PCA(n_components=3)
        # reduced_data = pca.fit_transform(df_pivot)
        # fig = plt.figure(figsize=(10, 8))
        # # Get the explained variance ratios for each component
        # explained_variance_ratios = pca.explained_variance_ratio_
        #
        # # Calculate cumulative variance for the first two components
        # cumulative_variance_two = sum(explained_variance_ratios[:2])
        #
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
        #            c=df_clusters.apply(lambda x: self.COLORS[x - 1]), edgecolors='w')
        # # Annotate in 3D with offset to reduce overlap
        # for i, name in enumerate(df_pivot.index):
        #     # Use ax.text for 3D annotations with a slight offset
        #     ax.text(
        #         reduced_data[i, 0] ,  # Small offset in x
        #         reduced_data[i, 1] ,  # Small offset in y
        #         reduced_data[i, 2] ,  # Small offset in z
        #         name,
        #         fontsize=8,
        #         alpha=0.7
        #     )
        # ax.set_title(f"\nExplained_variance ratios{explained_variance_ratios}\nCumulative variance of two components:{cumulative_variance_two}")
        #
        # # Set labels
        # ax.set_xlabel('PC1')
        # ax.set_ylabel('PC2')
        # ax.set_zlabel('PC3')
        col_plot.pyplot(fig)

    def k_means_gui_options(self, col_2, col_3,col_4):
        # Num clusters
        options = list(range(2, 11))
        stored_value = st.session_state.get("n_clusters_" + self.page_name, options[0])
        default_index = options.index(stored_value) if stored_value in options else 0
        st.session_state["n_clusters_" + self.page_name] = col_2.selectbox("Select K: number of clusters",
                                                                           options=options, index=default_index)

        # scaler
        options = ["MaxAbsScaler", "MinMaxScaler", "StandardScaler", "No scaling"]
        stored_value = st.session_state.get("scaler" + self.page_name, options[0])
        default_index = options.index(stored_value) if stored_value in options else 0
        st.session_state["scaler"] = col_3.radio("Select scaling option", options=options, index=default_index)
        st.session_state["optimal_k_analysis"] = col_4.checkbox("Enable cluster analysis", False)
        st.session_state["use_consensus_labels"] = col_4.checkbox("Use consensus labels", False)




