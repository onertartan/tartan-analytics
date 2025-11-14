#from pycirclize import Circos
#from pycirclize.parser import Matrix
from modules.base_page import BasePage
import locale
import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import extra_streamlit_components as stx
import numpy as np
from streamlit.components.v1 import html
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

import plotly.graph_objects as go
locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')


class Migration:
    page_name = "edu_migration"

    @staticmethod
    @st.cache_data
    def load_process_data(incoming_or_outgoing="outgoing"):
        df = pd.read_pickle('data/preprocessed/high_edu.pkl')
        # SELECTING PROVINCES
        df = df[df["General"]["scholarship"] != "AÖ-Ücretli"]  # exclude distant education
        mapping = {'General': ['city', "uni_type"],
                   'Provinces': list(set(df["Provinces"].columns) - {"Total", "Unknown", "Belirsiz", "Toplam"})}
        # Generate column list
        selected_columns = [(top, sub) for top, subs in mapping.items() for sub in subs]
        df_mig = df[selected_columns]
        df_mig.columns = df_mig.columns.droplevel()
        df_mig = df_mig.groupby(["city", "uni_type"]).sum()
        df_mig = df_mig.sort_index(key=lambda x: pd.Index([locale.strxfrm(e) for e in x]))

        df_mig = df_mig.T
        df_mig = df_mig.sort_index(key=lambda x: pd.Index([locale.strxfrm(e) for e in x]))
        all_provinces = df_mig.index.unique()
        df_mig.columns = df_mig.columns.set_levels( df_mig.columns.levels[1].str.replace('Devlet', 'State').str.replace('Vakıf', 'Foundation'), level=1)
        # = df_migration.loc[:, (slice(None), "State")].columns.get_level_values( 0)
        foundation_provinces = df_mig.loc[:, (slice(None), "Foundation")].columns.get_level_values(0)
        # Create df_migration
        mux = pd.MultiIndex.from_product([all_provinces, ["State", "Foundation", "All"]])
        df_mig = df_mig.reindex(columns=mux, fill_value=0)
        # add “All” sub-column (level-1) for every province ------------
        df_mig.loc[:, (slice(None), "All")] = df_mig.loc[:,  (slice(None), "State")].values + df_mig.loc[:,(slice(None), "Foundation")].values

        # Create df_migration_ratio
        mux = pd.MultiIndex.from_product([all_provinces, ["Combined", "State Only", "Foundation Only", "State vs All", "Foundation vs All"]])
        df_mig_ratio = pd.DataFrame(np.nan, index=df_mig.index, columns=mux, dtype='float64')
        if incoming_or_outgoing=="outgoing":
            df_mig_ratio.loc[:, (slice(None), "Combined")] = df_mig.loc[:, (slice(None), "All")].div( df_mig.loc[:, (slice(None), "All")].sum(axis=1), axis=0).values
            df_mig_ratio.loc[:, (slice(None), "State Only")] = df_mig.loc[:, (slice(None), "State")].div(df_mig.loc[:, (slice(None), "State")].sum(axis=1), axis=0).values
            df_mig_ratio.loc[:, (slice(None), "Foundation Only")] = df_mig.loc[:,(slice(None), "Foundation")].div(df_mig.loc[:, (slice(None), "Foundation")].sum(axis=1), axis=0).values
            df_mig_ratio.loc[:, (slice(None), "State vs All")] = df_mig.loc[:, (slice(None), "State")].div(df_mig.loc[:, (slice(None), "All")].sum(axis=1), axis=0).values
            df_mig_ratio.loc[:, (slice(None), "Foundation vs All")] = df_mig.loc[:,(slice(None), "Foundation")].div(df_mig.loc[:, (slice(None), "All")].sum(axis=1), axis=0).values
        # Create df_migration_ratio
        else:
            df_mig_ratio.loc[:, (slice(None), "Combined")] = df_mig.loc[:, (slice(None), "All")].div(df_mig.loc[:, (slice(None), "All")].sum(axis=0).values.reshape(1, -1), axis=1).values
            df_mig_ratio.loc[:, (slice(None), "State Only")] = df_mig.loc[:, (slice(None), "State")].div(df_mig.loc[:, (slice(None), "State")].sum(axis=0).values.reshape(1, -1), axis=1).values
            df_mig_ratio.loc[:, (slice(None), "Foundation Only")] = df_mig.loc[:, (slice(None), "Foundation")].div(df_mig.loc[:, (slice(None), "Foundation")].sum(axis=0).values.reshape(1, -1), axis=1).values
            df_mig_ratio.loc[:, (slice(None), "State vs All")] = df_mig.loc[:, (slice(None), "State")].div(df_mig.loc[:, (slice(None), "All")].sum(axis=0).values.reshape(1, -1), axis=1).values
            df_mig_ratio.loc[:, (slice(None), "Foundation vs All")] = df_mig.loc[:, (slice(None), "Foundation")].div(df_mig.loc[:, (slice(None), "All")].sum(axis=0).values.reshape(1, -1), axis=1).values
            df_mig_ratio.fillna(0, inplace=True)
        # SORT index
        df_mig = df_mig.sort_index(key=lambda x: pd.Index([locale.strxfrm(e) for e in x]))
        df_mig_ratio = df_mig_ratio.sort_index(key=lambda x: pd.Index([locale.strxfrm(e) for e in x]))
        return df_mig, df_mig_ratio, all_provinces, foundation_provinces

    def process_helper_filter(self, df, provinces_from, provinces_to):
        if provinces_from and provinces_to:
            # Filter both rows and columns
            df = df.loc[ df.index.isin(provinces_from),(df.columns.get_level_values(0).isin(provinces_to), slice(None)) ]
        elif provinces_from:
            # Filter only rows (origin provinces)
            df = df.loc[ df.index.isin(provinces_from),: ]
        elif provinces_to:
            # Filter only columns (destination provinces)
            df = df.loc[:, (df.columns.get_level_values(0).isin(provinces_to), slice(None))]
        else:
            return df.copy()
        return df

    def filter_provinces(self, data_all_in_one, col_origin, col_destination, option):

        df_migration_all, df_migration_ratio_all, all_provinces, foundation_provinces = data_all_in_one

        provinces_from = col_origin.multiselect("Select origin provinces", options=all_provinces)
        if option == "Foundation Only":
            provinces_to = col_destination.multiselect("Select destination provinces", options=foundation_provinces)
        else:
            provinces_to = col_destination.multiselect("Select destination provinces", options=all_provinces)

        # Filter if provinces are specified
        df_migration_filtered = self.process_helper_filter(df_migration_all,provinces_from,provinces_to)
        df_migration_ratio_filtered = self.process_helper_filter(df_migration_ratio_all,provinces_from,provinces_to )
        return df_migration_ratio_all, df_migration_filtered, df_migration_ratio_filtered

    def get_summary(self, migration_percentage_dict_filtered):
        upper_levels = ["Combined", "State Only", "Foundation Only", "State vs All", "Foundation vs All"]
        lower_levels = ["Internal", "External", "Total"]
        # Create multiindex columns
        columns = pd.MultiIndex.from_product([upper_levels, lower_levels])
        df_summary = pd.DataFrame(index=migration_percentage_dict_filtered["Combined"].index, columns=columns)
        for upper_level in upper_levels:
            df_summary.loc[:,(upper_level, "Internal")] = self.calculate_total_internal(migration_percentage_dict_filtered[upper_level].copy())
            df_summary.loc[:,(upper_level, "External")] = self.calculate_total_external(migration_percentage_dict_filtered[upper_level].copy())
            df_summary.loc[:,(upper_level, "Total")] = migration_percentage_dict_filtered[upper_level].sum(axis=1).to_frame("Total")
        return df_summary

    def render(self):
        col_origin, col_destination, col_uni, col_dep, _ = st.columns([1, 1, 1, 1, 2.5]) # col_uni and col_dep not used yet (reserved)
        tabs = [stx.TabBarItemData(id = "incoming", title="Incoming migration", description=""),
                stx.TabBarItemData(id = "outgoing", title="Outgoing migration", description="")]
        incoming_outgoing_tab = stx.tab_bar(data=tabs, default="outgoing")
        # Check if tab changed and reload data if needed
        # Use a dynamic cache key based on the current tab
        cache_key = f"migration_data_{incoming_outgoing_tab}"
        # Load data with the current tab as parameter
        if cache_key not in st.session_state:
            st.session_state[cache_key] = Migration.load_process_data(incoming_outgoing_tab)
        # Set the current migration data
        st.session_state["migration_data"] = st.session_state[cache_key]

        # Load data if not in session state
       # if "migration_data" not in st.session_state:
         #   st.session_state["migration_data"] = Migration.load_process_data(incoming_outgoing_tab)

        st.write("Tab changed in render as "+incoming_outgoing_tab)
        # departments = col_dep.multiselect("Select departments", list(df["General"]["dep_name"].unique()))
        # # Finally, select departments (this will apply the department filter if any department is selected)
        # if departments:
        #     df = df[df["General"]["dep_name"].isin(departments)]
        # if st.session_state["selected_tab_"+self.page_name] == "incoming":
        #     # Select universities
        #     universities = col_uni.multiselect("Select universities", options=list(df["General"]["uni_name"].unique()))
        # # # Update universities list to show only universities with the selected department
        # # # If universities are selected, filter them
        #     if universities:
        #         df = df[df["General"]["uni_name"].isin(universities) ]
        col_internal_external, col_ratio_type, col_minimum_ratio, col_line_type, col_line_width,_= st.columns([1,1,2,1,1,1])
        # Checkbox for inter-province migration (default: checked)
        show_inter_province = col_internal_external.checkbox("Show Migration Between Provinces", value=True)
        # Checkbox for self-loops (default: unselected)
        show_self_loops = col_internal_external.checkbox("Show Internal Choice (same province)", value=False)
        option = col_ratio_type.radio("Analysis Type:", options=["Combined", "State Only", "Foundation Only", "State vs All", "Foundation vs All"])

       # if "migration_data" not in st.session_state:
       #     st.session_state["migration_data"] = Migration.load_process_data(incoming_outgoing_tab)

        df_migration_ratio_all, df_migration_filtered, df_migration_ratio_filtered = self.filter_provinces(st.session_state["migration_data"],col_origin, col_destination, option)

        threshold = col_minimum_ratio.slider("Minimum Migration Value", min_value=0., max_value=100., value=1., step=0.10, help="Show only migration flows above this value", key="mig_threshold")
        layer_type = col_line_type.radio("Select Migration Flow Visualization Type", ["Arc Layer", "Line Layer"], help="Arc Layer shows curved flows, Line Layer shows straight lines" )
        selected_width = col_line_width.slider("Change line width", min_value=1., max_value=5., value=1., step=.1)
    #    st.dataframe((df_migration_ratio_filtered*100).round(1))
        if option == "Combined":
            cols = ["Foundation", "State", "All"]
        if "State" in option:
            cols = ["State"]
        elif "Foundation" in option:
            cols = ["Foundation"]
        if "All" in option:
            cols.append("All")

        # Filter ratios and counts
        df_migration_percentage = df_migration_ratio_filtered.loc[:, (slice(None), option)].droplevel(1, axis=1)*100
        if incoming_outgoing_tab=="incoming":
            df_migration_percentage=df_migration_percentage.T
        df_migration_percentage = self.calculate_internal_external_sums(df_migration_percentage, show_self_loops, show_inter_province)
        df_migration_filtered = self.calculate_internal_external_sums_counts(df_migration_filtered, show_self_loops, show_inter_province,cols)

        st.dataframe(df_migration_percentage.round(2))
        self.plot_on_map(df_migration_ratio_all,df_migration_ratio_filtered, show_inter_province, show_self_loops, layer_type, selected_width, threshold,option)

       # self.get_summary(migration_percentage_dict_filtered)

        st.dataframe(df_migration_filtered )

       # st.dataframe(df_migration[(df_migration > 0).all(axis=1)])

        #self.plot_network(df_migration_ratio)
        # Create and display the CHORD

    #  m = gdf_result.explore(**explore_kwargs)
    # folium_static(m, width=1100, height=450)
    #  self.plot_circos_chart(df,col_dep)

    def plot_circos_chart(self, df, col):
        threshold = col.slider("Minimum Migration Value",  1,  10, 1, help="Show only migration flows above this value")

        # Convert dataframe to from-to table format
        fromto_data = []
        for prov1 in df.index:
            for prov2 in df.columns:
                value = df.loc[prov1, prov2]
                if df.loc[prov1, prov2] > threshold:  # Only include significant migrations
                    fromto_data.append([prov1, prov2, value])

        # Create from-to table dataframe
        fromto_table_df = pd.DataFrame(fromto_data, columns=["from", "to", "value"])
        # Convert to matrix format for pycirclize
        matrix = Matrix.parse_fromto_table(fromto_table_df)

        # Initialize Circos plot
        circos = Circos.initialize_from_matrix(
            matrix,
            space=3,  # Space between sectors
            cmap="viridis",  # Color map for chords
            ticks_interval=5,  # Adjust based on your data scale
            label_kws=dict(size=10, r=110),  # Smaller font for 81 provinces
            link_kws=dict(direction=1, ec="black", lw=0.5),  # Chord styling
        )

       # Create a matplotlib figure for the chord diagram

        fig = circos.plotfig(figsize=(20, 20))
        col.pyplot(fig)

    import numpy as np

    import numpy as np

    def slice_polygon(self,center_lonlat, radius_deg, start_azimuth, end_azimuth, steps=30):
        """closed polygon [ [lon,lat], … ] for a circular sector"""
        cx, cy = center_lonlat
        az = np.linspace(start_azimuth, end_azimuth, steps)
        xs = cx + radius_deg * np.cos(az)
        ys = cy + radius_deg * np.sin(az)
        coords = np.column_stack([xs, ys]).tolist()
        coords.append([cx, cy])  # close at centre
        return coords
    def plot_on_map(self,df_migration_ratio_all, df_migration_ratio_filtered, show_inter_province, show_self_loops, layer_type, selected_width, threshold,option):
        df_migration_percentage_filtered = df_migration_ratio_filtered.loc[:, (slice(None), option)].droplevel(1, axis=1) * 100
        df_migration_percentage_all = df_migration_ratio_all.loc[:, (slice(None), option)].droplevel(1, axis=1) * 100
        # User selects layer type
        shapefile_path = "data/turkey_province_centers.geojson"
        gdf = gpd.read_file(shapefile_path, encoding='utf-8')
        city_to_coord = gdf.set_index('province')[['lon', 'lat']].to_dict('index')
        city_to_coord = {k: [v['lon'], v['lat']] for k, v in city_to_coord.items()}
        # Add tooltip text to nodes
        # ScatterplotLayer for province nodes
        diag = pd.DataFrame({"province": df_migration_percentage_all.index, "percentage": df_migration_percentage_all.values.diagonal()})
        df_internal = gdf.merge(diag, on="province")
        df_internal['tooltip_text'] = df_internal.apply(lambda row: f"<b>{row['province']}</b> Internal percentage: {row['percentage']:.1f}%", axis=1)

        df_nodes=df_internal.copy()
        df_nodes['tooltip_text'] = df_nodes['province'].apply(lambda x: f"<b>{x}</b> Province")
        df_internal = df_internal.set_index("province")
        scatter_layer = pdk.Layer(
            'ScatterplotLayer',
            data=df_nodes,
            get_position=['lon', 'lat'],
            get_radius=3000,
            get_fill_color=[0, 0, 255, 200],
            pickable=True
        )
        layers = []
        layers.append(scatter_layer)
        # Compute initial view state (centered on Turkey)
        view_state = pdk.ViewState(latitude=39.3, longitude=35.0, zoom=5.36, pitch=0, bearing=0)

        # Add coordinates for non-self-loops
        if show_inter_province:
            # Prepare flows data (filtering according to the threshold)
            df_flows = df_migration_percentage_filtered.copy()
            df_flows[df_flows < threshold] = 0.
            df_flows = df_flows.loc[:, (df_flows != 0).any(axis=0)]
            df_flows = df_flows.reset_index().melt(id_vars='index', var_name='to', value_name='percentage')
            df_flows.rename(columns={'index': 'from'}, inplace=True)
            # Split flows into self-loops and non-self-loops
            non_self_loops = df_flows[df_flows['from'] != df_flows['to']] if show_inter_province else pd.DataFrame()
            non_self_loops['from_lon'] = non_self_loops['from'].map(lambda c: city_to_coord.get(c, [None, None])[0])
            non_self_loops['from_lat'] = non_self_loops['from'].map(lambda c: city_to_coord.get(c, [None, None])[1])
            non_self_loops['to_lon'] = non_self_loops['to'].map(lambda c: city_to_coord.get(c, [None, None])[0])
            non_self_loops['to_lat'] = non_self_loops['to'].map(lambda c: city_to_coord.get(c, [None, None])[1])
            # Drop rows with missing coordinates
            non_self_loops = non_self_loops.dropna(subset=['from_lon', 'from_lat', 'to_lon', 'to_lat'])
            # Add tooltip text to non-self-loops
            non_self_loops['tooltip_text'] = non_self_loops.apply(lambda row: f"From <b>{row['from']}</b> to <b>{row['to']}</b>: {row['percentage']:.1f}%", axis=1 )
            if layer_type == "Arc Layer":
                flow_layer = pdk.Layer(
                    'ArcLayer',
                    data=non_self_loops,
                    get_source_position=['from_lon', 'from_lat'],
                    get_target_position=['to_lon', 'to_lat'],
                    get_source_color=[255, 0, 0, 160],
                    get_target_color=[0, 255, 0, 160],
                    get_width=f'{selected_width ** 2}*percentage/25',
                    pickable=True,
                    auto_highlight=True
                )
            else:  # Line Layer
                flow_layer = pdk.Layer(
                    'LineLayer',
                    data=non_self_loops,
                    get_source_position=['from_lon', 'from_lat'],
                    get_target_position=['to_lon', 'to_lat'],
                    get_color=[255, 165, 0, 160],
                    get_width=f'{selected_width ** 2}*percentage/25',
                    pickable=True,
                    auto_highlight=True
                )
            layers.append(flow_layer)

        if show_self_loops:

            df_internal=df_internal.loc[df_internal["percentage"] > threshold]
            print(df_internal)
            idx = cols = df_internal.index
            pie_rows = []  # list of dicts for the PolygonLayer
            df_migration_percentage_state_vs_all = df_migration_ratio_all.loc[idx, (cols, "State vs All")].droplevel(1, axis=1) * 100
            df_migration_percentage_foundation_vs_all = df_migration_ratio_all.loc[idx, (cols, "Foundation vs All")].droplevel(1, axis=1) * 100
            if option != "Combined":
                self_loop_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=df_internal,
                    get_position=['lon', 'lat'],
                    get_radius='percentage * 450',
                    get_fill_color=[237, 174, 73, 255],
                    pickable=True,
                    auto_highlight=True
                )
                layers.append(self_loop_layer)
            else:
                for city in df_internal.index:

                    lon, lat = city_to_coord[city]
                    combined_pct = df_internal.loc[city, "percentage"]  # Combined value
                    state_pct = df_migration_percentage_state_vs_all.loc[city, city]
                    found_pct = df_migration_percentage_foundation_vs_all.loc[city, city]
                    if city == "Tunceli":
                        print("Tunceli",state_pct,found_pct,combined_pct)
                    # central angles (radians)
                    state_angle = 2 * np.pi * state_pct / combined_pct
                    found_angle = 2 * np.pi * found_pct / combined_pct
                    # STATE slice (start at 0)
                    r_deg = combined_pct * 500 / 111_000  # km → deg scaling (tune 500)
                    combined_poly = self.slice_polygon((lon, lat), r_deg, 0, 2 * np.pi)
                    pie_rows.append({
                        "polygon": combined_poly,
                        "tooltip_text": f"<b>{city}</b> Combined-Internal Percentage: {combined_pct:.1f}%",
                        "fill_color": [237, 174, 73, 255]
                    })
                    if found_pct > 0:
                        state_poly = self.slice_polygon((lon, lat), 0.8*r_deg, 0, state_angle)
                        pie_rows.append({
                            "polygon": state_poly,
                            "tooltip_text": f"<b>{city}</b> State-Internal Percentage: {state_pct:.1f}%",
                            "fill_color": [0, 100, 0, 255]
                        })
                        # FOUNDATION slice (continue after state)
                        found_poly = self.slice_polygon((lon, lat), 0.8*r_deg, state_angle, state_angle + found_angle)
                        pie_rows.append({
                            "polygon": found_poly,
                            "tooltip_text": f"<b>{city}</b> Foundation-Internal Percentage: {found_pct:.1f}%",
                            "fill_color": [220, 20, 60, 255]
                        })

                    pie_df = pd.DataFrame(pie_rows)
                    # --- PolygonLayer for pie slices ---
                    pie_layer = pdk.Layer(
                        "PolygonLayer",
                        data=pie_df,
                        get_polygon="polygon",
                        get_fill_color="fill_color",
                        get_line_color=[0, 255, 0, 120],
                        line_width_min_pixels=1,
                        pickable=True,
                        auto_highlight=True,
                        id="pie-slices"
                    )
                    layers.append(pie_layer)

            # Simple tooltip that works for all layers
        tooltip = {
            "html": "{tooltip_text}",
            "style": {
                "background": "grey",
                "color": "white",
                "font-family": '"Helvetica Neue", Arial',
                "z-index": "10000",
                "font-size": "12px",  # Add this line for bigger text

            }
        }

        # Create Deck
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style='road'
        )

        # Display in Streamlit
        # Remove sliders
        # col_size, _ = st.columns([2, 8])
        # graph_width = col_size.slider(...)
        # graph_height = col_size.slider(...)

        # Display chart
        _, col_deck, _ = st.columns([1, 3, 1])
        col_deck.pydeck_chart(deck, use_container_width=False)


    def run(self):
        self.render()

    def calculate_total_internal(self, df_migration_percentage):
        for idx1 in df_migration_percentage.index:
            for idx2 in df_migration_percentage.columns:
                if not idx1 in (idx2,):
                    df_migration_percentage.loc[idx1, idx2] = 0
        # returns a column dataframe consisting of diagonal values
        return df_migration_percentage.sum(axis=1).to_frame("Internal")


    def calculate_total_external(self, df_migration_percentage):
        for idx1 in df_migration_percentage.index:
            for idx2 in df_migration_percentage.columns:
                if idx1 in (idx2,):
                    df_migration_percentage.loc[idx1, idx2] = 0
        # returns a column dataframe sum of non-diagonal values
        return df_migration_percentage.sum(axis=1).to_frame("External")

    def plot_network(self, df, layout="kamada_kawai"):
        # 0. Square the matrix (your existing code)
        all_nodes = df.columns.union(df.index)

        # 3) Build directed graph from rectangular df
        G = nx.DiGraph()
        G.add_nodes_from(all_nodes)
        for i in df.index:
            for j in df.columns:
                w = float(df.at[i, j])
                if w > 0:  # show only edges above threshold if desired
                    G.add_edge(i, j, weight=w)
        # 1. OPTIONAL: cap extreme edge weights before layout
        if G.number_of_edges():
            cap = np.percentile([d["weight"] for _, _, d in G.edges(data=True)], 95)
            for _, _, d in G.edges(data=True):
                d["weight"] = min(d["weight"], cap)

        # 2. Layout
        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G, weight="weight")
        elif layout == "spring":
            pos = nx.spring_layout(G, weight="weight", seed=42)
        else:
            pos = nx.spectral_layout(G)
        # 3. Node importance & compressed sizes
        node_importance = dict(G.degree(weight="weight"))
        max_imp = max(node_importance.values()) or 1.0

        # Log-scaled sizes (10 px floor, 100 px ceiling)
        node_size = [
            #10+ 100 * np.log1p(node_importance[n]) / np.log1p(max_imp)
             node_importance[n]/10
            for n in G.nodes()
        ]

        # Color-bar limits (5th–95th percentile)
        imp_vals = list(node_importance.values())
        p5, p95 = np.percentile(imp_vals, [5, 95])

        # 4. Edge trace
        edge_x, edge_y, edge_width = [], [], []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_width.append(d["weight"] * 5)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="rgba(100,100,100,0.5)"),
            hoverinfo="none",
            mode="lines",
        )

        # 5. Node trace
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = [f"{n}<br>Importance: {node_importance[n]:.2f}" for n in G.nodes()]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[str(n) for n in G.nodes()],
            textposition="bottom center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlOrRd",
                color=list(node_importance.values()),
                size=node_size,
                sizemode="area",
                cmin=p5,
                cmax=p95,
                colorbar=dict(
                    thickness=15,
                    title="Node Importance",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=2,
            ),
        )

        # 6. Figure & Streamlit display
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Province Migration Network",
                titlefont_size=20,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        fig.update_layout(height=1280, width=2048)

        col_network, _ = st.columns([9, 1])
        with col_network:
            st.plotly_chart(fig, use_container_width=True)

    def calculate_internal_external_sums(self, df, show_self_loops, show_inter_province):

        df = pd.concat([df, self.calculate_total_internal(df.copy()),
                        self.calculate_total_external(df.copy()),
                        df.sum(axis=1).to_frame("Total")],
                       axis=1)
        df = df[list(df.columns[-3:]) + list(df.columns[:-3])] #reorders columns (first 3 are internal,extertanl,total)
        if show_self_loops and not show_inter_province:
            df = self.calculate_total_internal(df.copy())
            #df_migration_ratio = df_migration_ratio[(df_migration_ratio > 0).all(axis=1)]
            df = df.sort_values(by="Internal", ascending=False)
            df = df[[df.columns[-1]] + list(df.columns[:-1])] # we don't drop columns until plotting map
        elif show_inter_province and not show_self_loops:
            df = df.sort_values(by="External", ascending=False)

        return df
    def calculate_internal_external_sums_counts(self, df, show_self_loops, show_inter_province,cols):
        df = df.loc[:, (slice(None), cols)]

        selected_columns = [(upper, lower) for upper in df.columns.levels[0] for lower in cols if (upper, lower) in df.columns]
        df = df.reindex(columns=selected_columns)
        internal = pd.concat({name:self.calculate_total_internal(df.loc[:, (slice(None), name)].droplevel(1, axis=1)) for name in cols},  axis=1 )
        internal = internal.swaplevel(axis=1).sort_index(axis=1)
        if show_self_loops and not show_inter_province:
            return internal
        else:
            external = pd.concat( {name: self.calculate_total_external(df.loc[:, (slice(None), name)].droplevel(1, axis=1)) for name in cols}, axis=1)
            external = external.swaplevel(axis=1).sort_index(axis=1)
            if show_inter_province and not show_self_loops:
                df = pd.concat([external, df], axis=1)
            else:
                df = pd.concat([internal, external, df],axis=1)
            return df


Migration().run()