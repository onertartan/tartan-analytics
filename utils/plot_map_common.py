import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from raceplotly.plots import barplot
from utils.helpers_ui import figure_setup
from matplotlib import colors, pyplot as plt
from streamlit_folium import folium_static
import folium
from utils.plot_pyramid import plot_pyramid_plotly, plot_pyramid_matplotlib
from utils.query import get_df_change, get_geo_df_result, get_df_result


def resize_folium_map(m):
    # Convert map to HTML string
    map_html = m._repr_html_()
    # Custom HTML with responsive iframe
    html = f"""
    <style>
        .responsive-map {{
            position: relative;
            padding-bottom: 45%; /* 4:3 Aspect Ratio */
            height: 0;
            overflow: hidden;
        }}
        .responsive-map iframe {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
    </style>
    <div class="responsive-map">
        {map_html}
    </div>
    """

    # Render the HTML
    components.html(html, height=400)

def plotter_folium_static(gdf_result, title, geo_scale, *args):
    st.markdown(f"<h3 style='text-align: center; color: grey;'>{title}</h1>", unsafe_allow_html=True)
   # df_result = df_result.reset_index()
    choropleth_map = folium.Map(location=[36.5, 35], zoom_start = 6.4, tiles="cartodb positron")
    gdf_result = gdf_result.reset_index()
    folium.Choropleth(
        geo_data=gdf_result,
        name="choropleth",
        data=gdf_result,
        columns=geo_scale + ["result"] if "district" not in geo_scale else ["id", "result"],
        key_on=f'feature.properties.{geo_scale[0]}' if "district" not in geo_scale else "feature.properties.id",
        fill_color=st.session_state["selected_cmap"],
        nan_fill_color="purple",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Result",
    ).add_to(choropleth_map)
    resize_folium_map(choropleth_map)


def plot_race(df_result,geo_scale):
    # Generate colors using Matplotlib
    pyplot_colors = plt.get_cmap('tab10').colors + plt.get_cmap('Accent').colors + plt.get_cmap('Pastel2').colors
    colors_255 = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in pyplot_colors]

    # Get unique geoscales (e.g. provinces)
    df_first_year = df_result.loc[st.session_state["slider_year_2"][0]]
    unique_geoscales = df_first_year.reset_index().sort_values(by="result", ascending=False).loc[:min(20,len(df_first_year)-1), geo_scale]

    # Create a mapping of provinces to colors
    color_mapping = {province: "rgb"+str(colors_255[i % len(colors_255)]) for i, province in enumerate(unique_geoscales)}
    # Assign colors to the 'color' column based on the province
    df_result = df_result.reset_index()
    my_raceplot = barplot(df_result,
                          item_column=geo_scale,
                          value_column='result',
                          time_column='year',
                          item_color = color_mapping)

    fig = my_raceplot.plot(title=f'Top 10 {geo_scale}s from {st.session_state["slider_year_2"][0]} to {st.session_state["slider_year_2"][1]} ',
                     item_label=f'Top 10 {geo_scale}s',
                     value_label='Result',
                     frame_duration=800)
    fig.update_layout( margin=dict(l=20, r=60, t=20, b=20))
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def plotter_folium_interactive(gdf_result, *args):
    gdf_result = gdf_result.reset_index()
    m = gdf_result.explore(
        column="result",  # make choropleth based on "BoroName" column
        tooltip="province",  # show "BoroName" value in tooltip (on hover)
        popup=True,  # show all values in popup (on click)
        tiles="CartoDB positron",  # use "CartoDB positron" tiles
        cmap=st.session_state["selected_cmap"],  # use "Set1" matplotlib colormap
        columns=[col for col in gdf_result.columns if col != "year"],
        style_kwds=dict(color="black", line_width=.01),  # use black outline
    )
    folium_static(m)


def plotter_matplotlib(gdf_result, title, geo_scale, ax):
    norm = None
    if st.session_state["display_percentage"] and gdf_result["result"].min() < 0 and gdf_result["result"].max() > 0:
        norm = colors.TwoSlopeNorm(vmin=gdf_result["result"].min(), vcenter=0, vmax=gdf_result["result"].max())
    gdf_result.plot(ax=ax, column=gdf_result["result"], cmap=st.session_state["selected_cmap"], edgecolor="black",
                    linewidth=.2, norm=norm,
                    legend=True, legend_kwds={"shrink": .6}, antialiased=True)  # column=selected_feature,
    ax.legend(fontsize=4, bbox_to_anchor=(.01, 0.01), loc='lower right', fancybox=True, shadow=True)
    ax.set_title(title)
    region_text_vertical_shift_dict = {"Akdeniz": -1.05, "Batı Marmara": -.6, "Doğu Marmara": .2,
                                       "Ortadoğu Anadolu": .25, "Doğu Karadeniz": -0.1}
    region_text_horizontal_shift_dict = {"Akdeniz": -.5}
    font_size_dict = {"region": 6.5, "sub-region": 5}
    if "district" not in geo_scale:
        gdf_result.reset_index().apply(
            lambda x: ax.annotate(text=x[geo_scale[0]], size=font_size_dict.get(geo_scale[0], 4.5),
                                  xy=(x.geometry.centroid.x + region_text_horizontal_shift_dict.get(x[geo_scale[0]], 0),
                                      x.geometry.centroid.y + region_text_vertical_shift_dict.get(x[geo_scale[0]], 0)),
                                  # x.geometry.centroid.coords[0]   ,
                                  ha='center', va="bottom"), axis=1)
    ax.axis("off")
    ax.margins(x=0, y=0)


def plot_generic(col_plot, col_df, gdf_borders, df_result, geo_scale, plotter_func, fig=None, axs=None):
    display_change = (st.session_state["year_1"] != st.session_state["year_2"])
    years = df_result.index.get_level_values(0).unique()
    col_df.write("""<style>.dataframe-margin { margin-bottom: 80px;} </style>""", unsafe_allow_html=True )
    for i, year in enumerate(years):
        print("tgb:\n", df_result.loc[year].head(), "\nres:\n", gdf_borders.head())
        gdf_result = get_geo_df_result(gdf_borders, df_result.loc[year], geo_scale)
        with col_plot:
            title = f'Results for {[st.session_state.year_1, st.session_state.year_2][i]}'
            plotter_func(gdf_result, title, geo_scale, axs[i, 0] if axs is not None else None)
            # Add title to the subplot
            col_df.markdown('<div class="dataframe-margin">', unsafe_allow_html=True)
            col_df.dataframe(df_result.loc[year].sort_values(by="result", ascending=False))
            st.markdown('</div>', unsafe_allow_html=True)

    if display_change:
        df_change = get_df_change(df_result)
        gdf_result = get_geo_df_result(gdf_borders, df_change, geo_scale)
        with col_plot:
            start_word = "Change"
            if st.session_state["display_percentage"]:
                start_word = "Relative change in ratio"
            title = f"{start_word} between years {st.session_state.year_1} and {st.session_state.year_2}"
            plotter_func(gdf_result, title, geo_scale, axs[2, 0] if axs is not None else None)
            col_df.markdown('<div class="dataframe-margin">', unsafe_allow_html=True)
            col_df.dataframe(df_change.sort_values(by="result", ascending=False))
            col_df.markdown('</div>', unsafe_allow_html=True)
    if fig:
        col_plot.pyplot(fig)

def plot(col_plot, col_df, df_data, gdf_borders, selected_features, geo_scale):

    if geo_scale == ["district"]:
        geo_scale = geo_scale + ["province"]  # geo_scale = ["province", "district"]
    if st.session_state["visualization_option"] == "Plotly":
        plot_pyramid_plotly(df_data)
    elif st.session_state["visualization_option"] == "Matplotlib":
        plot_pyramid_matplotlib(df_data)
    elif st.session_state["visualization_option"] == "Plotly race chart":
        df_result = get_df_result(df_data, selected_features, geo_scale, list(range(st.session_state["slider_year_2"][0], st.session_state["slider_year_2"][1] + 1)))
        plot_race(df_result, geo_scale[0])
    else:
        df_result = get_df_result(df_data, selected_features, geo_scale, sorted({st.session_state["year_1"], st.session_state["year_2"]}))
        print("OOO:",df_result.head())
        if st.session_state["visualization_option"] == "Matplotlib (static)":
            fig, axs = figure_setup((st.session_state["year_1"] != st.session_state["year_2"]))
            plot_generic(col_plot, col_df, gdf_borders, df_result, geo_scale, plotter_matplotlib, fig, axs)
        elif st.session_state["visualization_option"] == "Folium (static)":
            plot_generic(col_plot, col_df, gdf_borders, df_result, geo_scale, plotter_folium_static)
        else:
            plot_generic(col_plot, col_df, gdf_borders, df_result, geo_scale, plotter_folium_interactive)

