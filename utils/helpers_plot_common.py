import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from raceplotly.plots import barplot
from utils.helpers_ui import figure_setup
from matplotlib import colors, pyplot as plt
from streamlit_folium import folium_static
import folium
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

def create_folium_static(gdf_result,title,geo_scale):
    st.markdown(f"<h3 style='text-align: center; color: grey;'>{title}</h1>", unsafe_allow_html=True)
   # df_result = df_result.reset_index()
    choropleth_map = folium.Map(location=[36.5, 35], zoom_start = 6.4, tiles="cartodb positron")
    gdf_result = gdf_result.reset_index()
    # usmap_json_with_id = merged.set_index(keys = "id").to_json()
    #merged["id"]=  merged["id"].astype("str")
    folium.Choropleth(
        geo_data=gdf_result,
        name="choropleth",
        data=gdf_result,
        columns=geo_scale + ["result"] if "district" not in geo_scale else ["id","result"],
        key_on=f'feature.properties.{geo_scale[0]}' if "district" not in geo_scale else "feature.properties.id",
        fill_color=st.session_state["selected_cmap"],
        nan_fill_color="purple",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Result",
    ).add_to(choropleth_map)
    resize_folium_map(choropleth_map)

    """
    folium.Choropleth(
        geo_data=gdf_borders,
        name="choropleth",
        data=df_result.reset_index(),
        columns=["province", "result"],
        key_on='feature.properties.id',
        fill_color=st.session_state["selected_cmap"],
        nan_fill_color="purple",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Result",
    ).add_to(choropleth_map)
    resize_folium_map(choropleth_map)
    """
def get_geo_df_result(gdf_borders, df_result, geo_scale):
   # if "district" not in geo_scale:
     #   df_result = df_result.reset_index()
        print("\nAAA:\n",gdf_borders.head(),"\nBBB:\n",df_result.head())
        print("\nCCC:\n",gdf_borders.dissolve(by=geo_scale).head())
        print("before:\n",gdf_borders.shape,"\n")
        gdf_result = gdf_borders.dissolve(by=geo_scale)[["id","geometry"]].merge(df_result,left_index=True,right_index=True)  # after dissolving index becomes geo_scal,so common index is geo_scale(example:province)
        print("after:\n",gdf_result.shape,"\n")
        print("\nEEE:\n",gdf_result.head())
   # else:
   #     gdf_result = gdf_borders.merge(df_result, on=geo_scale)  # gdf_results.append(gdf_borders.merge(df_result, left_on=geo_scale, right_on=geo_scale)),
        return gdf_result

def plot_folium_static(col_plot, col_df, gdf_borders, df_result, geo_scale, input_color_theme="Viridis"):
    col_df.write(
        """<style>
        .dataframe-margin {
            margin-bottom: 0px;  margin-top: 0px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    display_change = (st.session_state["year_1"] != st.session_state["year_2"])
    years = df_result.index.get_level_values(0).unique()

    for i, year in enumerate(years):
        gdf_result = get_geo_df_result(gdf_borders, df_result.loc[year], geo_scale)
        with col_plot:
            create_folium_static(gdf_result,f"Result for year {year}", geo_scale)

        col_df.markdown('<div class="dataframe-margin">', unsafe_allow_html=True)
        col_df.dataframe(df_result.loc[year].sort_values(by="result",ascending=False))
        col_df.markdown('</div>', unsafe_allow_html=True)

    #      folium_static(choropleth_map, width=900, height=400)
    if display_change:
        df_change = get_df_change(df_result)
        gdf_result = get_geo_df_result(gdf_borders, df_change, geo_scale)

        start_word = "Change"
        if st.session_state["display_percentage"]:
            start_word = "Relative change in ratio"
        title = f"{start_word} between years {st.session_state.year_1} and {st.session_state.year_2}"
        with col_plot:
            create_folium_static(gdf_result, title,geo_scale)
        col_df.markdown('<div class="dataframe-margin">', unsafe_allow_html=True)
        col_df.dataframe(df_change)
        col_df.markdown('</div>', unsafe_allow_html=True)
def plot_race(df_result,geo_scale):
    print("ZXCV:", df_result.head(20))
    # Generate colors using Matplotlib
    colors = plt.get_cmap('tab10').colors + plt.get_cmap('Accent').colors + plt.get_cmap('Pastel2').colors
    colors_255 = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

    # Get unique provinces
    df_first_year= df_result.loc[st.session_state["slider_year_2"][0]]
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

def folium_interactive_plotter(gdf_result):
    gdf_result = gdf_result.reset_index()
    print("TRE:",gdf_result.dtypes,gdf_result.head())


    m = gdf_result.explore(
        column="result",  # make choropleth based on "BoroName" column
        tooltip="province",  # show "BoroName" value in tooltip (on hover)
        popup=True,  # show all values in popup (on click)
        tiles="CartoDB positron",  # use "CartoDB positron" tiles
        cmap= st.session_state["selected_cmap"],  # use "Set1" matplotlib colormap
    columns = [col  for col in gdf_result.columns if col!="year"],
        style_kwds=dict(color="black", line_width=.01),  # use black outline
    )

    folium_static(m)
def plot_folium_interactive(col_plot, col_df, gdf_borders, df_result, geo_scale):
    gdf_result = get_geo_df_result(gdf_borders, df_result, geo_scale)

    display_change = (st.session_state["year_1"] != st.session_state["year_2"])
    fig, axs = figure_setup(display_change)
    years = df_result.index.get_level_values(0).unique()
    for i,year in enumerate(years):
        ax = axs[i,0]
        folium_interactive_plotter(gdf_result)
    if display_change:
        ax = axs[2, 0]
        df_change = get_df_change(df_result)
        gdf_result = get_geo_df_result(gdf_borders, df_change, geo_scale)
        folium_interactive_plotter(gdf_result)

def plot_matplotlib_plotter(gdf_result,ax,geo_scale):
    norm = None
    if st.session_state["display_percentage"] and gdf_result["result"].min() < 0 and gdf_result["result"].max() > 0:
        norm = colors.TwoSlopeNorm(vmin=gdf_result["result"].min(), vcenter=0, vmax=gdf_result["result"].max())
    gdf_result.plot(ax=ax, column=gdf_result["result"], cmap=st.session_state["selected_cmap"], edgecolor="black",
                    linewidth=.2, norm=norm,
                    legend=True, legend_kwds={"shrink": .6}, antialiased=True)  # column=selected_feature,
    ax.legend(fontsize=4, bbox_to_anchor=(.01, 0.01), loc='lower right', fancybox=True, shadow=True)
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
    ax.margins(x=0,y=0)

def plot_matplotlib(col_plot, col_df, gdf_borders, df_result, geo_scale):

    display_change = (st.session_state["year_1"] != st.session_state["year_2"])
    fig, axs = figure_setup( display_change )
    years = df_result.index.get_level_values(0).unique()
    col_df.write(
        """<style>
        .dataframe-margin {
            margin-bottom: 80px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    for i,year in enumerate(years):
        ax = axs[i,0]
        print("tgb:\n", df_result.loc[year].head(), "\nres:\n", gdf_borders.head())
        gdf_result = get_geo_df_result(gdf_borders, df_result.loc[year], geo_scale)
        with col_plot:
            plot_matplotlib_plotter(gdf_result, ax, geo_scale)
            # Add title to the subplot
            ax.set_title(f'Results for {[st.session_state.year_1, st.session_state.year_2][i]}')
            col_df.markdown('<div class="dataframe-margin">', unsafe_allow_html=True)
            col_df.dataframe(df_result.loc[year].sort_values(by="result",ascending=False))
            st.markdown('</div>', unsafe_allow_html=True)

    if display_change:
        ax = axs[2, 0]
        df_change = get_df_change(df_result)
        gdf_result = get_geo_df_result(gdf_borders, df_change, geo_scale)
        with col_plot:
            plot_matplotlib_plotter(gdf_result, ax, geo_scale)
            start_word = "Change"
            if st.session_state["display_percentage"]:
                start_word = "Relative change in ratio"
            ax.set_title(f"{start_word} between years {st.session_state.year_1} and {st.session_state.year_2}")
            col_df.markdown('<div class="dataframe-margin">', unsafe_allow_html=True)

            col_df.dataframe(df_change.sort_values(by="result",ascending=False))
            col_df.markdown('</div>', unsafe_allow_html=True)


    """    
    if display_change: # animate
        fig, ax = plt.subplots(figsize=(12, 7))

        def animate(i):
            ax.cla()
            year = 1800 + i
            gdf_result = get_geo_df_result(gdf_borders, df_result, geo_scale)
            plot_matplotlib_plotter(gdf_result, ax, geo_scale)
            ax.set_title(f'Year {year}')
            ax.set_xscale('log')
            ax.set_ylim(25, 90)
            ax.set_xlim(100, 100000)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:5], labels[:5], loc='upper left')

        ani = FuncAnimation(fig=fig, func=animate, frames=220, interval=100)
        plt.show()
    """

    col_plot.pyplot(fig)

def get_df_year_and_features(df_data, nom_denom_selection, year, selected_features_dict, geo_scale):

    df_codes = pd.read_csv("data/preprocessed/region_codes.csv", index_col=0)
    #df_codes = pd.read_excel("data/updated_excel_file.xlsx", index_col=-1)


    df = df_data[nom_denom_selection]
    print("START:",df.head())
    selected_features = selected_features_dict[nom_denom_selection]
    if df.columns.nlevels > 1:  # Check if the DataFrame has multiple column levels
        print("ZZZ:\n",df.loc[year, selected_features].droplevel(1, axis=1).head())
        df = pd.DataFrame(
            df.loc[year, selected_features].droplevel(1, axis=1).sum(axis=1))  # .reset_index()
    else:
        if isinstance(df.loc[year, selected_features],    pd.Series):  # if it is a pandas series(for example it contains single column like sex=male)
            df = pd.DataFrame(df.loc[year, selected_features].to_frame().sum(axis=1))  # .reset_index()
        else:  # if it is a dataframe, then sum column values along horizontal axis
            df = pd.DataFrame(df.loc[year, selected_features].sum(axis=1))  # .reset_index()
    df.rename(columns={df.columns[-1]: "result"}, inplace=True)
    print("geo_scale:",geo_scale,"GGG:",df.head())
    print("\ndf_codes HHH:",df_codes.head())
    if geo_scale != ["district"]:
        df = df_codes.merge(df, left_index=True, right_on="province")
        print("\nJJJ:\n",df.head())
    if geo_scale == ["sub-region"]:
        agg_funs = {"province": lambda x: ",".join(x), "ibbs1 code": 'first', "region": "first",  "ibbs2 code": 'first', "result": "sum"}
        df = df.reset_index().groupby(["year", "sub-region"]).agg(agg_funs)
        # Replacing "alt bölgesi" with "sub-region" in the MultiIndex
     #   df.index = pd.MultiIndex.from_tuples([(year, region.replace("alt bölgesi", "sub-region")) for year, region in df.index] )
        print("FFF:\n", df.head())
    elif geo_scale == ["region"]:
        print("SSS:",df.head())
        agg_funs = {"province": lambda x: ",".join(x), "ibbs1 code": 'first', "sub-region": lambda x: ",".join(x),  "ibbs2 code": 'first', "result": "sum"}
        df = df.reset_index().groupby(["year", "region"]).agg(agg_funs)
    return df

def get_df_change(df_result):
    if st.session_state["year_1"] != st.session_state["year_2"]:  # display_change: Show the change between end and start years in the third figure
        df_change = pd.DataFrame({"result":df_result.loc[st.session_state["year_2"],"result"]- df_result.loc[st.session_state["year_1"],"result"] })
        if st.session_state["display_percentage"]:
            df_change["result"] = df_change["result"] / df_result.loc[st.session_state["year_1"],"result"] * 100

    return df_change


def get_df_results_yeni(df_data, selected_features, geo_scale, years):
    df_result = df_nom_result = get_df_year_and_features(df_data, "nominator", years, selected_features, geo_scale)
    if st.session_state["display_percentage"]:
        # df_data_nom and df_data_denom is same for maritial_status, sex-age pages, but different for birth
        df_denom_result = get_df_year_and_features(df_data, "denominator", years, selected_features, geo_scale)
        # Calculate the percentage
        df_result["result"] = df_nom_result["result"] / df_denom_result["result"]
    df_result = df_result[["result"] + df_result.columns[:-1].tolist()]  # Reorder dataframe (result to first column)
    return df_result

def plot(col_plot, col_df, df_data, gdf_borders, selected_features, geo_scale):

    if geo_scale == ["district"]:
        geo_scale = geo_scale + ["province"]  # geo_scale = ["province", "district"]

    if st.session_state["visualization_option"] == "Plotly race chart":
        df_result = get_df_results_yeni(df_data,selected_features, geo_scale, list(range(st.session_state["slider_year_2"][0], st.session_state["slider_year_2"][1]+1)))
        plot_race(df_result,geo_scale[0])
    else:
        df_result = get_df_results_yeni(df_data, selected_features, geo_scale, sorted({st.session_state["year_1"], st.session_state["year_2"]}))
        if st.session_state["visualization_option"] == "Matplotlib (static)":
            print("RRR:",df_result.head(),"PPP:",gdf_borders.head())
            plot_matplotlib(col_plot, col_df, gdf_borders, df_result, geo_scale)
        elif st.session_state["visualization_option"] == "Folium (static)":
            plot_folium_static(col_plot, col_df, gdf_borders, df_result, geo_scale)
        else:
            plot_folium_interactive(col_plot, col_df, gdf_borders, df_result, geo_scale)