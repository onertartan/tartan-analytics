from matplotlib import cm
import matplotlib.pyplot as plt
import geopandas as gpd
import streamlit as st
import numpy as np
from utils.helpers_ui import figure_setup


def plot_result(df_result, ax,names):
    # Create a color map
    cmap = cm.get_cmap('tab20', len(names))
    colors = {name: cmap(i) for i, name in enumerate(names)}
    colors[np.nan] = "gray"

    # Assign colors to each row in the GeoDataFrame
    df_result['color'] = df_result['name'].map(colors).fillna("gray")
    # After groupby df_result becomdes Pandas dataframe, we have to convert it to GeoPandas dataframe
    df_result = gpd.GeoDataFrame(df_result, geometry='geometry')
    # Plotting
    df_result.plot(ax=ax, color=df_result['color'], legend=True,legend_kwds={"shrink": .6 }, edgecolor="black", linewidth=.2)

    df_result.apply( lambda x: ax.annotate(text=x["province"].upper()+"\n"+x['name'].title() if isinstance(x['name'],str) else x["province"], size=4, xy=x.geometry.centroid.coords[0], ha='center',va="center"), axis=1)
    ax.axis("off")
    ax.margins(x=0)
    # Add a legend
    for name, color in  set(zip(df_result['name'], df_result['color'])):
        ax.plot([], [], color=color, label=name, linestyle='None', marker='o')
  #  ax.legend(title='Names', fontsize=4, bbox_to_anchor=(0.01, 0.01), loc='lower right', fancybox=True, shadow=True)


def plot_geopandas(col_plot, df_data, gdf_borders, page_name ):
    if page_name=="names or surnames" and st.session_state["name_surname_rb"] == "Surname":
        df = df_data["surname"]
    else:
        sex = st.session_state["sex_" + page_name].lower()
        df = df_data[sex]
    display_option = st.session_state["secondary_option_" + page_name]
    names_from_multi_select = st.session_state["names_" + page_name]
    top_n = int(st.session_state["top_n_selection_" + page_name])

    fig, axs = figure_setup()
    df_results = []
    year_1, year_2 = st.session_state["year_1"], st.session_state["year_2"]
    most_popular_names_5_year = sorted(df[df["rank"] == 1]["name"].unique())

    for i, year in enumerate(sorted({year_1, year_2})):
        if "most common" in display_option:
            df_year = df.loc[year].reset_index()
            df_result = df_year[df_year["rank"] == 1]
            df_result = gdf_borders.merge(df_result, left_on="province", right_on="province")
            df_result = df_result.groupby(["geometry", "province"])["name"].apply( lambda x: "%s" % '\n '.join(x)).to_frame().reset_index()
            df_results.append(df_result)
            plot_result(df_results[i], axs[i, 0],  names = most_popular_names_5_year )
            axs[i, 0].set_title(f'Results for {year}')
        else: # top_n option: Select single year, name(s) and top-n number to filter
            df_year = df.loc[year].reset_index()
            for name in names_from_multi_select:
                df_result = df_year[(df_year["name"] == name) & (df_year["rank"] <= top_n)]
                if df_result.empty:
                    st.write(f"{name} is not in the top {top_n} for the year {year_1}")
            if names_from_multi_select:
                df_result = df_year[(df_year["name"].isin(names_from_multi_select)) & (df_year["rank"] <= top_n)]
                df_result_not_null = gdf_borders.merge(df_result, left_on="province", right_on="province")

                df_result_not_null = df_result_not_null.groupby(["geometry", "province"])["name"].apply( lambda x: "%s" % '\n '.join(x)).to_frame().reset_index()
                df_results.append(df_result_not_null)
                df_result_with_nulls= gdf_borders.merge(df_result_not_null[["province","name"]], left_on="province", right_on="province",how="left")
                plot_result(df_result_with_nulls, axs[i, 0],  names = sorted(df_result_not_null['name'].unique()))
                axs[i, 0].set_title(f'Results for {year}\n top_n = {top_n}')

        if not df_results:
            st.write("No results found.")
    if df_results:
        col_plot.pyplot(fig)