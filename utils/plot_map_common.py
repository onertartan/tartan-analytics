import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from folium import GeoJsonPopup, GeoJsonTooltip
from raceplotly.plots import barplot
from matplotlib import colors, pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from streamlit_folium import folium_static
import folium
from utils.plot_pyramid import plot_pyramid_plotly, plot_pyramid_matplotlib
from utils.query import get_df_change, get_geo_df_result, get_df_result
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from kneed import KneeLocator

import seaborn as sns

"""
YAPILMASI GEREKENLER:
2- İSİMLER İÇİN ANİMASYON EKLENMELİ 
"""


def delete_temp_files(folder_path="temp"):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        # Delete each file
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file}")
    except FileNotFoundError:
        print(f"Folder not found: {folder_path}")
    except PermissionError:
        print("Permission denied")
def animate(col_plot):
    # Create a placeholder for the image
    image_placeholder = col_plot.empty()
    # Directory containing your JPEG images
    image_dir = "temp"  # Replace with your image directory path
    try:
        # Load all images
        images = load_images(image_dir)
        if not images:
            st.error("No JPEG images found in the specified directory!")
            return
        # Display animation
        while st.session_state["auto_play"] and st.session_state["animate"] :
            for image in images:
                image_placeholder.image(image, use_column_width=True)
                time.sleep(st.session_state["animation_speed"])
            # Break the loop if auto-play is unchecked
            if not st.session_state["auto_play"]:
                break
        # If auto-play is off, add a manual control
        if not st.session_state["auto_play"]:
            selected_frame = st.slider("Select Frame", min_value=0, max_value=len(images) - 1, value=0)
            image_placeholder.image(images[selected_frame], use_column_width=True)
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")


def load_images(image_dir):
    """
      Load all JPEG images from the specified directory
      """
    images = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            images.append(Image.open(image_path))
    return images


def figure_setup(display_change=False):
    if st.session_state["year_1"] == st.session_state["year_2"] or st.session_state["selected_slider"] == 1 or st.session_state["animate"]:
        n_rows = 1
    elif display_change:
        n_rows = 3
    else:
        n_rows = 2
    fig, axs = plt.subplots(n_rows, 1, squeeze=False, figsize=(10, 4 * n_rows),
                            gridspec_kw={'wspace': 0, 'hspace': 0.1})  # axs has size (3,1)
    # fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=0.5, hspace=0.5)

    return fig, axs

def resize_folium_map(m,height=450):
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
    components.html(html, height=450)

def plotter_folium_static(gdf_result, title, geo_scale, *args):
    st.markdown(f"<h3 style='text-align: center; color: grey;'>{title}</h1>", unsafe_allow_html=True)
    gdf_result = gdf_result.reset_index()
    # Create the map
    m = folium.Map(location=[36.5, 35], zoom_start=6.4, tiles="cartodb positron")
    # Create the choropleth layer
    if st.session_state["clustering_cb_" + st.session_state["page_name"]]:
        # For clustered data
        folium.GeoJson(
            data=gdf_result,
            style_function=lambda feature: {
                'fillColor': feature['properties']["color"],
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7,
                'line_opacity': 0.2
            },
            popup=GeoJsonPopup(
                fields=[col for col in gdf_result.columns if col not in ['geometry', 'year']],
                aliases=[col for col in gdf_result.columns if col not in ['geometry', 'year']],
                localize=True
            )
            #,    tooltip=GeoJsonTooltip(
              #  fields=[geo_scale[0], "result"],
              #  aliases=[geo_scale[0], "Result"],
              #  localize=True  )
        ).add_to(m)
    else:
        # For regular choropleth
        folium.Choropleth(
            geo_data=gdf_result,
            name="choropleth",
            data=gdf_result,
            columns=geo_scale + ["result"] if "district" not in geo_scale else ["id", "result"],
            key_on=f'feature.properties.{geo_scale[0]}' if "district" not in geo_scale else "feature.properties.id",
            fill_color=st.session_state["selected_cmap"],
            nan_fill_color="black",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Result"
        ).add_to(m)

        # Add tooltips
        style_function = lambda x: {'fillColor': '#ffffff',
                                    'color': '#000000',
                                    'fillOpacity': 0.1,
                                    'weight': 0.1}
        highlight_function = lambda x: {'fillColor': '#000000',
                                        'color': '#000000',
                                        'fillOpacity': 0.50,
                                        'weight': 0.1}

        NIL = folium.features.GeoJson(
            gdf_result,
            style_function=style_function,
            control=False,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=[geo_scale[0], 'result'],
                aliases=[geo_scale[0], 'Result'],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
            )
        )
        m.add_child(NIL)
        m.keep_in_front(NIL)

    # Convert to HTML with responsive container
    resize_folium_map(m)


def plotter_folium_static_old(gdf_result, title, geo_scale, *args):
    st.markdown(f"<h3 style='text-align: center; color: grey;'>{title}</h1>", unsafe_allow_html=True)
   # df_result = df_result.reset_index()
    choropleth_map = folium.Map(location=[36.5, 35], zoom_start = 6.4, tiles="cartodb positron")
    gdf_result = gdf_result.reset_index()
    if st.session_state["clustering_cb_"+st.session_state["page_name"]]:
        selected_column=["clusters"]+geo_scale
        print("SDFREW:",gdf_result.head())
        # Create the GeoJson layer
        folium.GeoJson(
            data=gdf_result,
            style_function=lambda feature: {
                'fillColor': feature['properties']["color"],
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7,
                'line_opacity': 0.2
            },
            # Add popup and tooltip if needed
            popup=GeoJsonPopup(
                fields=selected_column,
                aliases=selected_column,
                localize=True
            ),
            tooltip=GeoJsonTooltip(
                fields=selected_column,
                aliases=selected_column,
                localize=True
            )
        ).add_to(choropleth_map)
    else:
        folium.Choropleth(
            geo_data=gdf_result,
            name="choropleth",
            data=gdf_result,
            columns=geo_scale + ["result"] if "district" not in geo_scale else ["id","result"] ,
            key_on=f'feature.properties.{geo_scale[0]}' if "district" not in geo_scale else "feature.properties.id",
            fill_color=st.session_state["selected_cmap"],
            nan_fill_color="black",
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


def plotter_folium_interactive (gdf_result, *args):
    gdf_result = gdf_result.reset_index()
    print("ĞĞĞĞ:",gdf_result)

    m = gdf_result.explore(
        column="result",  # make choropleth based on "BoroName" column
        tooltip="province",  # show "BoroName" value in tooltip (on hover)
        popup=True,  # show all values in popup (on click)
        tiles="CartoDB positron",  # use "CartoDB positron" tiles
        cmap=st.session_state["selected_cmap"],  # use "Set1" matplotlib colormap
        columns=[col for col in gdf_result.columns if col != "year"],
        style_kwds=dict(color="black", line_width=.01),  # use black outline
    )

    folium_static(m, width=1100, height=450)


def optimal_k_analysis(df):
    # Assuming data is your dataset (numpy array or Pandas DataFrame)
    inertias, silhouette_scores, calinski_scores = [], [], []
    # Standardize the data
    scaler = StandardScaler()# MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    k_values = range(2, 15)
    fig,axs = plt.subplots(2,1,figsize=(8,10))
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertias.append(kmeans.inertia_)
        # Silhouette score
        sil_score = silhouette_score(df, kmeans.labels_)
        silhouette_scores.append(sil_score)

        # Calinski-Harabasz score
        cal_score = calinski_harabasz_score(df, kmeans.labels_)
        calinski_scores.append(cal_score)
    # Find optimal k using elbow method
    k_values = list(k_values)
    elbow = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
    optimal_k_elbow = elbow.elbow

    # Find optimal k using silhouette score
    optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]

    # Find optimal k using Calinski-Harabasz score
    optimal_k_calinski = k_values[np.argmax(calinski_scores)]

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Elbow plot
    ax1.plot(k_values, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_ylim(0)
    ax1.set_title('Number of clusters analysis')
    if optimal_k_elbow:
        ax1.axvline(x=optimal_k_elbow, color='r', linestyle='--')

    # Silhouette plot
    ax2.plot(k_values, silhouette_scores, 'go-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score')
    ax2.axvline(x=optimal_k_silhouette, color='r', linestyle='--')

    # Calinski-Harabasz plot
    ax3.plot(k_values, calinski_scores, 'mo-')
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Calinski-Harabasz Score')
    ax3.set_title('Calinski-Harabasz Score')
    ax3.axvline(x=optimal_k_calinski, color='r', linestyle='--')
    return  fig
def analyze_mortality_data(df):
    """
    Perform comprehensive mortality data analysis
    """
    # Separate male and female columns
    male_columns = df.columns[:12]
    female_columns = df.columns[12:24]
    months = [col.split('_')[-1] for col in male_columns]
    print("MORTAL",df)
    # Key statistics
    stats = {
        'Male': {
            'total_deaths': df[male_columns].sum(),
            'yearly_total': df[male_columns].sum(axis=1),
            'monthly_avg': df[male_columns].mean(),
            'yearly_avg': df[male_columns].mean(axis=1)
        },
        'Female': {
            'total_deaths': df[female_columns].sum(),
            'yearly_total': df[female_columns].sum(axis=1),
            'monthly_avg': df[female_columns].mean(),
            'yearly_avg': df[female_columns].mean(axis=1)
        }
    }

    # Sex ratio calculations
    stats['sex_ratio'] = {
        'total_ratio': stats['Male']['total_deaths'].sum() / stats['Female']['total_deaths'].sum(),
        'yearly_ratio': stats['Male']['yearly_total'] / stats['Female']['yearly_total']
    }

    return stats, months


def plot_mortality_trends(stats, months):
    """
    Create visualizations for mortality trends
    """
    # Yearly trends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Total deaths by year
    stats['Male']['yearly_total'].plot(ax=ax1, label='Male', marker='o')
    stats['Female']['yearly_total'].plot(ax=ax1, label='Female', marker='o')
    ax1.set_title('Total Deaths by Year and Sex')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total Deaths')
    ax1.legend()

    # Monthly distribution
    male_monthly = stats['Male']['total_deaths']
    female_monthly = stats['Female']['total_deaths']

    x = np.arange(len(months))
    width = 0.35

    ax2.bar(x - width / 2, male_monthly, width, label='Male')
    ax2.bar(x + width / 2, female_monthly, width, label='Female')
    ax2.set_title('Monthly Deaths by Sex')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Total Deaths')
    ax2.set_xticks(x)
    ax2.set_xticklabels(months)
    ax2.legend()

    plt.tight_layout()
    return fig

def analyse(df):

    # Sidebar for analysis options
    st.sidebar.header('Analysis Options')
    analysis_type = st.sidebar.selectbox('Select Analysis Type', [
        'Overview',
        'Yearly Trends',
        'Monthly Patterns',
        'Sex Comparison'
    ])

    # Perform analysis
    stats, months = analyze_mortality_data(df)

    # Display analysis based on selection
    if analysis_type == 'Overview':
        st.header('Mortality Overview')
        col1, col2 = st.columns(2)

        with col1:
            st.metric('Total Male Deaths', f"{stats['Male']['total_deaths'].sum():,.0f}")
            st.metric('Sex Ratio (M/F)', f"{stats['sex_ratio']['total_ratio']:.2f}")

        with col2:
            st.metric('Total Female Deaths', f"{stats['Female']['total_deaths'].sum():,.0f}")
            st.metric('Yearly Sex Ratio Avg', f"{stats['sex_ratio']['yearly_ratio'].mean():.2f}")

    elif analysis_type == 'Yearly Trends':
        st.header('Yearly Mortality Trends')
        fig = plot_mortality_trends(stats, months)
        st.pyplot(fig)

    elif analysis_type == 'Monthly Patterns':
        st.header('Monthly Mortality Patterns')
        month_select = st.selectbox('Select Month', months)

        male_month_data = stats['Male']['total_deaths'][months.index(month_select)]
        female_month_data = stats['Female']['total_deaths'][months.index(month_select)]

        col1, col2 = st.columns(2)
        with col1:
            st.metric(f'Male Deaths in {month_select}', f"{male_month_data:,.0f}")
        with col2:
            st.metric(f'Female Deaths in {month_select}', f"{female_month_data:,.0f}")

    elif analysis_type == 'Sex Comparison':
        st.header('Sex-based Mortality Comparison')
        comparison_metric = st.selectbox('Compare by', [
            'Yearly Total',
            'Monthly Average',
            'Sex Ratio'
        ])

        if comparison_metric == 'Yearly Total':
            fig, ax = plt.subplots(figsize=(10, 6))
            stats['Male']['yearly_total'].plot(ax=ax, label='Male', marker='o')
            stats['Female']['yearly_total'].plot(ax=ax, label='Female', marker='o')
            ax.set_title('Yearly Total Deaths by Sex')
            ax.set_xlabel('Year')
            ax.set_ylabel('Total Deaths')
            ax.legend()
            st.pyplot(fig)
def plotter_matplotlib(gdf_result, title, geo_scale, ax):

    ax.cla()

    print("ömnb2",ax.get_legend())
    norm = None
    if not st.session_state["clustering_cb_"+st.session_state["page_name"]] and st.session_state["display_percentage"] and gdf_result["result"].min() < 0 and gdf_result["result"].max() > 0:
        norm = colors.TwoSlopeNorm(vmin=gdf_result["result"].min(), vcenter=0, vmax=gdf_result["result"].max())
    if st.session_state["clustering_cb_"+st.session_state["page_name"]]:
        gdf_result.plot(ax=ax, color=gdf_result["color"], edgecolor="black",
                        linewidth=.2, norm=norm,  # legend=True, legend_kwds={"shrink": .6},
                        antialiased=True)  # column=selected_feature,
    else:
        gdf_result.plot(ax=ax, column=gdf_result["result"], cmap=st.session_state["selected_cmap"], edgecolor="black",
                    linewidth=.2, norm=norm,#legend=True, legend_kwds={"shrink": .6},
                    antialiased=True)  # column=selected_feature,
    print("ömnb3",ax.get_legend())

    ax.legend(fontsize=4, bbox_to_anchor=(.01, 0.01), loc='lower right', fancybox=True, shadow=True)
    ax.set_title(title)
    region_text_vertical_shift_dict = {"Akdeniz": -1.05, "Batı Marmara": -.6, "Doğu Marmara": .2,
                                       "Ortadoğu Anadolu": .25, "Doğu Karadeniz": -0.1}
    region_text_horizontal_shift_dict = {"Akdeniz": -.5}
    font_size_dict = {"region": 6.5, "sub-region": 5}
    if "district" not in geo_scale: #if geo_scale is region or sub-region
        gdf_result.reset_index().apply(
            lambda x: ax.annotate(text=x[geo_scale[0]], size=font_size_dict.get(geo_scale[0], 4.5),
                                  xy=(x.geometry.centroid.x + region_text_horizontal_shift_dict.get(x[geo_scale[0]], 0),
                                      x.geometry.centroid.y + region_text_vertical_shift_dict.get(x[geo_scale[0]], 0)),
                                  # x.geometry.centroid.coords[0]   ,
                                  ha='center', va="bottom"), axis=1)
    ax.axis("off")
    ax.margins(x=0, y=0)
    # Get the figure object
    fig = ax.get_figure()

    if st.session_state["animate"]:
        # Adjust the layout to prevent text cutoff
        fig.tight_layout()
        # Save the figure as JPEG
        fig.savefig( "temp/"+title+".jpg", dpi=150, bbox_inches='tight',format='jpg'  )# JPEG quality (0-100)
        # Clean up
        plt.close(fig)
def radar():

    categories = ['processing cost', 'mechanical properties', 'chemical stability',
                  'thermal stability', 'device integration']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[1, 5, 2, 2, 3],
        theta=categories,
        fill='toself',
        name='Product A'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[4, 3, 2.5, 1, 2],
        theta=categories,
        fill='toself',
        name='Product B'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )),
        showlegend=False
    )
    st.plotly_chart(fig)


def plot_map_generic(col_plot, col_df, gdf_borders, df_result, geo_scale, plotter_func,years_selected, fig=None, axs=None):
    # Add CSS for layout control
    st.markdown(""" <style>[role=checkbox]{ gap: 1rem; }</style>""", unsafe_allow_html=True)
    st.markdown("""
                  <div class="top-align">
                      <style>              
                          .top-align [data-testid="stHorizontalBlock"] {
                              align-items: top;
                          }
                      </style>
                  </div>
              """, unsafe_allow_html=True)

    print("ouı:",df_result)
    display_change = (st.session_state["year_1"] != st.session_state["year_2"])
    years = sorted(list(df_result.index.get_level_values(0).unique()))
    print("OIUYÇ",years)
    col_df.write("""<style>.dataframe-margin { margin-bottom: 80px;} </style>""", unsafe_allow_html=True )

    for i, year in enumerate(years):
        print("tgb:\n", df_result.loc[year].head(), "\nres:\n", gdf_borders.head())
        gdf_result = get_geo_df_result(gdf_borders, df_result.loc[year], geo_scale)
        title = f'Results for {years_selected[i]}'
        if axs is not None:
            if st.session_state["animate"]:
                ax = axs[0, 0]
            else:
                ax = axs[i, 0]
        else:
            ax = None
        with col_plot:
            col_df.markdown('<div class="dataframe-margin">', unsafe_allow_html=True)
            plotter_func(gdf_result, title, geo_scale, ax) # plot map or save figure for each year if the state of animate is True
            col_df.markdown('</div>', unsafe_allow_html=True)

        if not st.session_state["animate"] and not st.session_state["clustering_cb_"+st.session_state["page_name"]]:
            #display dataframe on the right side if not in animation mode
            col_df.dataframe(df_result.loc[year].sort_values(by="result", ascending=False))
        elif st.session_state["elbow"]:  # if k-means and elbow selected
            print("---")
            col_df.pyplot(optimal_k_analysis(df_result.iloc[:, :-5]))
        print("***", st.session_state["elbow"])

    if st.session_state["animate"]:
        st.session_state["animation_images_generated"] = True #generated and saved image for each year calling plotterFunction in the for loop above
    elif display_change:
        df_change = get_df_change(df_result)
        gdf_result = get_geo_df_result(gdf_borders, df_change, geo_scale)
        with col_plot:
            start_word = "Change"
            if st.session_state["display_percentage"]:
                start_word = "Relative change in ratio"
            title = f"{start_word} between years {st.session_state.year_1} and {st.session_state.year_2}"
            plotter_func(gdf_result, title, geo_scale, axs[2, 0] if axs is not None else None)
        if not st.session_state["clustering_cb_"+st.session_state["page_name"]]: # if not k-means clustering show result col
            print("+++")
            col_df.markdown('<div class="dataframe-margin">', unsafe_allow_html=True)
            col_df.dataframe(df_change.sort_values(by="result", ascending=False))
            col_df.markdown('</div>', unsafe_allow_html=True)
        elif st.session_state["elbow"]: # if k-means and elbow selected
            print("---")
            col_df.pyplot(optimal_k_analysis(df_result.iloc[:, :-5]))
        print("***",st.session_state["elbow"])
    if fig and not st.session_state["animate"]:
        col_plot.pyplot(fig)


def plot(col_plot, col_df, df_data, gdf_borders, selected_features, geo_scale ):
    print("123456", geo_scale,df_data["denominator"]["district"])
    if geo_scale == ["district"]:
        geo_scale = geo_scale + ["province"]  # geo_scale = ["province", "district"]
    # Population Pyramid - Plotly
    if st.session_state["selected_tab"]=="tab_pyramid":
        if st.session_state["visualization_option"] == "Plotly":
            plot_pyramid_plotly(df_data,selected_features)
        # Population pyramid - Matplotlib
        elif st.session_state["visualization_option"] == "Matplotlib":
            plot_pyramid_matplotlib(df_data,selected_features)
    elif st.session_state["selected_tab"] == "tab_map":
        # Racebar plot
        if st.session_state["visualization_option"] == "Raceplotly":
            df_result = get_df_result(df_data, selected_features, geo_scale, list(range(st.session_state["slider_year_2"][0], st.session_state["slider_year_2"][1] + 1)))
            plot_race(df_result, geo_scale[0])
        else:
            # Map plots
            if not st.session_state["animate"]: # if animation is not clicked plotter will work in the standard way
               # st.session_state["animation_images_generated"] = False
               # delete_temp_files()
                years_selected = sorted({st.session_state["year_1"], st.session_state["year_2"]})
                print("894632",geo_scale,df_data["denominator"]["district"])
                df_result = get_df_result(df_data, selected_features, geo_scale,years_selected)
                print("JNHU:",df_result.head())
            else: # In the next step plotter will generate images according to df_result for a range of years
                years_selected =  list(range(st.session_state["slider_year_2"][0], st.session_state["slider_year_2"][1] + 1))
                df_result = get_df_result(df_data, selected_features, geo_scale, years_selected)

            print("OOO:",df_result.head())
            if st.session_state["visualization_option"] == "Matplotlib":
                fig, axs = figure_setup((st.session_state["year_1"] != st.session_state["year_2"]))
                plot_map_generic(col_plot, col_df, gdf_borders, df_result, geo_scale, plotter_matplotlib, years_selected,fig, axs)
            elif st.session_state["visualization_option"] == "Folium":
                plot_map_generic(col_plot, col_df, gdf_borders, df_result, geo_scale, plotter_folium_static,years_selected)
            elif st.session_state["visualization_option"] == "Folium-interactive":
                plot_map_generic(col_plot, col_df, gdf_borders, df_result, geo_scale, plotter_folium_interactive,years_selected)

            if st.session_state["animate"]:
                animate(col_plot)