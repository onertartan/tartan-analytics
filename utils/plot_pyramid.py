import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt, patches

import plotly.express as px

def get_pyramid_dfs(df_data,selected_features,page_name):
    age_groups_ordered = st.session_state["age_group_keys"][page_name][1:] # exclude the 0th element "all"
    print("ÇÇ",age_groups_ordered)
    # Adding Male data to the figure
    df = df_data["nominator"].loc[st.session_state.slider_year_2[0]:st.session_state.slider_year_2[1],selected_features["nominator"]].reset_index().drop(columns=["province"]).groupby("year").sum()
    print("TTT:",df)

  #  df = df.loc[:,pd.IndexSlice["male",age_groups_ordered]]
    print("şş:",df.columns.nlevels)
    for i in range(df.columns.nlevels):
        df = df.stack(future_stack=True)
    df = df.reset_index()
    print("FGH:\n",df)
    df.rename(columns={df.columns[-1]:"Population"}, inplace=True)
    return df

def plot_pyramid_plotly(df_data,selected_features,page_name):


    df  = get_pyramid_dfs(df_data,selected_features,page_name)

    #total_population = df_male.sum() + df_female.sum()
    # Create the male and female bar traces
   # trace_male = go.Bar(x=df["male"], y=df_male.index,  name='Male', orientation='h', marker=dict(color="#1f77b4"))
   # trace_female = go.Bar(x=-df_female, y=df_female.index, name='Female', orientation='h', marker=dict(color="#d62728"))
   # max_population = max(df_female.max(), df_male.max()) * 1.2  # Find the max count either male or female
    layout_dict = {"title": "Population Pyramid", "title_font_size": 22, "barmode": 'overlay',
                   "yaxis": dict(title="Age"),"color":"#1f77b4",
                   "bargroupgap": 0,
                   "bargap": .3}
    df.loc[df["sex"]=="female","Population"]*=-1
    fig = px.bar(df, x="Population", y="age_group", orientation='h',color="sex",  animation_frame='year')

    fig.update_layout(barmode='overlay')

    layout_dict = {"title":"Population Pyramid","title_font_size" : 22, "barmode": 'overlay',
                       "yaxis":dict(title="Age"),
                        "bargroupgap":0,
                       "bargap":.3}
    # Create the layout
    """
    layout = go.Layout(title="Population Pyramid",title_font_size = 22, barmode = 'overlay',
                       yaxis=dict(title="Age"),
                        bargroupgap=0,   xaxis=go.layout.XAxis(
                     #  range=[-max_population, max_population],
                     #  ticktext = ['6M', '4M', '2M', '0',  '2M', '4M', '6M'],
                              title = 'Population in Millions',
                              title_font_size = 14 ),
                       bargap=.3)
    """
    # Create the figure
 #   fig = go.Figure(data=[trace_male, trace_female], layout=layout,)

  #  fig.update_xaxes(range=[-max_population*1.2,max_population*1.2])

    st.plotly_chart(fig)


def plot_pyramid_matplotlib(df_data,selected_features,page_name):
    cols = st.columns([1,8, 1])
    cols[1].write("Note: First slider is used for year selection.")
    df = df_data["nominator"].loc[st.session_state.slider_year_1, selected_features["nominator"]]

    df_male= df["male"].sum().reset_index()
    df_female = df["female"].sum().reset_index()
    df_male.rename(columns={df_male.columns[-1]:"Population"}, inplace=True)
    df_female.rename(columns={df_female.columns[-1]:"Population"}, inplace=True)

    print("jjj\n",df_male)
    # Calculate total population for percentage
    total_population = df_male["Population"].sum()+ df_female["Population"].sum()
    fig, ax = plt.subplots(figsize=(9, 6))

    shift =total_population/100  # Adjust this based on your data scale
    # Plot bars with a custom size and color
    bars_male = ax.barh(df_male["age_group"], df_male["Population"], color='#4583b5', align='center', height=0.7, left=shift, label='Male',
                        zorder=3)

    bars_female = ax.barh(df_female["age_group"], -df_female["Population"], color='#ef7a84', align='center', height=0.7, left=-shift,
                          label='Female',
                          zorder=3)
    ax.set_yticklabels([])

    # Set titles and labels
    fig.suptitle('Population Distribution by Age and Sex', fontsize=14, x=0.06, y=0.98, ha="left")  # Customize title
    ax.set_xlabel('Population', labelpad=10)

    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Y-axis
    ax.spines['left'].set_position(('data', shift))  # Center the y-axis
    ax.yaxis.set_ticks_position('none')  # Remove y-axis ticks
    for label in df_male["age_group"]:
        ax.text(0, label, f' {label} ', va='center', ha='center', color='black',
                backgroundcolor='#fafafa')  # Center y-axis labels

    # X-axis
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f'{int(abs(x))}'))  # Set custom tick labels to show absolute values
    max_population = max(df_female["Population"].max(), df_male["Population"].max()) * 1.2  # Find the max count either male or female
    ax.set_xlim(left=-max_population, right=max_population)  # Adjust x-axis limits for centering

    # Add data labels
    fontsize = 9  # Font size for the labels
    max_bar_width =max( max([bar.get_width() for bar in bars_male]),max([abs(bar.get_width()) for bar in bars_female]))
    label_shift = max_bar_width /4
    for bar in bars_female:
        width = bar.get_width()
        label_x_pos = bar.get_x() + width - label_shift  # Adjust position outside the bar
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                f'{abs(width) / total_population:.1%}', va='center', ha='left', color='#ef7a84', fontsize=fontsize,
                fontweight='bold')

    for bar in bars_male:
        width = bar.get_width()
        label_x_pos = bar.get_x() + width + label_shift   # Adjust position outside the bar
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                f'{width / total_population:.1%}', va='center', ha='right', color='#4583b5', fontsize=fontsize,
                fontweight='bold')


    # Adding a custom rectangle as a border around the figure
    border_radius = 0.015
    rect = patches.FancyBboxPatch((0.03, -0.03), 1, 1.08, transform=fig.transFigure, facecolor="#fafafa",
                                  edgecolor='black', linewidth=1, clip_on=False, zorder=-3, linestyle='-',
                                  boxstyle=f"round,pad=0.03,rounding_size={border_radius}")
    fig.patches.extend([rect])
    ax.set_facecolor('#fafafa')  # Set the background color of the axes

    # Adding data for males and females in corners
    ax.text(1.11, 0.95,
            f'Male: {df_male["Population"].sum()  } ({df_male["Population"].sum() / total_population:.1%})',
            transform=ax.transAxes,
            fontsize=9,
            ha='right',
            va='top',
            color="white",
            weight='bold',
            bbox=dict(facecolor='#4583b5', edgecolor='#4583b5', boxstyle=f"round,pad=1.2,rounding_size={0.4}"))

    ax.text(0.21, 0.95, f'Female: {df_female["Population"].sum()} ({df_female["Population"].sum()  / total_population:.1%})',
            transform=ax.transAxes,
            fontsize=9,
            ha='right',
            va='top',
            color="white",
            weight='bold',
            bbox=dict(facecolor='#ef7a84', edgecolor='#ef7a84', boxstyle=f"round,pad=1.2,rounding_size={0.4}"))

    cols[1].pyplot(fig)