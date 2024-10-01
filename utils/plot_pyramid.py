import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt, patches


def get_pyramid_dfs(df_data):
    # Adding Male data to the figure
    df = df_data["nominator"].loc[st.session_state["slider_year_1"]].sum()
    print("TTT:",df)
    age_groups_ordered = [f"{i}-{i + 4}" for i in range(0, 90, 5)] + ["90+"]
    df_male = df.loc["male"].loc[age_groups_ordered]
    df_female = df.loc["female"].loc[age_groups_ordered]
    return df_male, df_female

def plot_pyramid_plotly(df_data):


    df_male, df_female = get_pyramid_dfs(df_data)
    # Create the male and female bar traces
    trace_male = go.Bar(x=df_male, y=df_male.index,  name='Male', orientation='h', marker=dict(color="#1f77b4"))
    trace_female = go.Bar(x=-df_female, y=df_female.index, name='Female', orientation='h', marker=dict(color="#d62728"))

    # Create the layout
    layout = go.Layout(title="Population Pyramid",
                       xaxis=dict(title="Count"),
                       yaxis=dict(title="Age"),
                       barmode="overlay", bargroupgap=0,
                       bargap=0.1)

    # Create the figure
    fig = go.Figure(data=[trace_male, trace_female], layout=layout)
    st.plotly_chart(fig)


def plot_pyramid_matplotlib(df_data):
    cols = st.columns([1,8, 1])
    df_male, df_female = get_pyramid_dfs(df_data)
    # Calculate total population for percentage
    total_population = df_male + df_female
    fig, ax = plt.subplots(figsize=(9, 6))

    # Calculate total population for percentage
    total_population = df_male.sum() + df_female.sum()
    shift =300000  # Adjust this based on your data scale

    # Plot bars with a custom size and color

    bars_male = ax.barh(df_male.index, df_male, color='#4583b5', align='center', height=0.7, left=shift, label='Male',
                        zorder=3)

    bars_female = ax.barh(df_female.index, -df_female, color='#ef7a84', align='center', height=0.7, left=-shift, label='Female',
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
    for label in df_male.index:
        ax.text(0, label, f' {label} ', va='center', ha='center', color='black',
                backgroundcolor='#fafafa')  # Center y-axis labels

    # X-axis
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f'{int(abs(x))}'))  # Set custom tick labels to show absolute values
    max_population = max(abs(df_female).max(), df_male.max()) * 1.2  # Find the max count either male or female
    ax.set_xlim(left=-max_population, right=max_population)  # Adjust x-axis limits for centering

    # Add data labels
    fontsize = 9  # Font size for the labels
    max_bar_width = max([bar.get_width() for bar in bars_male])
    label_shift = max_bar_width / 5
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
            f'Male: {df_male.sum()} ({df_male.sum() / total_population:.1%})',
            transform=ax.transAxes,
            fontsize=9,
            ha='right',
            va='top',
            color="white",
            weight='bold',
            bbox=dict(facecolor='#4583b5', edgecolor='#4583b5', boxstyle=f"round,pad=1.2,rounding_size={0.4}"))

    ax.text(0.13, 0.95, f'Female: {df_female.sum()} ({df_female.sum()  / total_population:.1%})',
            transform=ax.transAxes,
            fontsize=9,
            ha='right',
            va='top',
            color="white",
            weight='bold',
            bbox=dict(facecolor='#ef7a84', edgecolor='#ef7a84', boxstyle=f"round,pad=1.2,rounding_size={0.4}"))

    cols[1].pyplot(fig)