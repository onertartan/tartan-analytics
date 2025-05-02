import numpy as np
import streamlit as st
from matplotlib import  pyplot as plt
import plotly.graph_objects as go
"""
YAPILMASI GEREKENLER:
2- İSİMLER İÇİN ANİMASYON EKLENMELİ 
"""

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


