from utils.helpers_names import names_main
import streamlit as st
import pandas as pd
import geopandas as gpd

# https://www.telepolis.de/features/Sieben-und-zehn-Jahre-3387465.html
# https://www.nature.com/articles/s41599-023-01584-3#citeas
# https://www.cambridge.org/core/journals/international-journal-of-middle-east-studies/article/abs/ethnic-kurds-in-turkey-a-demographic-study/690C9730D65FA10C4D3D9C30546097CB
#https://www.bpb.de/themen/europa/tuerkei/187953/bevoelkerungsgruppen-in-der-tuerkei/


@st.cache_data
def get_data():
    df_data={}
    file_path = "data/preprocessed/population/most_common_baby_names_male.csv"
    df_data["male"] = pd.read_csv(file_path,index_col=[0,1])
    file_path = "data/preprocessed/population/most_common_baby_names_female.csv"
    df_data["female"] = pd.read_csv(file_path, index_col=[0, 1])
    gdf_borders = gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")
    return df_data, gdf_borders


page_name = "baby names"
names_main(page_name, get_data)

