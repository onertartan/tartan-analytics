import streamlit as st
import pandas as pd
def get_geo_df_result(gdf_borders, df_result, geo_scale):
    print("\nAAA:\n",gdf_borders.head(),"\nBBB:\n",df_result.head())
    print("\nCCC:\n",gdf_borders.dissolve(by=geo_scale).head())
    print("before:\n",gdf_borders.shape,"\n")
    gdf_result = gdf_borders.dissolve(by=geo_scale)[["id","geometry"]].merge(df_result,left_index=True,right_index=True)  # after dissolving index becomes geo_scal,so common index is geo_scale(example:province)
    print("after:\n",gdf_result.shape,"\n")
    print("\nEEE:\n",gdf_result.head())
    return gdf_result

def get_df_year_and_features(df_data, nom_denom_selection, year, selected_features_dict, geo_scale):
    df_codes = pd.read_csv("data/preprocessed/region_codes.csv", index_col=0)
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


def get_df_result(df_data, selected_features, geo_scale, years):
    df_result = df_nom_result = get_df_year_and_features(df_data, "nominator", years, selected_features, geo_scale)
    if st.session_state["display_percentage"]:
        # df_data_nom and df_data_denom is same for maritial_status, sex-age pages, but different for birth
        df_denom_result = get_df_year_and_features(df_data, "denominator", years, selected_features, geo_scale)
        # Calculate the percentage
        df_result["result"] = df_nom_result["result"] / df_denom_result["result"]
    df_result = df_result[["result"] + df_result.columns[:-1].tolist()]  # Reorder dataframe (result to first column)
    return df_result
