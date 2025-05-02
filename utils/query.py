import streamlit as st
import pandas as pd



def get_df_year_and_features(df_data, nom_denom_selection, year, selected_features_dict, geo_scale,give_total=True):
    df_codes = pd.read_csv("data/preprocessed/region_codes.csv", index_col=0)
    print("YYY",year,"df_codesçç",df_codes.head(),"geo_scale",geo_scale)
    if "district" in geo_scale:
        print("1. çekpoint ",nom_denom_selection)
        df = df_data[nom_denom_selection]["district"]
        print("1. çekdf",df)
    else:
        print("2. çekpoint")
        df = df_data[nom_denom_selection]["province"]
    print("nom_denom_selection:",nom_denom_selection,"selected_features_dict:",selected_features_dict)
    selected_features = selected_features_dict[nom_denom_selection]

    print("nom_denom_selection:",nom_denom_selection,"START:\n",df.head(),"\nssselected_features:",selected_features)
    if len(selected_features)==1:
        selected_features=selected_features[0]
    df = pd.DataFrame(df.loc[year, selected_features]) # if it is Pandas Series it is converted to dataframe
    for i in range(df.columns.nlevels-1):
        df=df.droplevel(0,axis=1)
    if give_total:
        #if df.columns.nlevels > 1:  # Check if the DataFrame has multiple column levels
         #   df = df.droplevel(1, axis=1)  # .sum(axis=1))  # .reset_index()
        df=df.sum(axis=1).to_frame()
        print("nom_denom_selection:",nom_denom_selection,"nbhg",df.head())

        df.rename(columns={df.columns[-1]: "result"}, inplace=True) # aggregation result(sum of selected cols)
        df = df.sort_values(by=["year", "result"], ascending=False)

    # else:
    #    df = df.droplevel(0,axis=0) # k-means clustering drop year from index to merge with df_codes (make df compatible)
     #   df = df.droplevel(0, axis=1)  # .sum(axis=1))  # .reset_index()

    print("geo_scale:",geo_scale,"nom_denom_selection",nom_denom_selection,"GGG:",df.head(),df.shape)

    print("\ndf_codes HHH:",df_codes.head())
  #  print("ÜĞP:", df.loc[(slice(None), "İstanbul", slice(None))])
    numeric_cols = list(df.select_dtypes(include=['number']).columns)

    if geo_scale != ["district"]:
        df = df.join(df_codes, on="province")#left_index=True  DEĞİŞTİRİLDİ left_on="province" yapıldı
        print("\nÜPÜ:\n",df.head())
    if geo_scale == ["sub-region"]:
        agg_funs = {col:"sum" for col in numeric_cols}
        agg_funs.update({"province": lambda x: ",".join(x), "ibbs1 code": 'first', "region": "first",  "ibbs2 code": 'first'})
        df = df.reset_index().groupby(["year", "sub-region"]).agg(agg_funs)
        print("FFF:\n", df.head())
    elif geo_scale == ["region"]:
        print("SSS:",df.head())
        print("NNUMM:", numeric_cols)
        agg_funs = {col:"sum" for col in numeric_cols}
        agg_funs.update({"province": lambda x: ",".join(x), "ibbs1 code": 'first', "sub-region": lambda x: ",".join(x),  "ibbs2 code": 'first'})
        df = df.reset_index().groupby(["year", "region"]).agg(agg_funs)

    return df


def get_df_change(df_result):
    if st.session_state["year_1"] != st.session_state["year_2"]:  # display_change: Show the change between end and start years in the third figure
        if st.session_state["clustering_cb_"+st.session_state["page_name"]]:
            print("TTTT:",df_result.select_dtypes(include=['number']).loc[st.session_state["year_1"]].index)
            df1, df2 = df_result.loc[st.session_state["year_1"]],df_result.loc[st.session_state["year_2"]]
            df1, df2 = df1.align(df2, join="inner", axis=1)  # Align columns
            df1, df2 = df1.align(df2, join="inner", axis=0)  # Align rows
            df_change = df2.copy()
            print("PŞM:",df1.head())
            print("OIK:",df2.head())
            df_change =df2.select_dtypes(include=['number'])  -df1.select_dtypes(include=['number'])
            print("ÇÖKJ:", df1.shape,df2.shape)
            if st.session_state["display_percentage"]:
                numeric_cols = df_change.select_dtypes(include=['number']).columns
                df_change.loc[:, numeric_cols] = df_change.loc[:,numeric_cols] / df1.loc[:,numeric_cols]* 100
        else:
            df_change = pd.DataFrame({"result":df_result.loc[st.session_state["year_2"],"result"]- df_result.loc[st.session_state["year_1"],"result"] })
            if st.session_state["display_percentage"]:
                 df_change["result"] = df_change["result"] / df_result.loc[st.session_state["year_1"],"result"] * 100
    print("CCHHAA:",df_change.head())
    return df_change


def get_df_result(df_data, selected_features, geo_scale, years,give_total=True):
    k_means = st.session_state["clustering_cb_"+st.session_state["page_name"]]
    print("FFFF",st.session_state["clustering_cb_"+st.session_state["page_name"]],"ÜÜÜ",geo_scale)
    print("IUIU",df_data["denominator"]["district"])
    df_result = df_nom_result = get_df_year_and_features(df_data, "nominator", years, selected_features, geo_scale,not k_means)
    if st.session_state["display_percentage"]:
        # df_data_nom and df_data_denom is same for maritial_status, sex-age pages, but different for birth
        df_denom_result = get_df_year_and_features(df_data, "denominator", years, selected_features, geo_scale,give_total=True)
        print("vcxz",df_result)
        print("çömn:",df_denom_result)#df_denom_result.droplevel(0,axis=0))
        # Calculate the percentage
        if not k_means :
            print("BEFFFORE:",df_result.head())
            df_result["result"] = df_nom_result["result"] / df_denom_result["result"]
            print("AFFTERR:",df_result.head())
        else: # k-means
            print("FERT:",df_result.head(),"DENN:",df_denom_result.head())
            df_result.iloc[:,:-5] =df_result.iloc[:,:-5].div  (df_denom_result["result"],axis=0)#.div(df_denom_result.droplevel(0,axis=0)["result"],axis=0)
            print("SONUÇ:",df_result.head())
   # if not k_means:
    #    df_result = df_result[["result"] + df_result.columns[:-1].tolist()]  # Reorder dataframe (result to first column) ARTIK GEREKMİYOR GİBİ, result zaten ilk sütun olmuş
    return df_result
