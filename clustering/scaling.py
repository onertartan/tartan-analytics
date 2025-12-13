from sklearn.preprocessing import normalize
import pandas as pd
# Overridden method

def scale(scaler_method, df, total_counts):
    if "L1" in scaler_method:  # == "Share of Top 30 (L1 Norm)" in tab_geo_clustering or in name_clustering
        df_scaled = normalize(df, axis=1, norm='l1')
    elif scaler_method == "Share of Total Births":
        # 1. Clean the total_counts series
        # This groups by the Index Name (Province) and takes the first value found.
        # It reduces the duplicates down to exactly  81 unique provinces.
        total_counts_unique = total_counts.groupby(level=0).first()
        # 2. Perform the division
        # axis=0 ensures we divide row-by-row (matching the province index)
        df_scaled = df.div(total_counts_unique, axis=0)
    elif "L2" in scaler_method:
        df_scaled = normalize(df, axis=1, norm='l2')
    # elif: TF-IDF
    df = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)
    return df