import os
import streamlit as st
from utils.checkbox_group import Checkbox_Group
import re






# Function to extract years
def extract_years(filename):
    # Regex pattern to match two consecutive years (YYYY-YYYY)
    pattern = r'(\d{4})-(\d{4})'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None  # Return None if no match found


def get_file_path_by_suffix(folder_name,file_prefix, suffix):
    folder_path = "data/preprocessed/"+folder_name
    for filename in os.listdir(folder_path):
        if filename.startswith(file_prefix) and filename.endswith(suffix+".csv"):
            return os.path.join(folder_path, filename)


def feature_choice(col, feature_name, nom_denom_key_suffix, num_sub_cols=3):
    page_name = st.session_state["page_name"]
    disabled = not st.session_state["display_percentage"] if nom_denom_key_suffix == "denominator" else False
    print("FTYUI:",feature_name)
    if feature_name == "marital_status":
        selected_feature = col.radio("Choose marital status", ["All", "Never married", "Married", "Divorced", "Widowed"],key=nom_denom_key_suffix+"_"+feature_name,disabled=disabled).lower().replace(" ", "_")
    elif feature_name == "education":
        selected_feature = col.radio("Choose sex",  ['illiterate', 'literate but did not complete any school', 'primary', 'elementary',
                                                     'secondary school or equivalent', 'high school or equivalent', 'pre-license or bachelor', 'master', 'PhD'],
                                                     key = nom_denom_key_suffix+"_"+feature_name, disabled=disabled).lower()
    elif feature_name == "sex":
        selected_feature = col.radio("Choose education level", ["All", "Male", "Female"],
                                     key=nom_denom_key_suffix + "_" + feature_name, disabled=disabled).lower()

    elif feature_name == "age":
        print("PPP",st.session_state)
        if page_name+"_age_checkbox_group" not in st.session_state:
            print("ÜÜÜ")
            st.session_state[page_name+"_age_checkbox_group"] = Checkbox_Group(page_name, feature_name, num_sub_cols, st.session_state["age_group_keys"][page_name])
        st.session_state[page_name+"_age_checkbox_group"].place_checkboxes(col, nom_denom_key_suffix, disabled)

        selected_feature = st.session_state[page_name+"_age_checkbox_group"].get_checked_keys(nom_denom_key_suffix)

    if selected_feature == "all":
        return slice(None)

    return selected_feature
