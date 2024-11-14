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


