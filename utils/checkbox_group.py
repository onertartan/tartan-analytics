import streamlit as st
class Checkbox_Group:
    def __init__(self, page_name, feature_name, num_sub_cols, basic_keys, message="Select age group(s)"):
        self.page_name = page_name
        self.feature_name = feature_name
        self.num_sub_cols = num_sub_cols
        self.basic_keys = basic_keys
        self.checked_dict = {"nominator": {name: False for name in basic_keys},
                             "denominator": {name: False for name in basic_keys}}
        self.checked_dict["nominator"]["all"] = True
        self.checked_dict["denominator"]["all"] = True
        self.message = message


    def place_checkboxes(self, cols_nom_denom, nom_denom_key_suffix, disabled):
        cols_nom_denom.write(self.message)
        checkbox_group_sub_cols = cols_nom_denom.columns(self.num_sub_cols +1, gap="small")# extra +1 dummy col is to leave gap on the right
        for i, key_basic in enumerate(self.checked_dict[nom_denom_key_suffix].keys()):
            if i == 0:  # all checkbox
                on_change_fun = self.select_all
            else:
                on_change_fun = self.uncheck_all_option
            key = self.page_name+"_"+key_basic + "_"+nom_denom_key_suffix
            value = True if key_basic == "all" else False
            checkbox_group_sub_cols[i % self.num_sub_cols].checkbox(label=key_basic.capitalize(), key=key,
                                                    value=value,
                                                    disabled=disabled,
                                                    on_change=on_change_fun, args=[nom_denom_key_suffix])


    def get_checked_keys(self,nom_denom_key_suffix):
        if st.session_state[self.page_name+"_all_"+nom_denom_key_suffix]:
            return slice(None)
       # return [selected_key for selected_key, isChecked in self.checked_dict[nom_denom_key_suffix].items() if isChecked]
        checked_keys=[]
        for key_basic in self.basic_keys:
            if st.session_state[self.page_name+"_"+key_basic+"_"+nom_denom_key_suffix]:
                checked_keys.append(key_basic)
        return checked_keys

    def select_all(self,nom_denom_key_suffix):
        # uncheck options other than "all"
        if  st.session_state[self.page_name + "_" + "all" + "_" + nom_denom_key_suffix]:
            for key_basic in  self.basic_keys[1:]:
                st.session_state[self.page_name + "_" + key_basic + "_" + nom_denom_key_suffix] = False

    def uncheck_all_option(self, nom_denom_key_suffix,):
        if  st.session_state[self.page_name + "_" + "all" + "_" + nom_denom_key_suffix]:
            st.session_state[self.page_name + "_" + "all" + "_" + nom_denom_key_suffix] = False

    @staticmethod
    def age_group_quick_select():
        page_name = st.session_state["page_name"]
        print("GGG:", st.session_state["age_group_keys"].keys())
        basic_keys = st.session_state["age_group_keys"][page_name]
        if page_name == "sex_age":
            child_age_groups, elderly_age_groups = (["0-4", "5-9", "10-14"],
                                                    ["65-69", "70-74", "75-79", "80-84","85-89","90+"])
            working_age_groups = list(set(basic_keys) - set(child_age_groups + elderly_age_groups))
            age_group_selection = st.session_state[page_name + "_age_group_selection"]

            if age_group_selection != "Custom selection":
                if age_group_selection == "Quick selection for total age dependency ratio":
                    st.session_state[page_name + "_age_checkbox_group"].set_checkbox_values_for_quick_selection(page_name, basic_keys, child_age_groups + elderly_age_groups, "nominator")
                elif age_group_selection == "Quick selection for child age dependency ratio":
                    st.session_state[page_name + "_age_checkbox_group"].set_checkbox_values_for_quick_selection(page_name, basic_keys, child_age_groups, "nominator")
                elif age_group_selection == "Quick selection for old-age dependency ratio":
                    st.session_state[page_name + "_age_checkbox_group"].set_checkbox_values_for_quick_selection(page_name, basic_keys, elderly_age_groups, "nominator")

                st.session_state[page_name + "_age_checkbox_group"].set_checkbox_values_for_quick_selection(page_name, basic_keys, working_age_groups, "denominator")
        elif page_name == "birth":
            if st.session_state[page_name+"_age_group_selection"]!="Custom selection":# then general fertility rate age groups
                mid_age_groups = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
                st.session_state[page_name + "_age_checkbox_group"].set_checkbox_values_for_quick_selection(page_name, basic_keys, mid_age_groups, "denominator")

    @staticmethod
    def set_checkbox_values_for_quick_selection(page_name, basic_keys, keys_to_check, nom_denom_key_suffix):
        for key_basic in basic_keys:
            if key_basic in keys_to_check:
                val = True
            else:
                val = False
            st.session_state[page_name+"_"+key_basic+"_"+nom_denom_key_suffix] = val