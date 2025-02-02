import streamlit as st
class Checkbox_Group:
    def __init__(self, page_name, feature_name, num_sub_cols, basic_keys, message="Select age group(s)"):
        self.page_name = page_name
        self.feature_name = feature_name
        self.num_sub_cols = num_sub_cols
        self.basic_keys = basic_keys
      #  self.checked_dict = {"nominator": {name: False for name in basic_keys},
      #                       "denominator": {name: False for name in basic_keys}}
      #  self.checked_dict["nominator"]["all"] = True
      #  self.checked_dict["denominator"]["all"] = True
        self.reset_checked_values()
        self.message = message

    def reset_checked_values(self):
        # Initialize session state for checkboxes
        for nom_denom in ["nominator", "denominator"]:
            for key_basic in self.basic_keys:
                key = f"{self.page_name}_{nom_denom}_{self.feature_name}_{key_basic}"
                if key not in st.session_state:
                    st.session_state[key] = (key_basic == "all")  # "all" is initially True
    def place_checkboxes(self, cols_nom_denom, nom_denom_key_suffix, disabled,feature_name=""):
        print("PAGE::",st.session_state["page_name"])
        print("keys::",nom_denom_key_suffix)
        # Custom CSS to reduce checkbox spacing with custom margin

        cols_nom_denom.write(self.message)
        checkbox_group_sub_cols = cols_nom_denom.columns(self.num_sub_cols , gap="small")# sub columns for checkboxed
        #print("9999",self.checked_dict)
        for i, key_basic in enumerate(self.basic_keys):

       #1 for i, key_basic in enumerate(self.checked_dict[nom_denom_key_suffix].keys()):
            st.markdown(
                """<style>[data-testid=stVerticalBlock]{gap: 0rem;}</style>""",
                unsafe_allow_html=True)

            if i == 0:
                on_change_fun = self.select_all
            else: # if a checkbox other than "all" is clicked, then uncheck the "all" checkbox
                on_change_fun = self.uncheck_all_option
            key = self.page_name+"_"+nom_denom_key_suffix+"_"+feature_name+"_"+key_basic
            #value = True if key_basic == "all" else False  # 31 OCAK 2024 EN SON DEĞİŞİKLİK ÖNCESİ ORJİNAL
            #value = st.session_state.get(key, self.checked_dict[nom_denom_key_suffix][key_basic])
            value = st.session_state.get(key)

           # print("//",value,"//",key_basic,"&&",self.checked_dict[nom_denom_key_suffix][key_basic])
            checkbox_group_sub_cols[i % self.num_sub_cols].checkbox(label=key_basic.capitalize(), key=key,
                                                    value=value,
                                                    disabled=disabled,
                                                    on_change=on_change_fun, args=[nom_denom_key_suffix,feature_name])

    def get_checked_keys(self,nom_denom_key_suffix,feature_name=""):
        print("MARIT: page name:",self.page_name,",feature_name:",feature_name,",nom_denom_key_suffix:",nom_denom_key_suffix)
        if st.session_state[self.page_name+"_"+nom_denom_key_suffix+"_"+feature_name+"_all"]:
            return self.basic_keys[1:] # slice(None)
       # return [selected_key for selected_key, isChecked in self.checked_dict[nom_denom_key_suffix].items() if isChecked]
        checked_keys=[]
        for key_basic in self.basic_keys:
            if st.session_state[self.page_name+"_"+nom_denom_key_suffix+"_"+feature_name+"_"+key_basic]:
                checked_keys.append(key_basic)
        return checked_keys

    def select_all(self,nom_denom_key_suffix,feature_name):
        # uncheck options other than "all"
        if   st.session_state[self.page_name+"_"+nom_denom_key_suffix+"_"+feature_name+"_all"]:
            for key_basic in  self.basic_keys[1:]:
                st.session_state[self.page_name+"_"+nom_denom_key_suffix+"_"+feature_name+"_"+key_basic] = False
                # Update self.checked_dict
                #1self.checked_dict[nom_denom_key_suffix][key_basic] = False

    def uncheck_all_option(self, nom_denom_key_suffix,feature_name):
        if  st.session_state[self.page_name+"_"+nom_denom_key_suffix+"_"+feature_name+"_all"]:
            st.session_state[self.page_name + "_" + nom_denom_key_suffix + "_" + feature_name + "_all"] = False
            #self.checked_dict[nom_denom_key_suffix]["all"] = False

    # @staticmethod
    # def age_group_quick_select():
    #     page_name = st.session_state["page_name"]
    #     print("GGG:", st.session_state["age_group_keys"].keys())
    #     basic_keys = st.session_state["age_group_keys"][page_name]
    #     if page_name == "sex_age":
    #         child_age_groups, elderly_age_groups = (["0-4", "5-9", "10-14"],
    #                                                 ["65-69", "70-74", "75-79", "80-84","85-89","90+"])
    #         working_age_groups = list(set(basic_keys) - set(child_age_groups + elderly_age_groups))
    #         age_group_selection = st.session_state[page_name + "_age_group_selection"]
    #
    #         if age_group_selection != "Custom selection":
    #             if age_group_selection == "Quick selection for total age dependency ratio":
    #                 st.session_state[page_name + "_age_checkbox_group"].set_checkbox_values_for_quick_selection(page_name, basic_keys, child_age_groups + elderly_age_groups, "nominator")
    #             elif age_group_selection == "Quick selection for child age dependency ratio":
    #                 st.session_state[page_name + "_age_checkbox_group"].set_checkbox_values_for_quick_selection(page_name, basic_keys, child_age_groups, "nominator")
    #             elif age_group_selection == "Quick selection for old-age dependency ratio":
    #                 st.session_state[page_name + "_age_checkbox_group"].set_checkbox_values_for_quick_selection(page_name, basic_keys, elderly_age_groups, "nominator")
    #
    #             st.session_state[page_name + "_age_checkbox_group"].set_checkbox_values_for_quick_selection(page_name, basic_keys, working_age_groups, "denominator")
    #     elif page_name == "birth":
    #         if st.session_state[page_name+"_age_group_selection"]!="Custom selection":# then general fertility rate age groups
    #             mid_age_groups = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
    #             st.session_state[page_name + "_age_checkbox_group"].set_checkbox_values_for_quick_selection(page_name, basic_keys, mid_age_groups, "denominator")

    # @staticmethod
    # def set_checkbox_values_for_quick_selection(page_name, basic_keys, keys_to_check, nom_denom_key_suffix):
    #     for key_basic in basic_keys:
    #         if key_basic in keys_to_check:
    #             val = True
    #         else:
    #             val = False
    #         st.session_state[page_name+"_"+key_basic+"_"+nom_denom_key_suffix] = val
