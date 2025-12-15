import streamlit as st


def gui_basic_setup(col_weights):
    cols_title = st.columns(2)
    cols_title[0].markdown("<h3 style='color: red;'>Select primary parameters.</h3>", unsafe_allow_html=True)
    cols_title[0].markdown("<br><br><br>", unsafe_allow_html=True)

    # Checkbox to switch between population and percentage display
    cols_title[1].markdown("<h3 style='color: blue;'>Select secondary parameters.</h3>", unsafe_allow_html=True)
    cols_title[1].checkbox("Check to get ratio: primary parameters/secondary parameters.", key="display_percentage")
    cols_title[1].write("Uncheck to show counts of primary parameters.")

    cols_all = st.columns(col_weights)  # There are 2*n columns(for example: 3 for nominator,3 for denominator)
    with cols_all[len(cols_all) // 2]:
        st.html(
            '''
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border-left: 1px solid rgba(49, 51, 63, 0.2);
                        height: 180px;
                        margin: auto;
                    }
                </style>
            '''
        )
    cols_nom_denom = {"nominator": cols_all[0:len(col_weights) // 2],
                      "denominator": cols_all[len(col_weights) // 2 + 1:]}
    return cols_nom_denom