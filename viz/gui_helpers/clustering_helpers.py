import streamlit as st

def gui_clustering_up_col1():
    # First column of upper part in clustering showing scaling options
    options = ["Share of Top 30 (L1 Norm)",  # Denominator = Sum of the 30 columns
            "Share of Total",  # Denominator = Total births in province (External data)
            "TF-IDF",  # Best for emphasizing unique/rare names
            "L2 Normalization" ] # Best for pure cosine pattern matching
    st.radio("Select scaling option", options=options, key="scaler")


def gui_clustering():
        col1, col2, _ = st.columns([2, 2, 6])
        with col1:
            gui_clustering_up_col1()
        with col2:
            gui_clustering_up_col2("use_consensus_labels")
        selected_algo = gui_clustering_bottom()
        return selected_algo


def gui_clustering_up_col2():
    # scaler
    st.checkbox("Run cluster analysis", key="optimal_k_analysis")
    st.number_input("Number of seeds", min_value=1, max_value=100, value=1, key="number_of_seeds")
    st.checkbox("Use consensus labels", False, key="use_consensus_labels")

def gui_clustering_bottom():
    # Algorithm Selection
    algos = {
        "kmeans": {"label": "K-means", "gui_func": gui_clustering_kmeans},
        "gmm": {"label": "GMM", "gui_func": gui_clustering_gmm},
        "kmedoids": {"label": "K-medoids", "gui_func": kmeoids_gui_options},
      #  "dbscan": {"label": "DBSCAN", "gui_func": dbscan_gui_options},
    }
    cols = st.columns(len(algos))
    selected_algo = None

    for col, (key, config) in zip(cols, algos.items()):
        with col:
           # with st.form("submit_form_" + key):
               # submitted = st.form_submit_button(config["label"], use_container_width=True)
            clicked = st.button(config["label"], use_container_width=True,key=key)
            config["gui_func"]()
            if clicked:
                selected_algo = key
    return selected_algo

def kmeoids_gui_options():
    st.number_input("Number of clusters", 2, 15, 6, key="n_cluster_kmedoids")
    st.number_input("Maximum number of iteration", 10, 300, 100, key="max_iter_kmedoids")
    st.selectbox("Distance metric", ["cosine", "correlation"], help="Cosine: profile similarity; Correlation: shape similarity", key="pam_metric")


def gui_clustering_kmeans():
    st.number_input("Number of clusters", 2, 15, 6, key="n_cluster_kmeans")
    st.number_input("Random restarts (n_init)", 1, 100, 10, key="n_init_kmeans")


def gui_clustering():
    col1, col2, _ = st.columns([2, 2, 6])
    with col1:
        gui_clustering_up_col1()
    with col2:
        gui_clustering_up_col2()
    selected_algo = gui_clustering_bottom()
    return selected_algo


def gui_clustering_gmm():
    st.number_input("Number of clusters / components", 2, 15, 6, key="n_cluster_gmm")
    st.number_input("Random restarts (n_init)", 1, 100, 10, key="n_init_gmm")
    st.selectbox("Covariance", options=["diag", "full", "tied", "spherical"],key="gmm_covariance_type")

def dbscan_gui_options():
    """
            exp1, exp2, exp3 are the three columns you already pass in.
            """
    db_eps = st.number_input("ε (eps)",
                                  min_value=0.05, max_value=2.0,
                                  value=0.25, step=0.05,
                                  help="Max distance for neighbourhood")
    db_min = st.number_input("minPts",
                                  min_value=2, max_value=20,
                                  value=5, step=1,
                                  help="Min points to form a core region")
    # optional metric (keep Euclidean unless you need something else)
    db_metric = st.selectbox("Metric", options=["euclidean", "cosine"])