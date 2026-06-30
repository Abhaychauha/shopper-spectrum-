"""
Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce
-----------------------------------------------------------------------------------
Streamlit application implementing the two modules required by the project brief:

1. Product Recommendation Module
   - User enters a product name -> app returns the top 5 similar products
     using item-based collaborative filtering (cosine similarity on the
     CustomerID-Description / StockCode matrix).

2. Customer Segmentation Module
   - User enters Recency, Frequency and Monetary values -> app predicts the
     customer's RFM cluster and maps it to a business-friendly segment label
     (High-Value, Regular, Occasional, At-Risk).

This script expects three artifacts produced during model training (see the
"Artifacts expected" section below) to be present in the same folder as this
file, or in a path supplied via the sidebar. If they are not found, the app
falls back to a small demo dataset so the UI can still be explored.

Artifacts expected
-------------------
- rfm_kmeans_model.pkl   : trained KMeans (or other clustering) model fit on
                           scaled [Recency, Frequency, Monetary] features.
- rfm_scaler.pkl         : the StandardScaler (or similar) fit on the RFM
                           features before clustering.
- cluster_label_map.pkl  : dict mapping the raw cluster id (e.g. 0,1,2,3) to
                           the business label, e.g.
                           {0: "High-Value", 1: "Regular",
                            2: "Occasional", 3: "At-Risk"}
- product_similarity.pkl : a pandas DataFrame (square matrix) of cosine
                           similarities between products, indexed and
                           columned by product Description (or StockCode).
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
)

ARTIFACT_DIR = os.path.dirname(os.path.abspath("online_retail.csv"))

KMEANS_PATH = os.path.join(ARTIFACT_DIR, "rfm_kmeans_model.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "rfm_scaler.pkl")
LABEL_MAP_PATH = os.path.join(ARTIFACT_DIR, "cluster_label_map.pkl")
SIMILARITY_PATH = os.path.join(ARTIFACT_DIR, "product_similarity.pkl")


# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    kmeans = load_pickle(KMEANS_PATH)
    scaler = load_pickle(SCALER_PATH)
    label_map = load_pickle(LABEL_MAP_PATH)
    similarity_df = load_pickle(SIMILARITY_PATH)
    return kmeans, scaler, label_map, similarity_df


def default_label_map(n_clusters):
    """Fallback label map if cluster_label_map.pkl is missing.
    Assigns labels by sorting cluster centers loosely; in practice the
    real mapping should come from RFM-average interpretation done in the
    notebook (see project Step 4)."""
    fallback_labels = ["High-Value", "Regular", "Occasional", "At-Risk"]
    return {i: fallback_labels[i % len(fallback_labels)] for i in range(n_clusters)}


def make_demo_similarity_matrix():
    """Small demo product-similarity matrix used only when no trained
    artifact is found, so the UI remains usable for exploration/testing."""
    products = [
        "WHITE HANGING HEART T-LIGHT HOLDER",
        "RED WOOLLY HOTTIE WHITE HEART",
        "SET 7 BABUSHKA NESTING BOXES",
        "GLASS STAR FROSTED T-LIGHT HOLDER",
        "JUMBO BAG RED RETROSPOT",
        "REGENCY CAKESTAND 3 TIER",
        "PARTY BUNTING",
        "LUNCH BAG RED RETROSPOT",
    ]
    rng = np.random.default_rng(42)
    mat = rng.random((len(products), len(products)))
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 1.0)
    return pd.DataFrame(mat, index=products, columns=products)


# --------------------------------------------------------------------------------------
# Recommendation logic
# --------------------------------------------------------------------------------------
def get_recommendations(product_name, similarity_df, top_n=5):
    if similarity_df is None or product_name not in similarity_df.index:
        return None
    scores = similarity_df.loc[product_name].drop(labels=[product_name], errors="ignore")
    top_products = scores.sort_values(ascending=False).head(top_n)
    return top_products


def find_closest_matches(query, similarity_df, limit=8):
    if similarity_df is None:
        return []
    query_lower = query.strip().lower()
    matches = [p for p in similarity_df.index if query_lower in p.lower()]
    return matches[:limit]


# --------------------------------------------------------------------------------------
# Segmentation logic
# --------------------------------------------------------------------------------------
def predict_segment(recency, frequency, monetary, kmeans, scaler, label_map):
    features = np.array([[recency, frequency, monetary]])

    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        # Fallback: simple manual scaling (not equivalent to a fitted
        # StandardScaler, used only when no scaler artifact is available)
        features_scaled = features

    if kmeans is not None:
        cluster_id = int(kmeans.predict(features_scaled)[0])
    else:
        # Rule-based fallback so the UI still works without a trained model
        if recency <= 30 and frequency >= 10 and monetary >= 1000:
            cluster_id = 0
        elif frequency >= 5 and monetary >= 300:
            cluster_id = 1
        elif recency > 180:
            cluster_id = 3
        else:
            cluster_id = 2

    if label_map is not None and cluster_id in label_map:
        segment = label_map[cluster_id]
    else:
        segment = default_label_map(4).get(cluster_id, f"Cluster {cluster_id}")

    return cluster_id, segment


SEGMENT_INFO = {
    "High-Value": {
        "color": "#1b873f",
        "description": "Recent, frequent, and big-spending customers. Prioritize loyalty "
        "rewards and premium offers to retain them.",
    },
    "Regular": {
        "color": "#2563eb",
        "description": "Steady purchasers with moderate frequency and spend. Good targets "
        "for upsell and cross-sell campaigns.",
    },
    "Occasional": {
        "color": "#d97706",
        "description": "Infrequent, lower-spend customers. Consider engagement campaigns "
        "to increase purchase frequency.",
    },
    "At-Risk": {
        "color": "#dc2626",
        "description": "Haven't purchased in a long time. Strong candidates for win-back "
        "and retention offers.",
    },
}


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 Shopper Spectrum")
    st.caption("Customer Segmentation & Product Recommendations")
    page = st.radio(
        "Navigate",
        ["🎯 Product Recommendation", "🧩 Customer Segmentation", "ℹ️ About"],
    )
    st.divider()
    st.caption(
        "Place `rfm_kmeans_model.pkl`, `rfm_scaler.pkl`, `cluster_label_map.pkl` "
        "and `product_similarity.pkl` next to this script to use your trained "
        "models. Demo data is used otherwise."
    )

kmeans_model, rfm_scaler, cluster_label_map, similarity_matrix = load_artifacts()

using_demo_similarity = similarity_matrix is None
if using_demo_similarity:
    similarity_matrix = make_demo_similarity_matrix()

using_demo_model = kmeans_model is None or rfm_scaler is None


# --------------------------------------------------------------------------------------
# Page: Product Recommendation
# --------------------------------------------------------------------------------------
if page == "🎯 Product Recommendation":
    st.header("🎯 Product Recommendation Module")
    st.write(
        "Enter a product name to get the **top 5 similar products** based on "
        "item-based collaborative filtering (cosine similarity)."
    )

    if using_demo_similarity:
        st.info(
            "No trained `product_similarity.pkl` found — showing a small demo "
            "catalog so you can try out the UI.",
            icon="ℹ️",
        )

    product_input = st.text_input("Product Name", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER")

    col1, col2 = st.columns([1, 4])
    with col1:
        get_rec = st.button("Get Recommendations", type="primary")

    if get_rec:
        if not product_input.strip():
            st.warning("Please enter a product name.")
        else:
            exact_match = product_input.strip().upper() in [p.upper() for p in similarity_matrix.index]
            matched_name = None
            if product_input.strip() in similarity_matrix.index:
                matched_name = product_input.strip()
            else:
                upper_map = {p.upper(): p for p in similarity_matrix.index}
                if product_input.strip().upper() in upper_map:
                    matched_name = upper_map[product_input.strip().upper()]

            if matched_name is None:
                suggestions = find_closest_matches(product_input, similarity_matrix)
                st.error(f"Product '{product_input}' not found in catalog.")
                if suggestions:
                    st.write("Did you mean one of these?")
                    for s in suggestions:
                        st.write(f"- {s}")
            else:
                recommendations = get_recommendations(matched_name, similarity_matrix, top_n=5)
                st.success(f"Top 5 products similar to **{matched_name}**:")

                cols = st.columns(5)
                for i, (prod, score) in enumerate(recommendations.items()):
                    with cols[i % 5]:
                        st.markdown(
                            f"""
                            <div style="border:1px solid #e5e7eb; border-radius:10px;
                                        padding:14px; height:170px;
                                        display:flex; flex-direction:column;
                                        justify-content:space-between;">
                                <div style="font-weight:600; font-size:14px;">{prod}</div>
                                <div style="color:#6b7280; font-size:12px;">
                                    Similarity: {score:.2f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    with st.expander("Browse available products"):
        st.dataframe(
            pd.DataFrame({"Product": similarity_matrix.index}),
            use_container_width=True,
            height=250,
        )


# --------------------------------------------------------------------------------------
# Page: Customer Segmentation
# --------------------------------------------------------------------------------------
elif page == "🧩 Customer Segmentation":
    st.header("🧩 Customer Segmentation Module")
    st.write(
        "Enter a customer's RFM values to **predict their segment** "
        "(High-Value, Regular, Occasional, or At-Risk)."
    )

    if using_demo_model:
        st.info(
            "No trained KMeans model / scaler found — using a simple rule-based "
            "fallback so you can try out the UI.",
            icon="ℹ️",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input("Recency (in days)", min_value=0, value=30, step=1)
    with c2:
        frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5, step=1)
    with c3:
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0, step=10.0)

    predict = st.button("Predict Cluster", type="primary")

    if predict:
        cluster_id, segment = predict_segment(
            recency, frequency, monetary, kmeans_model, rfm_scaler, cluster_label_map
        )
        info = SEGMENT_INFO.get(segment, {"color": "#374151", "description": ""})

        st.markdown(
            f"""
            <div style="border-left: 6px solid {info['color']};
                        background:#f9fafb; padding:18px 20px; border-radius:8px;">
                <div style="font-size:13px; color:#6b7280;">Predicted Segment</div>
                <div style="font-size:26px; font-weight:700; color:{info['color']};">
                    {segment}
                </div>
                <div style="margin-top:8px; color:#374151;">{info['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Raw cluster id: {cluster_id}")


# --------------------------------------------------------------------------------------
# Page: About
# --------------------------------------------------------------------------------------
else:
    st.header("ℹ️ About this project")
    st.markdown(
        """
**Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce**

This app is the deliverable described in the project brief's *Streamlit Web
Application* section. It implements two modules:

1. **Product Recommendation** — item-based collaborative filtering using
   cosine similarity on a CustomerID-Description purchase matrix, returning
   the top 5 most similar products for a given product name.
2. **Customer Segmentation** — RFM (Recency, Frequency, Monetary) feature
   engineering, scaled and clustered (KMeans), with clusters mapped to
   business segments: **High-Value**, **Regular**, **Occasional**, and
   **At-Risk**.

To go live with real data, train the models in the accompanying notebook and
save these artifacts next to `app.py`:

- `rfm_kmeans_model.pkl`
- `rfm_scaler.pkl`
- `cluster_label_map.pkl`
- `product_similarity.pkl`
"""
    )