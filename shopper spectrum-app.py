"""
Shopper Spectrum — Customer Segmentation & Product Recommendation Engine
Matches the dark-purple UI shown in the design reference.
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — dark navy/purple theme ───────────────────────────────────────
st.markdown("""
<style>
/* ── Root / body ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f0e1a !important;
    color: #e2e0f0 !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stMain"] {
    background-color: #0f0e1a !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1a1830 !important;
    border-right: 1px solid #2e2b4a !important;
}
[data-testid="stSidebar"] * { color: #c8c4e8 !important; }
[data-testid="stSidebar"] .stRadio label { color: #c8c4e8 !important; font-size: 14px; }
[data-testid="stSidebarContent"] { padding-top: 1.5rem; }

/* ── Upload box ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #5b4fcf !important;
    border-radius: 10px !important;
    background: #1f1d35 !important;
    padding: 4px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6c4fcf, #a855c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1f1d35;
    border: 1px solid #2e2b4a;
    border-radius: 12px;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] { color: #a09dc8 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #e2e0f0 !important; font-size: 26px !important; font-weight: 700 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #1a1830; border-radius: 10px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #a09dc8 !important; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #3b2fa0 !important; color: white !important; }

/* ── Selectbox / inputs ── */
[data-testid="stSelectbox"] select,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background-color: #1f1d35 !important;
    color: #e2e0f0 !important;
    border: 1px solid #3b3760 !important;
    border-radius: 8px !important;
}

/* ── Info / success boxes ── */
.stAlert { border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: #2e2b4a !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#12102a",
    font_color="#c8c4e8",
    colorway=["#7c6fcd", "#a855c7", "#e879a0", "#f59e0b", "#34d399", "#60a5fa"],
    xaxis=dict(gridcolor="#2e2b4a", linecolor="#2e2b4a"),
    yaxis=dict(gridcolor="#2e2b4a", linecolor="#2e2b4a"),
)

# ── Data loading & caching ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.dropna(subset=["Description"])
    df["Description"] = df["Description"].str.strip()
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    return df

@st.cache_data(show_spinner=False)
def build_rfm(_df: pd.DataFrame) -> pd.DataFrame:
    snap = _df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = _df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snap - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()
    return rfm

@st.cache_data(show_spinner=False)
def cluster_rfm(_rfm: pd.DataFrame, k: int = 4):
    scaler = StandardScaler()
    X = scaler.fit_transform(_rfm[["Recency", "Frequency", "Monetary"]])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    _rfm = _rfm.copy()
    _rfm["Cluster"] = km.fit_predict(X)
    profile = _rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    # Auto-label
    label_map = {}
    remaining = list(profile.index)
    hv = profile.loc[remaining, "Frequency"].idxmax()
    label_map[hv] = "High-Value"; remaining.remove(hv)
    ar = profile.loc[remaining, "Recency"].idxmax()
    label_map[ar] = "At-Risk"; remaining.remove(ar)
    reg = profile.loc[remaining, "Monetary"].idxmax()
    label_map[reg] = "Regular"; remaining.remove(reg)
    for c in remaining:
        label_map[c] = "Occasional"
    _rfm["Segment"] = _rfm["Cluster"].map(label_map)
    return _rfm, km, scaler, label_map

@st.cache_data(show_spinner=False)
def build_collab_sim(_df: pd.DataFrame):
    basket = _df.groupby(["CustomerID", "Description"])["Quantity"].sum().unstack(fill_value=0)
    pop = (basket > 0).sum(axis=0)
    keep = pop[pop >= 3].index
    if len(keep) > 1000:
        keep = pop.loc[keep].sort_values(ascending=False).head(1000).index
    basket = basket[keep]
    sim = cosine_similarity(basket.T)
    return pd.DataFrame(sim, index=basket.columns, columns=basket.columns)

@st.cache_data(show_spinner=False)
def build_tfidf_sim(_df: pd.DataFrame):
    products = _df["Description"].dropna().unique()
    tfidf = TfidfVectorizer(stop_words="english")
    mat = tfidf.fit_transform(products)
    sim = cosine_similarity(mat)
    return pd.DataFrame(sim, index=products, columns=products)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:40px;'>🛒</div>
        <div style='font-size:18px; font-weight:700; color:#d4d0f5;'>Shopper Spectrum</div>
        <div style='font-size:11px; color:#7b78a8; margin-top:2px; letter-spacing:.5px;'>
            INNOVEXIS · Data Science and Gen AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Upload online_retail.csv**")
    uploaded = st.file_uploader(" ", type=["csv"], label_visibility="collapsed")
    st.caption("100MB per file • CSV")
    st.divider()

    st.markdown("**Navigate**")
    page = st.radio(
        "",
        ["🏠 Overview", "📊 EDA & Insights", "🎯 Customer Segments",
         "🔮 Recommendations", "🧪 Hypothesis Tests"],
        label_visibility="collapsed",
    )

# ── Load data ─────────────────────────────────────────────────────────────────
df = None
if uploaded:
    with st.spinner("Loading and cleaning data…"):
        df = load_and_clean(uploaded.getvalue())

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    # Hero banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #6c4fcf 0%, #a855c7 50%, #e879a0 100%);
                border-radius: 18px; padding: 48px 40px; text-align:center; margin-bottom:28px;">
        <div style="font-size:38px;">🛒</div>
        <div style="font-size:36px; font-weight:800; color:white; margin:8px 0 6px 0;">
            Shopper Spectrum
        </div>
        <div style="font-size:15px; color:rgba(255,255,255,0.80); letter-spacing:.5px;">
            Customer Segmentation &amp; Product Recommendation Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)
    cards = [
        ("🎯", "RFM Clustering",
         "Segment customers into High-Value, Regular, At-Risk &amp; Occasional groups using KMeans.",
         "#1d2a3a", "#5eaaff"),
        ("💛", "Collaborative Filtering",
         "Personalised product picks based on what similar shoppers bought.",
         "#2a1d35", "#c084fc"),
        ("📄", "Content-Based Recs",
         "TF-IDF + cosine similarity on product descriptions for item-to-item suggestions.",
         "#1d2a35", "#34d399"),
    ]
    for col, (icon, title, desc, bg, accent) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div style="background:{bg}; border:1px solid #2e2b4a; border-radius:14px;
                        padding:22px 20px; min-height:140px;">
                <div style="font-size:20px; margin-bottom:6px;">{icon}
                    <span style="color:{accent}; font-weight:700; font-size:15px;">{title}</span>
                </div>
                <div style="color:#a09dc8; font-size:13px; line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if df is None:
        st.markdown("""
        <div style="background:#1f1d35; border:1px solid #3b2fa0; border-radius:12px;
                    padding:20px 24px; color:#c8c4e8; font-size:14px;">
            👆 <b>Upload your <code style='background:#2e2b4a;padding:2px 6px;
            border-radius:4px;color:#a5f3fc'>online_retail.csv</code> in the sidebar to begin.</b><br>
            <span style='color:#7b78a8; font-size:12px;margin-top:4px;display:block;'>
            Dataset source: UCI ML Repository – Online Retail (or Kaggle mirror).</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ Data loaded — **{len(df):,}** clean transactions · **{df['CustomerID'].nunique():,}** customers · **{df['Description'].nunique():,}** products")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{len(df):,}")
        c2.metric("Unique Customers", f"{df['CustomerID'].nunique():,}")
        c3.metric("Unique Products", f"{df['Description'].nunique():,}")
        c4.metric("Total Revenue", f"£{df['TotalPrice'].sum():,.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.title("📊 EDA & Insights")
    if df is None:
        st.warning("Upload `online_retail.csv` in the sidebar to continue.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Customers", f"{df['CustomerID'].nunique():,}")
    c3.metric("Products", f"{df['Description'].nunique():,}")
    c4.metric("Revenue", f"£{df['TotalPrice'].sum():,.0f}")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Revenue Over Time", "Top Products", "Top Countries", "Price Distribution"])

    with tab1:
        monthly = df.groupby("Month")["TotalPrice"].sum().reset_index()
        fig = px.area(monthly, x="Month", y="TotalPrice",
                      title="Monthly Revenue",
                      labels={"TotalPrice": "Revenue (£)", "Month": ""},
                      color_discrete_sequence=["#7c6fcd"])
        fig.update_traces(fill="tozeroy", fillcolor="rgba(124,111,205,0.15)")
        fig.update_layout(**PLOTLY_THEME, title_font_size=16)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top_p = (df.groupby("Description")["TotalPrice"].sum()
                   .sort_values(ascending=False).head(15).reset_index())
        fig = px.bar(top_p, x="TotalPrice", y="Description", orientation="h",
                     title="Top 15 Products by Revenue",
                     labels={"TotalPrice": "Revenue (£)", "Description": ""},
                     color="TotalPrice",
                     color_continuous_scale=["#3b2fa0", "#a855c7", "#e879a0"])
        fig.update_layout(**PLOTLY_THEME, title_font_size=16,
                          yaxis=dict(autorange="reversed", gridcolor="#2e2b4a"),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        top_c = (df.groupby("Country")["TotalPrice"].sum()
                   .sort_values(ascending=False).head(12).reset_index())
        fig = px.pie(top_c, names="Country", values="TotalPrice",
                     title="Revenue by Country (Top 12)",
                     color_discrete_sequence=px.colors.sequential.Purpor)
        fig.update_layout(**PLOTLY_THEME, title_font_size=16)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        sample = df[df["UnitPrice"] < df["UnitPrice"].quantile(0.99)]
        fig = px.histogram(sample, x="UnitPrice", nbins=60,
                           title="Unit Price Distribution (99th percentile cap)",
                           labels={"UnitPrice": "Unit Price (£)"},
                           color_discrete_sequence=["#a855c7"])
        fig.update_layout(**PLOTLY_THEME, title_font_size=16)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🔢 Raw Data Sample")
    st.dataframe(df.head(200), use_container_width=True, height=280)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Customer Segments":
    st.title("🎯 Customer Segmentation")
    if df is None:
        st.warning("Upload `online_retail.csv` in the sidebar to continue.")
        st.stop()

    with st.spinner("Computing RFM & clustering…"):
        rfm = build_rfm(df)
        rfm_seg, km, scaler, label_map = cluster_rfm(rfm)

    SEGMENT_COLORS = {
        "High-Value": "#34d399", "Regular": "#60a5fa",
        "Occasional": "#f59e0b", "At-Risk": "#e879a0",
    }

    # Summary cards
    counts = rfm_seg["Segment"].value_counts()
    cols = st.columns(4)
    for col, seg in zip(cols, ["High-Value", "Regular", "Occasional", "At-Risk"]):
        n = counts.get(seg, 0)
        pct = n / len(rfm_seg) * 100
        col.markdown(f"""
        <div style="background:#1f1d35; border-left:4px solid {SEGMENT_COLORS[seg]};
                    border-radius:10px; padding:16px 18px;">
            <div style="font-size:11px; color:#7b78a8; text-transform:uppercase; letter-spacing:.8px;">{seg}</div>
            <div style="font-size:28px; font-weight:700; color:{SEGMENT_COLORS[seg]};">{n}</div>
            <div style="font-size:12px; color:#a09dc8;">{pct:.1f}% of customers</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Scatter Plot", "Segment Profiles", "Distribution", "Predict Segment"])

    with tab1:
        fig = px.scatter(rfm_seg, x="Recency", y="Monetary", size="Frequency",
                         color="Segment", hover_data=["CustomerID", "Frequency"],
                         color_discrete_map=SEGMENT_COLORS,
                         title="RFM Scatter — Recency vs Monetary (size = Frequency)",
                         opacity=0.7)
        fig.update_layout(**PLOTLY_THEME, title_font_size=15)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        profile = rfm_seg.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)
        profile["Customers"] = rfm_seg["Segment"].value_counts()
        st.dataframe(profile.style.format({"Recency": "{:.1f}", "Frequency": "{:.1f}", "Monetary": "£{:,.0f}"}),
                     use_container_width=True)

        fig = px.bar(profile.reset_index(), x="Segment",
                     y=["Recency", "Frequency", "Monetary"],
                     barmode="group", title="Average RFM per Segment",
                     color_discrete_sequence=["#7c6fcd", "#a855c7", "#e879a0"])
        fig.update_layout(**PLOTLY_THEME, title_font_size=15)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        metric = st.selectbox("Metric", ["Recency", "Frequency", "Monetary"])
        fig = px.box(rfm_seg, x="Segment", y=metric, color="Segment",
                     color_discrete_map=SEGMENT_COLORS,
                     title=f"{metric} Distribution by Segment")
        fig.update_layout(**PLOTLY_THEME, title_font_size=15, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("#### Predict a Customer's Segment")
        c1, c2, c3 = st.columns(3)
        rec = c1.number_input("Recency (days)", 0, 1000, 30)
        freq = c2.number_input("Frequency (orders)", 0, 500, 5)
        mon = c3.number_input("Monetary (£)", 0.0, 200000.0, 500.0, step=50.0)

        if st.button("Predict", type="primary"):
            x = scaler.transform([[rec, freq, mon]])
            cid = int(km.predict(x)[0])
            seg = label_map.get(cid, f"Cluster {cid}")
            color = SEGMENT_COLORS.get(seg, "#7c6fcd")
            st.markdown(f"""
            <div style="border-left:6px solid {color}; background:#1f1d35;
                        padding:20px 24px; border-radius:10px; margin-top:12px;">
                <div style="font-size:12px; color:#7b78a8;">Predicted Segment</div>
                <div style="font-size:30px; font-weight:800; color:{color};">{seg}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Recommendations":
    st.title("🔮 Product Recommendations")
    if df is None:
        st.warning("Upload `online_retail.csv` in the sidebar to continue.")
        st.stop()

    tab1, tab2 = st.tabs(["🤝 Collaborative Filtering", "📄 Content-Based (TF-IDF)"])

    with tab1:
        st.markdown("##### Item-based collaborative filtering — cosine similarity on the customer × product matrix")
        with st.spinner("Building similarity matrix…"):
            collab_sim = build_collab_sim(df)

        product = st.selectbox("Select a product", sorted(collab_sim.index.tolist()), key="cf")
        top_n = st.slider("Number of recommendations", 3, 10, 5, key="cf_n")
        if st.button("Get Recommendations", key="cf_btn", type="primary"):
            scores = collab_sim.loc[product].drop(labels=[product], errors="ignore")
            recs = scores.sort_values(ascending=False).head(top_n)
            cols = st.columns(min(top_n, 5))
            for i, (prod, score) in enumerate(recs.items()):
                with cols[i % 5]:
                    st.markdown(f"""
                    <div style="background:#1f1d35; border:1px solid #2e2b4a;
                                border-radius:12px; padding:16px 14px; min-height:120px;
                                display:flex; flex-direction:column; justify-content:space-between;">
                        <div style="font-size:12px; font-weight:600; color:#d4d0f5;">{prod}</div>
                        <div style="font-size:11px; color:#7c6fcd; margin-top:8px;">
                            Similarity: {score:.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.markdown("##### Content-based recommendations — TF-IDF on product description text")
        with st.spinner("Building TF-IDF matrix…"):
            tfidf_sim = build_tfidf_sim(df)

        product2 = st.selectbox("Select a product", sorted(tfidf_sim.index.tolist()), key="cb")
        top_n2 = st.slider("Number of recommendations", 3, 10, 5, key="cb_n")
        if st.button("Get Recommendations", key="cb_btn", type="primary"):
            scores2 = tfidf_sim.loc[product2].drop(labels=[product2], errors="ignore")
            recs2 = scores2.sort_values(ascending=False).head(top_n2)
            cols2 = st.columns(min(top_n2, 5))
            for i, (prod, score) in enumerate(recs2.items()):
                with cols2[i % 5]:
                    st.markdown(f"""
                    <div style="background:#1a1d2e; border:1px solid #2e2b4a;
                                border-radius:12px; padding:16px 14px; min-height:120px;
                                display:flex; flex-direction:column; justify-content:space-between;">
                        <div style="font-size:12px; font-weight:600; color:#d4d0f5;">{prod}</div>
                        <div style="font-size:11px; color:#34d399; margin-top:8px;">
                            Similarity: {score:.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — HYPOTHESIS TESTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Hypothesis Tests":
    st.title("🧪 Hypothesis Tests")
    if df is None:
        st.warning("Upload `online_retail.csv` in the sidebar to continue.")
        st.stop()

    with st.spinner("Running segmentation…"):
        rfm = build_rfm(df)
        rfm_seg, _, _, _ = cluster_rfm(rfm)

    def result_card(title, stat, pval, h0, h1, alpha=0.05):
        reject = pval < alpha
        color = "#34d399" if reject else "#e879a0"
        verdict = "Reject H₀" if reject else "Fail to Reject H₀"
        st.markdown(f"""
        <div style="background:#1f1d35; border:1px solid #2e2b4a; border-radius:14px;
                    padding:20px 22px; margin-bottom:16px;">
            <div style="font-size:15px; font-weight:700; color:#d4d0f5; margin-bottom:10px;">{title}</div>
            <div style="font-size:12px; color:#7b78a8;">H₀: {h0}</div>
            <div style="font-size:12px; color:#a09dc8; margin-bottom:12px;">H₁: {h1}</div>
            <div style="display:flex; gap:24px;">
                <div><div style="font-size:11px;color:#7b78a8;">Test Stat</div>
                     <div style="font-size:18px;font-weight:700;color:#c8c4e8;">{stat:.4f}</div></div>
                <div><div style="font-size:11px;color:#7b78a8;">p-value</div>
                     <div style="font-size:18px;font-weight:700;color:#c8c4e8;">{pval:.4f}</div></div>
                <div><div style="font-size:11px;color:#7b78a8;">Verdict</div>
                     <div style="font-size:18px;font-weight:700;color:{color};">{verdict}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Test 1 — UK vs non-UK avg order value
    uk = df[df["Country"] == "United Kingdom"]["TotalPrice"]
    non_uk = df[df["Country"] != "United Kingdom"]["TotalPrice"]
    stat1, p1 = stats.ttest_ind(uk, non_uk, equal_var=False)
    result_card(
        "T-Test: Average Order Value — UK vs Non-UK",
        stat1, p1,
        "Mean order value is the same for UK and non-UK customers",
        "Mean order value differs between UK and non-UK customers",
    )

    # Test 2 — High-Value vs At-Risk monetary
    hv = rfm_seg[rfm_seg["Segment"] == "High-Value"]["Monetary"]
    ar = rfm_seg[rfm_seg["Segment"] == "At-Risk"]["Monetary"]
    stat2, p2 = stats.ttest_ind(hv, ar, equal_var=False)
    result_card(
        "T-Test: Total Spend — High-Value vs At-Risk Customers",
        stat2, p2,
        "Mean total spend is equal for High-Value and At-Risk segments",
        "High-Value customers spend significantly more than At-Risk customers",
    )

    # Test 3 — Correlation: Frequency vs Monetary
    corr, p3 = stats.pearsonr(rfm_seg["Frequency"], rfm_seg["Monetary"])
    result_card(
        "Pearson Correlation: Purchase Frequency vs Total Spend",
        corr, p3,
        "There is no linear correlation between frequency and monetary value",
        "There is a significant positive correlation between frequency and spend",
    )

    # Test 4 — Chi-squared: Segment vs high/low recency
    rfm_seg["RecencyBand"] = pd.cut(rfm_seg["Recency"], bins=[0, 30, 90, 999],
                                     labels=["Recent", "Moderate", "Lapsed"])
    ct = pd.crosstab(rfm_seg["Segment"], rfm_seg["RecencyBand"])
    chi2, p4, dof, _ = stats.chi2_contingency(ct)
    result_card(
        "Chi-Squared Test: Segment vs Recency Band",
        chi2, p4,
        "Customer segment is independent of recency band",
        "Customer segment and recency band are significantly associated",
    )

    st.divider()
    st.subheader("Correlation Heatmap — RFM Features")
    corr_mat = rfm_seg[["Recency", "Frequency", "Monetary"]].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_mat.values, x=corr_mat.columns, y=corr_mat.columns,
        colorscale="Purpor", zmin=-1, zmax=1,
        text=corr_mat.values.round(2), texttemplate="%{text}",
    ))
    fig.update_layout(**PLOTLY_THEME, title="RFM Correlation Matrix", height=320)
    st.plotly_chart(fig, use_container_width=True)