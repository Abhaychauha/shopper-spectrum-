"""
Shopper Spectrum — Full Streamlit App
Dark ocean/teal theme · Fixed yaxis conflict · Easy charts
"""
import io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS — dark ocean theme ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    background:#071318 !important; color:#d1fae5 !important;
    font-family:'Inter',sans-serif !important;
}
[data-testid="stSidebar"]{
    background:#0a1f28 !important;
    border-right:1px solid #0e3344 !important;
}
[data-testid="stSidebar"] *{ color:#94d8cc !important; }
[data-testid="stFileUploader"]{
    border:2px dashed #14b8a6 !important;
    border-radius:10px !important; background:#0a1f28 !important;
}
.stButton>button{
    background:linear-gradient(135deg,#0d9488,#06b6d4) !important;
    color:#fff !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important;
}
.stButton>button:hover{ opacity:.85 !important; }
[data-testid="stMetric"]{
    background:#0d2233; border:1px solid #0e3a4a;
    border-radius:14px; padding:16px 18px !important;
}
[data-testid="stMetricLabel"]{ color:#5eead4 !important; font-size:11px !important;
    letter-spacing:.8px !important; text-transform:uppercase; }
[data-testid="stMetricValue"]{ color:#d1fae5 !important; font-size:28px !important;
    font-weight:800 !important; }
.stTabs [data-baseweb="tab-list"]{
    background:#0a1f28; border-radius:10px; gap:4px; padding:4px; border-bottom:none !important;
}
.stTabs [data-baseweb="tab"]{ border-radius:8px; color:#5eead4 !important;
    font-weight:500; padding:8px 18px; }
.stTabs [aria-selected="true"]{ background:#0e3344 !important;
    color:#2dd4bf !important; border-bottom:2px solid #f97316 !important; }
[data-testid="stExpander"]{ background:#0d2233 !important;
    border:1px solid #0e3a4a !important; border-radius:12px !important; margin-bottom:10px; }
[data-testid="stExpander"] summary{ color:#d1fae5 !important; font-weight:600; }
hr{ border-color:#0e3344 !important; }
.stAlert{ border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly base theme (NO xaxis/yaxis keys — pass them explicitly via lay()) ───
_PT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a1a24",
    font=dict(color="#94d8cc", family="Inter"),
    margin=dict(l=40, r=20, t=44, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#0e3344"),
)
_AXIS_DEFAULTS = dict(gridcolor="#0e3344", linecolor="#0e3344",
                      tickcolor="#0e3344", showgrid=True)

def lay(**overrides) -> dict:
    """Return a merged layout dict. Safely handles xaxis/yaxis overrides."""
    d = dict(**_PT_BASE)
    d["xaxis"] = dict(**_AXIS_DEFAULTS)
    d["yaxis"] = dict(**_AXIS_DEFAULTS)
    for k, v in overrides.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict):
            d[k] = {**_AXIS_DEFAULTS, **v}
        else:
            d[k] = v
    return d

COLORS  = ["#14b8a6","#06b6d4","#f97316","#facc15","#a78bfa","#fb7185","#34d399","#60a5fa"]
TEAL_CS = [[0,"#071318"],[0.5,"#0d9488"],[1,"#2dd4bf"]]
ORNG_CS = [[0,"#1a0a00"],[0.5,"#c2410c"],[1,"#fb923c"]]
CYAN_CS = [[0,"#071318"],[0.5,"#0369a1"],[1,"#38bdf8"]]
SEG_COLORS = {"High-Value":"#2dd4bf","Regular":"#38bdf8",
              "Occasional":"#facc15","At-Risk":"#fb7185"}

# ── Data helpers ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_clean(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.dropna(subset=["Description"])
    df["Description"]  = df["Description"].str.strip()
    df["CustomerID"]   = df["CustomerID"].astype(int)
    df["InvoiceDate"]  = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"]   = df["Quantity"] * df["UnitPrice"]
    df["Month"]        = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Weekday"]      = df["InvoiceDate"].dt.day_name()
    df["Hour"]         = df["InvoiceDate"].dt.hour
    return df

@st.cache_data(show_spinner=False)
def build_rfm(_df):
    snap = _df["InvoiceDate"].max() + pd.Timedelta(days=1)
    return _df.groupby("CustomerID").agg(
        Recency  = ("InvoiceDate",  lambda x: (snap - x.max()).days),
        Frequency= ("InvoiceNo",    "nunique"),
        Monetary = ("TotalPrice",   "sum"),
    ).reset_index()

@st.cache_data(show_spinner=False)
def run_clustering(_rfm, k=4):
    sc = StandardScaler()
    X  = sc.fit_transform(_rfm[["Recency","Frequency","Monetary"]])
    inertias, silhouettes = [], []
    for ki in range(2, 11):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        lb = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, lb))
    best_k = int(np.argmax(silhouettes) + 2)
    km_f   = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm2   = _rfm.copy()
    rfm2["Cluster"] = km_f.fit_predict(X)
    sil    = round(silhouette_score(X, rfm2["Cluster"]), 4)
    prof   = rfm2.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
    rem    = list(prof.index); lmap = {}
    hv = prof.loc[rem,"Frequency"].idxmax();  lmap[hv]="High-Value";  rem.remove(hv)
    ar = prof.loc[rem,"Recency"].idxmax();    lmap[ar]="At-Risk";     rem.remove(ar)
    rg = prof.loc[rem,"Monetary"].idxmax();   lmap[rg]="Regular";     rem.remove(rg)
    for c in rem: lmap[c] = "Occasional"
    rfm2["Segment"] = rfm2["Cluster"].map(lmap)
    return rfm2, km_f, sc, lmap, inertias, silhouettes, best_k, sil

@st.cache_data(show_spinner=False)
def build_collab(_df):
    b   = _df.groupby(["CustomerID","Description"])["Quantity"].sum().unstack(fill_value=0)
    pop = (b > 0).sum(0)
    keep = pop[pop >= 3].index
    if len(keep) > 800:
        keep = pop[keep].sort_values(ascending=False).head(800).index
    b   = b[keep]
    sim = cosine_similarity(b.T)
    return pd.DataFrame(sim, index=b.columns, columns=b.columns)

@st.cache_data(show_spinner=False)
def build_tfidf(_df):
    prods = _df["Description"].dropna().unique()
    tv    = TfidfVectorizer(stop_words="english")
    mat   = tv.fit_transform(prods)
    sim   = cosine_similarity(mat)
    return pd.DataFrame(sim, index=prods, columns=prods)

# ── UI helpers ─────────────────────────────────────────────────────────────────
def section(title: str):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:18px 0 10px 0;">
      <div style="width:4px;height:22px;background:linear-gradient(180deg,#0d9488,#06b6d4);
           border-radius:3px;"></div>
      <span style="font-size:14px;font-weight:600;color:#5eead4;">{title}</span>
    </div>""", unsafe_allow_html=True)

def hero(subtitle="Customer Segmentation &amp; Product Recommendation · P Suman Sangeet"):
    st.markdown(f"""
    <div style="background:linear-gradient(120deg,#0f4c75 0%,#1b6ca8 30%,#0d9488 65%,#f97316 100%);
         border-radius:16px;padding:38px 44px;margin-bottom:22px;text-align:center;">
      <div style="font-size:34px;font-weight:800;color:#fff;letter-spacing:-.5px;">🛒 Shopper Spectrum</div>
      <div style="color:rgba(255,255,255,.75);font-size:13px;margin-top:6px;letter-spacing:.3px;">{subtitle}</div>
    </div>""", unsafe_allow_html=True)

def kpi_card(col, label, value, sub, color="#2dd4bf"):
    col.markdown(f"""
    <div style="background:#0d2233;border:1px solid #0e3a4a;border-radius:14px;
         padding:20px 14px;text-align:center;">
      <div style="font-size:10px;font-weight:600;color:#5eead4;letter-spacing:1px;
           text-transform:uppercase;margin-bottom:8px;">{label}</div>
      <div style="font-size:28px;font-weight:800;color:#d1fae5;line-height:1;">{value}</div>
      <div style="font-size:11px;color:{color};margin-top:6px;">+{sub}</div>
    </div>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 20px 0;'>
      <div style='font-size:36px;'>🛒</div>
      <div style='font-size:17px;font-weight:700;color:#d1fae5;'>Shopper Spectrum</div>
      <div style='font-size:10px;color:#3d7a72;margin-top:3px;letter-spacing:.6px;'>
        INNOVEXIS · Data Science and Gen AI</div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("**📂 Upload online_retail.csv**")
    uploaded = st.file_uploader(" ", type=["csv"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Navigate**")
    page = st.radio("", ["🏠 Overview","📊 EDA & Insights","🎯 Customer Segments",
                         "🔮 Recommendations","🧪 Hypothesis Tests"],
                    label_visibility="collapsed")

df = None
if uploaded:
    with st.spinner("Cleaning data…"):
        df = load_clean(uploaded.getvalue())

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    hero()
    if df is None:
        st.markdown("""
        <div style="background:#0d2233;border:1px solid #14b8a6;border-radius:12px;
             padding:22px 26px;color:#94d8cc;font-size:14px;">
          👆 <b>Upload your <code style='background:#071318;padding:2px 7px;border-radius:4px;
          color:#2dd4bf'>online_retail.csv</code> in the sidebar to begin.</b><br>
          <span style='color:#3d7a72;font-size:12px;margin-top:6px;display:block;'>
          Dataset source: UCI ML Repository – Online Retail (or Kaggle mirror).</span>
        </div>""", unsafe_allow_html=True)
        st.stop()

    with st.spinner("Computing silhouette…"):
        rfm0 = build_rfm(df)
        _,_,_,_,_,_,best_k,sil = run_clustering(rfm0)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    rev = df["TotalPrice"].sum()
    kpi_card(c1,"Customers",  f"{df['CustomerID'].nunique():,}", "unique")
    kpi_card(c2,"Products",   f"{df['Description'].nunique():,}","SKUs",  "#38bdf8")
    kpi_card(c3,"Invoices",   f"{df['InvoiceNo'].nunique():,}",  "orders","#f97316")
    kpi_card(c4,"Revenue",    f"£{rev/1e6:.2f}M",               "total", "#facc15")
    kpi_card(c5,"Countries",  f"{df['Country'].nunique()}",      "markets","#a78bfa")
    kpi_card(c6,"Silhouette", str(sil),                          "cluster quality","#fb7185")

    st.markdown("<br>", unsafe_allow_html=True)
    section("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True, height=280)

    col1,col2 = st.columns(2)
    with col1:
        section("Missing Values (after clean)")
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Column","Missing"]
        fig = px.bar(mv, x="Column", y="Missing",
                     color="Missing", color_continuous_scale=TEAL_CS)
        fig.update_layout(**lay(height=230, coloraxis_showscale=False,
                                xaxis=dict(tickangle=-30)))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        section("Data Type Distribution")
        dt = df.dtypes.astype(str).value_counts().reset_index()
        dt.columns = ["Type","Count"]
        fig = px.pie(dt, names="Type", values="Count",
                     color_discrete_sequence=COLORS, hole=0.5)
        fig.update_layout(**lay(height=230))
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# EDA & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    hero()
    if df is None:
        st.warning("Upload data first."); st.stop()

    tab1,tab2,tab3,tab4,tab5 = st.tabs(
        ["🌍 Geography","📦 Products","📈 Time Trends","💰 Spend Patterns","📊 RFM Distributions"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            section("Top 10 Countries — Transaction Volume")
            tc = df["Country"].value_counts().head(10).reset_index()
            tc.columns = ["Country","Transactions"]
            fig = px.bar(tc, x="Transactions", y="Country", orientation="h",
                         color="Transactions", color_continuous_scale=TEAL_CS,
                         labels={"Transactions":"Transactions","Country":"Country"})
            # safe: pass yaxis via lay()
            fig.update_layout(**lay(height=370, coloraxis_showscale=False,
                                    yaxis=dict(autorange="reversed")))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            section("Avg Spend per Customer by Country")
            g = df.groupby("Country").agg(
                Revenue=("TotalPrice","sum"), Cust=("CustomerID","nunique"))
            g["AvgSpend"] = (g["Revenue"]/g["Cust"]).round(0)
            g = g.sort_values("AvgSpend",ascending=False).head(10).reset_index()
            fig = px.bar(g, x="AvgSpend", y="Country", orientation="h",
                         color="AvgSpend", color_continuous_scale=CYAN_CS,
                         labels={"AvgSpend":"Avg Spend (£)"})
            fig.update_layout(**lay(height=370, coloraxis_showscale=False,
                                    yaxis=dict(autorange="reversed")))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1,col2 = st.columns(2)
        with col1:
            section("Top 15 Products by Revenue")
            tp = (df.groupby("Description")["TotalPrice"].sum()
                    .sort_values(ascending=False).head(15).reset_index())
            fig = px.bar(tp, x="TotalPrice", y="Description", orientation="h",
                         color="TotalPrice", color_continuous_scale=TEAL_CS,
                         labels={"TotalPrice":"Revenue (£)","Description":""})
            fig.update_layout(**lay(height=420, coloraxis_showscale=False,
                                    yaxis=dict(autorange="reversed")))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            section("Top 15 Products by Units Sold")
            tp2 = (df.groupby("Description")["Quantity"].sum()
                     .sort_values(ascending=False).head(15).reset_index())
            fig = px.bar(tp2, x="Quantity", y="Description", orientation="h",
                         color="Quantity", color_continuous_scale=ORNG_CS,
                         labels={"Quantity":"Units Sold","Description":""})
            fig.update_layout(**lay(height=420, coloraxis_showscale=False,
                                    yaxis=dict(autorange="reversed")))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("Monthly Revenue Trend")
        mr = df.groupby("Month")["TotalPrice"].sum().reset_index()
        fig = px.line(mr, x="Month", y="TotalPrice", markers=True,
                      color_discrete_sequence=["#2dd4bf"],
                      labels={"TotalPrice":"Revenue (£)","Month":""})
        fig.update_traces(line_width=2.5, marker_size=6,
                          fill="tozeroy", fillcolor="rgba(45,212,191,0.10)")
        fig.update_layout(**lay(height=290))
        st.plotly_chart(fig, use_container_width=True)

        col1,col2 = st.columns(2)
        with col1:
            section("Orders by Day of Week")
            dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow = (df.groupby("Weekday")["InvoiceNo"].nunique()
                     .reindex(dow_order).reset_index())
            dow.columns = ["Day","Orders"]
            fig = px.bar(dow, x="Day", y="Orders",
                         color="Orders", color_continuous_scale=TEAL_CS)
            fig.update_layout(**lay(height=260, coloraxis_showscale=False))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            section("Orders by Hour of Day")
            hr = df.groupby("Hour")["InvoiceNo"].nunique().reset_index()
            hr.columns = ["Hour","Orders"]
            fig = px.bar(hr, x="Hour", y="Orders",
                         color="Orders", color_continuous_scale=CYAN_CS)
            fig.update_layout(**lay(height=260, coloraxis_showscale=False))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1,col2 = st.columns(2)
        with col1:
            section("Unit Price Distribution")
            cap = df[df["UnitPrice"] < df["UnitPrice"].quantile(0.97)]
            fig = px.histogram(cap, x="UnitPrice", nbins=50,
                               color_discrete_sequence=["#14b8a6"],
                               labels={"UnitPrice":"Unit Price (£)"})
            fig.update_layout(**lay(height=280, bargap=0.05))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            section("Revenue Share by Country (Top 8)")
            rs = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
            top7 = rs.head(7).to_dict()
            top7["Other"] = rs.iloc[7:].sum()
            pie_df = pd.DataFrame({"Country":list(top7.keys()),
                                   "Revenue":list(top7.values())})
            fig = px.pie(pie_df, names="Country", values="Revenue",
                         hole=0.45, color_discrete_sequence=COLORS)
            fig.update_layout(**lay(height=280))
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              textfont_size=11)
            st.plotly_chart(fig, use_container_width=True)

        section("Monthly Quantity Sold")
        mq = df.groupby("Month")["Quantity"].sum().reset_index()
        fig = px.bar(mq, x="Month", y="Quantity",
                     color="Quantity", color_continuous_scale=ORNG_CS,
                     labels={"Quantity":"Units Sold"})
        fig.update_layout(**lay(height=260, coloraxis_showscale=False))
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        with st.spinner("Building RFM…"):
            rfm_e = build_rfm(df)
        col1,col2,col3 = st.columns(3)
        for col, metric, cs in zip([col1,col2,col3],
                                    ["Recency","Frequency","Monetary"],
                                    [TEAL_CS, CYAN_CS, ORNG_CS]):
            with col:
                section(f"{metric} Distribution")
                cap_q = 0.99 if metric != "Frequency" else 1.0
                data = rfm_e[rfm_e[metric] < rfm_e[metric].quantile(cap_q)]
                fig = px.histogram(data, x=metric, nbins=40,
                                   color_discrete_sequence=[cs[1][1]])
                fig.update_layout(**lay(height=230, bargap=0.04))
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Customer Segments":
    hero()
    if df is None:
        st.warning("Upload data first."); st.stop()

    with st.spinner("Running KMeans…"):
        rfm = build_rfm(df)
        rfm_seg,km,sc,lmap,inertias,silhouettes,best_k,sil = run_clustering(rfm)

    tab1,tab2,tab3,tab4 = st.tabs(
        ["📐 Elbow & Silhouette","🗂 Cluster Profiles","📊 Visualizations","🔮 Predict Segment"])

    with tab1:
        col1,col2 = st.columns(2)
        ks = list(range(2,11))
        with col1:
            section("Elbow Method")
            fig = px.line(x=ks, y=inertias, markers=True,
                          labels={"x":"Clusters (k)","y":"Inertia"},
                          color_discrete_sequence=["#2dd4bf"])
            fig.update_traces(line_width=2.5, marker_size=7, marker_color="#5eead4")
            fig.update_layout(**lay(height=300))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            section("Silhouette Scores")
            fig = px.line(x=ks, y=silhouettes, markers=True,
                          labels={"x":"Clusters (k)","y":"Silhouette Score"},
                          color_discrete_sequence=["#f97316"])
            fig.update_traces(line_width=2.5, marker_size=7, marker_color="#fb923c")
            fig.update_layout(**lay(height=300))
            st.plotly_chart(fig, use_container_width=True)
        st.info(f"✅  Best k by silhouette = **{best_k}**  |  Current k = **4**  |  Silhouette = **{sil}**")

    with tab2:
        counts = rfm_seg["Segment"].value_counts()
        c1,c2,c3,c4 = st.columns(4)
        for col,seg in zip([c1,c2,c3,c4],["High-Value","Regular","Occasional","At-Risk"]):
            n=counts.get(seg,0); pct=n/len(rfm_seg)*100; color=SEG_COLORS[seg]
            col.markdown(f"""
            <div style="background:#0d2233;border-left:4px solid {color};
                 border-radius:10px;padding:14px 12px;">
              <div style="font-size:10px;color:#5eead4;letter-spacing:.8px;
                   text-transform:uppercase;">{seg}</div>
              <div style="font-size:26px;font-weight:800;color:{color};">{n:,}</div>
              <div style="font-size:11px;color:#94d8cc;">{pct:.1f}% of customers</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        section("Average RFM per Segment")
        prof = rfm_seg.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1)
        prof["Customers"] = rfm_seg["Segment"].value_counts()
        st.dataframe(prof.style.format({"Recency":"{:.1f}","Frequency":"{:.1f}",
                                         "Monetary":"£{:,.0f}","Customers":"{:,.0f}"}),
                     use_container_width=True)

    with tab3:
        col1,col2 = st.columns(2)
        with col1:
            section("Segment Customer Count")
            vc = rfm_seg["Segment"].value_counts().reset_index()
            vc.columns=["Segment","Count"]
            fig = px.bar(vc, x="Segment", y="Count", color="Segment",
                         color_discrete_map=SEG_COLORS, text="Count")
            fig.update_traces(textposition="outside", marker_cornerradius=5)
            fig.update_layout(**lay(height=310, showlegend=False))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            section("Recency vs Monetary (bubble = Frequency)")
            fig = px.scatter(rfm_seg, x="Recency", y="Monetary", color="Segment",
                             size="Frequency", hover_data=["CustomerID"],
                             opacity=0.65, color_discrete_map=SEG_COLORS,
                             labels={"Monetary":"Monetary (£)"})
            fig.update_layout(**lay(height=310))
            st.plotly_chart(fig, use_container_width=True)

        section("Average RFM per Segment — Grouped Bar")
        pr = rfm_seg.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1).reset_index()
        fig = px.bar(pr, x="Segment", y=["Recency","Frequency","Monetary"],
                     barmode="group",
                     color_discrete_sequence=["#2dd4bf","#38bdf8","#f97316"])
        fig.update_layout(**lay(height=290))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("#### Enter RFM values to predict the customer's segment")
        c1,c2,c3 = st.columns(3)
        rec  = c1.number_input("Recency (days)",   0, 1000, 30)
        freq = c2.number_input("Frequency (orders)",0, 500,  5)
        mon  = c3.number_input("Monetary (£)",  0.0, 200000.0, 500.0, step=50.0)
        if st.button("Predict Segment", type="primary"):
            x   = sc.transform([[rec,freq,mon]])
            cid = int(km.predict(x)[0])
            seg = lmap.get(cid, f"Cluster {cid}")
            color = SEG_COLORS.get(seg,"#2dd4bf")
            descs = {
                "High-Value": "Recent, frequent, big spender. Reward with loyalty perks.",
                "Regular":    "Steady buyer with moderate spend. Good for upselling.",
                "Occasional": "Infrequent buyer. Engage with re-activation campaigns.",
                "At-Risk":    "Long inactive. Win back with targeted offers.",
            }
            st.markdown(f"""
            <div style="border-left:6px solid {color};background:#0d2233;
                 padding:22px 26px;border-radius:12px;margin-top:14px;">
              <div style="font-size:11px;color:#5eead4;text-transform:uppercase;letter-spacing:.8px;">
                Predicted Segment</div>
              <div style="font-size:32px;font-weight:800;color:{color};margin:6px 0;">{seg}</div>
              <div style="color:#94d8cc;font-size:13px;">{descs.get(seg,"")}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Recommendations":
    hero()
    if df is None:
        st.warning("Upload data first."); st.stop()

    tab1,tab2 = st.tabs(["🤝 Collaborative Filtering","📄 Content-Based (TF-IDF)"])

    def rec_cards(recs, accent):
        cols = st.columns(min(len(recs), 5))
        for i,(p,s) in enumerate(recs.items()):
            with cols[i%5]:
                st.markdown(f"""
                <div style="background:#0d2233;border:1px solid #0e3a4a;border-radius:12px;
                     padding:14px 12px;min-height:110px;display:flex;flex-direction:column;
                     justify-content:space-between;">
                  <div style="font-size:12px;font-weight:600;color:#d1fae5;">{p}</div>
                  <div style="font-size:11px;color:{accent};margin-top:8px;">
                    Similarity: {s:.3f}</div>
                </div>""", unsafe_allow_html=True)

    with tab1:
        st.caption("Item-based collaborative filtering — cosine similarity on the customer × product purchase matrix.")
        with st.spinner("Building collaborative matrix…"):
            csim = build_collab(df)
        prod = st.selectbox("Select a Product", sorted(csim.index.tolist()), key="cf")
        n    = st.slider("Recommendations", 3, 10, 5, key="cfn")
        if st.button("Get Recommendations", key="cfb", type="primary"):
            recs = (csim.loc[prod].drop(labels=[prod], errors="ignore")
                        .sort_values(ascending=False).head(n))
            rec_cards(recs, "#2dd4bf")

    with tab2:
        st.caption("Content-based: TF-IDF vectors on product description text + cosine similarity.")
        with st.spinner("Building TF-IDF matrix…"):
            tsim = build_tfidf(df)
        prod2 = st.selectbox("Select a Product", sorted(tsim.index.tolist()), key="cb")
        n2    = st.slider("Recommendations", 3, 10, 5, key="cbn")
        if st.button("Get Recommendations", key="cbb", type="primary"):
            recs2 = (tsim.loc[prod2].drop(labels=[prod2], errors="ignore")
                         .sort_values(ascending=False).head(n2))
            rec_cards(recs2, "#f97316")

# ══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS TESTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Hypothesis Tests":
    hero()
    if df is None:
        st.warning("Upload data first."); st.stop()

    with st.spinner("Running segmentation…"):
        rfm_h = build_rfm(df)
        rfm_s,_,_,_,_,_,_,_ = run_clustering(rfm_h)

    def big_stats(stat, pval, stat_label="T-Statistic"):
        reject = pval < 0.05
        vcolor = "#2dd4bf" if reject else "#fb7185"
        verdict = "Reject H₀" if reject else "Fail to Reject H₀"
        pstr = f"{pval:.4e}" if pval < 0.001 else f"{pval:.4f}"
        c1,c2,c3 = st.columns(3)
        for col, lab, val in [(c1,stat_label,f"{stat:.4f}"),
                               (c2,"P-Value",pstr),
                               (c3,"Conclusion",verdict)]:
            vc = vcolor if lab=="Conclusion" else "#d1fae5"
            col.markdown(f"""
            <div style="padding:8px 0;">
              <div style="font-size:11px;color:#5eead4;">{lab}</div>
              <div style="font-size:32px;font-weight:800;color:{vc};line-height:1.1;">{val}</div>
            </div>""", unsafe_allow_html=True)
        return reject

    def insight(msg, reject):
        icon = "✅" if reject else "❌"
        bg   = "#071f18" if reject else "#1f0708"
        bdr  = "#2dd4bf" if reject else "#fb7185"
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {bdr};border-radius:10px;
             padding:14px 18px;margin-top:10px;font-size:13px;color:#d1fae5;">
          {icon} {msg}
        </div>""", unsafe_allow_html=True)

    # H1
    with st.expander("📌 H1 — UK vs Non-UK Customer Spending (Welch's t-test)", expanded=True):
        uk  = df[df["Country"]=="United Kingdom"]["TotalPrice"]
        nuk = df[df["Country"]!="United Kingdom"]["TotalPrice"]
        s1,p1 = stats.ttest_ind(uk, nuk, equal_var=False)
        r1 = big_stats(s1, p1)
        vdf = df.copy()
        vdf["Region"] = np.where(vdf["Country"]=="United Kingdom","UK","Non-UK")
        vdf2 = vdf[vdf["TotalPrice"] <= vdf["TotalPrice"].quantile(0.995)]
        fig = px.violin(vdf2, x="Region", y="TotalPrice", color="Region", box=True,
                        color_discrete_map={"UK":"#2dd4bf","Non-UK":"#f97316"},
                        labels={"TotalPrice":"Total Price (£)"})
        fig.update_layout(**lay(height=320, showlegend=True))
        st.plotly_chart(fig, use_container_width=True)
        insight("Reject H₀: UK and Non-UK customers spend significantly differently. Tailor pricing & promotions per region." if r1
                else "Fail to Reject H₀: No significant spending difference between regions.", r1)

    # H2
    with st.expander("📌 H2 — Quantity ↔ UnitPrice Correlation (Pearson)"):
        r2,p2 = stats.pearsonr(df["Quantity"], df["UnitPrice"])
        rej2  = big_stats(r2, p2, "Pearson r")
        cap1  = df["UnitPrice"].quantile(0.97)
        cap2  = df["Quantity"].quantile(0.97)
        ds    = df[(df["UnitPrice"]<=cap1)&(df["Quantity"]<=cap2)].sample(min(3000,len(df)),random_state=42)
        fig   = px.scatter(ds, x="Quantity", y="UnitPrice", opacity=0.35,
                           trendline="ols",
                           color_discrete_sequence=["#2dd4bf"],
                           trendline_color_override="#f97316",
                           labels={"UnitPrice":"Unit Price (£)"})
        fig.update_layout(**lay(height=290))
        st.plotly_chart(fig, use_container_width=True)
        insight("Reject H₀: Quantity and UnitPrice are significantly correlated." if rej2
                else "Fail to Reject H₀: No significant linear correlation between Quantity and UnitPrice.", rej2)

    # H3
    with st.expander("📌 H3 — Weekday Spending Variation (One-Way ANOVA)"):
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        grps = [df[df["Weekday"]==d]["TotalPrice"].dropna().values for d in dow_order]
        fs,p3 = stats.f_oneway(*grps)
        r3    = big_stats(fs, p3, "F-Statistic")
        da    = df.groupby("Weekday")["TotalPrice"].mean().reindex(dow_order).reset_index()
        da.columns = ["Day","AvgSpend"]
        fig = px.bar(da, x="Day", y="AvgSpend",
                     color="AvgSpend", color_continuous_scale=TEAL_CS,
                     labels={"AvgSpend":"Avg Spend (£)"})
        fig.update_layout(**lay(height=260, coloraxis_showscale=False))
        fig.update_traces(marker_cornerradius=4)
        st.plotly_chart(fig, use_container_width=True)
        insight("Reject H₀: Spending varies significantly across weekdays. Schedule campaigns on high-spend days." if r3
                else "Fail to Reject H₀: No significant spending variation across weekdays.", r3)

    # H4
    with st.expander("📌 H4 — High-Value vs At-Risk Total Spend (Welch's t-test)"):
        hv = rfm_s[rfm_s["Segment"]=="High-Value"]["Monetary"]
        ar = rfm_s[rfm_s["Segment"]=="At-Risk"]["Monetary"]
        s4,p4 = stats.ttest_ind(hv, ar, equal_var=False)
        r4    = big_stats(s4, p4)
        seg2  = rfm_s[rfm_s["Segment"].isin(["High-Value","At-Risk"])]
        fig   = px.box(seg2, x="Segment", y="Monetary", color="Segment",
                       color_discrete_map=SEG_COLORS, points="outliers",
                       labels={"Monetary":"Total Spend (£)"})
        fig.update_layout(**lay(height=270, showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
        insight("Reject H₀: High-Value customers spend significantly more than At-Risk customers." if r4
                else "Fail to Reject H₀: No significant difference between High-Value and At-Risk spend.", r4)