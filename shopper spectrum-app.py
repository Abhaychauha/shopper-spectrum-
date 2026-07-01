"""
Shopper Spectrum — Full Streamlit App
Dark slate theme · Easy-to-read charts · Matches all 4 screenshot pages
"""
import io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒",
                   layout="wide", initial_sidebar_state="expanded")

# ── Theme — dark slate with teal/amber accents ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0f1117 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2d3d !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stFileUploader"] {
    border: 2px dashed #6366f1 !important;
    border-radius: 10px !important;
    background: #1a2035 !important;
}
.stButton > button {
    background: linear-gradient(135deg,#6366f1,#06b6d4) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
}
.stButton > button:hover { opacity:.85 !important; }
[data-testid="stMetric"] {
    background:#1a2035; border:1px solid #2d3a55;
    border-radius:14px; padding:16px 20px !important;
}
[data-testid="stMetricLabel"] { color:#94a3b8 !important; font-size:11px !important; letter-spacing:.8px !important; text-transform:uppercase; }
[data-testid="stMetricValue"] { color:#e2e8f0 !important; font-size:28px !important; font-weight:800 !important; }
[data-testid="stMetricDelta"] { font-size:11px !important; }
.stTabs [data-baseweb="tab-list"] {
    background:#161b27; border-radius:10px; gap:4px; padding:4px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] { border-radius:8px; color:#94a3b8 !important; font-weight:500; padding:8px 18px; }
.stTabs [aria-selected="true"] { background:#1e3a5f !important; color:#38bdf8 !important; border-bottom:2px solid #f87171 !important; }
.stSelectbox > div, .stNumberInput > div { background:#1a2035 !important; }
[data-testid="stExpander"] { background:#1a2035 !important; border:1px solid #2d3a55 !important; border-radius:12px !important; margin-bottom:10px; }
[data-testid="stExpander"] summary { color:#e2e8f0 !important; font-weight:600; }
div[data-testid="stHorizontalBlock"] { gap:14px; }
hr { border-color: #1e2d3d !important; }
.stAlert { border-radius:10px !important; }
.stDataFrame { border-radius:10px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Plotly base theme ──────────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
    font=dict(color="#cbd5e1", family="Inter"),
    xaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d", showgrid=True),
    yaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d", showgrid=True),
    margin=dict(l=40,r=20,t=40,b=40),
)
COLORS = ["#6366f1","#06b6d4","#10b981","#f59e0b","#f87171","#a78bfa","#34d399","#60a5fa"]
SEG_COLORS = {"High-Value":"#10b981","Regular":"#38bdf8","Occasional":"#f59e0b","At-Risk":"#f87171"}

# ── Cached helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_clean(raw: bytes) -> pd.DataFrame:
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
    df["Weekday"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    return df

@st.cache_data(show_spinner=False)
def build_rfm(_df):
    snap = _df["InvoiceDate"].max() + pd.Timedelta(days=1)
    return _df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snap-x.max()).days),
        Frequency=("InvoiceNo","nunique"),
        Monetary=("TotalPrice","sum"),
    ).reset_index()

@st.cache_data(show_spinner=False)
def run_clustering(_rfm, k=4):
    sc = StandardScaler()
    X = sc.fit_transform(_rfm[["Recency","Frequency","Monetary"]])
    inertias, silhouettes = [], []
    for ki in range(2,11):
        km = KMeans(n_clusters=ki,random_state=42,n_init=10)
        lb = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X,lb))
    best_k = int(np.argmax(silhouettes)+2)
    km_final = KMeans(n_clusters=k,random_state=42,n_init=10)
    rfm2 = _rfm.copy()
    rfm2["Cluster"] = km_final.fit_predict(X)
    sil = silhouette_score(X, rfm2["Cluster"])
    # Auto-label
    prof = rfm2.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
    rem = list(prof.index)
    lmap={}
    hv=prof.loc[rem,"Frequency"].idxmax(); lmap[hv]="High-Value"; rem.remove(hv)
    ar=prof.loc[rem,"Recency"].idxmax(); lmap[ar]="At-Risk"; rem.remove(ar)
    rg=prof.loc[rem,"Monetary"].idxmax(); lmap[rg]="Regular"; rem.remove(rg)
    for c in rem: lmap[c]="Occasional"
    rfm2["Segment"]=rfm2["Cluster"].map(lmap)
    return rfm2, km_final, sc, lmap, inertias, silhouettes, best_k, round(sil,4)

@st.cache_data(show_spinner=False)
def build_collab(_df):
    b = _df.groupby(["CustomerID","Description"])["Quantity"].sum().unstack(fill_value=0)
    pop=(b>0).sum(0)
    keep=pop[pop>=3].index
    if len(keep)>800: keep=pop[keep].sort_values(ascending=False).head(800).index
    b=b[keep]
    sim=cosine_similarity(b.T)
    return pd.DataFrame(sim,index=b.columns,columns=b.columns)

@st.cache_data(show_spinner=False)
def build_tfidf(_df):
    prods=_df["Description"].dropna().unique()
    tv=TfidfVectorizer(stop_words="english")
    mat=tv.fit_transform(prods)
    sim=cosine_similarity(mat)
    return pd.DataFrame(sim,index=prods,columns=prods)

# ── Reusable section header ────────────────────────────────────────────────────
def section(title):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:18px 0 12px 0;">
      <div style="width:4px;height:22px;background:linear-gradient(180deg,#6366f1,#06b6d4);border-radius:3px;"></div>
      <span style="font-size:15px;font-weight:600;color:#94a3b8;">{title}</span>
    </div>""", unsafe_allow_html=True)

def hero():
    st.markdown("""
    <div style="background:linear-gradient(120deg,#3730a3 0%,#6366f1 40%,#db2777 80%,#f97316 100%);
         border-radius:16px;padding:40px 44px;margin-bottom:24px;text-align:center;">
      <div style="font-size:34px;font-weight:800;color:#fff;letter-spacing:-0.5px;">🛒 Shopper Spectrum</div>
      <div style="color:rgba(255,255,255,.75);font-size:14px;margin-top:6px;letter-spacing:.3px;">
        Customer Segmentation &amp; Product Recommendation · P Suman Sangeet
      </div>
    </div>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 22px 0;'>
      <div style='font-size:38px;'>🛒</div>
      <div style='font-size:17px;font-weight:700;color:#e2e8f0;'>Shopper Spectrum</div>
      <div style='font-size:10px;color:#64748b;margin-top:3px;letter-spacing:.6px;'>INNOVEXIS · Data Science and Gen AI</div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("**📂 Upload online_retail.csv**")
    uploaded = st.file_uploader(" ", type=["csv"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Navigate**")
    page = st.radio("", ["🏠 Overview","📊 EDA & Insights","🎯 Customer Segments",
                         "🔮 Recommendations","🧪 Hypothesis Tests"],
                    label_visibility="collapsed")

# ── Load data ──────────────────────────────────────────────────────────────────
df = None
if uploaded:
    with st.spinner("Cleaning data…"):
        df = load_clean(uploaded.getvalue())

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    hero()
    if df is None:
        st.markdown("""
        <div style="background:#1a2035;border:1px solid #3b4fcf;border-radius:12px;
             padding:22px 26px;color:#cbd5e1;font-size:14px;">
          👆 <b>Upload your <code style='background:#111827;padding:2px 7px;border-radius:4px;
          color:#38bdf8'>online_retail.csv</code> in the sidebar to begin.</b><br>
          <span style='color:#64748b;font-size:12px;margin-top:6px;display:block;'>
          Dataset source: UCI ML Repository – Online Retail (or Kaggle mirror).</span>
        </div>""", unsafe_allow_html=True)
        st.stop()

    with st.spinner("Computing cluster quality…"):
        rfm0 = build_rfm(df)
        _, _, _, _, _, sils, best_k, sil_score = run_clustering(rfm0)

    # 6 KPI cards
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    def kpi(col, label, value, sub):
        col.markdown(f"""
        <div style="background:#1a2035;border:1px solid #2d3a55;border-radius:14px;
             padding:20px 16px;text-align:center;">
          <div style="font-size:10px;font-weight:600;color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">{label}</div>
          <div style="font-size:30px;font-weight:800;color:#e2e8f0;line-height:1;">{value}</div>
          <div style="font-size:11px;color:#38bdf8;margin-top:6px;">+{sub}</div>
        </div>""", unsafe_allow_html=True)

    kpi(c1,"Customers",f"{df['CustomerID'].nunique():,}","unique")
    kpi(c2,"Products",f"{df['Description'].nunique():,}","SKUs")
    kpi(c3,"Invoices",f"{df['InvoiceNo'].nunique():,}","orders")
    rev = df["TotalPrice"].sum()
    kpi(c4,"Revenue",f"£{rev/1e6:.2f}M","total")
    kpi(c5,"Countries",f"{df['Country'].nunique()}","markets")
    kpi(c6,"Silhouette",str(sil_score),"cluster quality")

    st.markdown("<br>", unsafe_allow_html=True)
    section("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True, height=290)

    col1, col2 = st.columns(2)
    with col1:
        section("Missing Values (after clean)")
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Column","Missing"]
        fig = px.bar(mv, x="Column", y="Missing", color="Missing",
                     color_continuous_scale=["#1e3a5f","#6366f1"],
                     labels={"Missing":"Count"})
        fig.update_layout(**PT, height=240, coloraxis_showscale=False,
                          xaxis_tickangle=-30, title_font_size=13)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        section("Data Type Distribution")
        dtypes = df.dtypes.astype(str).value_counts().reset_index()
        dtypes.columns = ["Type","Count"]
        fig = px.pie(dtypes, names="Type", values="Count",
                     color_discrete_sequence=COLORS, hole=0.5)
        fig.update_layout(**PT, height=240, showlegend=True)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2  EDA & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    hero()
    if df is None:
        st.warning("Upload `online_retail.csv` in the sidebar to continue."); st.stop()

    tab1,tab2,tab3,tab4,tab5 = st.tabs(
        ["🌍 Geography","📦 Products","📈 Time Trends","💰 Spend Patterns","📊 RFM Distributions"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            section("Top 10 Countries — Transaction Volume")
            tc = df["Country"].value_counts().head(10).reset_index()
            tc.columns = ["Country","Transactions"]
            fig = px.bar(tc, x="Transactions", y="Country", orientation="h",
                         color="Transactions", color_continuous_scale=["#1e3a5f","#6366f1"],
                         labels={"Transactions":"Transactions","Country":"Country"})
            fig.update_layout(**PT, height=380, yaxis=dict(autorange="reversed",gridcolor="#1e2d3d"),
                              coloraxis_showscale=False)
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            section("Avg Spend per Customer by Country")
            avg_spend = df.groupby("Country").agg(
                AvgSpend=("TotalPrice","sum"), Customers=("CustomerID","nunique")
            )
            avg_spend["AvgSpend"] = (avg_spend["AvgSpend"]/avg_spend["Customers"]).round(0)
            avg_spend = avg_spend.sort_values("AvgSpend",ascending=False).head(10).reset_index()
            fig = px.bar(avg_spend, x="AvgSpend", y="Country", orientation="h",
                         color="AvgSpend", color_continuous_scale=["#0d3d3d","#06b6d4"],
                         labels={"AvgSpend":"AvgSpend"})
            fig.update_layout(**PT, height=380, yaxis=dict(autorange="reversed",gridcolor="#1e2d3d"),
                              coloraxis_showscale=False)
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1,col2 = st.columns(2)
        with col1:
            section("Top 15 Products by Revenue")
            tp = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(15).reset_index()
            fig = px.bar(tp, x="TotalPrice", y="Description", orientation="h",
                         color="TotalPrice", color_continuous_scale=["#1e3a5f","#6366f1"],
                         labels={"TotalPrice":"Revenue (£)","Description":""})
            fig.update_layout(**PT, height=420, yaxis=dict(autorange="reversed",gridcolor="#1e2d3d"),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            section("Top 15 Products by Units Sold")
            tp2 = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(15).reset_index()
            fig = px.bar(tp2, x="Quantity", y="Description", orientation="h",
                         color="Quantity", color_continuous_scale=["#0d3d3d","#10b981"],
                         labels={"Quantity":"Units Sold","Description":""})
            fig.update_layout(**PT, height=420, yaxis=dict(autorange="reversed",gridcolor="#1e2d3d"),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("Monthly Revenue Trend")
        mr = df.groupby("Month")["TotalPrice"].sum().reset_index()
        fig = px.line(mr, x="Month", y="TotalPrice", markers=True,
                      labels={"TotalPrice":"Revenue (£)","Month":""},
                      color_discrete_sequence=["#6366f1"])
        fig.update_traces(line_width=2.5, marker_size=6,
                          fill="tozeroy", fillcolor="rgba(99,102,241,0.12)")
        fig.update_layout(**PT, height=300)
        st.plotly_chart(fig, use_container_width=True)

        col1,col2 = st.columns(2)
        with col1:
            section("Orders by Day of Week")
            dow_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow=df.groupby("Weekday")["InvoiceNo"].nunique().reindex(dow_order).reset_index()
            dow.columns=["Day","Orders"]
            fig=px.bar(dow,x="Day",y="Orders",color="Orders",
                       color_continuous_scale=["#1e3a5f","#06b6d4"])
            fig.update_layout(**PT,height=270,coloraxis_showscale=False)
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig,use_container_width=True)
        with col2:
            section("Orders by Hour of Day")
            hr=df.groupby("Hour")["InvoiceNo"].nunique().reset_index()
            hr.columns=["Hour","Orders"]
            fig=px.bar(hr,x="Hour",y="Orders",color="Orders",
                       color_continuous_scale=["#1e3a5f","#10b981"])
            fig.update_layout(**PT,height=270,coloraxis_showscale=False)
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig,use_container_width=True)

    with tab4:
        col1,col2 = st.columns(2)
        with col1:
            section("Unit Price Distribution")
            cap=df[df["UnitPrice"]<df["UnitPrice"].quantile(0.97)]
            fig=px.histogram(cap,x="UnitPrice",nbins=50,
                             color_discrete_sequence=["#6366f1"],
                             labels={"UnitPrice":"Unit Price (£)"})
            fig.update_layout(**PT,height=290,bargap=0.05)
            st.plotly_chart(fig,use_container_width=True)
        with col2:
            section("Revenue Share by Country (Top 8)")
            rs=df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
            top8=rs.head(7)
            top8["Other"]=rs.iloc[7:].sum()
            top8=top8.reset_index()
            top8.columns=["Country","Revenue"]
            fig=px.pie(top8,names="Country",values="Revenue",hole=0.45,
                       color_discrete_sequence=COLORS)
            fig.update_layout(**PT,height=290)
            fig.update_traces(textposition="inside",textinfo="percent+label",textfont_size=11)
            st.plotly_chart(fig,use_container_width=True)

    with tab5:
        with st.spinner("Building RFM…"):
            rfm_eda=build_rfm(df)
        col1,col2,col3=st.columns(3)
        for col,metric,color in zip([col1,col2,col3],
                                     ["Recency","Frequency","Monetary"],
                                     ["#6366f1","#06b6d4","#10b981"]):
            with col:
                section(f"{metric} Distribution")
                cap_pct = 0.99 if metric!="Frequency" else 1.0
                data=rfm_eda[rfm_eda[metric]<rfm_eda[metric].quantile(cap_pct)]
                fig=px.histogram(data,x=metric,nbins=40,color_discrete_sequence=[color])
                fig.update_layout(**PT,height=240,showlegend=False,bargap=0.04)
                st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3  CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Customer Segments":
    hero()
    if df is None:
        st.warning("Upload `online_retail.csv` in the sidebar to continue."); st.stop()

    with st.spinner("Running KMeans (k=2…10)…"):
        rfm=build_rfm(df)
        rfm_seg,km,sc,lmap,inertias,silhouettes,best_k,sil_score=run_clustering(rfm)

    tab1,tab2,tab3,tab4=st.tabs(
        ["📐 Elbow & Silhouette","🗂 Cluster Profiles","📊 Visualizations","🔮 Predict Segment"])

    with tab1:
        col1,col2=st.columns(2)
        with col1:
            section("Elbow Method")
            ks=list(range(2,11))
            fig=px.line(x=ks,y=inertias,markers=True,
                        labels={"x":"Clusters (k)","y":"Inertia"},
                        color_discrete_sequence=["#6366f1"])
            fig.update_traces(line_width=2.5,marker_size=7,marker_color="#a78bfa")
            fig.update_layout(**PT,height=300)
            st.plotly_chart(fig,use_container_width=True)
        with col2:
            section("Silhouette Scores")
            fig=px.line(x=ks,y=silhouettes,markers=True,
                        labels={"x":"Clusters (k)","y":"Silhouette Score"},
                        color_discrete_sequence=["#10b981"])
            fig.update_traces(line_width=2.5,marker_size=7,marker_color="#34d399")
            fig.update_layout(**PT,height=300)
            st.plotly_chart(fig,use_container_width=True)

        st.info(f"✅  Best k by silhouette = **{best_k}**  |  Current k = **4**  |  Silhouette = **{sil_score}**")

    with tab2:
        counts=rfm_seg["Segment"].value_counts()
        c1,c2,c3,c4=st.columns(4)
        for col,seg in zip([c1,c2,c3,c4],["High-Value","Regular","Occasional","At-Risk"]):
            n=counts.get(seg,0); pct=n/len(rfm_seg)*100
            color=SEG_COLORS[seg]
            col.markdown(f"""
            <div style="background:#1a2035;border-left:4px solid {color};
                 border-radius:10px;padding:16px 14px;">
              <div style="font-size:10px;color:#64748b;letter-spacing:.8px;text-transform:uppercase;">{seg}</div>
              <div style="font-size:26px;font-weight:800;color:{color};">{n:,}</div>
              <div style="font-size:11px;color:#94a3b8;">{pct:.1f}% of customers</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        section("Segment Profile (Average RFM)")
        prof=rfm_seg.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1)
        prof["Count"]=rfm_seg["Segment"].value_counts()
        st.dataframe(prof.style.format({"Recency":"{:.1f}","Frequency":"{:.1f}",
                                         "Monetary":"£{:,.0f}","Count":"{:,.0f}"}),
                     use_container_width=True)

    with tab3:
        col1,col2=st.columns(2)
        with col1:
            section("Customer Segments — Count")
            vc=rfm_seg["Segment"].value_counts().reset_index()
            vc.columns=["Segment","Count"]
            fig=px.bar(vc,x="Segment",y="Count",color="Segment",
                       color_discrete_map=SEG_COLORS,text="Count")
            fig.update_traces(textposition="outside",marker_cornerradius=5)
            fig.update_layout(**PT,height=310,showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
        with col2:
            section("RFM Scatter — Recency vs Monetary")
            fig=px.scatter(rfm_seg,x="Recency",y="Monetary",color="Segment",
                           size="Frequency",hover_data=["CustomerID"],
                           opacity=0.65,color_discrete_map=SEG_COLORS,
                           labels={"Monetary":"Monetary (£)"})
            fig.update_layout(**PT,height=310)
            st.plotly_chart(fig,use_container_width=True)

        section("Average RFM by Segment")
        prof2=rfm_seg.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1).reset_index()
        fig=px.bar(prof2,x="Segment",y=["Recency","Frequency","Monetary"],
                   barmode="group",color_discrete_sequence=["#6366f1","#06b6d4","#10b981"],
                   labels={"value":"Value","variable":"Metric"})
        fig.update_layout(**PT,height=300)
        st.plotly_chart(fig,use_container_width=True)

    with tab4:
        st.markdown("#### 🔍 Enter RFM values to predict the customer segment")
        c1,c2,c3=st.columns(3)
        rec=c1.number_input("Recency (days since last purchase)",0,1000,30)
        freq=c2.number_input("Frequency (number of orders)",0,500,5)
        mon=c3.number_input("Monetary (total spend £)",0.0,200000.0,500.0,step=50.0)
        if st.button("Predict Segment",type="primary"):
            x=sc.transform([[rec,freq,mon]])
            cid=int(km.predict(x)[0])
            seg=lmap.get(cid,f"Cluster {cid}")
            color=SEG_COLORS.get(seg,"#6366f1")
            desc_map={
                "High-Value":"Recent buyer, frequent orders, high spend. Reward with loyalty perks.",
                "Regular":"Steady buyer with moderate spend. Good target for upselling.",
                "Occasional":"Infrequent buyer, lower spend. Engage with re-activation campaigns.",
                "At-Risk":"Long time since last purchase. Win back with targeted offers.",
            }
            st.markdown(f"""
            <div style="border-left:6px solid {color};background:#1a2035;
                 padding:22px 26px;border-radius:12px;margin-top:14px;">
              <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.8px;">Predicted Segment</div>
              <div style="font-size:34px;font-weight:800;color:{color};margin:6px 0;">{seg}</div>
              <div style="color:#94a3b8;font-size:13px;">{desc_map.get(seg,"")}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Recommendations":
    hero()
    if df is None:
        st.warning("Upload `online_retail.csv` in the sidebar to continue."); st.stop()

    tab1,tab2=st.tabs(["🤝 Collaborative Filtering","📄 Content-Based (TF-IDF)"])

    with tab1:
        st.markdown("##### Item-based collaborative filtering using cosine similarity on the customer × product purchase matrix.")
        with st.spinner("Building collaborative similarity matrix…"):
            csim=build_collab(df)
        prod=st.selectbox("Select a Product",sorted(csim.index.tolist()),key="cf")
        n=st.slider("Number of recommendations",3,10,5,key="cfn")
        if st.button("Get Recommendations",key="cfb",type="primary"):
            recs=csim.loc[prod].drop(labels=[prod],errors="ignore").sort_values(ascending=False).head(n)
            cols=st.columns(min(n,5))
            for i,(p,s) in enumerate(recs.items()):
                with cols[i%5]:
                    st.markdown(f"""
                    <div style="background:#1a2035;border:1px solid #2d3a55;border-radius:12px;
                         padding:16px 12px;min-height:110px;display:flex;flex-direction:column;
                         justify-content:space-between;">
                      <div style="font-size:12px;font-weight:600;color:#e2e8f0;">{p}</div>
                      <div style="font-size:11px;color:#6366f1;margin-top:8px;">Similarity: {s:.3f}</div>
                    </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("##### Content-based recommendations using TF-IDF vectors on product description text.")
        with st.spinner("Building TF-IDF matrix…"):
            tsim=build_tfidf(df)
        prod2=st.selectbox("Select a Product",sorted(tsim.index.tolist()),key="cb")
        n2=st.slider("Number of recommendations",3,10,5,key="cbn")
        if st.button("Get Recommendations",key="cbb",type="primary"):
            recs2=tsim.loc[prod2].drop(labels=[prod2],errors="ignore").sort_values(ascending=False).head(n2)
            cols2=st.columns(min(n2,5))
            for i,(p,s) in enumerate(recs2.items()):
                with cols2[i%5]:
                    st.markdown(f"""
                    <div style="background:#1a2035;border:1px solid #2d3a55;border-radius:12px;
                         padding:16px 12px;min-height:110px;display:flex;flex-direction:column;
                         justify-content:space-between;">
                      <div style="font-size:12px;font-weight:600;color:#e2e8f0;">{p}</div>
                      <div style="font-size:11px;color:#10b981;margin-top:8px;">Similarity: {s:.3f}</div>
                    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5  HYPOTHESIS TESTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Hypothesis Tests":
    hero()
    if df is None:
        st.warning("Upload `online_retail.csv` in the sidebar to continue."); st.stop()

    with st.spinner("Running segmentation…"):
        rfm_h=build_rfm(df)
        rfm_seg_h,_,_,_,_,_,_,_=run_clustering(rfm_h)

    def big_stats(stat, pval, label="T-Statistic"):
        reject=pval<0.05
        verdict="Reject H₀" if reject else "Fail to Reject H₀"
        vcolor="#10b981" if reject else "#f87171"
        pval_str=f"{pval:.4e}" if pval<0.001 else f"{pval:.4f}"
        c1,c2,c3=st.columns(3)
        c1.markdown(f"<div style='font-size:11px;color:#64748b;'>{label}</div><div style='font-size:36px;font-weight:800;color:#e2e8f0;'>{stat:.4f}</div>",unsafe_allow_html=True)
        c2.markdown(f"<div style='font-size:11px;color:#64748b;'>P-Value</div><div style='font-size:36px;font-weight:800;color:#e2e8f0;'>{pval_str}</div>",unsafe_allow_html=True)
        c3.markdown(f"<div style='font-size:11px;color:#64748b;'>Conclusion</div><div style='font-size:36px;font-weight:800;color:{vcolor};'>{verdict}</div>",unsafe_allow_html=True)
        return reject

    def insight(msg, reject):
        icon="✅" if reject else "❌"
        bg="#1a2a1f" if reject else "#2a1a1f"
        border="#10b981" if reject else "#f87171"
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:10px;
             padding:14px 18px;margin-top:12px;font-size:13px;color:#cbd5e1;">
          {icon} {msg}
        </div>""", unsafe_allow_html=True)

    # H1
    with st.expander("📌 H1 — UK vs Non-UK Customer Spending (Welch's t-test)", expanded=True):
        uk=df[df["Country"]=="United Kingdom"]["TotalPrice"]
        nuk=df[df["Country"]!="United Kingdom"]["TotalPrice"]
        stat1,p1=stats.ttest_ind(uk,nuk,equal_var=False)
        reject1=big_stats(stat1,p1)
        df_vio=df.copy(); df_vio["Region"]=np.where(df_vio["Country"]=="United Kingdom","UK","Non-UK")
        cap=df_vio["TotalPrice"].quantile(0.995)
        df_vio2=df_vio[df_vio["TotalPrice"]<=cap]
        fig=px.violin(df_vio2,x="Region",y="TotalPrice",color="Region",box=True,
                      color_discrete_map={"UK":"#6366f1","Non-UK":"#06b6d4"},
                      labels={"TotalPrice":"TotalPrice"})
        fig.update_layout(**PT,height=330,showlegend=True)
        st.plotly_chart(fig,use_container_width=True)
        insight("Reject H₀: UK and Non-UK customers spend significantly differently. Tailor pricing &amp; promotions per region." if reject1
                else "Fail to Reject H₀: No significant spending difference between UK and Non-UK customers.", reject1)

    # H2
    with st.expander("📌 H2 — Quantity ↔ UnitPrice Correlation (Pearson)"):
        corr2,p2=stats.pearsonr(df["Quantity"],df["UnitPrice"])
        reject2=big_stats(corr2,p2,"Pearson r")
        cap=df["UnitPrice"].quantile(0.98); cap2=df["Quantity"].quantile(0.98)
        df_s=df[(df["UnitPrice"]<=cap)&(df["Quantity"]<=cap2)].sample(min(3000,len(df)),random_state=42)
        fig=px.scatter(df_s,x="Quantity",y="UnitPrice",opacity=0.35,
                       trendline="ols",color_discrete_sequence=["#6366f1"],
                       trendline_color_override="#f59e0b",
                       labels={"UnitPrice":"Unit Price (£)"})
        fig.update_layout(**PT,height=300)
        st.plotly_chart(fig,use_container_width=True)
        insight("Reject H₀: Quantity and UnitPrice are significantly correlated." if reject2
                else "Fail to Reject H₀: No significant linear correlation between Quantity and UnitPrice.", reject2)

    # H3
    with st.expander("📌 H3 — Weekday Spending Variation (One-Way ANOVA)"):
        dow_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        groups=[df[df["Weekday"]==d]["TotalPrice"].dropna().values for d in dow_order]
        fstat,p3=stats.f_oneway(*groups)
        reject3=big_stats(fstat,p3,"F-Statistic")
        dow_avg=df.groupby("Weekday")["TotalPrice"].mean().reindex(dow_order).reset_index()
        dow_avg.columns=["Weekday","AvgSpend"]
        fig=px.bar(dow_avg,x="Weekday",y="AvgSpend",color="AvgSpend",
                   color_continuous_scale=["#1e3a5f","#10b981"],
                   labels={"AvgSpend":"Avg Spend (£)"})
        fig.update_layout(**PT,height=280,coloraxis_showscale=False)
        fig.update_traces(marker_cornerradius=4)
        st.plotly_chart(fig,use_container_width=True)
        insight("Reject H₀: Spending varies significantly across weekdays. Schedule campaigns on high-spend days." if reject3
                else "Fail to Reject H₀: No significant spending variation across weekdays.", reject3)

    # H4
    with st.expander("📌 H4 — High-Value vs At-Risk Total Spend (Welch's t-test)"):
        hv=rfm_seg_h[rfm_seg_h["Segment"]=="High-Value"]["Monetary"]
        ar=rfm_seg_h[rfm_seg_h["Segment"]=="At-Risk"]["Monetary"]
        stat4,p4=stats.ttest_ind(hv,ar,equal_var=False)
        reject4=big_stats(stat4,p4)
        fig=px.box(rfm_seg_h[rfm_seg_h["Segment"].isin(["High-Value","At-Risk"])],
                   x="Segment",y="Monetary",color="Segment",
                   color_discrete_map=SEG_COLORS,points="outliers",
                   labels={"Monetary":"Total Spend (£)"})
        fig.update_layout(**PT,height=280,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
        insight("Reject H₀: High-Value customers spend significantly more than At-Risk customers." if reject4
                else "Fail to Reject H₀: No significant spending difference between High-Value and At-Risk.", reject4)