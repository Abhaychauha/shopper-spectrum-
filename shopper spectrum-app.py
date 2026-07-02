"""Shopper Spectrum — rebuilt with all loops unrolled & unique hardcoded chart keys."""
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

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    background:#071318 !important;color:#d1fae5 !important;
    font-family:'Inter',sans-serif !important;}
[data-testid="stSidebar"]{background:#0a1f28 !important;border-right:1px solid #0e3344 !important;}
[data-testid="stSidebar"] *{color:#94d8cc !important;}
[data-testid="stFileUploader"]{border:2px dashed #14b8a6 !important;border-radius:10px !important;background:#0a1f28 !important;}
.stButton>button{background:linear-gradient(135deg,#0d9488,#06b6d4) !important;color:#fff !important;
    border:none !important;border-radius:8px !important;font-weight:600 !important;}
[data-testid="stMetric"]{background:#0d2233;border:1px solid #0e3a4a;border-radius:14px;padding:16px 18px !important;}
[data-testid="stMetricLabel"]{color:#5eead4 !important;font-size:11px !important;letter-spacing:.8px !important;text-transform:uppercase;}
[data-testid="stMetricValue"]{color:#d1fae5 !important;font-size:28px !important;font-weight:800 !important;}
.stTabs [data-baseweb="tab-list"]{background:#0a1f28;border-radius:10px;gap:4px;padding:4px;border-bottom:none !important;}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#5eead4 !important;font-weight:500;padding:8px 18px;}
.stTabs [aria-selected="true"]{background:#0e3344 !important;color:#2dd4bf !important;border-bottom:2px solid #f97316 !important;}
[data-testid="stExpander"]{background:#0d2233 !important;border:1px solid #0e3a4a !important;border-radius:12px !important;margin-bottom:10px;}
hr{border-color:#0e3344 !important;}
</style>""", unsafe_allow_html=True)

_PT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a1a24",
           font=dict(color="#94d8cc",family="Inter"), margin=dict(l=40,r=20,t=44,b=40),
           legend=dict(bgcolor="rgba(0,0,0,0)"))
_AX = dict(gridcolor="#0e3344",linecolor="#0e3344",showgrid=True)

def lay(**ov):
    d={**_PT,"xaxis":{**_AX},"yaxis":{**_AX}}
    for k,v in ov.items():
        d[k]={**_AX,**v} if k in("xaxis","yaxis") and isinstance(v,dict) else v
    return d

COLORS=["#14b8a6","#06b6d4","#f97316","#facc15","#a78bfa","#fb7185","#34d399","#60a5fa"]
TC=[[0,"#071318"],[.5,"#0d9488"],[1,"#2dd4bf"]]
OC=[[0,"#1a0a00"],[.5,"#c2410c"],[1,"#fb923c"]]
CC=[[0,"#071318"],[.5,"#0369a1"],[1,"#38bdf8"]]
SC={"High-Value":"#2dd4bf","Regular":"#38bdf8","Occasional":"#facc15","At-Risk":"#fb7185"}

@st.cache_data(show_spinner=False)
def load_clean(raw):
    df=pd.read_csv(io.BytesIO(raw))
    df["InvoiceNo"]=df["InvoiceNo"].astype(str)
    df=df.dropna(subset=["CustomerID"])
    df=df[~df["InvoiceNo"].str.startswith("C")]
    df=df[(df["Quantity"]>0)&(df["UnitPrice"]>0)]
    df=df.dropna(subset=["Description"])
    df["Description"]=df["Description"].str.strip()
    df["CustomerID"]=df["CustomerID"].astype(int)
    df["InvoiceDate"]=pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"]=df["Quantity"]*df["UnitPrice"]
    df["Month"]=df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Weekday"]=df["InvoiceDate"].dt.day_name()
    df["Hour"]=df["InvoiceDate"].dt.hour
    return df

@st.cache_data(show_spinner=False)
def build_rfm(_df):
    snap=_df["InvoiceDate"].max()+pd.Timedelta(days=1)
    return _df.groupby("CustomerID").agg(
        Recency=("InvoiceDate",lambda x:(snap-x.max()).days),
        Frequency=("InvoiceNo","nunique"),Monetary=("TotalPrice","sum")).reset_index()

@st.cache_data(show_spinner=False)
def run_clustering(_rfm,k=4):
    sc=StandardScaler(); X=sc.fit_transform(_rfm[["Recency","Frequency","Monetary"]])
    inertias,silhouettes=[],[]
    for ki in range(2,11):
        km=KMeans(n_clusters=ki,random_state=42,n_init=10); lb=km.fit_predict(X)
        inertias.append(km.inertia_); silhouettes.append(silhouette_score(X,lb))
    best_k=int(np.argmax(silhouettes)+2)
    km_f=KMeans(n_clusters=k,random_state=42,n_init=10)
    r2=_rfm.copy(); r2["Cluster"]=km_f.fit_predict(X)
    sil=round(silhouette_score(X,r2["Cluster"]),4)
    prof=r2.groupby("Cluster")[["Recency","Frequency","Monetary"]].mean()
    rem=list(prof.index); lmap={}
    hv=prof.loc[rem,"Frequency"].idxmax(); lmap[hv]="High-Value"; rem.remove(hv)
    ar=prof.loc[rem,"Recency"].idxmax(); lmap[ar]="At-Risk"; rem.remove(ar)
    rg=prof.loc[rem,"Monetary"].idxmax(); lmap[rg]="Regular"; rem.remove(rg)
    for c in rem: lmap[c]="Occasional"
    r2["Segment"]=r2["Cluster"].map(lmap)
    return r2,km_f,sc,lmap,inertias,silhouettes,best_k,sil

@st.cache_data(show_spinner=False)
def build_collab(_df):
    b=_df.groupby(["CustomerID","Description"])["Quantity"].sum().unstack(fill_value=0)
    pop=(b>0).sum(0); keep=pop[pop>=3].index
    if len(keep)>800: keep=pop[keep].sort_values(ascending=False).head(800).index
    b=b[keep]; sim=cosine_similarity(b.T)
    return pd.DataFrame(sim,index=b.columns,columns=b.columns)

@st.cache_data(show_spinner=False)
def build_tfidf(_df):
    prods=_df["Description"].dropna().unique()
    tv=TfidfVectorizer(stop_words="english"); mat=tv.fit_transform(prods)
    sim=cosine_similarity(mat)
    return pd.DataFrame(sim,index=prods,columns=prods)

def sec(t):
    st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;margin:18px 0 10px 0;">
      <div style="width:4px;height:22px;background:linear-gradient(180deg,#0d9488,#06b6d4);border-radius:3px;"></div>
      <span style="font-size:14px;font-weight:600;color:#5eead4;">{t}</span></div>""",unsafe_allow_html=True)

def hero():
    st.markdown("""<div style="background:linear-gradient(120deg,#0f4c75,#1b6ca8 30%,#0d9488 65%,#f97316);
      border-radius:16px;padding:38px 44px;margin-bottom:22px;text-align:center;">
      <div style="font-size:34px;font-weight:800;color:#fff;">🛒 Shopper Spectrum</div>
      <div style="color:rgba(255,255,255,.75);font-size:13px;margin-top:6px;">
        Customer Segmentation &amp; Product Recommendation · P Suman Sangeet</div>
    </div>""",unsafe_allow_html=True)

def kpi(col,label,value,sub,color="#2dd4bf"):
    col.markdown(f"""<div style="background:#0d2233;border:1px solid #0e3a4a;border-radius:14px;
         padding:20px 14px;text-align:center;">
      <div style="font-size:10px;font-weight:600;color:#5eead4;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">{label}</div>
      <div style="font-size:28px;font-weight:800;color:#d1fae5;line-height:1;">{value}</div>
      <div style="font-size:11px;color:{color};margin-top:6px;">+{sub}</div>
    </div>""",unsafe_allow_html=True)

def seg_card(col,seg,n,pct):
    c=SC[seg]
    col.markdown(f"""<div style="background:#0d2233;border-left:4px solid {c};border-radius:10px;padding:14px 12px;">
      <div style="font-size:10px;color:#5eead4;letter-spacing:.8px;text-transform:uppercase;">{seg}</div>
      <div style="font-size:26px;font-weight:800;color:{c};">{n:,}</div>
      <div style="font-size:11px;color:#94d8cc;">{pct:.1f}% of customers</div></div>""",unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:12px 0 20px 0;'>
      <div style='font-size:36px;'>🛒</div>
      <div style='font-size:17px;font-weight:700;color:#d1fae5;'>Shopper Spectrum</div>
      <div style='font-size:10px;color:#3d7a72;margin-top:3px;letter-spacing:.6px;'>INNOVEXIS · Data Science and Gen AI</div>
    </div>""",unsafe_allow_html=True)
    st.divider()
    st.markdown("**📂 Upload online_retail.csv**")
    uploaded=st.file_uploader(" ",type=["csv"],label_visibility="collapsed")
    st.divider()
    st.markdown("**Navigate**")
    page=st.radio("",["🏠 Overview","📊 EDA & Insights","📈 RFM Analysis",
                      "🎯 Customer Segments","🔮 Recommendations","🧪 Hypothesis Tests"],
                  label_visibility="collapsed")

df=None
if uploaded:
    with st.spinner("Cleaning data…"): df=load_clean(uploaded.getvalue())

# ══════════════════════════════════════════════════════════════════════════════
if page=="🏠 Overview":
    hero()
    if df is None:
        st.markdown("""<div style="background:#0d2233;border:1px solid #14b8a6;border-radius:12px;
             padding:22px 26px;color:#94d8cc;font-size:14px;">
          👆 <b>Upload your <code style='background:#071318;padding:2px 7px;border-radius:4px;
          color:#2dd4bf'>online_retail.csv</code> in the sidebar to begin.</b><br>
          <span style='color:#3d7a72;font-size:12px;margin-top:6px;display:block;'>
          Dataset source: UCI ML Repository – Online Retail (or Kaggle mirror).</span></div>""",unsafe_allow_html=True)
        st.stop()
    with st.spinner("Computing silhouette…"):
        rfm0=build_rfm(df); _,_,_,_,_,_,_,sil=run_clustering(rfm0)
    c1,c2,c3,c4,c5,c6=st.columns(6)
    kpi(c1,"Customers",f"{df['CustomerID'].nunique():,}","unique")
    kpi(c2,"Products",f"{df['Description'].nunique():,}","SKUs","#38bdf8")
    kpi(c3,"Invoices",f"{df['InvoiceNo'].nunique():,}","orders","#f97316")
    kpi(c4,"Revenue",f"£{df['TotalPrice'].sum()/1e6:.2f}M","total","#facc15")
    kpi(c5,"Countries",f"{df['Country'].nunique()}","markets","#a78bfa")
    kpi(c6,"Silhouette",str(sil),"cluster quality","#fb7185")
    st.markdown("<br>",unsafe_allow_html=True)
    sec("Dataset Preview"); st.dataframe(df.head(100),use_container_width=True,height=280)
    c1,c2=st.columns(2)
    with c1:
        sec("Missing Values (after clean)")
        mv=df.isnull().sum().reset_index(); mv.columns=["Column","Missing"]
        fig=px.bar(mv,x="Column",y="Missing",color="Missing",color_continuous_scale=TC)
        fig.update_layout(**lay(height=230,coloraxis_showscale=False,xaxis=dict(tickangle=-30)))
        st.plotly_chart(fig,use_container_width=True,key="ov_missing")
    with c2:
        sec("Data Type Distribution")
        dt=df.dtypes.astype(str).value_counts().reset_index(); dt.columns=["Type","Count"]
        fig=px.pie(dt,names="Type",values="Count",color_discrete_sequence=COLORS,hole=0.5)
        fig.update_layout(**lay(height=230))
        fig.update_traces(textposition="inside",textinfo="percent+label")
        st.plotly_chart(fig,use_container_width=True,key="ov_dtype")

# ══════════════════════════════════════════════════════════════════════════════
elif page=="📊 EDA & Insights":
    hero()
    if df is None: st.warning("Upload data first."); st.stop()
    t1,t2,t3,t4,t5=st.tabs(["🌍 Geography","📦 Products","📈 Time Trends","💰 Spend Patterns","📊 RFM Distributions"])

    with t1:
        c1,c2=st.columns(2)
        with c1:
            sec("Top 10 Countries — Transaction Volume")
            tc=df["Country"].value_counts().head(10).reset_index(); tc.columns=["Country","Transactions"]
            fig=px.bar(tc,x="Transactions",y="Country",orientation="h",color="Transactions",color_continuous_scale=TC)
            fig.update_layout(**lay(height=370,coloraxis_showscale=False,yaxis=dict(autorange="reversed")))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig,use_container_width=True,key="geo_txn_vol")
        with c2:
            sec("Avg Spend per Customer by Country")
            g=df.groupby("Country").agg(Rev=("TotalPrice","sum"),Cust=("CustomerID","nunique"))
            g["AvgSpend"]=(g["Rev"]/g["Cust"]).round(0)
            g=g.sort_values("AvgSpend",ascending=False).head(10).reset_index()
            fig=px.bar(g,x="AvgSpend",y="Country",orientation="h",color="AvgSpend",color_continuous_scale=CC)
            fig.update_layout(**lay(height=370,coloraxis_showscale=False,yaxis=dict(autorange="reversed")))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig,use_container_width=True,key="geo_avg_spend")

    with t2:
        c1,c2=st.columns(2)
        with c1:
            sec("Top 15 Products by Revenue")
            tp=df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(15).reset_index()
            fig=px.bar(tp,x="TotalPrice",y="Description",orientation="h",color="TotalPrice",color_continuous_scale=TC,labels={"TotalPrice":"Revenue (£)","Description":""})
            fig.update_layout(**lay(height=420,coloraxis_showscale=False,yaxis=dict(autorange="reversed")))
            st.plotly_chart(fig,use_container_width=True,key="prod_rev")
        with c2:
            sec("Top 15 Products by Units Sold")
            tp2=df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(15).reset_index()
            fig=px.bar(tp2,x="Quantity",y="Description",orientation="h",color="Quantity",color_continuous_scale=OC,labels={"Quantity":"Units Sold","Description":""})
            fig.update_layout(**lay(height=420,coloraxis_showscale=False,yaxis=dict(autorange="reversed")))
            st.plotly_chart(fig,use_container_width=True,key="prod_units")

    with t3:
        sec("Monthly Revenue Trend")
        mr=df.groupby("Month")["TotalPrice"].sum().reset_index()
        fig=px.line(mr,x="Month",y="TotalPrice",markers=True,color_discrete_sequence=["#2dd4bf"],labels={"TotalPrice":"Revenue (£)","Month":""})
        fig.update_traces(line_width=2.5,marker_size=6,fill="tozeroy",fillcolor="rgba(45,212,191,0.10)")
        fig.update_layout(**lay(height=290))
        st.plotly_chart(fig,use_container_width=True,key="time_monthly_rev")
        c1,c2=st.columns(2)
        with c1:
            sec("Orders by Day of Week")
            dow_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow=df.groupby("Weekday")["InvoiceNo"].nunique().reindex(dow_order).reset_index()
            dow.columns=["Day","Orders"]
            fig=px.bar(dow,x="Day",y="Orders",color="Orders",color_continuous_scale=TC)
            fig.update_layout(**lay(height=260,coloraxis_showscale=False))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig,use_container_width=True,key="time_dow")
        with c2:
            sec("Orders by Hour of Day")
            hr=df.groupby("Hour")["InvoiceNo"].nunique().reset_index(); hr.columns=["Hour","Orders"]
            fig=px.bar(hr,x="Hour",y="Orders",color="Orders",color_continuous_scale=CC)
            fig.update_layout(**lay(height=260,coloraxis_showscale=False))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig,use_container_width=True,key="time_hour")

    with t4:
        c1,c2=st.columns(2)
        with c1:
            sec("Unit Price Distribution")
            cap=df[df["UnitPrice"]<df["UnitPrice"].quantile(0.97)]
            fig=px.histogram(cap,x="UnitPrice",nbins=50,color_discrete_sequence=["#14b8a6"],labels={"UnitPrice":"Unit Price (£)"})
            fig.update_layout(**lay(height=280,bargap=0.05))
            st.plotly_chart(fig,use_container_width=True,key="spend_price_dist")
        with c2:
            sec("Revenue Share by Country (Top 8)")
            rs=df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
            top7=rs.head(7).to_dict(); top7["Other"]=rs.iloc[7:].sum()
            pie_df=pd.DataFrame({"Country":list(top7.keys()),"Revenue":list(top7.values())})
            fig=px.pie(pie_df,names="Country",values="Revenue",hole=0.45,color_discrete_sequence=COLORS)
            fig.update_layout(**lay(height=280))
            fig.update_traces(textposition="inside",textinfo="percent+label",textfont_size=11)
            st.plotly_chart(fig,use_container_width=True,key="spend_country_pie")
        sec("Monthly Quantity Sold")
        mq=df.groupby("Month")["Quantity"].sum().reset_index()
        fig=px.bar(mq,x="Month",y="Quantity",color="Quantity",color_continuous_scale=OC,labels={"Quantity":"Units Sold"})
        fig.update_layout(**lay(height=260,coloraxis_showscale=False))
        st.plotly_chart(fig,use_container_width=True,key="spend_monthly_qty")

    with t5:
        with st.spinner("Building RFM…"): rfm_e=build_rfm(df)
        # ── Unrolled — Recency ──
        c1,c2,c3=st.columns(3)
        with c1:
            sec("Recency Distribution")
            d=rfm_e[rfm_e["Recency"]<rfm_e["Recency"].quantile(0.99)]
            fig=px.histogram(d,x="Recency",nbins=40,color_discrete_sequence=["#2dd4bf"])
            fig.update_layout(**lay(height=230,bargap=0.04))
            st.plotly_chart(fig,use_container_width=True,key="eda_dist_recency")
        with c2:
            sec("Frequency Distribution")
            d=rfm_e.copy()
            fig=px.histogram(d,x="Frequency",nbins=40,color_discrete_sequence=["#38bdf8"])
            fig.update_layout(**lay(height=230,bargap=0.04))
            st.plotly_chart(fig,use_container_width=True,key="eda_dist_frequency")
        with c3:
            sec("Monetary Distribution")
            d=rfm_e[rfm_e["Monetary"]<rfm_e["Monetary"].quantile(0.99)]
            fig=px.histogram(d,x="Monetary",nbins=40,color_discrete_sequence=["#f97316"])
            fig.update_layout(**lay(height=230,bargap=0.04))
            st.plotly_chart(fig,use_container_width=True,key="eda_dist_monetary")

# ══════════════════════════════════════════════════════════════════════════════
elif page=="📈 RFM Analysis":
    hero()
    if df is None: st.warning("Upload data first."); st.stop()
    with st.spinner("Computing RFM…"):
        rfm_a=build_rfm(df); rfm_s,_,_,_,_,_,_,sil_a=run_clustering(rfm_a)

    c1,c2,c3,c4=st.columns(4)
    kpi(c1,"Avg Recency",f"{rfm_a['Recency'].mean():.0f}d","days since last purchase")
    kpi(c2,"Avg Frequency",f"{rfm_a['Frequency'].mean():.1f}","orders per customer","#38bdf8")
    kpi(c3,"Avg Monetary",f"£{rfm_a['Monetary'].mean():,.0f}","spend per customer","#f97316")
    kpi(c4,"Silhouette",str(sil_a),"cluster quality","#facc15")
    st.markdown("<br>",unsafe_allow_html=True)

    ta,tb,tc_tab,td,te=st.tabs(["📊 Distributions","🔢 RFM Scoring","🗺️ 2D Maps","📦 3D View","📋 Customer Table"])

    with ta:
        # Unrolled — 3 histograms
        c1,c2,c3=st.columns(3)
        with c1:
            sec("Recency Distribution")
            d=rfm_a[rfm_a["Recency"]<rfm_a["Recency"].quantile(0.99)]
            fig=px.histogram(d,x="Recency",nbins=45,color_discrete_sequence=["#2dd4bf"])
            fig.update_layout(**lay(height=230,bargap=0.04))
            st.plotly_chart(fig,use_container_width=True,key="rfm_hist_recency")
        with c2:
            sec("Frequency Distribution")
            fig=px.histogram(rfm_a,x="Frequency",nbins=45,color_discrete_sequence=["#38bdf8"])
            fig.update_layout(**lay(height=230,bargap=0.04))
            st.plotly_chart(fig,use_container_width=True,key="rfm_hist_frequency")
        with c3:
            sec("Monetary Distribution")
            d=rfm_a[rfm_a["Monetary"]<rfm_a["Monetary"].quantile(0.99)]
            fig=px.histogram(d,x="Monetary",nbins=45,color_discrete_sequence=["#f97316"])
            fig.update_layout(**lay(height=230,bargap=0.04))
            st.plotly_chart(fig,use_container_width=True,key="rfm_hist_monetary")
        c1,c2=st.columns(2)
        with c1:
            sec("Recency vs Frequency")
            d=rfm_s[rfm_s["Frequency"]<rfm_s["Frequency"].quantile(0.99)]
            fig=px.scatter(d,x="Recency",y="Frequency",color="Segment",opacity=0.5,
                           color_discrete_map=SC,labels={"Frequency":"Frequency (orders)"})
            fig.update_layout(**lay(height=290))
            st.plotly_chart(fig,use_container_width=True,key="rfm_scatter_r_f")
        with c2:
            sec("Frequency vs Monetary")
            d=rfm_s[(rfm_s["Frequency"]<rfm_s["Frequency"].quantile(0.99))&(rfm_s["Monetary"]<rfm_s["Monetary"].quantile(0.99))]
            fig=px.scatter(d,x="Frequency",y="Monetary",color="Segment",opacity=0.5,
                           color_discrete_map=SC,labels={"Monetary":"Monetary (£)"})
            fig.update_layout(**lay(height=290))
            st.plotly_chart(fig,use_container_width=True,key="rfm_scatter_f_m")

    with tb:
        st.markdown("""<div style="background:#0d2233;border:1px solid #0e3a4a;border-radius:10px;
             padding:16px 20px;margin-bottom:16px;font-size:13px;color:#94d8cc;">
          <b style="color:#2dd4bf;">How RFM scoring works:</b> Each customer receives a score of
          <b>1–5</b> for Recency (R), Frequency (F), and Monetary (M), combined into an
          <b>RFM Score (3–15)</b>. Higher = better customer. Scores use quintile-based binning.
        </div>""",unsafe_allow_html=True)
        rs=rfm_a.copy()
        rs["R_Score"]=pd.qcut(rs["Recency"],5,labels=[5,4,3,2,1]).astype(int)
        rs["F_Score"]=pd.qcut(rs["Frequency"].rank(method="first"),5,labels=[1,2,3,4,5]).astype(int)
        rs["M_Score"]=pd.qcut(rs["Monetary"].rank(method="first"),5,labels=[1,2,3,4,5]).astype(int)
        rs["RFM_Score"]=rs["R_Score"]+rs["F_Score"]+rs["M_Score"]
        c1,c2=st.columns(2)
        with c1:
            sec("RFM Score Distribution")
            sd=rs["RFM_Score"].value_counts().sort_index().reset_index(); sd.columns=["RFM Score","Customers"]
            fig=px.bar(sd,x="RFM Score",y="Customers",color="RFM Score",color_continuous_scale=TC,text="Customers")
            fig.update_traces(textposition="outside",marker_cornerradius=4)
            fig.update_layout(**lay(height=280,coloraxis_showscale=False))
            st.plotly_chart(fig,use_container_width=True,key="rfm_score_dist")
        with c2:
            sec("Avg Monetary by RFM Score")
            am=rs.groupby("RFM_Score")["Monetary"].mean().reset_index(); am.columns=["RFM Score","Avg Monetary"]
            fig=px.bar(am,x="RFM Score",y="Avg Monetary",color="Avg Monetary",color_continuous_scale=OC,labels={"Avg Monetary":"Avg Spend (£)"})
            fig.update_layout(**lay(height=280,coloraxis_showscale=False))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig,use_container_width=True,key="rfm_avg_monetary")
        # Unrolled — R / F / M score bars
        c1,c2,c3=st.columns(3)
        with c1:
            sec("Recency Score (R)")
            vc=rs["R_Score"].value_counts().sort_index().reset_index(); vc.columns=["Score","Count"]
            fig=px.bar(vc,x="Score",y="Count",color="Count",color_continuous_scale=TC)
            fig.update_layout(**lay(height=220,coloraxis_showscale=False))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig,use_container_width=True,key="rfm_r_score_bar")
        with c2:
            sec("Frequency Score (F)")
            vc=rs["F_Score"].value_counts().sort_index().reset_index(); vc.columns=["Score","Count"]
            fig=px.bar(vc,x="Score",y="Count",color="Count",color_continuous_scale=TC)
            fig.update_layout(**lay(height=220,coloraxis_showscale=False))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig,use_container_width=True,key="rfm_f_score_bar")
        with c3:
            sec("Monetary Score (M)")
            vc=rs["M_Score"].value_counts().sort_index().reset_index(); vc.columns=["Score","Count"]
            fig=px.bar(vc,x="Score",y="Count",color="Count",color_continuous_scale=TC)
            fig.update_layout(**lay(height=220,coloraxis_showscale=False))
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(fig,use_container_width=True,key="rfm_m_score_bar")

    with tc_tab:
        c1,c2=st.columns(2)
        with c1:
            sec("Recency vs Monetary — Segment Map")
            d=rfm_s[rfm_s["Monetary"]<rfm_s["Monetary"].quantile(0.98)]
            fig=px.scatter(d,x="Recency",y="Monetary",color="Segment",size="Frequency",
                           opacity=0.65,color_discrete_map=SC,hover_data=["CustomerID","Frequency"],labels={"Monetary":"Monetary (£)"})
            fig.update_layout(**lay(height=340))
            st.plotly_chart(fig,use_container_width=True,key="map2d_r_m")
        with c2:
            sec("Recency vs Frequency — Segment Map")
            d=rfm_s[rfm_s["Frequency"]<rfm_s["Frequency"].quantile(0.98)]
            fig=px.scatter(d,x="Recency",y="Frequency",color="Segment",size="Monetary",
                           opacity=0.65,color_discrete_map=SC,hover_data=["CustomerID","Monetary"],labels={"Frequency":"Frequency (orders)"})
            fig.update_layout(**lay(height=340))
            st.plotly_chart(fig,use_container_width=True,key="map2d_r_f")
        sec("Heatmap — Avg Monetary by R-Score × F-Score")
        rs2=rfm_a.copy()
        rs2["R_Score"]=pd.qcut(rs2["Recency"],5,labels=[5,4,3,2,1]).astype(int)
        rs2["F_Score"]=pd.qcut(rs2["Frequency"].rank(method="first"),5,labels=[1,2,3,4,5]).astype(int)
        pivot=rs2.pivot_table(values="Monetary",index="R_Score",columns="F_Score",aggfunc="mean").round(0)
        fig=go.Figure(go.Heatmap(z=pivot.values,x=[f"F{c}" for c in pivot.columns],
            y=[f"R{r}" for r in pivot.index],colorscale="Teal",
            text=pivot.values.astype(int),texttemplate="£%{text}",textfont_size=11))
        fig.update_layout(**lay(height=300,xaxis=dict(title="Frequency Score"),yaxis=dict(title="Recency Score")))
        st.plotly_chart(fig,use_container_width=True,key="map2d_heatmap")

    with td:
        sec("3D RFM Scatter — Recency · Frequency · Monetary")
        d=rfm_s[rfm_s["Monetary"]<rfm_s["Monetary"].quantile(0.98)].copy()
        fig=px.scatter_3d(d,x="Recency",y="Frequency",z="Monetary",color="Segment",
                          opacity=0.65,color_discrete_map=SC,hover_data=["CustomerID"],
                          labels={"Monetary":"Monetary (£)","Frequency":"Frequency (orders)"})
        fig.update_traces(marker_size=3)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(bgcolor="#0a1a24",
                xaxis=dict(backgroundcolor="#0a1a24",gridcolor="#0e3344",showbackground=True,title="Recency"),
                yaxis=dict(backgroundcolor="#0a1a24",gridcolor="#0e3344",showbackground=True,title="Frequency"),
                zaxis=dict(backgroundcolor="#0a1a24",gridcolor="#0e3344",showbackground=True,title="Monetary (£)")),
            font=dict(color="#94d8cc"),height=520,margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig,use_container_width=True,key="rfm_3d")
        st.caption("Drag to rotate · Scroll to zoom · Click legend to toggle segments.")

    with te:
        rd=rfm_s[["CustomerID","Recency","Frequency","Monetary","Segment"]].sort_values("Monetary",ascending=False).reset_index(drop=True)
        c1,c2,c3=st.columns(3)
        sf=c1.selectbox("Filter by Segment",["All"]+list(SC.keys()))
        rm=c2.slider("Max Recency (days)",1,int(rd["Recency"].max()),int(rd["Recency"].max()))
        ms=c3.number_input("Min Monetary (£)",0.0,float(rd["Monetary"].max()),0.0,step=100.0)
        fl=rd.copy()
        if sf!="All": fl=fl[fl["Segment"]==sf]
        fl=fl[(fl["Recency"]<=rm)&(fl["Monetary"]>=ms)]
        st.caption(f"Showing **{len(fl):,}** customers")
        st.dataframe(fl.style.format({"Monetary":"£{:,.2f}","Recency":"{:.0f}d","Frequency":"{:.0f}"}),
                     use_container_width=True,height=400)
        st.download_button("⬇ Download filtered CSV",fl.to_csv(index=False).encode(),"rfm_customers.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════════════
elif page=="🎯 Customer Segments":
    hero()
    if df is None: st.warning("Upload data first."); st.stop()
    with st.spinner("Running KMeans…"):
        rfm=build_rfm(df); rfm_seg,km,sc_m,lmap,inertias,silhouettes,best_k,sil=run_clustering(rfm)
    ta,tb,tc_tab,td=st.tabs(["📐 Elbow & Silhouette","🗂 Cluster Profiles","📊 Visualizations","🔮 Predict Segment"])

    with ta:
        c1,c2=st.columns(2); ks=list(range(2,11))
        with c1:
            sec("Elbow Method")
            fig=px.line(x=ks,y=inertias,markers=True,labels={"x":"Clusters (k)","y":"Inertia"},color_discrete_sequence=["#2dd4bf"])
            fig.update_traces(line_width=2.5,marker_size=7,marker_color="#5eead4")
            fig.update_layout(**lay(height=300))
            st.plotly_chart(fig,use_container_width=True,key="seg_elbow")
        with c2:
            sec("Silhouette Scores")
            fig=px.line(x=ks,y=silhouettes,markers=True,labels={"x":"Clusters (k)","y":"Silhouette Score"},color_discrete_sequence=["#f97316"])
            fig.update_traces(line_width=2.5,marker_size=7,marker_color="#fb923c")
            fig.update_layout(**lay(height=300))
            st.plotly_chart(fig,use_container_width=True,key="seg_silhouette")
        st.info(f"✅  Best k by silhouette = **{best_k}**  |  Current k = **4**  |  Silhouette = **{sil}**")

    with tb:
        cnts=rfm_seg["Segment"].value_counts()
        c1,c2,c3,c4=st.columns(4)
        for col,seg in zip([c1,c2,c3,c4],["High-Value","Regular","Occasional","At-Risk"]):
            seg_card(col,seg,cnts.get(seg,0),cnts.get(seg,0)/len(rfm_seg)*100)
        st.markdown("<br>",unsafe_allow_html=True)
        sec("Average RFM per Segment")
        prof=rfm_seg.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1)
        prof["Customers"]=rfm_seg["Segment"].value_counts()
        st.dataframe(prof.style.format({"Recency":"{:.1f}","Frequency":"{:.1f}","Monetary":"£{:,.0f}","Customers":"{:,.0f}"}),use_container_width=True)

    with tc_tab:
        c1,c2=st.columns(2)
        with c1:
            sec("Segment Customer Count")
            vc=rfm_seg["Segment"].value_counts().reset_index(); vc.columns=["Segment","Count"]
            fig=px.bar(vc,x="Segment",y="Count",color="Segment",color_discrete_map=SC,text="Count")
            fig.update_traces(textposition="outside",marker_cornerradius=5)
            fig.update_layout(**lay(height=310,showlegend=False))
            st.plotly_chart(fig,use_container_width=True,key="seg_count_bar")
        with c2:
            sec("Recency vs Monetary (bubble = Frequency)")
            fig=px.scatter(rfm_seg,x="Recency",y="Monetary",color="Segment",size="Frequency",
                           opacity=0.65,color_discrete_map=SC,hover_data=["CustomerID"],labels={"Monetary":"Monetary (£)"})
            fig.update_layout(**lay(height=310))
            st.plotly_chart(fig,use_container_width=True,key="seg_scatter")
        sec("Average RFM per Segment — Grouped Bar")
        pr=rfm_seg.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1).reset_index()
        fig=px.bar(pr,x="Segment",y=["Recency","Frequency","Monetary"],barmode="group",
                   color_discrete_sequence=["#2dd4bf","#38bdf8","#f97316"])
        fig.update_layout(**lay(height=290))
        st.plotly_chart(fig,use_container_width=True,key="seg_grouped_bar")

    with td:
        st.markdown("#### Enter RFM values to predict the customer's segment")
        c1,c2,c3=st.columns(3)
        rec=c1.number_input("Recency (days)",0,1000,30)
        freq=c2.number_input("Frequency (orders)",0,500,5)
        mon=c3.number_input("Monetary (£)",0.0,200000.0,500.0,step=50.0)
        if st.button("Predict Segment",type="primary"):
            x=sc_m.transform([[rec,freq,mon]]); cid=int(km.predict(x)[0])
            seg=lmap.get(cid,f"Cluster {cid}"); color=SC.get(seg,"#2dd4bf")
            descs={"High-Value":"Recent, frequent, big spender. Reward with loyalty perks.",
                   "Regular":"Steady buyer with moderate spend. Good for upselling.",
                   "Occasional":"Infrequent buyer. Engage with re-activation campaigns.",
                   "At-Risk":"Long inactive. Win back with targeted offers."}
            st.markdown(f"""<div style="border-left:6px solid {color};background:#0d2233;
                 padding:22px 26px;border-radius:12px;margin-top:14px;">
              <div style="font-size:11px;color:#5eead4;text-transform:uppercase;letter-spacing:.8px;">Predicted Segment</div>
              <div style="font-size:32px;font-weight:800;color:{color};margin:6px 0;">{seg}</div>
              <div style="color:#94d8cc;font-size:13px;">{descs.get(seg,"")}</div></div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page=="🔮 Recommendations":
    hero()
    if df is None: st.warning("Upload data first."); st.stop()
    ta,tb=st.tabs(["🤝 Collaborative Filtering","📄 Content-Based (TF-IDF)"])

    def rec_cards(recs,accent):
        cols=st.columns(min(len(recs),5))
        for i,(p,s) in enumerate(recs.items()):
            with cols[i%5]:
                st.markdown(f"""<div style="background:#0d2233;border:1px solid #0e3a4a;border-radius:12px;
                     padding:14px 12px;min-height:110px;display:flex;flex-direction:column;justify-content:space-between;">
                  <div style="font-size:12px;font-weight:600;color:#d1fae5;">{p}</div>
                  <div style="font-size:11px;color:{accent};margin-top:8px;">Similarity: {s:.3f}</div>
                </div>""",unsafe_allow_html=True)

    with ta:
        st.caption("Item-based collaborative filtering — cosine similarity on the customer × product purchase matrix.")
        with st.spinner("Building collaborative matrix…"): csim=build_collab(df)
        prod=st.selectbox("Select a Product",sorted(csim.index.tolist()),key="cf_prod")
        n=st.slider("Recommendations",3,10,5,key="cf_n")
        if st.button("Get Recommendations",key="cf_btn",type="primary"):
            recs=csim.loc[prod].drop(labels=[prod],errors="ignore").sort_values(ascending=False).head(n)
            rec_cards(recs,"#2dd4bf")

    with tb:
        st.caption("Content-based: TF-IDF vectors on product description text + cosine similarity.")
        with st.spinner("Building TF-IDF matrix…"): tsim=build_tfidf(df)
        prod2=st.selectbox("Select a Product",sorted(tsim.index.tolist()),key="cb_prod")
        n2=st.slider("Recommendations",3,10,5,key="cb_n")
        if st.button("Get Recommendations",key="cb_btn",type="primary"):
            recs2=tsim.loc[prod2].drop(labels=[prod2],errors="ignore").sort_values(ascending=False).head(n2)
            rec_cards(recs2,"#f97316")

# ══════════════════════════════════════════════════════════════════════════════
elif page=="🧪 Hypothesis Tests":
    hero()
    if df is None: st.warning("Upload data first."); st.stop()
    with st.spinner("Running segmentation…"):
        rfm_h=build_rfm(df); rfm_s2,_,_,_,_,_,_,_=run_clustering(rfm_h)

    def big_stats(stat,pval,stat_label="T-Statistic"):
        reject=pval<0.05; vc="#2dd4bf" if reject else "#fb7185"
        verdict="Reject H₀" if reject else "Fail to Reject H₀"
        pstr=f"{pval:.4e}" if pval<0.001 else f"{pval:.4f}"
        c1,c2,c3=st.columns(3)
        for col,lab,val in[(c1,stat_label,f"{stat:.4f}"),(c2,"P-Value",pstr),(c3,"Conclusion",verdict)]:
            vc2=vc if lab=="Conclusion" else "#d1fae5"
            col.markdown(f"""<div style="padding:8px 0;">
              <div style="font-size:11px;color:#5eead4;">{lab}</div>
              <div style="font-size:32px;font-weight:800;color:{vc2};line-height:1.1;">{val}</div></div>""",unsafe_allow_html=True)
        return reject

    def insight(msg,reject):
        bg="#071f18" if reject else "#1f0708"; bdr="#2dd4bf" if reject else "#fb7185"
        icon="✅" if reject else "❌"
        st.markdown(f"""<div style="background:{bg};border:1px solid {bdr};border-radius:10px;
             padding:14px 18px;margin-top:10px;font-size:13px;color:#d1fae5;">{icon} {msg}</div>""",unsafe_allow_html=True)

    with st.expander("📌 H1 — UK vs Non-UK Customer Spending (Welch's t-test)",expanded=True):
        uk=df[df["Country"]=="United Kingdom"]["TotalPrice"]
        nuk=df[df["Country"]!="United Kingdom"]["TotalPrice"]
        s1,p1=stats.ttest_ind(uk,nuk,equal_var=False); r1=big_stats(s1,p1)
        vdf=df.copy(); vdf["Region"]=np.where(vdf["Country"]=="United Kingdom","UK","Non-UK")
        vdf2=vdf[vdf["TotalPrice"]<=vdf["TotalPrice"].quantile(0.995)]
        fig=px.violin(vdf2,x="Region",y="TotalPrice",color="Region",box=True,
                      color_discrete_map={"UK":"#2dd4bf","Non-UK":"#f97316"},labels={"TotalPrice":"Total Price (£)"})
        fig.update_layout(**lay(height=320,showlegend=True))
        st.plotly_chart(fig,use_container_width=True,key="h1_violin")
        insight("Reject H₀: UK and Non-UK customers spend significantly differently. Tailor pricing & promotions per region." if r1
                else "Fail to Reject H₀: No significant spending difference between regions.",r1)

    with st.expander("📌 H2 — Quantity ↔ UnitPrice Correlation (Pearson)"):
        r2,p2=stats.pearsonr(df["Quantity"],df["UnitPrice"]); rej2=big_stats(r2,p2,"Pearson r")
        cap1=df["UnitPrice"].quantile(0.97); cap2=df["Quantity"].quantile(0.97)
        ds=df[(df["UnitPrice"]<=cap1)&(df["Quantity"]<=cap2)].sample(min(3000,len(df)),random_state=42)
        fig=px.scatter(ds,x="Quantity",y="UnitPrice",opacity=0.35,color_discrete_sequence=["#2dd4bf"],labels={"UnitPrice":"Unit Price (£)"})
        m,b=np.polyfit(ds["Quantity"],ds["UnitPrice"],1); xr=np.linspace(ds["Quantity"].min(),ds["Quantity"].max(),200)
        fig.add_trace(go.Scatter(x=xr,y=m*xr+b,mode="lines",line=dict(color="#f97316",width=2),name="Trend"))
        fig.update_layout(**lay(height=290))
        st.plotly_chart(fig,use_container_width=True,key="h2_scatter")
        insight("Reject H₀: Quantity and UnitPrice are significantly correlated." if rej2
                else "Fail to Reject H₀: No significant linear correlation between Quantity and UnitPrice.",rej2)

    with st.expander("📌 H3 — Weekday Spending Variation (One-Way ANOVA)"):
        dow_order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        grps=[df[df["Weekday"]==d]["TotalPrice"].dropna().values for d in dow_order]
        fs,p3=stats.f_oneway(*grps); r3=big_stats(fs,p3,"F-Statistic")
        da=df.groupby("Weekday")["TotalPrice"].mean().reindex(dow_order).reset_index(); da.columns=["Day","AvgSpend"]
        fig=px.bar(da,x="Day",y="AvgSpend",color="AvgSpend",color_continuous_scale=TC,labels={"AvgSpend":"Avg Spend (£)"})
        fig.update_layout(**lay(height=260,coloraxis_showscale=False))
        fig.update_traces(marker_cornerradius=4)
        st.plotly_chart(fig,use_container_width=True,key="h3_dow_bar")
        insight("Reject H₀: Spending varies significantly across weekdays. Schedule campaigns on high-spend days." if r3
                else "Fail to Reject H₀: No significant spending variation across weekdays.",r3)

    with st.expander("📌 H4 — High-Value vs At-Risk Total Spend (Welch's t-test)"):
        hv=rfm_s2[rfm_s2["Segment"]=="High-Value"]["Monetary"]
        ar=rfm_s2[rfm_s2["Segment"]=="At-Risk"]["Monetary"]
        s4,p4=stats.ttest_ind(hv,ar,equal_var=False); r4=big_stats(s4,p4)
        seg2=rfm_s2[rfm_s2["Segment"].isin(["High-Value","At-Risk"])]
        fig=px.box(seg2,x="Segment",y="Monetary",color="Segment",color_discrete_map=SC,points="outliers",labels={"Monetary":"Total Spend (£)"})
        fig.update_layout(**lay(height=270,showlegend=False))
        st.plotly_chart(fig,use_container_width=True,key="h4_box")
        insight("Reject H₀: High-Value customers spend significantly more than At-Risk customers." if r4
                else "Fail to Reject H₀: No significant difference between High-Value and At-Risk spend.",r4)