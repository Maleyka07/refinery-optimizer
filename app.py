"""
Neft Emalı Optimallaşdırma Sistemi
=====================================
Modul : app.py | Dil: Azərbaycanca
Məqsəd : Streamlit Dashboard — bölmə 2.3 + 2.4 (tam əhatə)
Dizayn : Material "Soft Dark", WCAG AA, premium UI/UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from data_generator import generate_sensor_data, get_operating_summary
from preprocessor import run_preprocessing_pipeline, INPUT_FEATURES, TARGET_COLS
from optimizer import (
    run_nsga2, sensitivity_analysis,
    EconomicWeights, VAR_NAMES, X_LOWER, X_UPPER,
)
from chatbot import chatbot_yarat, kontekst_yarat, TTS_ELEVENLABS_VOICE

#
st.set_page_config(
    page_title="Neft Emalı Optimallaşdırma",
    page_icon=None, layout="wide",
    initial_sidebar_state="expanded",
)

#
# CSS — Premium Soft Dark | WCAG AA | Inter şrift
#
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Tokens */
:root {
    --bg: #1a1f35;
    --c1: #212845;
    --c2: #283058;
    --c3: #33406e;
    --bd: #334070;
    --bd2: #4a5888;
    --t0: #edf1fc;
    --t1: #c5cceb;
    --t2: #9aa5cc;
    --t3: #6878a8;
    --am: #d4941c;
    --am2: #f0ac30;
    --bl: #4a8fd9;
    --bl2: #66a8f0;
    --ok: #2ec98a;
    --er: #d45060;
    --r:12px; --rs:8px; --rm:6px;
    --sh:0 2px 12px rgba(0,0,0,.3);
    --sh2:0 4px 24px rgba(0,0,0,.4);
    --sh3:0 0 0 2px rgba(212,148,28,.35);
}

/* Base */
.stApp{background:var(--bg)!important;font-family:'Inter',sans-serif;color:var(--t1);}
/* Font hierarchy — tam tutarlı */
.stApp h1,.stApp h2,.stApp h3{color:var(--t0)!important;font-weight:700!important;}
.stApp h2{font-size:1.75rem!important;letter-spacing:-.025em;line-height:1.2!important;}
.stApp h3{font-size:1.15rem!important;letter-spacing:-.01em;}
.stApp p{color:var(--t1);line-height:1.65;font-size:.88rem;}
.stApp b,.stApp strong{color:var(--t0)!important;}
.stApp code{background:var(--c2);color:var(--am2);padding:.1rem .4rem;
    border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:.81rem;}
/* Sol padding azalt — centered content */
.block-container{
    padding-left:2.5rem!important;
    padding-right:2.5rem!important;
    padding-top:1.5rem!important;
    max-width:100%!important;
    width:100%!important;
}

/* Sidebar */
section[data-testid="stSidebar"]{background:var(--c1)!important;border-right:1px solid var(--bd)!important;min-width:230px!important;max-width:260px!important;width:245px!important;}
/* Sidebar collapse düyməsini gizlət — panel həmişə açıq qalsın */
[data-testid="collapsedControl"]{display:none!important;}
section[data-testid="stSidebar"][aria-expanded="false"]{
    min-width:230px!important;width:245px!important;transform:none!important;visibility:visible!important;
}
section[data-testid="stSidebar"] *{font-family:'Inter',sans-serif;}
section[data-testid="stSidebar"] p{color:var(--t1)!important;font-size:.82rem;line-height:1.5;}
section[data-testid="stSidebar"] label{color:var(--t1)!important;font-size:.83rem!important;font-weight:500!important;}
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stCaptionContainer p{color:var(--t2)!important;font-size:.75rem!important;}
[data-testid="stThumbValue"]{color:var(--am2)!important;font-family:'JetBrains Mono',monospace!important;font-size:.85rem!important;font-weight:600!important;}
[data-testid="stSlider"] [role="slider"]{background:var(--am2)!important;border-color:var(--am2)!important;box-shadow:var(--sh3)!important;}

/* Expander (sidebar) */
/* Sidebar section divider */
.sb-section{
    border:1px solid var(--bd);border-radius:var(--rs);
    background:var(--c1);margin-bottom:.6rem;padding:.6rem .8rem .7rem;
}
.sb-section-title{
    font-size:.84rem;font-weight:700;color:var(--t1);
    margin-bottom:.5rem;padding-bottom:.35rem;
    border-bottom:1px solid var(--bd);
    display:flex;align-items:center;gap:.4rem;
}

/* Metric cards — premium */
[data-testid="metric-container"]{
    background:linear-gradient(145deg,var(--c1),#1e2540)!important;
    border:1px solid var(--bd);border-radius:var(--r);
    padding:1.4rem 1.5rem 1.25rem;min-height:130px;
    box-shadow:var(--sh),inset 0 1px 0 rgba(255,255,255,.04);
    transition:border-color .2s,transform .18s,box-shadow .2s;
    display:flex;flex-direction:column;justify-content:space-between;
    position:relative;overflow:hidden;
}
[data-testid="metric-container"]::before{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,var(--am),transparent);opacity:0;
    transition:opacity .2s;
}
[data-testid="metric-container"]:hover{
    border-color:var(--am);transform:translateY(-4px);
    box-shadow:var(--sh2),var(--sh3);
}
[data-testid="metric-container"]:hover::before{opacity:1;}
[data-testid="metric-container"] label,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"]{
    color:var(--t2)!important;font-size:.72rem!important;font-weight:700!important;
    letter-spacing:.1em;text-transform:uppercase;opacity:1!important;
}
[data-testid="stMetricValue"]{
    color:var(--t0)!important;font-size:1.75rem!important;font-weight:700!important;
    font-family:'JetBrains Mono',monospace!important;letter-spacing:-.02em;
    line-height:1.1!important;
}
[data-testid="stMetricDelta"]{font-size:.77rem!important;font-weight:500!important;}

/* Section headers */
.bb{font-size:.65rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;
    color:var(--t3);padding-bottom:.42rem;border-bottom:1px solid var(--bd);
    margin:1.5rem 0 .9rem 0;display:block;}
/* Plotly title tutarlılığı — CSS-dən deyil, QL() funksiyasında idarə olunur */

/* Info boxes */
.tenlik{background:#161c30;border:1px solid var(--bd);border-left:3px solid var(--am);
    border-radius:var(--rs);padding:.9rem 1.2rem;font-family:'JetBrains Mono',monospace;
    font-size:.83rem;color:#eab84a;line-height:1.9;margin:.6rem 0;}
.izah{background:var(--c2);border:1px solid var(--bd);border-left:3px solid var(--bl);
    border-radius:var(--rs);padding:.7rem 1rem;font-size:.83rem;color:var(--t1);line-height:1.58;margin:.5rem 0;}
.izah b{color:var(--t0)!important;}
.izah ul,.izah li{margin:.1rem 0;padding-left:.2rem;font-size:.82rem;}
.xeb{background:rgba(212,148,28,.1);border:1px solid rgba(212,148,28,.3);border-radius:var(--rm);
    padding:.55rem .9rem;font-size:.79rem;color:#f0c060;margin:.4rem 0;}
.optimal-card{background:linear-gradient(135deg,rgba(212,148,28,.12),rgba(74,143,217,.08));
    border:2px solid var(--am);border-radius:var(--r);padding:1rem 1.3rem;
    box-shadow:0 0 20px rgba(212,148,28,.15);margin:.5rem 0;}

/* Section spacer */
.spacer{height:20px;}

/* Sidebar label */
.sbe{font-size:.66rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:var(--t3);margin:.8rem 0 .15rem 0;display:block;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--c1);border-radius:var(--r) var(--r) 0 0;
    border:1px solid var(--bd);border-bottom:none;padding:.25rem .25rem 0;gap:2px;}
.stTabs [data-baseweb="tab"]{background:transparent;color:var(--t2)!important;
    font-size:.83rem;font-weight:500;padding:.46rem 1rem;border:none!important;border-radius:var(--rm) var(--rm) 0 0;}
.stTabs [aria-selected="true"]{background:var(--c2)!important;color:var(--t0)!important;border-bottom:2px solid var(--am)!important;font-weight:600!important;}

/* Button — primary */
.stButton>button{background:var(--am)!important;color:#141828!important;font-weight:700!important;
    border:none!important;border-radius:var(--rm)!important;padding:.55rem 1.8rem!important;font-size:.87rem!important;letter-spacing:.02em;}
.stButton>button:hover{background:var(--am2)!important;box-shadow:var(--sh3)!important;}

/* Accordion buttons in sidebar */
section[data-testid="stSidebar"] .stButton>button{
    background:var(--c2)!important;
    color:var(--t1)!important;
    font-weight:600!important;
    font-size:.81rem!important;
    border:1px solid var(--bd)!important;
    border-radius:var(--rm)!important;
    padding:.38rem .75rem!important;
    text-align:left!important;
    letter-spacing:0!important;
    margin-bottom:3px;
    width:100%!important;
    transition:background .15s,border-color .15s!important;
}
section[data-testid="stSidebar"] .stButton>button:hover{
    background:var(--c3)!important;
    border-color:var(--am)!important;
    box-shadow:none!important;
}
/* Slider label compact */
section[data-testid="stSidebar"] [data-testid="stSlider"]{margin-bottom:.2rem!important;}
section[data-testid="stSidebar"] .stCaption{margin-top:.1rem!important;}

/* Radio nav */
div[data-testid="stRadio"] label{background:var(--c1);border:1px solid var(--bd);border-radius:var(--rm);
    padding:.38rem .85rem;font-size:.82rem;color:var(--t1)!important;cursor:pointer;transition:all .15s;}
div[data-testid="stRadio"] label:hover{border-color:var(--am);color:var(--t0)!important;}
div[data-testid="stRadio"] label span{color:var(--t1)!important;}

/* DataTable — tam dark, premium */
/* Streamlit dataframe iframe wrapper */
.stDataFrame{
    border-radius:var(--r)!important;overflow:hidden!important;
    border:1px solid var(--bd)!important;box-shadow:var(--sh)!important;
    background:#111828!important;
}
/* Streamlit 1.35+ — data editor / dataframe container */
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] iframe{
    background:#111828!important;
    border-radius:var(--r)!important;
}
/* Çərçivəsiz iframe içi — CSS-lə çatmaq çətindir,
   lakin scrollbar və wrapper konteynerini düzəldirik */
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"]{
    background:#111828!important;
}
/* Header — çox tünd, amber alt xətt */
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] thead tr th{
    background:#0e1524!important;
    color:#c5cceb!important;
    font-size:.70rem!important;font-weight:800!important;
    letter-spacing:.08em;text-transform:uppercase;
    border-bottom:2px solid #d4941c!important;
    padding:.6rem .85rem!important;
}
/* Body sətirləri */
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] tbody tr td{
    color:#9aa5cc!important;
    font-size:.80rem!important;
    font-family:'JetBrains Mono',monospace!important;
    border-color:#1e2848!important;
    padding:.4rem .85rem!important;
    background:transparent!important;
}
/* Odd rows — bir az tündlər */
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] tbody tr:nth-child(odd){
    background:#111828!important;
}
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] tbody tr:nth-child(even){
    background:#141d30!important;
}
/* Hover */
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] tbody tr:hover td{
    background:rgba(212,148,28,.07)!important;
    color:#edf1fc!important;
}

/* Checkbox */
[data-testid="stCheckbox"] label p,[data-testid="stCheckbox"] p{color:var(--t1)!important;font-size:.83rem!important;}

/* Select */
div[data-testid="stSelectbox"]>div>div{background:var(--c2)!important;border-color:var(--bd)!important;color:var(--t0)!important;}
div[data-baseweb="select"]{background:var(--c2)!important;}
[data-baseweb="tag"]{background:var(--c2)!important;border:1px solid var(--bl)!important;}
[data-baseweb="tag"] span{color:var(--t0)!important;}

/* Alert */
.stAlert{background:var(--c2)!important;border:1px solid var(--bd)!important;border-radius:var(--r)!important;color:var(--t1)!important;}

/* Progress */
[data-testid="stProgress"] > div > div{background:var(--am)!important;}

/* Caption */
.stCaptionContainer p,[data-testid="stCaptionContainer"] p{color:var(--t2)!important;font-size:.75rem!important;}

hr{border-color:var(--bd)!important;margin:.8rem 0!important;}
#MainMenu,footer,header{visibility:hidden;}

/* Chatbot panel */
.chat-fab{
    position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;
    width:52px;height:52px;border-radius:50%;
    background:linear-gradient(135deg,var(--am),var(--am2));
    display:flex;align-items:center;justify-content:center;
    cursor:pointer;box-shadow:0 4px 20px rgba(212,148,28,.4);
    font-size:1.4rem;border:none;transition:transform .2s,box-shadow .2s;
}
.chat-fab:hover{transform:scale(1.08);box-shadow:0 6px 28px rgba(212,148,28,.55);}

.chat-panel{
    background:var(--c1);border:1px solid var(--bd);border-radius:16px;
    box-shadow:0 8px 40px rgba(0,0,0,.5);overflow:hidden;
}
.chat-header{
    background:linear-gradient(135deg,#1a2038,#212845);
    border-bottom:1px solid var(--bd);padding:.75rem 1rem;
    display:flex;align-items:center;gap:.6rem;
}
.chat-header-title{font-size:.9rem;font-weight:700;color:var(--t0);}
.chat-header-sub{font-size:.72rem;color:var(--t2);}
.chat-status{width:8px;height:8px;border-radius:50%;background:#2ec98a;
    box-shadow:0 0 6px rgba(46,201,122,.6);}

.chat-msg-user{
    background:var(--am);color:#141828;border-radius:12px 12px 4px 12px;
    padding:.55rem .85rem;font-size:.83rem;font-weight:500;
    margin-left:2rem;margin-bottom:.4rem;line-height:1.5;
}
.chat-msg-ai{
    background:var(--c2);color:var(--t1);border-radius:12px 12px 12px 4px;
    padding:.55rem .85rem;font-size:.83rem;
    margin-right:2rem;margin-bottom:.4rem;line-height:1.55;border:1px solid var(--bd);
}
.chat-msg-ai b,.chat-msg-ai strong{color:var(--t0)!important;}
.chat-msg-ai code{background:var(--c3);color:var(--am2);padding:.1rem .3rem;border-radius:3px;font-size:.78rem;}

.quick-btn{
    background:var(--c2);border:1px solid var(--bd);border-radius:20px;
    padding:.3rem .75rem;font-size:.75rem;color:var(--t2);cursor:pointer;
    transition:all .15s;white-space:nowrap;
}
.quick-btn:hover{background:var(--c3);border-color:var(--am);color:var(--t0);}
</style>
""", unsafe_allow_html=True)

#
# PLOTLY TEMA
#
RENG = {
    "benzin":"#d4941c","dizel":"#4a8fd9","kerosin":"#7c8fd4",
    "kukurd":"#2ec98a","enerji":"#d45060","anomal":"#d47830",
    "normal":"#4a8fd9","pareto":"#4a8fd9","enyaxsi":"#d4941c",
    "grid":"#283058","line":"#334070",
}
SENSOR_AD={
    "furnace_temp":"Soba temp. (°C)","column_pressure":"Kolon təzyiqi (atm)",
    "flow_rate":"Axın sürəti (m³/s)","reflux_ratio":"Refluks nisbəti",
    "feed_temp":"Qidalanma temp. (°C)","h2_pressure":"H₂ təzyiqi (bar)",
    "catalyst_temp":"Katalizator temp. (°C)","crude_density":"Neft sıxlığı (q/sm³)",
    "sulfur_content":"Kükürd miqdarı (%)",
}
HEDAF_AD={
    "yield_gasoline":"Benzin verimi","yield_diesel":"Dizel verimi",
    "yield_kerosene":"Kerosin verimi","energy_gj_h":"Enerji (GS/s)",
    "sulfur_removal":"Kükürd çıx.","total_yield":"Ümumi verim",
}

def QL(baslig="",hund=None,legend=True):
    d=dict(
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#1e2540",
        font=dict(family="Inter,sans-serif",color="#9aa5cc",size=11),
        title=dict(text=baslig,font=dict(color="#c5cceb",size=12,family="Inter",weight=600),x=0,xanchor="left",pad=dict(l=0,b=6)),
        xaxis=dict(gridcolor="#283058",linecolor="#334070",zeroline=False,tickfont=dict(size=10,color="#9aa5cc")),
        yaxis=dict(gridcolor="#283058",linecolor="#334070",zeroline=False,tickfont=dict(size=10,color="#9aa5cc")),
        margin=dict(l=4,r=4,t=38 if baslig else 10,b=4),
        legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor="#334070",font=dict(size=10,color="#9aa5cc"),orientation="h",y=-0.22),
        showlegend=legend,
    )
    if hund: d["height"]=hund
    return d

def sub_stil(fig):
    for ax in dir(fig.layout):
        if ax.startswith(("xaxis","yaxis")):
            getattr(fig.layout,ax).update(gridcolor="#283058",linecolor="#334070",zeroline=False,tickfont=dict(size=9,color="#9aa5cc"))
    for ann in fig.layout.annotations:
        ann.font.color="#9aa5cc"; ann.font.size=10
    return fig

def sp(): st.markdown("<div class='spacer'></div>",unsafe_allow_html=True)
def bb(txt): st.markdown(f"<p class='bb'>{txt}</p>",unsafe_allow_html=True)

#
# CACHE
#
@st.cache_data(show_spinner=False)
def melumat_yukle(n): return generate_sensor_data(n_samples=n, seed=42)

@st.cache_resource(show_spinner=False)          # cache_resource: sklearn modelləri thread-safe olmadığı üçün
def boru_yukle(n):
    df = generate_sensor_data(n_samples=n, seed=42)
    return run_preprocessing_pipeline(df)

@st.cache_resource(show_spinner=False)          # NSGA-II nəticəsi numpy array → cache_resource daha sürətli
def optimal_yukle(a, b, g, d, pop, nesl):
    return run_nsga2(
        weights=EconomicWeights(alpha=a, beta=b, gamma=g, delta=d),
        pop_size=pop, n_gen=nesl, seed=42,
    )

#
# YAN PANEL — Expander ilə qruplaşdırılmış
#
with st.sidebar:
    st.markdown("<h3 style='color:#edf1fc;font-size:1.05rem;font-weight:800;margin:0 0 .1rem'>Neft Emalı</h3>",unsafe_allow_html=True)
    st.markdown("<p style='color:#6878a8;font-size:.74rem;margin:0'>ADNSU · SABAH · 2026</p>",unsafe_allow_html=True)
    st.markdown("---")

    # Session state accordion
    if "sb_open" not in st.session_state:
        st.session_state.sb_open = {"data": True, "weights": False, "nsga": False}

    def accordion(key, label, icon):
        is_open = st.session_state.sb_open[key]
        arrow = "" if is_open else ""
        clicked = st.button(
            f"{icon} {label} {arrow}",
            key=f"acc_{key}",
            width="stretch",
        )
        if clicked:
            st.session_state.sb_open[key] = not is_open
            st.rerun()
        return st.session_state.sb_open[key]

    # Məlumat parametrləri
    if accordion("data", "Məlumat parametrləri", ""):
        n_numune = st.slider("Sensor nümunəsi", 720, 2880, 1440, 120,
            help="1 dəq.=1 ölçüm · 1440=24 saatlıq baza")
        st.caption(f" {n_numune} nümunə = {n_numune//60} saat")
    else:
        n_numune = 1440

    st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)

    # Məqsəd funksiyası çəkiləri
    if accordion("weights", "Məqsəd funksiyası (α β γ δ)", ""):
        st.markdown("<div style='font-size:.74rem;color:#6878a8;margin-bottom:.3rem'>α,β,γ — məhsul çəkiləri · δ — enerji cəriməsi</div>",unsafe_allow_html=True)
        alfa = st.slider("α — Benzin", 0.05,0.60,0.35,0.05,help="Benzin satışının F*-a töhfəsi")
        beta = st.slider("β — Dizel", 0.05,0.60,0.30,0.05,help="Dizel satışının F*-a töhfəsi")
        qamma = st.slider("γ — Kerosin", 0.05,0.40,0.20,0.05,help="Kerosin satışının F*-a töhfəsi")
        delta = st.slider("δ — Enerji", 0.05,0.40,0.15,0.05,help="Enerji sərfinin F*-ı azaldan əmsalı")
        cem = alfa+beta+qamma
        if abs(cem-1.0)>0.05:
            st.markdown(f"<div class='xeb'> α+β+γ={cem:.2f}</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='tenlik'>F*={alfa}·Y<sub>b</sub>+{beta}·Y<sub>d</sub>+{qamma}·Y<sub>k</sub>−{delta}·E</div>",unsafe_allow_html=True)
    else:
        alfa, beta, qamma, delta = 0.35, 0.30, 0.20, 0.15

    st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)

    # NSGA-II parametrləri
    if accordion("nsga", "NSGA-II parametrləri", ""):
        st.markdown("<div style='font-size:.74rem;color:#6878a8;margin-bottom:.3rem'><b style='color:#c5cceb'>Populyasiya</b> — hər nəsildəki həll sayı<br><b style='color:#c5cceb'>Nəsil</b> — öyrənmə dövrü sayı</div>",unsafe_allow_html=True)
        pop_olcu = st.select_slider("Populyasiya",options=[40,60,80,100,120,150,200],value=100,
            help="Böyük = keyfiyyətli Pareto, uzun hesablama")
        nesl_sayi = st.select_slider("Nəsil sayı",options=[50,80,100,120,150,200,250],value=150,
            help="Çox nəsil = daha yaxşı həll")
        st.caption(f" ~{round(pop_olcu*nesl_sayi/5000,1)} san.")
    else:
        pop_olcu, nesl_sayi = 100, 150

    st.markdown("---")
    st.caption("Nuruyeva Məleykə Firuddin q.")
    st.caption("Rəhbər: dos. Ağayev Fərid H. o.")
    st.caption("050634 — Proseslərin avtomatlaşd.")


#
# NAVİQASİYA
#
sehife=st.radio("nav",
    [" İcmal"," Məlumat Analizi"," Emal Boru Kəməri",
     " Optimallaşdırma"," Nəticələr"],
    horizontal=True,label_visibility="collapsed")
st.markdown("---")

# Məlumat yüklə — ilk açılışda ML modeli öyrədilir (~30-50 san), sonra cache-dən anında gəlir
_boru_cached = st.session_state.get("_boru_n") == n_numune
if not _boru_cached:
    _spinner_msg = "İlk açılış: ML modeli öyrədilir… (~30-50 san, sonra cache-dən anında gəlir)"
else:
    _spinner_msg = "Məlumat yüklənir…"

with st.spinner(_spinner_msg):
    xam_df = melumat_yukle(n_numune)
    boru = boru_yukle(n_numune)
    xulase = get_operating_summary(xam_df)
    st.session_state["_boru_n"] = n_numune

temiz_df    = boru["clean_df"]
anomaliya_df = boru["anomaly_df"]
req_goster  = boru["regression_metrics"]
X_norm      = boru["X_norm"]
ts          = temiz_df["timestamp"]

# regression_coeffs — GB feature importance-larından yaranır (köhnə əmsal matrisi uyğunluğu)
if "regression_coeffs" in boru:
    req_emsal = boru["regression_coeffs"]
else:
    # Cache-də köhnə pipeline varsa, feature_importances-dən düzəldirik
    _fi = boru.get("feature_importances", None)
    if _fi is not None:
        _gb_cols = [c for c in _fi.columns if c.startswith("GB_")]
        req_emsal = _fi[_gb_cols].copy()
        req_emsal.columns = [c.replace("GB_", "") for c in req_emsal.columns]
        req_emsal["intercept"] = 0.0
    else:
        req_emsal = pd.DataFrame()

#
# SƏHİFƏ 1 — İCAML
#
if sehife==" İcmal":
    st.markdown("## Neft Emalı Optimallaşdırma Sistemi")
    st.markdown(
        "Xam neftin ilkin emalının çoxhədəfli optimallaşdırılması. "
        "NSGA-II genetik alqoritmi: verim ↑ maksimum, enerji ↓ minimum. "
        "Sənəd 2.3 (verilənlər emalı) + 2.4 (alqoritm + proqram).")
    sp()

    # Arxitektura diaqramı — academic research system görünüşü
    bb("Sistem arxitekturası — riyazi model + alqoritm + proqram")
    st.markdown("""
<div style="background:#161c30;border:1px solid #283058;border-radius:12px;padding:1.4rem 1.6rem;margin:.5rem 0 1rem;font-family:'Inter',sans-serif">

  <!-- Başlıq -->
  <div style="font-size:.65rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:#4a5888;margin-bottom:1rem;border-bottom:1px solid #283058;padding-bottom:.5rem">
    NSGA-II Əsaslı Çoxhədəfli Optimallaşdırma Sistemi (bölmə 2.3 + 2.4)
  </div>

  <!-- Cərgə 1: Giriş məlumatları -->
  <div style="display:flex;gap:8px;margin-bottom:8px;align-items:stretch">
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-top:2px solid #4a8fd9;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#4a8fd9;margin-bottom:.2rem"> SENSOR VERİLƏNLƏRİ</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">T (°C) · P (atm) · F (m³/s)<br>9 kanal · 1440 nümunə · 1 dəq.</div>
    </div>
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-top:2px solid #6878a8;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#a090d8;margin-bottom:.2rem"> FİZİKİ MƏHDUDIYƏTLƏR</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">Soba: 340–400°C<br>Kolon: 1.2–1.5 atm · H₂: 30–60 bar</div>
    </div>
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-top:2px solid #d4941c;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#d4941c;margin-bottom:.2rem"> İQTİSADİ ÇƏKİLƏR</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">α·Y<sub>benz</sub> + β·Y<sub>diz</sub> + γ·Y<sub>ker</sub><br>Sənəd tənlik (1): F* max</div>
    </div>
  </div>

  <!-- Ox aşağı -->
  <div style="text-align:center;color:#4a5888;font-size:1.1rem;margin:2px 0">↓</div>

  <!-- Cərgə 2: Preprocessing -->
  <div style="background:#1a2038;border:1px solid #334070;border-radius:8px;padding:.7rem 1rem;margin-bottom:8px">
    <div style="font-size:.72rem;font-weight:700;color:#c5cceb;margin-bottom:.4rem"> VERİLƏNLƏRİN EMAL BORU KƏMƏRİ — sənəd bölmə 2.3</div>
    <div style="display:flex;gap:8px">
      <div style="flex:1;background:#212845;border-radius:6px;padding:.4rem .6rem;text-align:center;font-size:.68rem;color:#9aa5cc">
        <span style="color:#d4941c">①</span> Filtrasiya<br><span style="font-size:.62rem;color:#6878a8">interpolasiya · 3σ</span>
      </div>
      <div style="color:#4a5888;align-self:center">→</div>
      <div style="flex:1;background:#212845;border-radius:6px;padding:.4rem .6rem;text-align:center;font-size:.68rem;color:#9aa5cc">
        <span style="color:#d4941c">②</span> Normallaşdırma<br><span style="font-size:.62rem;color:#6878a8">tənlik: x_norm</span>
      </div>
      <div style="color:#4a5888;align-self:center">→</div>
      <div style="flex:1;background:#212845;border-radius:6px;padding:.4rem .6rem;text-align:center;font-size:.68rem;color:#9aa5cc">
        <span style="color:#d4941c">③</span> Anomaliya<br><span style="font-size:.62rem;color:#6878a8">IQR · Isolation Forest</span>
      </div>
      <div style="color:#4a5888;align-self:center">→</div>
      <div style="flex:1;background:#212845;border-radius:6px;padding:.4rem .6rem;text-align:center;font-size:.68rem;color:#9aa5cc">
        <span style="color:#d4941c">④</span> ARO Reqressiya<br><span style="font-size:.62rem;color:#6878a8">Y=a₁T+a₂P+a₃F+ε</span>
      </div>
    </div>
  </div>

  <!-- Ox aşağı -->
  <div style="text-align:center;color:#4a5888;font-size:1.1rem;margin:2px 0">↓</div>

  <!-- Cərgə 3: NSGA-II — vurğulanmış -->
  <div style="background:linear-gradient(135deg,rgba(212,148,28,.1),rgba(74,143,217,.08));border:2px solid #d4941c;border-radius:10px;padding:.8rem 1rem;margin-bottom:8px;box-shadow:0 0 20px rgba(212,148,28,.12)">
    <div style="display:flex;align-items:center;gap:.8rem">
      <div style="font-size:1.5rem"></div>
      <div>
        <div style="font-size:.78rem;font-weight:800;color:#f0ac30">NSGA-II — çoxhədəfli optimallaşdırma — sənəd bölmə 2.4</div>
        <div style="font-size:.69rem;color:#9aa5cc;margin-top:.2rem">
          f₁=−F*→min &nbsp;·&nbsp; f₂=E→min &nbsp;·&nbsp; KKT: g(x)≤0, x_min≤x≤x_max &nbsp;·&nbsp; 100 fərd × 150 nəsil
        </div>
      </div>
    </div>
  </div>

  <!-- Ox aşağı -->
  <div style="text-align:center;color:#4a5888;font-size:1.1rem;margin:2px 0">↓</div>

  <!-- Cərgə 4: Çıxış -->
  <div style="display:flex;gap:8px;align-items:stretch">
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-bottom:2px solid #2ec98a;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#2ec98a;margin-bottom:.2rem"> PARETO FRONTU</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">100 dominant həll<br>kompromis seçim</div>
    </div>
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-bottom:2px solid #d4941c;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#d4941c;margin-bottom:.2rem"> OPTİMAL HƏLL</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">Verim ↑ +8.7%<br>Enerji ↓ 1.05 GS/s</div>
    </div>
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-bottom:2px solid #4a8fd9;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#4a8fd9;margin-bottom:.2rem"> DASHBOARD</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">5 interaktiv səhifə<br>Streamlit · Plotly</div>
    </div>
  </div>

</div>""",unsafe_allow_html=True)
    sp()

    bb("Əsas əməliyyat göstəriciləri")
    k1,k2,k3,k4,k5,k6=st.columns(6)
    k1.metric("Soba temp.", f"{xulase['avg_furnace_temp']:.1f} °C", help="Borulu soba çıxış temp. — distillasiya üçün kritik")
    k2.metric("Benzin", f"{xulase['avg_yield_gasoline']:.1f} %", help="Benzin fraksiyasının ortalama verimi")
    k3.metric("Dizel", f"{xulase['avg_yield_diesel']:.1f} %", help="Dizel fraksiyasının ortalama verimi")
    k4.metric("Kerosin", f"{xulase['avg_yield_kerosene']:.1f} %", help="Kerosin fraksiyasının ortalama verimi")
    k5.metric("Enerji", f"{xulase['avg_energy']:.2f} GS/s", help="Ortalama enerji sərfi — azalması hədəfdir")
    k6.metric("Kükürd çıx.", f"{xulase['avg_sulfur_removal']:.1f} %", help="Desulfurizasiya səmərəliliyi")
    sp()

    bb("24 saatlıq proses monitorinqi")
    ca,cb=st.columns(2,gap="medium")
    with ca:
        fig=go.Figure()
        for sc,ad,reng in[("yield_gasoline","Benzin (%)",RENG["benzin"]),
                          ("yield_diesel","Dizel (%)",RENG["dizel"]),
                          ("yield_kerosene","Kerosin (%)",RENG["kerosin"])]:
            fig.add_trace(go.Scatter(x=ts,y=temiz_df[sc]*100,name=ad,
                line=dict(color=reng,width=1.8),
                hovertemplate=f"<b>{ad}</b>: %{{y:.2f}}%<extra></extra>"))
        fig.update_layout(**QL("Fraksiya verimləri (%)",hund=300))
        fig.update_yaxes(title_text="Verim (%)")
        st.plotly_chart(fig,width="stretch")
    with cb:
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=ts,y=temiz_df["energy_gj_h"],name="Enerji (GS/s)",
            line=dict(color=RENG["enerji"],width=1.8),fill="tozeroy",fillcolor="rgba(212,80,96,.07)",
            hovertemplate="<b>Enerji</b>: %{y:.3f} GS/s<extra></extra>"))
        fig2.add_trace(go.Scatter(x=ts,y=temiz_df["sulfur_removal"]*100,name="Kükürd çıx. (%)",
            line=dict(color=RENG["kukurd"],width=1.8),yaxis="y2",
            hovertemplate="<b>Kükürd</b>: %{y:.1f}%<extra></extra>"))
        fig2.update_layout(**QL("Enerji sərfi + kükürd çıxarılması",hund=300))
        fig2.update_layout(
            yaxis=dict(title_text="Enerji (GS/s)",gridcolor="#283058",linecolor="#334070",zeroline=False),
            yaxis2=dict(overlaying="y",side="right",gridcolor="#283058",zeroline=False,
                        tickfont=dict(size=9,color="#9aa5cc"),title_text="Kükürd çıx. (%)"))
        st.plotly_chart(fig2,width="stretch")
    sp()

    bb("Məhsul paylanması · Statistik analiz · Temperatur")
    cp,cq,cr=st.columns([1.1,1.4,1.5],gap="medium")
    with cp:
        fig_pi=go.Figure(go.Pie(
            labels=["Benzin","Dizel","Kerosin","Qalıq (mazut)"],
            values=[xulase["avg_yield_gasoline"],xulase["avg_yield_diesel"],
                    xulase["avg_yield_kerosene"],100-xulase["avg_total_yield"]],
            hole=0.58,
            marker=dict(colors=[RENG["benzin"],RENG["dizel"],RENG["kerosin"],"#33406e"],
                        line=dict(color="#1a1f35",width=2)),
            textfont=dict(size=11),pull=[0.04,0,0,0],
            hovertemplate="<b>%{label}</b>: %{value:.1f}%<extra></extra>"))
        fig_pi.add_annotation(
            text=f"<b>{xulase['avg_total_yield']:.1f}%</b><br><span style='font-size:10px'>ümumi verim</span>",
            x=0.5,y=0.5,showarrow=False,
            font=dict(color="#edf1fc",size=13,family="JetBrains Mono"))
        fig_pi.update_layout(**QL("Məhsul paylanması",hund=280))
        st.plotly_chart(fig_pi,width="stretch")
    with cq:
        fig_b=go.Figure()
        for sc,ad,reng in[("yield_gasoline","Benzin",RENG["benzin"]),
                          ("yield_diesel","Dizel",RENG["dizel"]),
                          ("yield_kerosene","Kerosin",RENG["kerosin"])]:
            fig_b.add_trace(go.Box(y=temiz_df[sc]*100,name=ad,marker_color=reng,
                boxmean="sd",line=dict(width=1.5),
                hovertemplate=f"<b>{ad}</b>: %{{y:.2f}}%<extra></extra>"))
        fig_b.update_layout(**QL("Verim statistik analizi",hund=280,legend=False))
        fig_b.update_yaxes(title_text="Verim (%)")
        st.plotly_chart(fig_b,width="stretch")
    with cr:
        tv=temiz_df["furnace_temp"].dropna()
        fig_h=go.Figure()
        fig_h.add_trace(go.Histogram(x=tv,nbinsx=40,
            marker=dict(color=RENG["benzin"],opacity=.75,line=dict(color="#1a1f35",width=.5)),
            hovertemplate="T: %{x:.1f}°C | Say: %{y}<extra></extra>"))
        fig_h.add_vline(x=tv.mean(),line_color="#edf1fc",line_dash="dash",line_width=1.5,
            annotation_text=f"Ort: {tv.mean():.1f}°C",
            annotation_font_color="#edf1fc",annotation_font_size=10)
        fig_h.update_layout(**QL("Soba temp. paylanması",hund=280,legend=False))
        fig_h.update_xaxes(title_text="Temperatur (°C)")
        fig_h.update_yaxes(title_text="Tezlik")
        st.plotly_chart(fig_h,width="stretch")
    sp()

    bb("Sənəd tənlikləri — Python kodunda tətbiqi")
    eq1,eq2=st.columns(2,gap="medium")
    with eq1:
        st.markdown('<div class="tenlik"><b>Məqsəd funksiyası (tənlik 1):</b><br>'
            'F* = α·Y<sub>benz</sub>+β·Y<sub>diz</sub>+γ·Y<sub>ker</sub>−δ·E<br><br>'
            '<b>Normallaşdırma (tənlik 2):</b><br>'
            'x<sub>norm</sub>=(x−x<sub>min</sub>)/(x<sub>max</sub>−x<sub>min</sub>)<br><br>'
            '<b>Reqressiya modeli:</b><br>'
            'Y = a₁T + a₂P + a₃F + a₄ + ε</div>',unsafe_allow_html=True)
    with eq2:
        st.markdown('<div class="tenlik"><b>Kütlə balansı (tənlik 3):</b><br>'
            'ΣF<sub>giriş</sub>=ΣF<sub>çıxış</sub><br><br>'
            '<b>Enerji balansı (tənlik 4):</b><br>'
            'Q<sub>verilən</sub>=Q<sub>istifadə</sub>+Q<sub>itki</sub><br><br>'
            '<b>Temperatur profili (tənlik 5):</b><br>'
            'T(z)=T<sub>alt</sub>−k·z<br><br>'
            '<b>Laqranj (KKT, tənlik 10):</b><br>'
            'L(x,λ,μ)=f(x)+Σλᵢgᵢ(x)+Σμh(x)</div>',unsafe_allow_html=True)

#
# SƏHİFƏ 2 — MƏLUMAT ANALİZİ
#
elif sehife==" Məlumat Analizi":
    st.markdown("## Kəşfiyyat Məlumat Analizi (EDA)")
    st.markdown("Sənəd Cədvəl 2.1: T (temp.), P (təzyiq), F (axın) — 9 sensor, 24 saat, 1 dəq. interval.")
    sp()

    # Raw Data Preview
    bb("Ham məlumat — dataset icmalı")
    col_i1,col_i2,col_i3=st.columns(3)
    col_i1.info(f"{xam_df.shape[0]} sətir x {xam_df.shape[1]} sütun")
    col_i2.info(f"{int(xulase['n_anomalies'])} anomaliya ({xulase['anomaly_rate_pct']:.1f}%)")
    col_i3.info(f"{len(temiz_df)} təmiz nümunə saxlanıldı")
    sp()

    t_raw,t_clean=st.tabs([" Ham Məlumat"," Təmiz Məlumat"])
    with t_raw:
        st.markdown('<div class="izah">Sensorlardan birbaşa alınan xam məlumat — filtrasiyadan <b>əvvəl</b>. NaN və anomal dəyərlər mövcuddur.</div>',unsafe_allow_html=True)
        disp_cols=["timestamp"]+INPUT_FEATURES[:6]
        st.dataframe(xam_df[disp_cols].head(20).style.format({c:"{:.3f}" for c in INPUT_FEATURES[:6]}),
                     width="stretch",height=320)
    with t_clean:
        st.markdown('<div class="izah">Preprocessing boru kəmərindən keçmiş məlumat — filtrasiya, normallaşdırma, anomaliya aşkarlanmasından <b>sonra</b>.</div>',unsafe_allow_html=True)
        st.dataframe(temiz_df[disp_cols+["yield_gasoline","yield_diesel","energy_gj_h"]].head(20)
                     .style.format({c:"{:.3f}" for c in INPUT_FEATURES[:6]+["yield_gasoline","yield_diesel","energy_gj_h"]}),
                     width="stretch",height=320)
    sp()

    t1,t2,t3=st.tabs([" Zaman Seriyaları"," Korrelyasiya"," Statistika"])

    with t1:
        csel,cplt=st.columns([1,3],gap="medium")
        with csel:
            sec=st.multiselect("Sensor kanalları",INPUT_FEATURES,
                default=["furnace_temp","column_pressure","flow_rate"],
                format_func=lambda x:SENSOR_AD.get(x,x),label_visibility="collapsed")
            pen=st.slider("Hərəkətli ort. (dəq.)",1,60,15,
                help="Sensor küyünü hamarlamaq üçün")
            ag=st.checkbox("Anomaliyaları göstər",True,
                help="IQR + Isolation Forest ilə aşkar edilmiş nöqtələr")
        with cplt:
            if not sec:
                st.info("Sol tərəfdən sensor seçin.")
            else:
                RCOL=[RENG["benzin"],RENG["dizel"],RENG["kerosin"],RENG["kukurd"],
                      RENG["anomal"],"#a78bfa","#f472b6","#38bdf8","#fb923c"]
                fig_ts=make_subplots(rows=len(sec),cols=1,shared_xaxes=True,
                    subplot_titles=[SENSOR_AD.get(s,s) for s in sec],vertical_spacing=.06)
                for i,sensor in enumerate(sec):
                    reng=RCOL[i%len(RCOL)]
                    vals=temiz_df[sensor]; roll=vals.rolling(pen).mean()
                    fig_ts.add_trace(go.Scatter(x=ts,y=vals,opacity=.28,line=dict(color=reng,width=1),
                        showlegend=False,hovertemplate=f"%{{y:.3f}}<extra>{SENSOR_AD.get(sensor,sensor)}</extra>"),row=i+1,col=1)
                    fig_ts.add_trace(go.Scatter(x=ts,y=roll,name=SENSOR_AD.get(sensor,sensor),
                        line=dict(color=reng,width=2),hovertemplate=f"MA{pen}: %{{y:.3f}}<extra></extra>"),row=i+1,col=1)
                    if ag and "anomaly_final" in anomaliya_df.columns:
                        am=anomaliya_df["anomaly_final"]
                        ad2=temiz_df[am.values[:len(temiz_df)]]
                        if len(ad2):
                            fig_ts.add_trace(go.Scatter(x=ad2["timestamp"],y=ad2[sensor],
                                mode="markers",name="Anomaliya" if i==0 else None,showlegend=(i==0),
                                marker=dict(color=RENG["anomal"],size=5,symbol="x-thin",
                                    line=dict(width=2,color=RENG["anomal"])),
                                hovertemplate="Anomaliya: %{y:.3f}<extra></extra>"),row=i+1,col=1)
                fig_ts.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#1e2540",
                    font=dict(family="Inter",color="#9aa5cc",size=10),
                    height=max(220*len(sec),280),margin=dict(l=8,r=8,t=30,b=8),
                    legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.06,font=dict(size=10,color="#9aa5cc")))
                sub_stil(fig_ts)
                st.plotly_chart(fig_ts,width="stretch")

    with t2:
        bb("Pearson korrelyasiya matrisi")
        st.markdown('<div class="izah">r=+1: güclü müsbət · r=0: əlaqə yox · r=−1: güclü mənfi.<br>'
            '<b style="color:#d4941c">Amber=müsbət</b> · <b style="color:#4a8fd9">Mavi=mənfi</b></div>',unsafe_allow_html=True)
        cc=INPUT_FEATURES+["yield_gasoline","yield_diesel","yield_kerosene","energy_gj_h"]
        ce=[SENSOR_AD.get(c,HEDAF_AD.get(c,c)) for c in cc]
        corr=temiz_df[cc].corr()
        fig_c=go.Figure(go.Heatmap(z=corr.values,x=ce,y=ce,
            colorscale=[[0,"#d45060"],[0.5,"#1e2540"],[1,"#d4941c"]],
            zmid=0,zmin=-1,zmax=1,text=corr.values.round(2),texttemplate="%{text}",
            textfont=dict(size=8.5,family="JetBrains Mono")))
        fig_c.update_layout(**QL("Korrelyasiya matrisi (Pearson r)",hund=500,legend=False))
        fig_c.update_xaxes(tickangle=-35,tickfont=dict(size=9,color="#9aa5cc"))
        st.plotly_chart(fig_c,width="stretch")
        sp()
        bb("Sensor–verim scatter analizi")
        s1,s2,s3=st.columns(3,gap="medium")
        for col_o,(xc,yc,xl,yl,reng) in zip([s1,s2,s3],[
            ("furnace_temp","yield_gasoline","Soba temp. (°C)","Benzin verimi (%)",RENG["benzin"]),
            ("reflux_ratio","yield_diesel","Refluks nisbəti","Dizel verimi (%)",RENG["dizel"]),
            ("h2_pressure","sulfur_removal","H₂ təzyiqi (bar)","Kükürd çıx. (%)",RENG["kukurd"])]):
            sf=go.Figure(go.Scatter(x=temiz_df[xc],y=temiz_df[yc]*100,mode="markers",
                marker=dict(color=reng,size=3,opacity=.5),
                hovertemplate=f"{xl}: %{{x:.2f}}<br>{yl}: %{{y:.2f}}<extra></extra>"))
            sf.update_layout(**QL(f"{xl} → {yl}",hund=230,legend=False))
            sf.update_xaxes(title_text=xl); sf.update_yaxes(title_text=yl)
            col_o.plotly_chart(sf,width="stretch")

    with t3:
        bb("Təsviri statistika")
        st.markdown('<div class="izah"><b>count</b>=say · <b>mean</b>=orta · <b>std</b>=kənarlaşma · <b>min/max</b>=həddlər · <b>25/50/75%</b>=kvartillər</div>',unsafe_allow_html=True)
        ss=INPUT_FEATURES+["yield_gasoline","yield_diesel","yield_kerosene","energy_gj_h"]
        sdf=temiz_df[ss].describe().round(4)
        sdf.columns=[SENSOR_AD.get(c,HEDAF_AD.get(c,c)) for c in sdf.columns]
        sdf.index=["Say","Orta","Std","Min","25%","50%","75%","Maks"]
        st.dataframe(sdf.style.format("{:.4f}"),width="stretch",height=310)
        sp()
        bb("Paylanma histoqramları — 9 sensor")
        fig_mul=make_subplots(rows=3,cols=3,
            subplot_titles=[SENSOR_AD.get(s,s) for s in INPUT_FEATURES],
            vertical_spacing=.14,horizontal_spacing=.08)
        R9=[RENG["benzin"],RENG["dizel"],RENG["kerosin"],RENG["kukurd"],
            RENG["anomal"],"#a78bfa","#f472b6","#38bdf8","#fb923c"]
        for i,sensor in enumerate(INPUT_FEATURES):
            r,c=divmod(i,3)
            fig_mul.add_trace(go.Histogram(x=temiz_df[sensor],nbinsx=35,
                marker=dict(color=R9[i],opacity=.82,line=dict(color="#1a1f35",width=.3)),
                showlegend=False),row=r+1,col=c+1)
        fig_mul.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#1e2540",
            font=dict(family="Inter",color="#9aa5cc",size=9),
            height=520,margin=dict(l=4,r=4,t=40,b=4),showlegend=False)
        sub_stil(fig_mul)
        st.plotly_chart(fig_mul,width="stretch")

#
# SƏHİFƏ 3 — EMAL BORU KƏMƏRİ
#
elif sehife==" Emal Boru Kəməri":
    st.markdown("## Verilənlərin Emal Boru Kəməri")
    st.markdown("Sənəd bölmə 2.3: filtrasiya → normallaşdırma → anomaliya → reqressiya → ARO / E2E / DT metodları.")
    sp()

    tf,tn,ta,tr,tvm=st.tabs([
        "1⃣ Filtrasiya","2⃣ Normallaşdırma",
        "3⃣ Anomaliya","4⃣ Reqressiya (ARO)",
        "5⃣ E2E · DT · Vakuum"])

    with tf:
        bb("Xətti interpolasiya + 3-sigma kəsmə")
        col_fi1,col_fi2=st.columns(2,gap="medium")
        with col_fi1:
            st.markdown('<div class="izah"><b>① Filtrasiya</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">Ardıcıl ≤5 boşluq → xətti interpolasiya</span><br><br>'
                '<b>② 3-sigma kəsmə</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">[μ−3σ, μ+3σ] xarici dəyərlər kəsilir</span>'
                '</div>',unsafe_allow_html=True)
        with col_fi2:
            st.markdown('<div class="izah"><b>③ NaN silmə</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">Hələ NaN olan sətrlər çıxarılır</span><br><br>'
                '<b>Nəticə</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">Optimallaşdırıcıya etibarlı məlumat</span>'
                '</div>',unsafe_allow_html=True)
        xs,ts2=len(xam_df),len(temiz_df)
        a1,a2,a3,a4=st.columns(4)
        a1.metric("Xam nümunə",f"{xs:,}",help="Orijinal sensor ölçümləri")
        a2.metric("Temiz nümunə",f"{ts2:,}",help="Filtrasiyadan sonra qalan etibarlı nümunələr")
        a3.metric("Çıxarılan",f"{xs-ts2:,}",help="Nasaz sensor qeydləri")
        a4.metric("Saxlanma",f"{ts2/xs*100:.1f}%",help="≥95% ideal sayılır")
        sp()
        ss=st.selectbox("Sensor seçin",INPUT_FEATURES,format_func=lambda x:SENSOR_AD.get(x,x))
        fig_f=make_subplots(rows=1,cols=2,subplot_titles=["Xam məlumat","Filtrə edilmiş"],horizontal_spacing=.06)
        for ci,(dg,ad,reng) in enumerate([(xam_df,"Xam",RENG["anomal"]),(temiz_df,"Temiz",RENG["benzin"])]):
            fig_f.add_trace(go.Scatter(x=dg["timestamp"],y=dg[ss],name=ad,
                line=dict(color=reng,width=1.3),hovertemplate=f"{ad}: %{{y:.3f}}<extra></extra>"),row=1,col=ci+1)
        fig_f.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#1e2540",
            font=dict(family="Inter",color="#9aa5cc",size=10),
            height=280,margin=dict(l=8,r=8,t=36,b=8),showlegend=False)
        sub_stil(fig_f)
        st.plotly_chart(fig_f,width="stretch")

    with tn:
        bb("Min-Maks normallaşdırma — sənəd tənliyi")
        st.markdown('<div class="izah">Müxtəlif ölçü vahidli sensorları [0,1] aralığına çevirir:<br>'
            '<code>x_norm = (x − x_min) / (x_max − x_min)</code></div>',unsafe_allow_html=True)
        ns=st.selectbox("Xüsusiyyət",INPUT_FEATURES,format_func=lambda x:SENSOR_AD.get(x,x),key="ns")
        xv=temiz_df[ns].values; nv=X_norm[ns].values
        fig_n=make_subplots(rows=1,cols=2,subplot_titles=["Xam dəyərlər","Normallaşdırılmış [0,1]"],horizontal_spacing=.08)
        for ci,(vals,reng,ad) in enumerate([(xv,RENG["dizel"],"Xam"),(nv,RENG["benzin"],"Norm")]):
            fig_n.add_trace(go.Histogram(x=vals,nbinsx=45,name=ad,
                marker=dict(color=reng,opacity=.82,line=dict(color="#1a1f35",width=.4)),
                hovertemplate=f"{ad}: %{{x:.4f}} | Say: %{{y}}<extra></extra>"),row=1,col=ci+1)
            xk="x" if ci==0 else "x2"
            fig_n.add_shape(type="line",x0=vals.mean(),x1=vals.mean(),y0=0,y1=1,
                yref="paper",xref=xk,line=dict(color="#edf1fc",dash="dot",width=1.2))
            fig_n.add_annotation(x=vals.mean(),y=.95,yref="paper",xref=xk,
                text=f"Ort:{vals.mean():.3f}",showarrow=False,
                font=dict(color="#edf1fc",size=9,family="JetBrains Mono"),bgcolor="rgba(26,31,53,.85)")
        fig_n.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#1e2540",
            font=dict(family="Inter",color="#9aa5cc",size=10),
            height=300,margin=dict(l=8,r=8,t=36,b=8),showlegend=False)
        sub_stil(fig_n)
        st.plotly_chart(fig_n,width="stretch")

    with ta:
        bb("İki metodlu anomaliya aşkarlanması")
        col_ai1,col_ai2=st.columns(2,gap="medium")
        with col_ai1:
            st.markdown('<div class="izah"><b>Metod A — IQR</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">Q1−1.5·IQR ~ Q3+1.5·IQR<br>Statistik, modelsiz, sürətli</span>'
                '</div>',unsafe_allow_html=True)
        with col_ai2:
            st.markdown('<div class="izah"><b>Metod B — Isolation Forest</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">ML əsaslı, çoxölçülü<br>Yekun: hər iki metodun birliyi</span>'
                '</div>',unsafe_allow_html=True)
        adf=anomaliya_df
        ni=int(adf["anomaly_iqr"].sum()) if "anomaly_iqr" in adf.columns else 0
        nf=int(adf["anomaly_iforest"].sum()) if "anomaly_iforest" in adf.columns else 0
        ny=int(adf["anomaly_final"].sum()) if "anomaly_final" in adf.columns else 0
        b1,b2,b3,b4=st.columns(4)
        b1.metric("IQR anomaliya",f"{ni}",help="Statistik üsul — univariate")
        b2.metric("Isolation Forest",f"{nf}",help="ML üsulu — multivariate")
        b3.metric("Yekun anomaliya",f"{ny}",help="İki metodun birliyindən")
        b4.metric("Anomaliya faizi",f"{ny/len(adf)*100:.2f}%",help="<5% arzu olunan dəyərdir")
        sp()
        ase=st.selectbox("Sensor",INPUT_FEATURES,format_func=lambda x:SENSOR_AD.get(x,x),key="ase")
        fig_a=go.Figure()
        nm=~adf["anomaly_final"] if "anomaly_final" in adf.columns else pd.Series(True,index=adf.index)
        am=adf["anomaly_final"] if "anomaly_final" in adf.columns else pd.Series(False,index=adf.index)
        fig_a.add_trace(go.Scatter(
            x=adf.loc[nm,"timestamp"] if "timestamp" in adf.columns else adf.index[nm],
            y=adf.loc[nm,ase],name="Normal",line=dict(color=RENG["normal"],width=1.4),
            hovertemplate="Normal: %{y:.3f}<extra></extra>"))
        fig_a.add_trace(go.Scatter(
            x=adf.loc[am,"timestamp"] if "timestamp" in adf.columns else adf.index[am],
            y=adf.loc[am,ase],name="Anomaliya",mode="markers",
            marker=dict(color=RENG["anomal"],size=7,symbol="x-thin",line=dict(width=2,color=RENG["anomal"])),
            hovertemplate=" %{y:.3f}<extra></extra>"))
        fig_a.update_layout(**QL(f"{SENSOR_AD.get(ase,ase)} — Anomaliya aşkarlanması",hund=300))
        st.plotly_chart(fig_a,width="stretch")
        if "anomaly_score" in adf.columns:
            fsk=go.Figure(go.Histogram(x=adf["anomaly_score"],nbinsx=50,
                marker=dict(color=RENG["normal"],opacity=.82,line=dict(color="#1a1f35",width=.4))))
            fsk.add_vline(x=0,line_color=RENG["anomal"],line_dash="dash",line_width=2,
                annotation_text="Qərar həddi (0)",annotation_font_color=RENG["anomal"],annotation_font_size=10)
            fsk.update_layout(**QL("Isolation Forest anomaliya skoru",hund=220,legend=False))
            fsk.update_xaxes(title_text="Bal (aşağı = daha anomal)")
            st.plotly_chart(fsk,width="stretch")

    with tr:
        bb("ARO reqressiya modeli — GB + RF Ensemble (CV çəkiləri ilə)")
        st.markdown('<div class="izah">'
            '<b>ARO Ensemble:</b> hər hədəf üçün iki model paralel öyrədilir — '
            '<b>GradientBoosting</b> (aşağı bias) + <b>RandomForest</b> (aşağı dispersiya).<br>'
            'Çəkilər 5-qatlı CV R² skoruna əsasən: <code>w_gb = cv_gb/(cv_gb+cv_rf)</code><br>'
            'Yekun: <code>ensemble = w_gb·GB + w_rf·RF</code>'
            '</div>',unsafe_allow_html=True)

        raw_metrics = req_goster.copy()
        has_new_cols = "Ensemble_R²" in raw_metrics.columns

        if has_new_cols:
            show_cols = ["GB_CV_R²","RF_CV_R²","W_GB","W_RF","GB_R²","RF_R²","Ensemble_R²","Ensemble_RMSE"]
            show_cols = [c for c in show_cols if c in raw_metrics.columns]
            gdf = raw_metrics[show_cols].copy()
            r2_col = "Ensemble_R²"
        else:
            gdf = raw_metrics.copy()
            r2_col = "R²" if "R²" in gdf.columns else gdf.columns[0]
        gdf.index = [HEDAF_AD.get(i,i) for i in gdf.index]

        rc1,rc2=st.columns([1,1.6],gap="medium")
        with rc1:
            st.markdown("**Model göstəriciləri**")
            fmt_dict = {}
            for c in gdf.columns:
                if "RMSE" in c: fmt_dict[c] = "{:.6f}"
                elif "W_" in c: fmt_dict[c] = "{:.3f}"
                else: fmt_dict[c] = "{:.4f}"
            styled = gdf.style.format(fmt_dict)
            if r2_col in gdf.columns:
                styled = styled.background_gradient(subset=[r2_col],cmap="YlOrRd")
            st.dataframe(styled,width="stretch")
            st.markdown('<div class="izah" style="font-size:.76rem">'
                '<b>CV R²</b>=cross-val. açıqlama gücü · <b>W</b>=çəki · '
                '<b>Ensemble R²</b>=yekun model · <b>RMSE</b>=orta kvadrat xəta</div>',
                unsafe_allow_html=True)

        with rc2:
            if has_new_cols and "Ensemble_R²" in raw_metrics.columns:
                r2v = raw_metrics["Ensemble_R²"].values
                r2e = [HEDAF_AD.get(i,i) for i in raw_metrics.index]
                fig_r2 = go.Figure()
                if "GB_R²" in raw_metrics.columns:
                    fig_r2.add_trace(go.Bar(name="GB R²",x=r2e,y=raw_metrics["GB_R²"].values,
                        marker=dict(color=RENG["dizel"],opacity=.8),
                        hovertemplate="GB R²: %{y:.4f}<extra></extra>"))
                if "RF_R²" in raw_metrics.columns:
                    fig_r2.add_trace(go.Bar(name="RF R²",x=r2e,y=raw_metrics["RF_R²"].values,
                        marker=dict(color=RENG["kerosin"],opacity=.8),
                        hovertemplate="RF R²: %{y:.4f}<extra></extra>"))
                fig_r2.add_trace(go.Bar(name="Ensemble R²",x=r2e,y=r2v,
                    marker=dict(color=["#d4941c" if v>=.95 else "#4a8fd9" if v>=.85 else "#d45060" for v in r2v],
                        opacity=.95,line=dict(color="#1a1f35",width=.5)),
                    text=[f"{v:.4f}" for v in r2v],textposition="outside",
                    textfont=dict(size=9,family="JetBrains Mono"),
                    hovertemplate="Ensemble R²: %{y:.4f}<extra></extra>"))
                _ql_r2 = QL("GB vs RF vs Ensemble R²",hund=300)
                _ql_r2["legend"] = dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9,color="#9aa5cc"),orientation="h",y=-0.28)
                fig_r2.update_layout(**_ql_r2,barmode="group")
            else:
                r2v = gdf[r2_col].values; r2e = list(gdf.index)
                fig_r2 = go.Figure(go.Bar(x=r2e,y=r2v,
                    marker=dict(color=["#d4941c" if v>=.95 else "#4a8fd9" if v>=.85 else "#d45060" for v in r2v],
                        opacity=.9,line=dict(color="#1a1f35",width=.5)),
                    text=[f"{v:.4f}" for v in r2v],textposition="outside",
                    textfont=dict(size=10,family="JetBrains Mono"),
                    hovertemplate="%{x}: R²=%{y:.4f}<extra></extra>"))
            fig_r2.add_hline(y=.95,line_dash="dot",line_color=RENG["kukurd"],line_width=1.2,
                annotation_text="0.95 keyfiyyət həddi",
                annotation_font_color=RENG["kukurd"],annotation_font_size=9)
            fig_r2.update_yaxes(range=[0,1.12],title_text="R²")
            fig_r2.update_xaxes(tickangle=-20)
            st.plotly_chart(fig_r2,width="stretch")
        sp()

        if has_new_cols and "W_GB" in raw_metrics.columns and "W_RF" in raw_metrics.columns:
            bb("CV çəkiləri — GB vs RF modelin töhfəsi (hər hədəf üçün)")
            st.markdown('<div class="izah">Çəkilər 5-qatlı cross-validation R²-ə əsasən avtomatik hesablanır. '
                'Daha yüksək CV R² → daha böyük çəki. <b>W_GB + W_RF = 1.0</b></div>',unsafe_allow_html=True)
            wlabels = [HEDAF_AD.get(i,i) for i in raw_metrics.index]
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(name="W_GB (GradientBoosting)",x=wlabels,
                y=raw_metrics["W_GB"].values,
                marker=dict(color=RENG["benzin"],opacity=.85),
                hovertemplate="W_GB: %{y:.3f}<extra></extra>"))
            fig_w.add_trace(go.Bar(name="W_RF (RandomForest)",x=wlabels,
                y=raw_metrics["W_RF"].values,
                marker=dict(color=RENG["dizel"],opacity=.85),
                hovertemplate="W_RF: %{y:.3f}<extra></extra>"))
            _ql_w = QL("Ensemble çəkiləri: W_GB vs W_RF",hund=260)
            _ql_w["legend"] = dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color="#9aa5cc"),orientation="h",y=-0.28)
            fig_w.update_layout(**_ql_w,barmode="stack")
            fig_w.add_hline(y=0.5,line_dash="dot",line_color="#6878a8",line_width=1,
                annotation_text="0.5 bərabər çəki",annotation_font_color="#6878a8",annotation_font_size=9)
            fig_w.update_yaxes(range=[0,1.08],title_text="Çəki")
            fig_w.update_xaxes(tickangle=-20)
            st.plotly_chart(fig_w,width="stretch")
            sp()

        bb("Feature Importance — hansı sensor hansı verimi idarə edir?")
        st.markdown('<div class="izah">GradientBoosting və RandomForest modelləri üçün '
            'hər sensorun hər verim/çıxış dəyişəninə təsiri. '
            '<b style="color:#d4941c">Amber = GB</b> · '
            '<b style="color:#4a8fd9">Mavi = RF</b> · '
            'Yüksək dəyər → həmin sensor həmin verimi daha çox izah edir.</div>',unsafe_allow_html=True)

        feat_imp_df = boru["feature_importances"]

        fi_t1,fi_t2 = st.tabs([" GradientBoosting"," RandomForest"])

        def _plot_fi(prefix,color_main,color_accent,tab_key):
            cols_fi = [c for c in feat_imp_df.columns if c.startswith(prefix+"_")]
            if not cols_fi:
                st.info(f"{prefix} importance sütunları tapılmadı."); return
            feat_labels = [SENSOR_AD.get(c.replace(prefix+"_",""),c.replace(prefix+"_","")) for c in cols_fi]
            fi_target_sel = st.selectbox("Hədəf dəyişəni",options=list(feat_imp_df.index),
                format_func=lambda x:HEDAF_AD.get(x,x),key=f"fi_sel_{tab_key}")
            row_vals = feat_imp_df.loc[fi_target_sel,cols_fi].values.astype(float)
            sort_idx = np.argsort(row_vals)[::-1]
            sorted_vals = row_vals[sort_idx]
            sorted_labels = [feat_labels[i] for i in sort_idx]
            fig_fi = go.Figure(go.Bar(
                x=sorted_vals,y=sorted_labels,orientation="h",
                marker=dict(color=[color_main if v>=sorted_vals.max()*0.5 else color_accent for v in sorted_vals],
                    opacity=0.88,line=dict(color="#1a1f35",width=0.5)),
                text=[f"{v:.4f}" for v in sorted_vals],textposition="outside",
                textfont=dict(size=9,family="JetBrains Mono"),
                hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>"))
            fig_fi.update_layout(**QL(f"{prefix} Feature Importance — {HEDAF_AD.get(fi_target_sel,fi_target_sel)}",hund=350,legend=False))
            fig_fi.update_xaxes(title_text="Importance",range=[0,sorted_vals.max()*1.22])
            fig_fi.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_fi,width="stretch")
            st.markdown("**Bütün hədəflər — istilik xəritəsi**")
            z_mat = feat_imp_df[cols_fi].values.astype(float)
            target_labels = [HEDAF_AD.get(t,t) for t in feat_imp_df.index]
            fig_fih = go.Figure(go.Heatmap(z=z_mat,x=feat_labels,y=target_labels,
                colorscale=[[0,"#1e2540"],[0.4,"#283058"],[0.7,"#4a8fd9"],[1,"#d4941c"]],
                text=z_mat.round(4),texttemplate="%{text}",
                textfont=dict(size=8,family="JetBrains Mono"),
                hovertemplate="<b>%{y}</b> ← %{x}<br>Importance: %{z:.4f}<extra></extra>"))
            fig_fih.update_layout(**QL(f"{prefix} — tam importance matrisi",hund=320,legend=False))
            fig_fih.update_xaxes(tickangle=-30,tickfont=dict(size=9,color="#9aa5cc"))
            st.plotly_chart(fig_fih,width="stretch")

        with fi_t1:
            _plot_fi("GB",RENG["benzin"],"#c07820","gb")
        with fi_t2:
            _plot_fi("RF",RENG["dizel"],"#2a5f99","rf")
        sp()

        bb("GB − RF importance fərqi istilik xəritəsi")
        st.markdown('<div class="izah"><b style="color:#d4941c">Amber</b>=GB üstün · '
            '<b style="color:#4a8fd9">Mavi</b>=RF üstün · Sıfıra yaxın=hər iki model razılaşır</div>',unsafe_allow_html=True)
        gb_cols_fi=[c for c in feat_imp_df.columns if c.startswith("GB_")]
        rf_cols_fi=[c for c in feat_imp_df.columns if c.startswith("RF_")]
        if gb_cols_fi and rf_cols_fi:
            sensor_names=[SENSOR_AD.get(c.replace("GB_",""),c.replace("GB_","")) for c in gb_cols_fi]
            target_names=[HEDAF_AD.get(t,t) for t in feat_imp_df.index]
            diff_mat=feat_imp_df[gb_cols_fi].values-feat_imp_df[rf_cols_fi].values
            fig_diff=go.Figure(go.Heatmap(z=diff_mat,x=sensor_names,y=target_names,
                colorscale=[[0,"#4a8fd9"],[0.5,"#1e2540"],[1,"#d4941c"]],zmid=0,
                text=diff_mat.round(4),texttemplate="%{text}",textfont=dict(size=8,family="JetBrains Mono"),
                hovertemplate="<b>%{y}</b> ← %{x}<br>GB−RF: %{z:.4f}<extra></extra>"))
            fig_diff.update_layout(**QL("GB − RF importance fərqi",hund=300,legend=False))
            fig_diff.update_xaxes(tickangle=-30,tickfont=dict(size=9,color="#9aa5cc"))
            st.plotly_chart(fig_diff,width="stretch")
        sp()

        bb("Reqressiya əmsalları istilik xəritəsi (GB Feature Importance)")
        ex=[SENSOR_AD.get(c,c) for c in req_emsal.columns if c!="intercept"]
        ez=req_emsal[[c for c in req_emsal.columns if c!="intercept"]].values
        fig_em=go.Figure(go.Heatmap(z=ez,x=ex,y=[HEDAF_AD.get(i,i) for i in req_emsal.index],
            colorscale=[[0,"#d45060"],[0.5,"#1e2540"],[1,"#d4941c"]],
            zmid=0,text=ez.round(4),texttemplate="%{text}",textfont=dict(size=8.5,family="JetBrains Mono")))
        fig_em.update_layout(**QL("Feature Importance əmsalları (GB, normallaşdırılmış)",hund=310,legend=False))
        fig_em.update_xaxes(tickangle=-30,tickfont=dict(size=9,color="#9aa5cc"))
        st.plotly_chart(fig_em,width="stretch")

    with tvm:
        bb("E2E, DT metodları və Vakuum Distilləsi")

        st.markdown("### Başdan-Başa Öyrənmə (E2E — End-to-End Learning)")
        st.markdown('<div class="izah">'
            'Sənəd bölmə 2.3, Şəkil 2.7. ARO-dan fərqli olaraq E2E proqnozlaşdırma '
            'və qərar qəbuletməni <b>vahid çərçivədə</b> birləşdirir.<br>'
            '• Hər iterasiyada optimallaşdırma nəticəsi geri yayılaraq modeli kalibrləyir<br>'
            '• Qərar mərkəzli itki funksiyası (decision-focused loss)<br>'
            '• ARO-dakı iki mərhələli ayrılmanın yaratdığı qərəzləri azaldır<br>'
            '• Neft emalında: real vaxt rejimli adaptiv idarəetmə sistemlərində istifadə olunur'
            '</div>',unsafe_allow_html=True)

        col_e1,col_e2=st.columns(2,gap="medium")
        with col_e1:
            st.markdown("ARO vs E2E müqayisəsi:")
            comp_df=pd.DataFrame({
                "Xüsusiyyət":["Struktur","Öyrənmə","Sürət","Dəqiqlik","Tətbiq"],
                "ARO":["Ardıcıl (2 mərhələ)","Proqnoz → Qərar","Sürətli","Yüksək R²","Sənaye standartı"],
                "E2E":["Vahid çərçivə","Birgə öyrənmə","Yavaş","Daha optimal","Tədqiqat sahəsi"],
            })
            st.dataframe(comp_df.style.set_properties(**{"color":"#c5cceb"}),width="stretch",hide_index=True)
        with col_e2:
            # E2E simulyasiyası — reqressiya + iterativ yaxşılaşma
            nesl=list(range(1,21))
            aro_r2=[0.82+0.015*i-0.0003*i**2 for i in nesl]
            e2e_r2=[0.78+0.019*i-0.0002*i**2 for i in nesl]
            fig_e2e=go.Figure()
            fig_e2e.add_trace(go.Scatter(x=nesl,y=aro_r2,name="ARO",
                line=dict(color=RENG["benzin"],width=2,dash="dot")))
            fig_e2e.add_trace(go.Scatter(x=nesl,y=e2e_r2,name="E2E",
                line=dict(color=RENG["dizel"],width=2)))
            fig_e2e.update_layout(**QL("ARO vs E2E — öyrənmə əyrisi (simulyasiya)",hund=250))
            fig_e2e.update_xaxes(title_text="İterasiya")
            fig_e2e.update_yaxes(title_text="R² dəyəri",range=[0.8,1.0])
            st.plotly_chart(fig_e2e,width="stretch")
        sp()

        st.markdown("### Birbaşa Öyrənmə (DT — Decision-Tree / Direct Learning)")
        st.markdown('<div class="izah">'
            'Sənəd bölmə 2.3, Şəkil 2.8. DT məlumatları birbaşa qərarlara inteqrasiya edir.<br>'
            '• Açıq optimallaşdırma formulalarına ehtiyac yoxdur<br>'
            '• Ənənəvi optimallaşdırmanın mümkün olmadığı mühitlərdə üstünlük<br>'
            '• Son məqsəd performans metrikləri ilə birbaşa uyğunlaşdırma<br>'
            '• Neft emalında: real vaxt qərar dəstəyi sistemlərində tətbiq olunur'
            '</div>',unsafe_allow_html=True)

        # DT üçün Decision Tree vizualizasiyası
        dt_data={
            "Soba temp. > 370°C":{"Bəli":"Benzin verimi > 25%","Xeyr":"Benzin verimi < 20%"},
            "Benzin verimi > 25%":{"Bəli":" Optimal rejim","Xeyr":" Temperaturu artır"},
            "Benzin verimi < 20%":{"Bəli":" Refluks artır","Xeyr":" Parametr düzəlt"},
        }
        col_dt1,col_dt2=st.columns([1,1],gap="medium")
        with col_dt1:
            st.markdown("DT qərar ağacı — sadələşdirilmiş nümunə:")
            for node,branches in dt_data.items():
                st.markdown(
                    f"<div style='background:#212845;border:1px solid #334070;border-radius:8px;"
                    f"padding:.6rem .9rem;margin:.3rem 0;font-size:.82rem;color:#c5cceb'>"
                    f"<b>{node}</b></div>",unsafe_allow_html=True)
                for label,result in branches.items():
                    color="#2ec98a" if "" in result else "#d4941c" if "" in result else "#d45060"
                    st.markdown(
                        f"<div style='margin-left:20px;background:#1e2540;border-left:2px solid {color};"
                        f"padding:.4rem .8rem;margin-bottom:.2rem;font-size:.78rem;color:#9aa5cc'>"
                        f"→ <b style='color:{color}'>{label}:</b> {result}</div>",unsafe_allow_html=True)
        with col_dt2:
            # 3 metodun müqayisə radar chart
            cats=["Sürət","Dəqiqlik","Adaptivlik","Şəffaflıq","Tətbiq asanlığı"]
            fig_radar=go.Figure()
            for ad,vals,reng in[
                ("ARO",[85,82,60,90,95],RENG["benzin"]),
                ("E2E",[50,92,88,60,55],RENG["dizel"]),
                ("DT", [90,75,70,85,80],RENG["kukurd"]),
            ]:
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]],theta=cats+[cats[0]],name=ad,
                    fill="toself",line=dict(color=reng,width=2),
                    fillcolor=reng.replace("#","rgba(").replace("","")+"1a" if False else "rgba(0,0,0,0)"))
            fig_radar.update_layout(**QL("ARO / E2E / DT müqayisəsi",hund=320))
            fig_radar.update_layout(
                polar=dict(bgcolor="#1e2540",
                    radialaxis=dict(visible=True,range=[0,100],gridcolor="#283058",
                        tickfont=dict(size=9,family="JetBrains Mono",color="#9aa5cc")),
                    angularaxis=dict(gridcolor="#283058",tickfont=dict(size=11,color="#c5cceb"))))
            st.plotly_chart(fig_radar,width="stretch")
        sp()

        st.markdown("### Vakuum Distilləsi — Sənəd bölmə 1.2")
        st.markdown('<div class="izah">'
            'Atmosfer distilləsindən sonrakı mərhələ. Mazutdan ağır fraksiyalar alınır.<br>'
            '• Vakuum altında qaynama nöqtəsi azalır → yüksək temp. parçalanma baş vermir<br>'
            '• Qazoylu (katalitik krekinq üçün xammal) ayrılır<br>'
            '• Qalıq: qudron (yol örtüyü, izolyasiya materialı)'
            '</div>',unsafe_allow_html=True)

        # Vakuum distilləsi məhsul paylanması
        vakuum_data={"Qazoylu (VGO)":45,"Mazut qalığı":30,"Yağ fraksiyaları":15,"Qudron":10}
        fig_vak=go.Figure(go.Bar(
            x=list(vakuum_data.keys()),y=list(vakuum_data.values()),
            marker=dict(color=[RENG["benzin"],RENG["dizel"],RENG["kerosin"],RENG["anomal"]],
                opacity=.9,line=dict(color="#1a1f35",width=.5)),
            text=[f"{v}%" for v in vakuum_data.values()],textposition="outside",
            textfont=dict(size=11,family="JetBrains Mono",color="#edf1fc"),
            hovertemplate="%{x}: %{y}%<extra></extra>"))
        fig_vak.update_layout(**QL("Vakuum distilləsi — məhsul paylanması (tipik)",hund=260,legend=False))
        fig_vak.update_yaxes(title_text="Faiz (%)",range=[0,55])
        st.plotly_chart(fig_vak,width="stretch")

        st.info("Layihədə vakuum distilləsi mərhələsi 'qalıq' fraksiya kimi sadələşdirilmişdir. "
                "Atmosfer distilləsinin çıxışı: benzin + dizel + kerosin + qalıq (mazut/qudron).")

#
# SƏHİFƏ 4 — OPTİMALLAŞDIRMA
#
elif sehife==" Optimallaşdırma":
    st.markdown("## NSGA-II Çoxhədəfli Optimallaşdırma")
    st.markdown(
        "NSGA-II — çoxhədəfli optimallaşdırma üçün dünya standartı genetik alqoritm. "
        "İki hədəf eyni vaxtda: F* maksimumlaşdır (məhsul gəliri), E minimumlaşdır (enerji).")
    st.markdown('<div class="izah"><b>Pareto frontu:</b> Bir hədəfi yaxşılaşdırmaq digərini pisləşdirirsə, '
        'bu həllər Pareto-bərabərdir. Front kompromis həllərin məcmusudur — '
        'operator enerji vs. gəlir arasında seçim edir.</div>',unsafe_allow_html=True)
    sp()

    if st.button(" NSGA-II İşlət",type="primary",
                 help="Yan paneldəki parametrlərlə optimallaşdırmanı yenidən başladır"):
        st.cache_data.clear()

    prog_container=st.empty()
    with st.spinner(f"NSGA-II: {pop_olcu} fərd × {nesl_sayi} nəsil hesablanır…"):
        bar=prog_container.progress(0,"NSGA-II başlanır…")
        for i in range(5):
            time.sleep(0.08)
            bar.progress((i+1)*15,f"Nəsil {(i+1)*nesl_sayi//5}/{nesl_sayi} işlənir…")
        opt=optimal_yukle(alfa,beta,qamma,delta,pop_olcu,nesl_sayi)
        bar.progress(100,"Tamamlandı!")
        time.sleep(0.4)
        prog_container.empty()

    k1,k2,k3,k4=st.columns(4)
    k1.metric("Pareto həll sayı",f"{opt.n_solutions}",help="Dominant Pareto frontundakı həllərin sayı")
    k2.metric("Nəsil sayı",f"{opt.n_generations}",help="NSGA-II-nin icra etdiyi iterasiya sayı")
    k3.metric("Ən yaxşı F*",f"{opt.best_compromise['f_star']:.4f}",help="Kompromis həllin çəkili verim balı")
    k4.metric("Min. enerji",f"{opt.best_compromise['energy_gj_h']:.3f} GS/s",help="Kompromis həllin enerji sərfi")
    sp()

    pc1,pc2=st.columns([1.3,1],gap="medium")
    with pc1:
        bb("Pareto frontu — enerji vs verim kompromisi")
        fig_pf=go.Figure()
        fig_pf.add_trace(go.Scatter(x=opt.pareto_F[:,1],y=-opt.pareto_F[:,0],mode="markers",
            name="Pareto həllər",
            marker=dict(color=-opt.pareto_F[:,0],
                colorscale=[[0,"#283058"],[0.5,"#4a8fd9"],[1,"#d4941c"]],
                size=10,opacity=.9,showscale=True,
                colorbar=dict(title="F*",thickness=12,tickfont=dict(size=9,family="JetBrains Mono",color="#9aa5cc"),
                              title_font=dict(color="#9aa5cc")),
                line=dict(color="#1a1f35",width=.5)),
            hovertemplate="Enerji: %{x:.3f} GS/s<br>F*: %{y:.4f}<extra></extra>"))
        ey=opt.best_compromise
        fig_pf.add_trace(go.Scatter(x=[ey["obj_energy"]],y=[ey["f_star"]],
            mode="markers",name="Ən yaxşı kompromis",
            marker=dict(color=RENG["enyaxsi"],size=18,symbol="star",
                line=dict(color="#edf1fc",width=2)),
            hovertemplate=f"Ən yaxşı<br>Enerji: {ey['obj_energy']:.3f}<br>F*: {ey['f_star']:.4f}<extra></extra>"))
        fig_pf.update_layout(**QL(f"Pareto frontu — {opt.n_solutions} dominant həll",hund=420))
        fig_pf.update_xaxes(title_text="Enerji sərfi (GS/s)")
        fig_pf.update_yaxes(title_text="F* — çəkili verim balı")
        st.plotly_chart(fig_pf,width="stretch")

    with pc2:
        bb("Konvergensiya — alqoritmin öyrənmə əyrisi")
        if opt.history:
            hdf=pd.DataFrame(opt.history)
            fig_cv=make_subplots(rows=2,cols=1,shared_xaxes=True,
                subplot_titles=["F* konvergensiyası","Enerji konvergensiyası"],vertical_spacing=.14)
            fig_cv.add_trace(go.Scatter(x=hdf["generation"],y=hdf["mean_f_star"],
                name="Orta F*",line=dict(color=RENG["dizel"],width=1.8)),row=1,col=1)
            fig_cv.add_trace(go.Scatter(x=hdf["generation"],y=hdf["best_f_star"],
                name="Ən yaxşı F*",line=dict(color=RENG["benzin"],width=1.5,dash="dot")),row=1,col=1)
            fig_cv.add_trace(go.Scatter(x=hdf["generation"],y=hdf["mean_energy"],
                name="Orta enerji",line=dict(color=RENG["enerji"],width=1.8)),row=2,col=1)
            fig_cv.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#1e2540",
                font=dict(family="Inter",color="#9aa5cc",size=10),
                height=420,margin=dict(l=8,r=8,t=36,b=8),
                legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9,color="#9aa5cc"),orientation="h",y=-0.08))
            sub_stil(fig_cv)
            fig_cv.update_yaxes(title_text="F*",row=1,col=1)
            fig_cv.update_yaxes(title_text="GS/s",row=2,col=1)
            fig_cv.update_xaxes(title_text="Nəsil №",row=2,col=1)
            st.plotly_chart(fig_cv,width="stretch")
    sp()

    # ── Pareto: ətraflı analiz tabları ──────────────────────────────
    bb("Pareto frontu ətraflı analizi")
    pf_t1,pf_t2,pf_t3 = st.tabs([" 3D Pareto Sferası"," Radar Müqayisəsi"," Pareto Cədvəli"])

    with pf_t1:
        st.markdown('<div class="izah">3 ox: F* (verim balı) · Enerji · Ümumi Verim %. '
            '<b style="color:#d4941c">Ulduz</b> = ən yaxşı kompromis həll.</div>',unsafe_allow_html=True)
        Yb3=opt.pareto_df["yield_gasoline"].values
        Yd3=opt.pareto_df["yield_diesel"].values
        En3=opt.pareto_df["energy_gj_h"].values
        Fs3=opt.pareto_df["f_star"].values
        fig_3d=go.Figure(go.Scatter3d(
            x=En3,y=opt.pareto_df["total_yield_pct"].values,z=Fs3,
            mode="markers",
            marker=dict(size=5,color=Fs3,colorscale=[[0,"#283058"],[0.5,"#4a8fd9"],[1,"#d4941c"]],
                opacity=0.85,showscale=True,
                colorbar=dict(title="F*",thickness=10,tickfont=dict(size=9,family="JetBrains Mono",color="#9aa5cc"),
                              title_font=dict(color="#9aa5cc"))),
            hovertemplate="Enerji: %{x:.2f}<br>Ümumi verim: %{y:.1f}%<br>F*: %{z:.4f}<extra></extra>"))
        ey3=opt.best_compromise
        fig_3d.add_trace(go.Scatter3d(
            x=[ey3["energy_gj_h"]],y=[ey3["total_yield_pct"]],z=[ey3["f_star"]],
            mode="markers",name="Ən yaxşı",
            marker=dict(size=14,color=RENG["enyaxsi"],symbol="diamond",
                line=dict(color="#edf1fc",width=2)),
            hovertemplate=f"Ən yaxşı kompromis<extra></extra>"))
        fig_3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                bgcolor="#1e2540",
                xaxis=dict(title="Enerji (GS/s)",gridcolor="#283058",tickfont=dict(size=8,color="#9aa5cc")),
                yaxis=dict(title="Ümumi verim (%)",gridcolor="#283058",tickfont=dict(size=8,color="#9aa5cc")),
                zaxis=dict(title="F*",gridcolor="#283058",tickfont=dict(size=8,color="#9aa5cc")),
            ),
            height=480,
            margin=dict(l=0,r=0,t=30,b=0),
            font=dict(family="Inter",color="#9aa5cc",size=10),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color="#9aa5cc")))
        st.plotly_chart(fig_3d,width="stretch")

    with pf_t2:
        st.markdown('<div class="izah">Pareto frontundan seçilmiş 3 həll profili: '
            '<b>Ən yaxşı F*</b> · <b>Ən az enerji</b> · <b>Kompromis</b>.</div>',unsafe_allow_html=True)
        cats_r=["Benzin verimi","Dizel verimi","Kerosin verimi","Enerji (çevrilmiş)","Kükürd çıx.","F*"]
        def _norm_row(row):
            vals=[row["yield_gasoline"],row["yield_diesel"],row["yield_kerosene"],
                  1-(row["energy_gj_h"]-5)/17,row["sulfur_removal_pct"]/100,row["f_star"]]
            return [max(0,min(1,v))*100 for v in vals]
        idx_best_f = opt.pareto_df["f_star"].idxmax()
        idx_best_e = opt.pareto_df["energy_gj_h"].idxmin()
        idx_comp   = opt.best_compromise.name
        fig_rad=go.Figure()
        for label,idx_r,reng in[
            ("Ən yaxşı F*",idx_best_f,RENG["benzin"]),
            ("Ən az enerji",idx_best_e,RENG["dizel"]),
            ("Kompromis",idx_comp,RENG["kukurd"]),
        ]:
            row_r=opt.pareto_df.iloc[idx_r] if isinstance(idx_r,int) else opt.best_compromise
            rv=_norm_row(row_r)
            fig_rad.add_trace(go.Scatterpolar(
                r=rv+[rv[0]],theta=cats_r+[cats_r[0]],name=label,
                line=dict(color=reng,width=2),
                hovertemplate="%{theta}: %{r:.1f}<extra>"+label+"</extra>"))
        fig_rad.update_layout(**QL("Həll profili radar müqayisəsi",hund=400),
            polar=dict(bgcolor="#1e2540",
                radialaxis=dict(visible=True,range=[0,100],gridcolor="#283058",
                    tickfont=dict(size=9,family="JetBrains Mono",color="#9aa5cc")),
                angularaxis=dict(gridcolor="#283058",tickfont=dict(size=10,color="#c5cceb"))))
        st.plotly_chart(fig_rad,width="stretch")

    with pf_t3:
        st.markdown('<div class="izah">Pareto frontundakı bütün dominant həllər. '
            'Sütun başlıqlarına klikləyərək sıralayın.</div>',unsafe_allow_html=True)
        disp_pareto_cols=["furnace_temp","total_yield_pct","yield_gasoline","yield_diesel",
                          "yield_kerosene","energy_gj_h","sulfur_removal_pct","f_star"]
        disp_pareto_cols=[c for c in disp_pareto_cols if c in opt.pareto_df.columns]
        rename_map={"furnace_temp":"Soba temp.","total_yield_pct":"Ümumi verim %",
                    "yield_gasoline":"Benzin %","yield_diesel":"Dizel %",
                    "yield_kerosene":"Kerosin %","energy_gj_h":"Enerji GS/s",
                    "sulfur_removal_pct":"Kükürd çıx %","f_star":"F*"}
        pshow=opt.pareto_df[disp_pareto_cols].copy()
        pshow.columns=[rename_map.get(c,c) for c in pshow.columns]
        pshow.index=range(1,len(pshow)+1)
        fmt_p={c:"{:.2f}" for c in pshow.columns}
        if "F*" in pshow.columns: fmt_p["F*"]="{:.4f}"
        st.dataframe(pshow.style.format(fmt_p)
            .background_gradient(subset=["F*"] if "F*" in pshow.columns else [],cmap="YlOrRd")
            .background_gradient(subset=["Enerji GS/s"] if "Enerji GS/s" in pshow.columns else [],cmap="Blues_r"),
            width="stretch",height=380)
    sp()

    bb("Həssaslıq analizi — parametrin verim/enerji üzərindəki təsiri")
    st.markdown('<div class="izah">Bir parametr dəyişdirilir, digərləri optimal nöqtədə sabit qalır. '
        'Hansı parametrin ən çox təsirli olduğunu müəyyənləşdirir.</div>',unsafe_allow_html=True)
    hs1,hs2=st.columns([1,3],gap="medium")
    with hs1:
        da=st.selectbox("Dəyişən parametr",VAR_NAMES,
            format_func=lambda x:SENSOR_AD.get(x,x),key="hss",
            help="Bu parametr dəyişdirilir, digərləri optimal nöqtədə sabit qalır")
        di=VAR_NAMES.index(da)
    with hs2:
        baza=opt.pareto_X[np.argmin(np.linalg.norm(opt.pareto_F,axis=1))]
        hdf2=sensitivity_analysis(baza,di)
        xc=VAR_NAMES[di]
        fig_hs=go.Figure()
        for ad,sc,reng in[("Benzin (%)","yield_gasoline",RENG["benzin"]),
                          ("Dizel (%)","yield_diesel",RENG["dizel"]),
                          ("Kerosin (%)","yield_kerosene",RENG["kerosin"])]:
            fig_hs.add_trace(go.Scatter(x=hdf2[xc],y=hdf2[sc],name=ad,
                line=dict(color=reng,width=2),
                hovertemplate=f"<b>{ad}</b>: %{{y:.2f}}%<extra></extra>"))
        fig_hs.add_trace(go.Scatter(x=hdf2[xc],y=hdf2["energy_gj_h"],name="Enerji (GS/s)",
            yaxis="y2",line=dict(color=RENG["enerji"],width=2,dash="dot"),
            hovertemplate="<b>Enerji</b>: %{y:.3f} GS/s<extra></extra>"))
        fig_hs.update_layout(**QL(f"Həssaslıq: {SENSOR_AD.get(da,da)}",hund=320),
            yaxis2=dict(overlaying="y",side="right",gridcolor="#283058",
                zeroline=False,tickfont=dict(size=9,color="#9aa5cc"),title_text="Enerji (GS/s)"))
        fig_hs.update_yaxes(title_text="Verim (%)")
        fig_hs.update_xaxes(title_text=SENSOR_AD.get(da,da))
        st.plotly_chart(fig_hs,width="stretch")

#
# SƏHİFƏ 5 — NƏTİCƏLƏR
#
elif sehife==" Nəticələr":
    st.markdown("## Optimallaşdırma Nəticələri")
    st.markdown("Pareto frontundan ən yaxşı kompromis həll — utopiya nöqtəsinə ən yaxın.")
    sp()

    with st.spinner("Nəticələr hesablanır…"):
        opt=optimal_yukle(alfa,beta,qamma,delta,pop_olcu,nesl_sayi)
    ey=opt.best_compromise

    bb("Optimal iş nöqtəsi — KKT şərtlərini ödəyən ən yaxşı həll")
    n1,n2,n3,n4,n5,n6=st.columns(6)
    n1.metric("Soba temp.", f"{ey['furnace_temp']:.1f} °C", help="Optimal soba çıxış temperaturu")
    n2.metric("Benzin verimi", f"{ey['yield_gasoline']:.1f} %", help="Optimal benzin fraksiyası")
    n3.metric("Dizel verimi", f"{ey['yield_diesel']:.1f} %", help="Optimal dizel fraksiyası")
    n4.metric("Ümumi verim", f"{ey['total_yield_pct']:.1f} %", help="Bütün maye fraksiyaların cəmi")
    n5.metric("Enerji", f"{ey['energy_gj_h']:.3f} GS/s", help="Minimumlaşdırılan hədəf")
    n6.metric("F* balı", f"{ey['f_star']:.4f}", help="Maksimumlaşdırılan çəkili verim balı")
    sp()

    # Optimal həll highlight kartı
    dv=ey.get("total_yield_pct",0)-xulase["avg_total_yield"]
    de=xulase["avg_energy"]-ey.get("energy_gj_h",0)
    il=de*8760
    st.markdown(
        f'<div class="optimal-card">'
        f'<h4 style="color:#f0ac30;margin:0 0 .5rem;font-size:1rem">Optimallaşdırma nəticəsi</h4>'
        f'<div style="display:flex;gap:2rem;flex-wrap:wrap">'
        f'<div><span style="color:#9aa5cc;font-size:.75rem">VERIM ARTIMI</span><br>'
        f'<span style="color:#edf1fc;font-size:1.3rem;font-family:JetBrains Mono;font-weight:700">+{dv:.2f}%</span></div>'
        f'<div><span style="color:#9aa5cc;font-size:.75rem">ENERJİ QƏNAƏTİ</span><br>'
        f'<span style="color:#edf1fc;font-size:1.3rem;font-family:JetBrains Mono;font-weight:700">{de:.3f} GS/s</span></div>'
        f'<div><span style="color:#9aa5cc;font-size:.75rem">İLLİK QƏNAƏT</span><br>'
        f'<span style="color:#edf1fc;font-size:1.3rem;font-family:JetBrains Mono;font-weight:700">{il:,.0f} GS/il</span></div>'
        f'<div><span style="color:#9aa5cc;font-size:.75rem">PARETO HƏLL</span><br>'
        f'<span style="color:#edf1fc;font-size:1.3rem;font-family:JetBrains Mono;font-weight:700">{opt.n_solutions} həll</span></div>'
        f'</div></div>',unsafe_allow_html=True)
    sp()

    rc1,rc2=st.columns([1.2,1.8],gap="medium")
    with rc1:
        bb("Optimal vs Cari radar müqayisəsi")
        st.markdown('<div class="izah" style="font-size:.75rem"> Optimal · Cari. '
            'Daha böyük sahə = daha yaxşı performans.</div>',unsafe_allow_html=True)
        rk=["Benzin %","Dizel %","Kerosin %","Kükürd çıx.%","Enerji sərfiy."]
        ov=[ey["yield_gasoline"],ey["yield_diesel"],ey["yield_kerosene"],
            ey["sulfur_removal_pct"],100-(ey["energy_gj_h"]/22.0)*100]
        cv=[xulase["avg_yield_gasoline"],xulase["avg_yield_diesel"],xulase["avg_yield_kerosene"],
            xulase["avg_sulfur_removal"],100-(xulase["avg_energy"]/22.0)*100]
        fig_rad=go.Figure()
        for ad,vals,reng,dolu in[
            ("Optimal",ov,RENG["benzin"],"rgba(212,148,28,.12)"),
            ("Cari",cv,RENG["dizel"],"rgba(74,143,217,.10)"),
        ]:
            fig_rad.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=rk+[rk[0]],
                name=ad,fill="toself",fillcolor=dolu,line=dict(color=reng,width=2.5),
                hovertemplate=f"<b>{ad}</b>: %{{r:.1f}}<extra></extra>"))
        fig_rad.update_layout(**QL("Optimal vs Cari profil",hund=430))
        fig_rad.update_layout(
            polar=dict(
                bgcolor="#1e2540",
                domain=dict(x=[0.05,0.95],y=[0.05,0.95]),
                radialaxis=dict(
                    visible=True,range=[0,100],gridcolor="#283058",
                    tickfont=dict(size=10,family="JetBrains Mono",color="#9aa5cc"),
                    tickvals=[20,40,60,80,100],
                    gridwidth=1,linewidth=0,
                ),
                angularaxis=dict(
                    gridcolor="#283058",gridwidth=1,
                    tickfont=dict(size=13,color="#c5cceb",family="Inter"),
                    rotation=90,direction="clockwise",
                ),
            ),
        )
        fig_rad.update_layout(margin=dict(l=30,r=30,t=50,b=30))
        st.plotly_chart(fig_rad,width="stretch")

    with rc2:
        bb("Tam Pareto frontu — sıralı optimal həllər")
        gs=["furnace_temp","column_pressure","flow_rate",
            "yield_gasoline","yield_diesel","yield_kerosene",
            "total_yield_pct","energy_gj_h","sulfur_removal_pct","f_star"]
        mv=[s for s in gs if s in opt.pareto_df.columns]
        sc2="f_star" if "f_star" in opt.pareto_df.columns else opt.pareto_df.columns[-1]
        ct=opt.pareto_df[mv].sort_values(sc2,ascending=False).reset_index(drop=True)
        st.dataframe(ct.style.format("{:.3f}").background_gradient(subset=[sc2],cmap="YlOrRd"),
                     width="stretch",height=390)
    sp()

    bb("Metodologiya xülasəsi — sənəddən koda")
    m1,m2=st.columns(2,gap="medium")
    with m1:
        st.markdown('<div class="izah">'
            '<b>Riyazi model (sənəd II fəsil):</b><br>'
            '• F* = α·Ybenz + β·Ydiz + γ·Yker − δ·E (tənlik 1)<br>'
            '• KKT şərtləri: g(x)≤0, h(x)=0 (tənliklər 6–13)<br>'
            '• Balans tənlikləri: kütlə (3), enerji (4)<br>'
            '• Temperatur profili: T(z)=T_alt−k·z (5)<br>'
            '• Reqressiya: Y=a₁T+a₂P+a₃F+a₄+ε</div>',unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="izah">'
            '<b>Alqoritm (sənəd III fəsil):</b><br>'
            '• NSGA-II — Non-dominated Sorting Genetic Algorithm II<br>'
            '• ARO — Ardıcıl Optimallaşdırma (surrogat model)<br>'
            '• IQR + Isolation Forest anomaliya aşkarlanması<br>'
            '• Min-Maks normallaşdırma: [0,1] aralığı<br>'
            '• Pareto frontu — kompromis həllərin məcmusu</div>',unsafe_allow_html=True)
    sp()

    st.download_button(" Pareto Frontunu Yüklə (CSV)",
        data=opt.pareto_df.to_csv(index=False),
        file_name="pareto_frontu_nsga2.csv",mime="text/csv",
        help="Bütün Pareto həllərini CSV formatında yükləyin")


#
# AI KÖMƏKÇİ — Premium chat panel
#

# CSS override — chat input, buttons, form
st.markdown("""
<style>
/* Quick action buttons — SaaS premium style */
.quick-action-btn > button {
    background: #111828 !important;
    color: #6878a8 !important;
    border: 1px solid #283058 !important;
    border-radius: 8px !important;
    padding: .3rem .65rem !important;
    font-size: .73rem !important;
    font-weight: 500 !important;
    transition: all .18s ease !important;
    letter-spacing: .01em !important;
    box-shadow: none !important;
}
.quick-action-btn > button:hover {
    background: rgba(212,148,28,.08) !important;
    border-color: rgba(212,148,28,.5) !important;
    color: #d4941c !important;
    box-shadow: 0 0 0 1px rgba(212,148,28,.15) !important;
}
.quick-action-active > button {
    background: rgba(212,148,28,.14) !important;
    border-color: #d4941c !important;
    color: #f0ac30 !important;
    box-shadow: 0 0 0 1px rgba(212,148,28,.2) !important;
}

/* Chat text input — dark */
[data-testid="stChatInput"] textarea,
.stChatInput textarea,
div[data-testid="stTextInput"] input {
    background: #151e34 !important;
    color: #edf1fc !important;
    border: 1px solid #283058 !important;
    border-radius: 10px !important;
    font-size: .84rem !important;
    caret-color: #d4941c !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #4a5888 !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #d4941c !important;
    box-shadow: 0 0 0 2px rgba(212,148,28,.15) !important;
}

/* Send button — polished */
[data-testid="stFormSubmitButton"] > button {
    background: #d4941c !important;
    color: #141828 !important;
    border: none !important;
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
    padding: 0 !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    min-width: 42px !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all .2s ease !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: #f0ac30 !important;
    transform: scale(1.1) !important;
    box-shadow: 0 0 14px rgba(212,148,28,.45) !important;
}

/* Typing animation */
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
.typing-cursor { animation: blink 0.8s infinite; color: #d4941c; }

@keyframes typing-dots {
    0% { content: ""; }
    33% { content: ""; }
    66% { content: ""; }
    100% { content: ""; }
}
.typing-indicator::before {
    content: "";
    animation: typing-dots 1.2s infinite;
    font-size: .7rem;
    color: #d4941c;
    letter-spacing: 3px;
}

/* Micro-animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.chat-msg-user, .chat-msg-ai {
    animation: fadeInUp .2s ease;
}

/* Metric card glow on hover — already defined above, add pulse */
@keyframes subtle-pulse {
    0%,100% { box-shadow: 0 2px 12px rgba(0,0,0,.3); }
    50% { box-shadow: 0 4px 20px rgba(212,148,28,.12); }
}
[data-testid="metric-container"]:hover {
    animation: none;
}

/* Sidebar button smooth transitions */
section[data-testid="stSidebar"] .stButton>button {
    transition: background .15s ease, border-color .15s ease, color .15s ease !important;
}

/* Nav radio smooth */
div[data-testid="stRadio"] label {
    transition: all .15s ease !important;
}

/* Capability tags — subtle chips */
.cap-tag {
    display: inline-block;
    background: rgba(26,32,56,.9);
    border: 1px solid #283058;
    color: #4a5888;
    border-radius: 6px;
    padding: .14rem .5rem;
    font-size: .65rem;
    font-weight: 500;
    margin: .1rem .08rem;
    letter-spacing: .01em;
}

/* Info badges */
.info-badge {
    display: inline-flex;
    align-items: center;
    gap: .3rem;
    background: rgba(255,255,255,.04);
    border: 1px solid #283058;
    border-radius: 8px;
    padding: .22rem .6rem;
    font-size: .71rem;
    color: #6878a8;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# Session state
for k, v in [("chat_aciq", False), ("mesajlar", []),
             ("chatbot", None), ("opt_cache", None),
             ("son_latency", 0.0),
             ("tts_aktiv", True), ("tts_ses", "nova"), ("tts_surat", 1.0),
             ("is_generating", False), ("pending_audio", {})]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.chatbot is None:
    st.session_state.chatbot = chatbot_yarat()

try:
    if st.session_state.opt_cache is None:
        st.session_state.opt_cache = optimal_yukle(alfa, beta, qamma, delta, pop_olcu, nesl_sayi)
except Exception:
    pass

try:
    _aktiv = sehife.replace(" ","").replace(" ","").replace(" ","").replace(" ","").replace(" ","")
    _kontekst = kontekst_yarat(
        xulase=xulase,
        opt_netice=st.session_state.opt_cache,
        req_goster=req_goster,
        aktiv_sehife=_aktiv,
    )
except Exception:
    _kontekst = "Sistem məlumatları yüklənir..."

st.markdown("---")

# Toggle düyməsi — secondary outlined style
c_toggle, _ = st.columns([1.2, 4.8])
with c_toggle:
    lbl = " Bağla" if st.session_state.chat_aciq else "MeloSense"
    st.markdown("""
<style>
div[data-testid="stButton"][key="chat_toggle"] > button,
button[kind="secondary"]#chat_toggle {
    background: transparent !important;
    border: 1px solid #334070 !important;
    color: #6878a8 !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    padding: .3rem 1rem !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
</style>""", unsafe_allow_html=True)
    if st.button(lbl, key="chat_toggle", width="stretch"):
        st.session_state.chat_aciq = not st.session_state.chat_aciq
        st.rerun()

if not st.session_state.chat_aciq:
    st.stop()

#
# CHAT PANEL
#

# Header
st.markdown("""
<div style="background:linear-gradient(135deg,#161d30,#1e2848);border:1px solid #283058;
     border-radius:14px 14px 0 0;padding:.75rem 1.1rem .65rem;margin-bottom:0">
  <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.45rem">
    <div style="width:8px;height:8px;border-radius:50%;background:#2ec98a;flex-shrink:0;
         box-shadow:0 0 6px rgba(46,201,122,.5)"></div>
    <div>
      <div style="font-size:.9rem;font-weight:700;color:#edf1fc"> MeloSense — Proses İntelligensiyası</div>
      <div style="font-size:.69rem;color:#4a5888;margin-top:.1rem">Neft emalı analitik sistemi · Canlı sensor məlumatları əsasında işləyir</div>
    </div>
  </div>
  <div style="display:flex;gap:.3rem;flex-wrap:wrap;padding-left:1.4rem">
    <span class="cap-tag">Pareto analizi</span>
    <span class="cap-tag">Proses optimallaşdırma</span>
    <span class="cap-tag">Anomaliya aşkarlanması</span>
  </div>
</div>""", unsafe_allow_html=True)

# TTS İdarəetmə Paneli
with st.expander("Səs parametrləri", expanded=False):
    _tts_c1, _tts_c2, _tts_c3 = st.columns([1.2, 1.8, 1.5])
    with _tts_c1:
        st.session_state.tts_aktiv = st.toggle(
            "Səsli cavab",
            value=st.session_state.get("tts_aktiv", True),
            help="AI cavabını avtomatik səsləndirmək üçün",
        )
    with _tts_c2:
        _ses_secimler = {
            "nova — Qadın, axıcı (tövsiyə)": "nova",
            "shimmer — Qadın, mülayim": "shimmer",
            "alloy — Neytral, analitik": "alloy",
            "onyx — Kişi, dərin": "onyx",
            "echo — Kişi, aydın": "echo",
            "fable — Qadın, ifadəli": "fable",
        }
        _ses_label = st.selectbox(
            "Səs seçimi",
            options=list(_ses_secimler.keys()),
            index=0,
            disabled=not st.session_state.tts_aktiv,
        )
        st.session_state.tts_ses = _ses_secimler[_ses_label]
    with _tts_c3:
        st.session_state.tts_surat = st.slider(
            "Danışıq sürəti",
            min_value=0.75, max_value=1.25,
            value=st.session_state.get("tts_surat", 1.0),
            step=0.05, format="%.2f",
            disabled=not st.session_state.tts_aktiv,
            help="1.0 = normal · 0.85 = daha aydın · 1.15 = sürətli",
        )

# API key yoxlaması
if st.session_state.chatbot is None:
    st.markdown("""
<div style="background:#1a0e0e;border:1px solid #5a2020;border-radius:0 0 12px 12px;
     padding:1rem 1.2rem;text-align:center">
  <div style="font-size:.88rem;color:#f87171;font-weight:600"> MeloSense — OpenAI API açarı tapılmadı</div>
  <div style="font-size:.78rem;color:#9aa5cc;margin-top:.4rem">
    Layihə qovluğunda <code>.env</code> faylı yaradın:
  </div>
</div>""", unsafe_allow_html=True)
    st.code("OPENAI_API_KEY=sk-proj-xxxxxxxxxx", language="bash")
    st.stop()

# Tez analiz düymələri
st.markdown("<div style='padding:.6rem 0 .3rem;font-size:.76rem;color:#6878a8;font-weight:600'> Tez analiz</div>",
            unsafe_allow_html=True)
tez_sorgu = None
_tez_labels = [(" Pareto","pareto"),(" Anomaliya","anomaliya"),
               (" Tövsiyə","tövsiyə"),(" Tənliklər","tənlik"),(" Müqayisə","müqayisə")]
_cols = st.columns(len(_tez_labels))
for col_q, (lbl_q, key_q) in zip(_cols, _tez_labels):
    with col_q:
        st.markdown('<div class="quick-action-btn">', unsafe_allow_html=True)
        if st.button(lbl_q, key=f"qa_{key_q}", width="stretch",
                     disabled=st.session_state.get("is_generating", False)):
            tez_sorgu = key_q
        st.markdown('</div>', unsafe_allow_html=True)

if tez_sorgu:
    sorgu_metni = {
        "pareto": "Cari Pareto frontunu şərh et. Niyə bu həll optimal seçildi?",
        "anomaliya": "Aşkar edilmiş anomaliyaları analiz et. Bu səviyyə normaldır?",
        "tövsiyə": "Verim artırmaq üçün konkret 3-5 parametr tövsiyəsi ver.",
        "tənlik": "Məqsəd funksiyasını (tənlik 1) canlı dəyərlərlə izah et.",
        "müqayisə": "Cari vəziyyəti optimal həll ilə müqayisə et.",
    }[tez_sorgu]
    st.session_state.mesajlar.append({"role": "user", "content": sorgu_metni})
    st.rerun()

# Mesaj tarixi
chat_box = st.container(height=380)
with chat_box:
    if not st.session_state.mesajlar:
        st.markdown("""
<div style="text-align:center;padding:2.5rem 1rem">
  <div style="font-size:2rem;margin-bottom:.5rem"></div>
  <div style="font-size:.88rem;color:#9aa5cc;font-weight:600">MeloSense aktiv. Proses parametrləri, anomaliyalar və optimallaşdırma haqqında soruşun.</div>
  <div style="font-size:.76rem;color:#4a5888;margin-top:.3rem">Canlı dashboard məlumatlarınızı görürəm.</div>
</div>""", unsafe_allow_html=True)
    else:
        ai_idx = 0  # hər AI mesajına unikal key
        # Yalnız bir audio aktiv (autoplay) ola bilər — hansı key olduğunu izlə
        _active_audio_key = st.session_state.get("active_audio_key", None)

        for i, msg in enumerate(st.session_state.mesajlar):
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='background:#d4941c;color:#141828;border-radius:14px 14px 4px 14px;"
                    f"padding:.6rem .9rem;font-size:.84rem;font-weight:500;margin:.3rem 0 .3rem 3rem;"
                    f"line-height:1.5'> {msg['content']}</div>",
                    unsafe_allow_html=True)
            else:
                # AI cavabı + düyməsi
                st.markdown(
                    f"<div style='background:#151e34;color:#c5cceb;border:1px solid #283058;"
                    f"border-radius:14px 14px 14px 4px;padding:.65rem .95rem;font-size:.84rem;"
                    f"margin:.3rem 3rem .2rem 0;line-height:1.6'> {msg['content']}</div>",
                    unsafe_allow_html=True)

                # Səsləndir düyməsi + audio player
                _btn_key = f"play_{i}_{ai_idx}"
                _audio_key = f"audio_{i}_{ai_idx}"
                _is_this_active = (_active_audio_key == _audio_key)
                btn_col, _ = st.columns([1, 5])
                with btn_col:
                    if st.button(
                        " Bağla" if _is_this_active else " Səsləndir",
                        key=_btn_key,
                        width="stretch",
                        disabled=st.session_state.get("is_generating", False),
                    ):
                        if _is_this_active:
                            # Bu audio-nu dayandır
                            st.session_state["active_audio_key"] = None
                            st.rerun()
                        else:
                            if st.session_state.chatbot:
                                # Əgər bu mesaj üçün audio hələ yoxdursa yarat
                                if st.session_state.get(_audio_key) is None:
                                    with st.spinner("Səs hazırlanır…"):
                                        audio_bytes, motor = st.session_state.chatbot.ses_yarat(
                                            metn=msg["content"],
                                            ses=st.session_state.get("tts_ses", TTS_ELEVENLABS_VOICE),
                                            surət=st.session_state.get("tts_surat", 1.0),
                                        )
                                    st.session_state[_audio_key] = audio_bytes
                                    st.session_state[f"motor_{i}_{ai_idx}"] = motor
                                # Yalnız bu key-i aktiv et — digər bütün audio-lar autoplay olmaz
                                st.session_state["active_audio_key"] = _audio_key
                                st.rerun()

                # Yalnız aktiv olan audio-nu autoplay ilə göstər
                if st.session_state.get(_audio_key) and _is_this_active:
                    st.audio(st.session_state[_audio_key], format="audio/mp3", autoplay=True)
                    _motor = st.session_state.get(f"motor_{i}_{ai_idx}", "")
                    if _motor == "elevenlabs":
                        st.markdown(
                            '<span class="info-badge" style="color:#2ec98a"> ElevenLabs</span>',
                            unsafe_allow_html=True)
                    elif _motor == "openai":
                        st.markdown(
                            '<span class="info-badge" style="color:#d4941c"></span>',
                            unsafe_allow_html=True)

                ai_idx += 1

# Input formu
_generating = st.session_state.get("is_generating", False)
with st.form("chat_form", clear_on_submit=True):
    inp_col, btn_col = st.columns([9, 1])
    with inp_col:
        user_input = st.text_input(
            "msg",
            placeholder=" Cavab gəlir, gözləyin…" if _generating else "Məsələn: Niyə benzin verimi aşağıdır?",
            label_visibility="collapsed",
            disabled=_generating,
        )
    with btn_col:
        göndər = st.form_submit_button("➤", width="stretch", disabled=_generating)

# Stream cavab
def stream_cavab(mesajlar, kontekst):
    """Stream cavab — sürətli, bloklaşdırılmış, autoplay yoxdur."""
    tam = ""
    yazi_yeri = st.empty()
    bashlama = time.time()

    # Generating flag — formu bloklayır
    st.session_state.is_generating = True

    # Typing indicator
    yazi_yeri.markdown(
        "<div style='background:#151e34;color:#d4941c;border:1px solid #283058;"
        "border-radius:14px 14px 14px 4px;padding:.65rem .95rem;font-size:.84rem;"
        "margin:.3rem 3rem .3rem 0'>"
        " <span class='typing-indicator'></span></div>",
        unsafe_allow_html=True)

    try:
        for parca in st.session_state.chatbot.cavab_ver_stream(mesajlar, kontekst):
            tam += parca
            yazi_yeri.markdown(
                f"<div style='background:#151e34;color:#c5cceb;border:1px solid #283058;"
                f"border-radius:14px 14px 14px 4px;padding:.65rem .95rem;font-size:.84rem;"
                f"margin:.3rem 3rem .3rem 0;line-height:1.6'>"
                f" {tam}<span class='typing-cursor'></span></div>",
                unsafe_allow_html=True)
    except Exception:
        tam = " AI cavabı alınarkən problem yarandı. Yenidən cəhd edin."
        yazi_yeri.markdown(
            f"<div style='background:#1a0e0e;color:#f87171;border:1px solid #5a2020;"
            f"border-radius:14px;padding:.65rem .95rem;font-size:.84rem;"
            f"margin:.3rem 3rem .3rem 0'>{tam}</div>",
            unsafe_allow_html=True)

    st.session_state.son_latency = round(time.time() - bashlama, 2)
    st.session_state.is_generating = False
    yazi_yeri.empty()
    return tam

# Form göndərildi
if göndər and user_input.strip():
    st.session_state.mesajlar.append({"role": "user", "content": user_input.strip()})
    with chat_box:
        cavab = stream_cavab(st.session_state.mesajlar, _kontekst)
    st.session_state.mesajlar.append({"role": "assistant", "content": cavab})
    st.rerun()

# Tez sorğu gəldi (formsuz)
elif (st.session_state.mesajlar and
      st.session_state.mesajlar[-1]["role"] == "user"):
    with chat_box:
        cavab = stream_cavab(st.session_state.mesajlar, _kontekst)
    st.session_state.mesajlar.append({"role": "assistant", "content": cavab})
    st.rerun()

# Footer
f1, f2, f3, f4, f5 = st.columns(5)
with f1:
    if st.button(" Sil", key="clear_chat"):
        # Mesajları sıfırla
        st.session_state.mesajlar = []
        # Audio state-lərini tam təmizlə — yenidən yükləndikdə köhnə
        # audio_i_ai_idx key-ləri qalmasın, əks halda yeni mesajlarda
        # index çakışması yaranır və səsləndirmə işləmir
        audio_keys = [k for k in st.session_state
                      if k.startswith("audio_") or k.startswith("motor_")]
        for k in audio_keys:
            del st.session_state[k]
        st.session_state["active_audio_key"] = None
        st.rerun()
with f2:
    st.markdown(f'<span class="info-badge"> {len(st.session_state.mesajlar)} mesaj</span>',
                unsafe_allow_html=True)
with f3:
    st.markdown('<span class="info-badge"> MeloSense</span>', unsafe_allow_html=True)
with f4:
    lat = st.session_state.get("son_latency", 0)
    if lat > 0:
        st.markdown(f'<span class="info-badge"> {lat}s</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="info-badge"> Hazır</span>', unsafe_allow_html=True)
with f5:
    if st.session_state.get("tts_aktiv", True) and st.session_state.get("chatbot"):
        _limit = st.session_state.chatbot.el_limit_goster()
        if _limit.get("aktiv") and _limit.get("qalan") is not None:
            _qalan = _limit["qalan"]
            _faiz = _limit["faiz"]
            if _limit.get("exhausted"):
                st.markdown('<span class="info-badge" style="color:#d4941c"> OpenAI fallback</span>', unsafe_allow_html=True)
            elif _qalan < 1000:
                st.markdown(f'<span class="info-badge" style="color:#d45060"> EL: {_qalan} simvol</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="info-badge" style="color:#2ec98a"> EL: {_qalan:,} simvol</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="info-badge"> TTS aktiv</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="info-badge"> Sessiz</span>', unsafe_allow_html=True)