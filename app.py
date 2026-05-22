"""
Neft Emalı Optimallaşdırma Sistemi
Modul : app.py
Məqsəd: Streamlit Dashboard — bölmə 2.3 + 2.4
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
    EconomicWeights, VAR_NAMES, VAR_LABELS, X_LOWER, X_UPPER,
)
from chatbot import chatbot_yarat, kontekst_yarat, TTS_ELEVENLABS_VOICE

st.set_page_config(
    page_title="Neft Emalı Optimallaşdırma",
    page_icon=None, layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

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

.stApp{background:var(--bg)!important;font-family:'Inter',sans-serif;color:var(--t1);}
.stApp h1,.stApp h2,.stApp h3{color:var(--t0)!important;font-weight:700!important;}
.stApp h2{font-size:1.75rem!important;letter-spacing:-.025em;line-height:1.2!important;}
.stApp h3{font-size:1.15rem!important;letter-spacing:-.01em;}
.stApp p{color:var(--t1);line-height:1.65;font-size:.88rem;}
.stApp b,.stApp strong{color:var(--t0)!important;}
.stApp code{background:var(--c2);color:var(--am2);padding:.1rem .4rem;
    border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:.81rem;}
.block-container{
    padding-left:2.5rem!important;padding-right:2.5rem!important;
    padding-top:1.5rem!important;max-width:100%!important;width:100%!important;
}

section[data-testid="stSidebar"]{background:var(--c1)!important;border-right:1px solid var(--bd)!important;min-width:230px!important;max-width:260px!important;width:245px!important;}
[data-testid="collapsedControl"]{display:none!important;}
section[data-testid="stSidebar"][aria-expanded="false"]{min-width:230px!important;width:245px!important;transform:none!important;visibility:visible!important;}
section[data-testid="stSidebar"] *{font-family:'Inter',sans-serif;}
section[data-testid="stSidebar"] p{color:var(--t1)!important;font-size:.82rem;line-height:1.5;}
section[data-testid="stSidebar"] label{color:var(--t1)!important;font-size:.83rem!important;font-weight:500!important;}
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stCaptionContainer p{color:var(--t2)!important;font-size:.75rem!important;}
[data-testid="stThumbValue"]{color:var(--am2)!important;font-family:'JetBrains Mono',monospace!important;font-size:.85rem!important;font-weight:600!important;}
[data-testid="stSlider"] [role="slider"]{background:var(--am2)!important;border-color:var(--am2)!important;box-shadow:var(--sh3)!important;}

.sb-section{border:1px solid var(--bd);border-radius:var(--rs);background:var(--c1);margin-bottom:.6rem;padding:.6rem .8rem .7rem;}
.sb-section-title{font-size:.84rem;font-weight:700;color:var(--t1);margin-bottom:.5rem;padding-bottom:.35rem;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:.4rem;}

[data-testid="metric-container"]{
    background:linear-gradient(145deg,var(--c1),#1e2540)!important;
    border:1px solid var(--bd);border-radius:var(--r);
    padding:1.4rem 1.5rem 1.25rem;min-height:130px;
    box-shadow:var(--sh),inset 0 1px 0 rgba(255,255,255,.04);
    transition:border-color .2s,transform .18s,box-shadow .2s;
    display:flex;flex-direction:column;justify-content:space-between;position:relative;overflow:hidden;
}
[data-testid="metric-container"]::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--am),transparent);opacity:0;transition:opacity .2s;}
[data-testid="metric-container"]:hover{border-color:var(--am);transform:translateY(-4px);box-shadow:var(--sh2),var(--sh3);}
[data-testid="metric-container"]:hover::before{opacity:1;}
[data-testid="metric-container"] label,[data-testid="stMetricLabel"] p,[data-testid="stMetricLabel"]{color:var(--t2)!important;font-size:.72rem!important;font-weight:700!important;letter-spacing:.1em;text-transform:uppercase;opacity:1!important;}
[data-testid="stMetricValue"]{color:var(--t0)!important;font-size:1.75rem!important;font-weight:700!important;font-family:'JetBrains Mono',monospace!important;letter-spacing:-.02em;line-height:1.1!important;}
[data-testid="stMetricDelta"]{font-size:.77rem!important;font-weight:500!important;}

.bb{font-size:.65rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:var(--t3);padding-bottom:.42rem;border-bottom:1px solid var(--bd);margin:1.5rem 0 .9rem 0;display:block;}

.tenlik{background:#161c30;border:1px solid var(--bd);border-left:3px solid var(--am);border-radius:var(--rs);padding:.9rem 1.2rem;font-family:'JetBrains Mono',monospace;font-size:.83rem;color:#eab84a;line-height:1.9;margin:.6rem 0;}
.izah{background:var(--c2);border:1px solid var(--bd);border-left:3px solid var(--bl);border-radius:var(--rs);padding:.7rem 1rem;font-size:.83rem;color:var(--t1);line-height:1.58;margin:.5rem 0;}
.izah b{color:var(--t0)!important;}
.izah ul,.izah li{margin:.1rem 0;padding-left:.2rem;font-size:.82rem;}
.xeb{background:rgba(212,148,28,.1);border:1px solid rgba(212,148,28,.3);border-radius:var(--rm);padding:.55rem .9rem;font-size:.79rem;color:#f0c060;margin:.4rem 0;}
.optimal-card{background:linear-gradient(135deg,rgba(212,148,28,.12),rgba(74,143,217,.08));border:2px solid var(--am);border-radius:var(--r);padding:1rem 1.3rem;box-shadow:0 0 20px rgba(212,148,28,.15);margin:.5rem 0;}
.spacer{height:20px;}
.sbe{font-size:.66rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:var(--t3);margin:.8rem 0 .15rem 0;display:block;}

.stTabs [data-baseweb="tab-list"]{background:var(--c1);border-radius:var(--r) var(--r) 0 0;border:1px solid var(--bd);border-bottom:none;padding:.25rem .25rem 0;gap:2px;}
.stTabs [data-baseweb="tab"]{background:transparent;color:var(--t2)!important;font-size:.83rem;font-weight:500;padding:.46rem 1rem;border:none!important;border-radius:var(--rm) var(--rm) 0 0;}
.stTabs [aria-selected="true"]{background:var(--c2)!important;color:var(--t0)!important;border-bottom:2px solid var(--am)!important;font-weight:600!important;}

.stButton>button{background:var(--am)!important;color:#141828!important;font-weight:700!important;border:none!important;border-radius:var(--rm)!important;padding:.55rem 1.8rem!important;font-size:.87rem!important;letter-spacing:.02em;}
.stButton>button:hover{background:var(--am2)!important;box-shadow:var(--sh3)!important;}

section[data-testid="stSidebar"] .stButton>button{background:var(--c2)!important;color:var(--t1)!important;font-weight:600!important;font-size:.81rem!important;border:1px solid var(--bd)!important;border-radius:var(--rm)!important;padding:.38rem .75rem!important;text-align:left!important;letter-spacing:0!important;margin-bottom:3px;width:100%!important;transition:background .15s,border-color .15s!important;}
section[data-testid="stSidebar"] .stButton>button:hover{background:var(--c3)!important;border-color:var(--am)!important;box-shadow:none!important;}
section[data-testid="stSidebar"] [data-testid="stSlider"]{margin-bottom:.2rem!important;}
section[data-testid="stSidebar"] .stCaption{margin-top:.1rem!important;}

div[data-testid="stRadio"] label{background:var(--c1);border:1px solid var(--bd);border-radius:var(--rm);padding:.38rem .85rem;font-size:.82rem;color:var(--t1)!important;cursor:pointer;transition:all .15s;}
div[data-testid="stRadio"] label:hover{border-color:var(--am);color:var(--t0)!important;}
div[data-testid="stRadio"] label span{color:var(--t1)!important;}

.stDataFrame{border-radius:var(--r)!important;overflow:hidden!important;border:1px solid var(--bd)!important;box-shadow:var(--sh)!important;background:#111828!important;}
[data-testid="stDataFrame"] > div,[data-testid="stDataFrame"] iframe{background:#111828!important;border-radius:var(--r)!important;}

[data-testid="stCheckbox"] label p,[data-testid="stCheckbox"] p{color:var(--t1)!important;font-size:.83rem!important;}
div[data-testid="stSelectbox"]>div>div{background:var(--c2)!important;border-color:var(--bd)!important;color:var(--t0)!important;}
div[data-baseweb="select"]{background:var(--c2)!important;}
.stAlert{background:var(--c2)!important;border:1px solid var(--bd)!important;border-radius:var(--r)!important;color:var(--t1)!important;}
[data-testid="stProgress"] > div > div{background:var(--am)!important;}
.stCaptionContainer p,[data-testid="stCaptionContainer"] p{color:var(--t2)!important;font-size:.75rem!important;}
hr{border-color:var(--bd)!important;margin:.8rem 0!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stToolbar"]{display:none!important;visibility:hidden!important;}
.stDeployButton{display:none!important;visibility:hidden!important;}
[data-testid="stHeader"]{display:none!important;}
section.main > div:first-child{padding-top:0!important;}
iframe[title="keyboard_double_arrow_right"]{display:none!important;}
span.material-symbols-rounded{display:none!important;}
[class*="keyboard"]{display:none!important;}
.eyeqlp50{display:none!important;}

.chat-panel{background:var(--c1);border:1px solid var(--bd);border-radius:16px;box-shadow:0 8px 40px rgba(0,0,0,.5);overflow:hidden;}
.chat-header{background:linear-gradient(135deg,#1a2038,#212845);border-bottom:1px solid var(--bd);padding:.75rem 1rem;display:flex;align-items:center;gap:.6rem;}
.chat-header-title{font-size:.9rem;font-weight:700;color:var(--t0);}
.chat-header-sub{font-size:.72rem;color:var(--t2);}
.chat-status{width:8px;height:8px;border-radius:50%;background:#2ec98a;box-shadow:0 0 6px rgba(46,201,122,.6);}
.info-badge{font-size:.72rem;color:var(--t3);padding:.15rem .55rem;background:var(--c1);border:1px solid var(--bd);border-radius:20px;}

@keyframes typing-blink{0%,100%{opacity:0}50%{opacity:1}}
.typing-cursor{display:inline-block;width:2px;height:1em;background:var(--am);margin-left:2px;animation:typing-blink 1s infinite;vertical-align:text-bottom;}
</style>
""", unsafe_allow_html=True)

# ── Rəng paleti ────────────────────────────────────────────────────────────
RENG = {
    "benzin":"#d4941c","dizel":"#4a8fd9","kerosin":"#7c8fd4",
    "kukurd":"#2ec98a","enerji":"#d45060","anomal":"#d47830",
    "normal":"#4a8fd9","pareto":"#4a8fd9","enyaxsi":"#d4941c",
    "grid":"#283058","line":"#334070",
}
SENSOR_AD = {
    "furnace_temp":"Soba temp. (°C)","column_pressure":"Kolon təzyiqi (atm)",
    "flow_rate":"Axın sürəti (m³/s)","reflux_ratio":"Refluks nisbəti",
    "feed_temp":"Qidalanma temp. (°C)","h2_pressure":"H₂ təzyiqi (bar)",
    "catalyst_temp":"Katalizator temp. (°C)","crude_density":"Neft sıxlığı (q/sm³)",
    "sulfur_content":"Kükürd miqdarı (%)",
}
HEDAF_AD = {
    "yield_gasoline":"Benzin verimi","yield_diesel":"Dizel verimi",
    "yield_kerosene":"Kerosin verimi","energy_gj_h":"Enerji (GS/s)",
    "sulfur_removal":"Kükürd çıx.","total_yield":"Ümumi verim",
}

def sp(): st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
def bb(t): st.markdown(f"<span class='bb'>{t}</span>", unsafe_allow_html=True)

def QL(title, hund=350, legend=True):
    cfg = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e2540",
        font=dict(family="Inter", color="#9aa5cc", size=10),
        height=hund, margin=dict(l=8, r=8, t=44, b=8),
        title=dict(text=title, font=dict(size=13, color="#c5cceb", family="Inter"), x=0, xref="paper"),
        xaxis=dict(gridcolor="#283058", linecolor="#334070", zeroline=False),
        yaxis=dict(gridcolor="#283058", linecolor="#334070", zeroline=False),
    )
    if legend:
        cfg["legend"] = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9, color="#9aa5cc"), orientation="h", y=-0.22)
    return cfg

def sub_stil(fig):
    fig.update_xaxes(gridcolor="#283058", linecolor="#334070", zeroline=False)
    fig.update_yaxes(gridcolor="#283058", linecolor="#334070", zeroline=False)


# ── Sidebar accordion (state-i qoruyur) ────────────────────────────────────
def accordion(key, label, icon=""):
    btn_key = f"_acc_{key}"
    if btn_key not in st.session_state:
        st.session_state[btn_key] = False
    ico_open = "▾" if st.session_state[btn_key] else "▸"
    if st.button(f"{ico_open} {icon} {label}".strip(), key=f"_accbtn_{key}"):
        st.session_state[btn_key] = not st.session_state[btn_key]
    return st.session_state[btn_key]


# ── Cache ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def melumat_yukle(n):
    return generate_sensor_data(n_samples=n)

@st.cache_data(show_spinner=False)
def boru_yukle(n):
    raw = generate_sensor_data(n_samples=n)
    return run_preprocessing_pipeline(raw)

@st.cache_data(show_spinner=False)
def optimal_yukle(a, b, g, d, e, pop, nesl, _ver="v3"):
    w = EconomicWeights(alpha=a, beta=b, gamma=g, delta=d, epsilon=e)
    return run_nsga2(weights=w, pop_size=pop, n_gen=nesl)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sb-section-title'>Parametrlər</div>", unsafe_allow_html=True)

    span_n = st.markdown("<span class='sbe'>Nümunə sayı</span>", unsafe_allow_html=True)
    n_numune = st.selectbox("n", options=[720, 1440, 2160], index=1, label_visibility="collapsed")
    st.caption("1440 = 24 saat · 1 dəq. interval")

    st.markdown("---")

    # İqtisadi çəkilər — accordion
    if accordion("iq", "İqtisadi çəkilər", ""):
        st.markdown("<div style='font-size:.74rem;color:#6878a8;margin-bottom:.3rem'>F* = α·Yb+β·Yd+γ·Yk−δ·E<br>α+β+γ ≤ 1.0 olmalıdır</div>", unsafe_allow_html=True)
        alfa  = st.slider("α (benzin)",  0.1, 0.6, 0.35, 0.05)
        beta  = st.slider("β (dizel)",   0.1, 0.5, 0.30, 0.05)
        qamma = st.slider("γ (kerosin)", 0.05, 0.4, 0.20, 0.05)
        delta   = st.slider("δ (enerji)",    0.05, 0.3,  0.15, 0.05)
        epsilon = st.slider("ε (kükürd çıx.)", 0.0, 0.2, 0.10, 0.05,
            help="Kükürd çıxarılmasının iqtisadi mükafatı — HDS prosesi")
        cem   = alfa + beta + qamma
        if cem > 1.0:
            st.markdown(f"<div class='xeb'>α+β+γ={cem:.2f} — 1.0 aşılır</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tenlik'>F*={alfa}·Yb+{beta}·Yd+{qamma}·Yk−{delta}·E+{epsilon}·S</div>", unsafe_allow_html=True)
    else:
        alfa, beta, qamma, delta, epsilon = 0.35, 0.30, 0.20, 0.15, 0.10

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # NSGA-II parametrləri — accordion (state qorunur, səhifə keçidini yoxlamır)
    if accordion("nsga", "NSGA-II parametrləri", ""):
        st.markdown("<div style='font-size:.74rem;color:#6878a8;margin-bottom:.3rem'><b style='color:#c5cceb'>Populyasiya</b> — hər nəsildəki həll sayı<br><b style='color:#c5cceb'>Nəsil</b> — öyrənmə dövrü sayı</div>", unsafe_allow_html=True)
        pop_olcu  = st.selectbox("Populyasiya", options=[40,60,80,100,120,150,200], index=3,
            help="Böyük = keyfiyyətli Pareto, uzun hesablama")
        nesl_sayi = st.selectbox("Nəsil sayı", options=[50,80,100,120,150,200,250], index=4,
            help="Çox nəsil = daha yaxşı həll")
        st.caption(f"~{round(pop_olcu*nesl_sayi/5000,1)} san.")
    else:
        pop_olcu, nesl_sayi = 100, 150

    st.markdown("---")
    st.caption("Nuruyeva Məleykə Firuddin q.")
    st.caption("Rəhbər: dos. Ağayev Fərid H. o.")
    st.caption("050634 — Proseslərin avtomatlaşd.")


# ── Naviqasiya ─────────────────────────────────────────────────────────────
sehife = st.radio("nav",
    [" İcmal"," Məlumat Analizi"," Emal Boru Kəməri",
     " Optimallaşdırma"," Nəticələr"],
    horizontal=True, label_visibility="collapsed")
st.markdown("---")

# ── Məlumat yüklə ──────────────────────────────────────────────────────────
_boru_cached = st.session_state.get("_boru_n") == n_numune
_spinner_msg = "İlk açılış: model öyrədilir…" if not _boru_cached else "Məlumat yüklənir…"

with st.spinner(_spinner_msg):
    xam_df  = melumat_yukle(n_numune)
    boru    = boru_yukle(n_numune)
    xulase  = get_operating_summary(xam_df)
    st.session_state["_boru_n"] = n_numune

# Optimal nəticəni bir dəfə hesabla — bütün səhifələr eyni nəticəni istifadə edir
_opt_cache_key = f"opt_{alfa}_{beta}_{qamma}_{delta}_{epsilon}_{pop_olcu}_{nesl_sayi}"
if st.session_state.get("_opt_cache_key") != _opt_cache_key:
    try:
        _opt_result = optimal_yukle(alfa, beta, qamma, delta, epsilon, pop_olcu, nesl_sayi, _ver="v3")
        st.session_state["_opt_result"]    = _opt_result
        st.session_state["_opt_cache_key"] = _opt_cache_key
    except Exception:
        st.session_state["_opt_result"]    = None
        st.session_state["_opt_cache_key"] = _opt_cache_key

_OPT = st.session_state.get("_opt_result")

temiz_df     = boru["clean_df"]
anomaliya_df = boru["anomaly_df"]
req_goster   = boru["regression_metrics"]
X_norm       = boru["X_norm"]
ts           = temiz_df["timestamp"]

# Linear regression əmsalları
lr_coeffs = boru.get("lr_coefficients", boru.get("regression_coeffs", pd.DataFrame()))
# regression_coeffs uyğunluq üçün
req_emsal = boru.get("regression_coeffs", pd.DataFrame())


# ══════════════════════════════════════════════════════════════════════════
# SƏHİFƏ 1 — İCAML
# ══════════════════════════════════════════════════════════════════════════
if sehife == " İcmal":
    st.markdown("## Neft Emalı Optimallaşdırma Sistemi")
    st.markdown(
        "Xam neftin ilkin emalının çoxhədəfli optimallaşdırılması. "
        "NSGA-II genetik alqoritmi: verim ↑ maksimum, enerji ↓ minimum. "
        "Sənəd 2.3 (verilənlər emalı) + 2.4 (alqoritm + proqram).")
    sp()

    bb("Sistem arxitekturası")
    st.markdown("""
<div style="background:#161c30;border:1px solid #283058;border-radius:12px;padding:1.4rem 1.6rem;margin:.5rem 0 1rem;font-family:'Inter',sans-serif">
  <div style="font-size:.65rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:#4a5888;margin-bottom:1rem;border-bottom:1px solid #283058;padding-bottom:.5rem">
    NSGA-II Əsaslı Çoxhədəfli Optimallaşdırma Sistemi (bölmə 2.3 + 2.4)
  </div>
  <div style="display:flex;gap:8px;margin-bottom:8px;align-items:stretch">
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-top:2px solid #4a8fd9;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#4a8fd9;margin-bottom:.2rem">SENSOR VERİLƏNLƏRİ</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">T (°C) · P (atm) · F (m³/s)<br>9 kanal · 1440 nümunə · 1 dəq.</div>
    </div>
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-top:2px solid #6878a8;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#a090d8;margin-bottom:.2rem">FİZİKİ MƏHDUDIYƏTLƏR</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">Soba: 340–400°C<br>Kolon: 1.2–1.5 atm · H₂: 30–60 bar</div>
    </div>
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-top:2px solid #d4941c;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#d4941c;margin-bottom:.2rem">İQTİSADİ ÇƏKİLƏR</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">α·Yb + β·Yd + γ·Yk − δ·E + ε·S<br>Tənlik (1): F* max</div>
    </div>
  </div>
  <div style="text-align:center;color:#4a5888;font-size:1.1rem;margin:2px 0">↓</div>
  <div style="background:#1a2038;border:1px solid #334070;border-radius:8px;padding:.7rem 1rem;margin-bottom:8px">
    <div style="font-size:.72rem;font-weight:700;color:#c5cceb;margin-bottom:.4rem">VERİLƏNLƏRİN EMAL BORU KƏMƏRİ — sənəd bölmə 2.3</div>
    <div style="display:flex;gap:8px">
      <div style="flex:1;background:#212845;border-radius:6px;padding:.4rem .6rem;text-align:center;font-size:.68rem;color:#9aa5cc">
        <span style="color:#d4941c">①</span> Filtrasiya<br><span style="font-size:.62rem;color:#6878a8">interpolasiya · 3σ</span>
      </div>
      <div style="color:#4a5888;align-self:center">→</div>
      <div style="flex:1;background:#212845;border-radius:6px;padding:.4rem .6rem;text-align:center;font-size:.68rem;color:#9aa5cc">
        <span style="color:#d4941c">②</span> Normallaşdırma<br><span style="font-size:.62rem;color:#6878a8">x_norm = (x−min)/(max−min)</span>
      </div>
      <div style="color:#4a5888;align-self:center">→</div>
      <div style="flex:1;background:#212845;border-radius:6px;padding:.4rem .6rem;text-align:center;font-size:.68rem;color:#9aa5cc">
        <span style="color:#d4941c">③</span> Anomaliya<br><span style="font-size:.62rem;color:#6878a8">IQR · Isolation Forest</span>
      </div>
      <div style="color:#4a5888;align-self:center">→</div>
      <div style="flex:1;background:#212845;border-radius:6px;padding:.4rem .6rem;text-align:center;font-size:.68rem;color:#9aa5cc">
        <span style="color:#d4941c">④</span> Xətti Reqressiya<br><span style="font-size:.62rem;color:#6878a8">Y=a₁T+a₂P+a₃F+ε | E+T<sub>f</sub></span>
      </div>
    </div>
  </div>
  <div style="text-align:center;color:#4a5888;font-size:1.1rem;margin:2px 0">↓</div>
  <div style="background:linear-gradient(135deg,rgba(212,148,28,.1),rgba(74,143,217,.08));border:2px solid #d4941c;border-radius:10px;padding:.8rem 1rem;margin-bottom:8px;box-shadow:0 0 20px rgba(212,148,28,.12)">
    <div style="display:flex;align-items:center;gap:.8rem">
      <div>
        <div style="font-size:.78rem;font-weight:800;color:#f0ac30">NSGA-II — çoxhədəfli optimallaşdırma — sənəd bölmə 2.4</div>
        <div style="font-size:.69rem;color:#9aa5cc;margin-top:.2rem">
          f₁=−F*→min · f₂=E→min · KKT: g(x)≤0, x_min≤x≤x_max · 100 fərd × 150 nəsil
        </div>
      </div>
    </div>
  </div>
  <div style="text-align:center;color:#4a5888;font-size:1.1rem;margin:2px 0">↓</div>
  <div style="display:flex;gap:8px;align-items:stretch">
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-bottom:2px solid #2ec98a;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#2ec98a;margin-bottom:.2rem">PARETO FRONTU</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">100 dominant həll<br>kompromis seçim</div>
    </div>
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-bottom:2px solid #d4941c;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#d4941c;margin-bottom:.2rem">OPTİMAL HƏLL</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">Verim ↑ maksimum<br>Enerji ↓ minimum</div>
    </div>
    <div style="flex:1;background:#1e2540;border:1px solid #283058;border-bottom:2px solid #4a8fd9;border-radius:8px;padding:.65rem .8rem;text-align:center">
      <div style="font-size:.72rem;font-weight:700;color:#4a8fd9;margin-bottom:.2rem">DASHBOARD</div>
      <div style="font-size:.7rem;color:#9aa5cc;line-height:1.5">5 interaktiv səhifə<br>Streamlit · Plotly</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
    sp()

    bb("Əsas əməliyyat göstəriciləri")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Soba temp.",  f"{xulase['avg_furnace_temp']:.1f} °C")
    k2.metric("Benzin",      f"{xulase['avg_yield_gasoline']:.1f} %")
    k3.metric("Dizel",       f"{xulase['avg_yield_diesel']:.1f} %")
    k4.metric("Kerosin",     f"{xulase['avg_yield_kerosene']:.1f} %")
    k5.metric("Enerji",      f"{xulase['avg_energy']:.2f} GS/s")
    k6.metric("Kükürd çıx.", f"{xulase['avg_sulfur_removal']:.1f} %")
    sp()

    bb("24 saatlıq proses monitorinqi")
    ca, cb = st.columns(2, gap="medium")
    with ca:
        fig = go.Figure()
        for sc, ad, reng in [
            ("yield_gasoline","Benzin (%)",RENG["benzin"]),
            ("yield_diesel","Dizel (%)",RENG["dizel"]),
            ("yield_kerosene","Kerosin (%)",RENG["kerosin"]),
        ]:
            fig.add_trace(go.Scatter(x=ts, y=temiz_df[sc]*100, name=ad,
                line=dict(color=reng, width=1.8),
                hovertemplate=f"<b>{ad}</b>: %{{y:.2f}}%<extra></extra>"))
        fig.update_layout(**QL("Fraksiya verimləri (%)", hund=300))
        fig.update_yaxes(title_text="Verim (%)")
        st.plotly_chart(fig, width="stretch")
    with cb:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ts, y=temiz_df["energy_gj_h"], name="Enerji (GS/s)",
            line=dict(color=RENG["enerji"], width=1.8), fill="tozeroy", fillcolor="rgba(212,80,96,.07)",
            hovertemplate="<b>Enerji</b>: %{y:.3f} GS/s<extra></extra>"))
        fig2.add_trace(go.Scatter(x=ts, y=temiz_df["sulfur_removal"]*100, name="Kükürd çıx. (%)",
            line=dict(color=RENG["kukurd"], width=1.8), yaxis="y2",
            hovertemplate="<b>Kükürd</b>: %{y:.1f}%<extra></extra>"))
        fig2.update_layout(**QL("Enerji sərfi + kükürd çıxarılması", hund=300))
        fig2.update_layout(
            yaxis=dict(title_text="Enerji (GS/s)", gridcolor="#283058", linecolor="#334070", zeroline=False),
            yaxis2=dict(overlaying="y", side="right", gridcolor="#283058", zeroline=False,
                        tickfont=dict(size=9, color="#9aa5cc"), title_text="Kükürd çıx. (%)"))
        st.plotly_chart(fig2, width="stretch")
    sp()

    bb("Sənəd tənlikləri")
    eq1, eq2 = st.columns(2, gap="medium")
    with eq1:
        st.markdown('<div class="tenlik"><b>Məqsəd funksiyası (tənlik 1, genişləndirilmiş):</b><br>'
            'F* = α·Y<sub>b</sub>+β·Y<sub>d</sub>+γ·Y<sub>k</sub>−δ·(E/E<sub>max</sub>)+ε·S<sub>removal</sub><br><br>'
            '<b>Enerji balansı (tənlik 4, genişləndirilmiş):</b><br>'
            'E = 0.042·T + 0.018·F + 0.008·T<sub>f</sub> − 11.5<br><br>'
            '<b>Kükürd çıxarılması:</b><br>'
            'S = 0.55·((H₂−30)/30) + 0.30·((T<sub>c</sub>−280)/80) + 0.40</div>', unsafe_allow_html=True)
    with eq2:
        st.markdown('<div class="tenlik"><b>Normallaşdırma (tənlik 2):</b><br>'
            'x<sub>norm</sub>=(x−x<sub>min</sub>)/(x<sub>max</sub>−x<sub>min</sub>)<br><br>'
            '<b>Kütlə balansı (tənlik 3):</b><br>'
            'ΣF<sub>giriş</sub>=ΣF<sub>çıxış</sub><br><br>'
            '<b>Temperatur profili (tənlik 5):</b><br>'
            'T(z)=T<sub>alt</sub>−k·z<br><br>'
            '<b>KKT (tənliklər 6–8):</b><br>'
            'g(x)≤0, h(x)=0, x<sub>min</sub>≤x≤x<sub>max</sub><br><br>'
            '<b>Xətti reqressiya:</b><br>'
            'Y=a₁T+a₂P+a₃F+a₄+ε</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SƏHİFƏ 2 — MƏLUMAT ANALİZİ
# ══════════════════════════════════════════════════════════════════════════
elif sehife == " Məlumat Analizi":
    st.markdown("## Kəşfiyyat Məlumat Analizi (EDA)")
    st.markdown("Sənəd Cədvəl 2.1: T (temp.), P (təzyiq), F (axın) — 9 sensor, 24 saat, 1 dəq. interval.")
    sp()

    # Anomaliya statistikası — bir dəfə hesabla
    adf = anomaliya_df
    n_anomaly_iqr     = int(adf["anomaly_iqr"].sum())    if "anomaly_iqr"     in adf.columns else 0
    n_anomaly_iforest = int(adf["anomaly_iforest"].sum()) if "anomaly_iforest" in adf.columns else 0
    n_anomaly_final   = int(adf["anomaly_final"].sum())   if "anomaly_final"   in adf.columns else 0
    anomaly_rate_pct  = n_anomaly_final / len(adf) * 100 if len(adf) > 0 else 0.0

    bb("Xam məlumat — dataset icmalı")
    col_i1, col_i2, col_i3 = st.columns(3)
    col_i1.info(f"{xam_df.shape[0]} sətir x {xam_df.shape[1]} sütun")
    col_i2.info(f"{n_anomaly_final} anomaliya ({anomaly_rate_pct:.1f}%)")
    col_i3.info(f"{len(temiz_df)} təmiz nümunə saxlanıldı")
    sp()

    t_raw, t_clean = st.tabs([" Xam Məlumat"," Təmiz Məlumat"])
    with t_raw:
        st.markdown('<div class="izah">Sensorlardan birbaşa alınan xam məlumat — filtrasiyadan <b>əvvəl</b>. NaN və anomal dəyərlər mövcuddur.</div>', unsafe_allow_html=True)
        disp_cols = ["timestamp"] + INPUT_FEATURES[:6]
        st.dataframe(xam_df[disp_cols].head(20).style.format({c:"{:.3f}" for c in INPUT_FEATURES[:6]}),
                     width="stretch", height=320)
    with t_clean:
        st.markdown('<div class="izah">Preprocessing boru kəmərindən keçmiş məlumat — filtrasiya, normallaşdırma, anomaliya aşkarlanmasından <b>sonra</b>.</div>', unsafe_allow_html=True)
        st.dataframe(temiz_df[disp_cols+["yield_gasoline","yield_diesel","energy_gj_h"]].head(20)
                     .style.format({c:"{:.3f}" for c in INPUT_FEATURES[:6]+["yield_gasoline","yield_diesel","energy_gj_h"]}),
                     width="stretch", height=320)
    sp()

    t1, t2, t3 = st.tabs([" Zaman Seriyaları"," Korrelyasiya"," Statistika"])

    with t1:
        csel, cplt = st.columns([1, 3], gap="medium")
        with csel:
            sec = st.multiselect("Sensor kanalları", INPUT_FEATURES,
                default=["furnace_temp","column_pressure","flow_rate"],
                format_func=lambda x: SENSOR_AD.get(x, x), label_visibility="collapsed")
            pen = st.slider("Hərəkətli ort. (dəq.)", 1, 60, 15)
            ag  = st.checkbox("Anomaliyaları göstər", True)
        with cplt:
            if not sec:
                st.info("Sol tərəfdən sensor seçin.")
            else:
                RCOL = [RENG["benzin"],RENG["dizel"],RENG["kerosin"],RENG["kukurd"],
                        RENG["anomal"],"#a78bfa","#f472b6","#38bdf8","#fb923c"]
                fig_ts = make_subplots(rows=len(sec), cols=1, shared_xaxes=True,
                    subplot_titles=[SENSOR_AD.get(s,s) for s in sec], vertical_spacing=.06)
                for i, sensor in enumerate(sec):
                    reng = RCOL[i % len(RCOL)]
                    vals = temiz_df[sensor]
                    roll = vals.rolling(pen).mean()
                    fig_ts.add_trace(go.Scatter(x=ts, y=vals, opacity=.28, line=dict(color=reng, width=1),
                        showlegend=False), row=i+1, col=1)
                    fig_ts.add_trace(go.Scatter(x=ts, y=roll, name=SENSOR_AD.get(sensor, sensor),
                        line=dict(color=reng, width=2)), row=i+1, col=1)
                    if ag and "anomaly_final" in adf.columns:
                        am_mask = adf["anomaly_final"].values[:len(temiz_df)]
                        am_df_full = adf[am_mask].copy()
                        am_df   = temiz_df[am_mask]
                        if len(am_df):
                            # Anomaliya tipini müəyyən et
                            def _anom_tip(row):
                                iqr_v = row.get("anomaly_iqr", False)
                                iso_v = row.get("anomaly_iforest", False)
                                if iqr_v and iso_v: return "IQR + IsolationForest"
                                if iqr_v: return "IQR (statistik hüdud)"
                                if iso_v: return "IsolationForest (kombinasiya)"
                                return "Anomaliya"
                            tips = am_df_full.apply(_anom_tip, axis=1).values
                            sensor_vals = am_df[sensor].values
                            hover_texts = [
                                f"<b>Anomaliya</b><br>{sensor}: {v:.3f}<br>Tip: {t}"
                                for v, t in zip(sensor_vals, tips)
                            ]
                            fig_ts.add_trace(go.Scatter(
                                x=am_df["timestamp"], y=am_df[sensor],
                                mode="markers", name="Anomaliya" if i==0 else None,
                                showlegend=(i==0),
                                text=hover_texts,
                                hovertemplate="%{text}<extra></extra>",
                                marker=dict(color=RENG["anomal"], size=6, symbol="x-thin",
                                    line=dict(width=2, color=RENG["anomal"]))), row=i+1, col=1)
                fig_ts.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e2540",
                    font=dict(family="Inter", color="#9aa5cc", size=10),
                    height=max(220*len(sec), 280), margin=dict(l=8,r=8,t=30,b=8),
                    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.06, font=dict(size=10,color="#9aa5cc")))
                sub_stil(fig_ts)
                st.plotly_chart(fig_ts, width="stretch")

    with t2:
        bb("Pearson korrelyasiya matrisi")
        st.markdown('<div class="izah">r=+1: güclü müsbət · r=0: əlaqə yox · r=−1: güclü mənfi.<br>'
            '<b style="color:#c0613a">Narıncı=müsbət</b> · <b style="color:#3a6186">Mavi=mənfi</b> · Ağa yaxın=zəif əlaqə</div>', unsafe_allow_html=True)
        cc   = INPUT_FEATURES + ["yield_gasoline","yield_diesel","yield_kerosene","energy_gj_h"]
        ce   = [SENSOR_AD.get(c, HEDAF_AD.get(c, c)) for c in cc]
        corr = temiz_df[cc].corr()
        fig_c = go.Figure(go.Heatmap(z=corr.values, x=ce, y=ce,
            colorscale=[[0,"#3a6186"],[0.25,"#6a9fb5"],[0.5,"#e8ecf0"],[0.75,"#e8a87c"],[1,"#c0613a"]],
            zmid=0, zmin=-1, zmax=1, text=corr.values.round(2), texttemplate="%{text}",
            textfont=dict(size=8.5, family="JetBrains Mono")))
        fig_c.update_layout(**QL("Korrelyasiya matrisi (Pearson r)", hund=500, legend=False))
        fig_c.update_xaxes(tickangle=-35, tickfont=dict(size=9, color="#9aa5cc"))
        st.plotly_chart(fig_c, width="stretch")

    with t3:
        bb("Təsviri statistika")
        ss  = INPUT_FEATURES + ["yield_gasoline","yield_diesel","yield_kerosene","energy_gj_h"]
        sdf = temiz_df[ss].describe().round(4)
        sdf.columns = [SENSOR_AD.get(c, HEDAF_AD.get(c,c)) for c in sdf.columns]
        sdf.index   = ["Say","Orta","Std","Min","25%","50%","75%","Maks"]
        st.dataframe(sdf.style.format("{:.4f}"), width="stretch", height=310)


# ══════════════════════════════════════════════════════════════════════════
# SƏHİFƏ 3 — EMAL BORU KƏMƏRİ
# ══════════════════════════════════════════════════════════════════════════
elif sehife == " Emal Boru Kəməri":
    st.markdown("## Verilənlərin Emal Boru Kəməri")
    st.markdown("Sənəd bölmə 2.3: filtrasiya → normallaşdırma → anomaliya → xətti reqressiya → ARO / E2E / DT metodları.")
    sp()

    # Anomaliya statistikası — bir dəfə hesabla
    adf = anomaliya_df
    n_iqr     = int(adf["anomaly_iqr"].sum())    if "anomaly_iqr"     in adf.columns else 0
    n_iforest = int(adf["anomaly_iforest"].sum()) if "anomaly_iforest" in adf.columns else 0
    n_final   = int(adf["anomaly_final"].sum())   if "anomaly_final"   in adf.columns else 0
    a_rate    = n_final / len(adf) * 100          if len(adf) > 0 else 0.0

    tf, tn, ta, tr, tvm = st.tabs([
        "1 Filtrasiya","2 Normallaşdırma",
        "3 Anomaliya","4 Reqressiya (ARO)",
        "5 E2E · DT"])

    # ── Tab 1: Filtrasiya ──────────────────────────────────────────────────
    with tf:
        bb("Xətti interpolasiya + 3-sigma kəsmə")
        col_fi1, col_fi2 = st.columns(2, gap="medium")
        with col_fi1:
            st.markdown('<div class="izah"><b>① Filtrasiya</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">Ardıcıl ≤5 boşluq → xətti interpolasiya</span><br><br>'
                '<b>② 3-sigma kəsmə</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">[μ−3σ, μ+3σ] xarici dəyərlər kəsilir</span>'
                '</div>', unsafe_allow_html=True)
        with col_fi2:
            st.markdown('<div class="izah"><b>③ NaN silmə</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">Hələ NaN olan sətrlər çıxarılır</span><br><br>'
                '<b>Nəticə</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">Optimallaşdırıcıya etibarlı məlumat</span>'
                '</div>', unsafe_allow_html=True)
        xs, ts2 = len(xam_df), len(temiz_df)
        a1,a2,a3,a4 = st.columns(4)
        a1.metric("Xam nümunə",  f"{xs:,}")
        a2.metric("Təmiz nümunə", f"{ts2:,}")
        a3.metric("Çıxarılan",   f"{xs-ts2:,}")
        a4.metric("Saxlanma",    f"{ts2/xs*100:.1f}%")
        sp()
        ss_sel = st.selectbox("Sensor seçin", INPUT_FEATURES, format_func=lambda x: SENSOR_AD.get(x,x))
        fig_f  = make_subplots(rows=1, cols=2, subplot_titles=["Xam məlumat","Filtrlənmiş"], horizontal_spacing=.06)
        for ci, (dg, ad, reng) in enumerate([(xam_df,"Xam",RENG["anomal"]), (temiz_df,"Təmiz",RENG["benzin"])]):
            fig_f.add_trace(go.Scatter(x=dg["timestamp"], y=dg[ss_sel], name=ad,
                line=dict(color=reng, width=1.3),
                hovertemplate=f"{ad}: %{{y:.3f}}<extra></extra>"), row=1, col=ci+1)
        fig_f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e2540",
            font=dict(family="Inter", color="#9aa5cc", size=10),
            height=280, margin=dict(l=8,r=8,t=36,b=8), showlegend=False)
        sub_stil(fig_f)
        st.plotly_chart(fig_f, width="stretch")

    # ── Tab 2: Normallaşdırma ──────────────────────────────────────────────
    with tn:
        bb("Min-Maks normallaşdırma — sənəd tənliyi")
        st.markdown('<div class="izah">Müxtəlif ölçü vahidli sensorları [0,1] aralığına çevirir:<br>'
            '<code>x_norm = (x − x_min) / (x_max − x_min)</code></div>', unsafe_allow_html=True)
        ns  = st.selectbox("Xüsusiyyət", INPUT_FEATURES, format_func=lambda x: SENSOR_AD.get(x,x), key="ns")
        xv  = temiz_df[ns].values
        nv  = X_norm[ns].values
        fig_n = make_subplots(rows=1, cols=2, subplot_titles=["Xam dəyərlər","Normallaşdırılmış [0,1]"], horizontal_spacing=.08)
        for ci, (vals, reng, ad) in enumerate([(xv, RENG["dizel"],"Xam"), (nv, RENG["benzin"],"Norm")]):
            fig_n.add_trace(go.Histogram(x=vals, nbinsx=45, name=ad,
                marker=dict(color=reng, opacity=.82, line=dict(color="#1a1f35", width=.4))), row=1, col=ci+1)
            xk = "x" if ci==0 else "x2"
            fig_n.add_shape(type="line", x0=vals.mean(), x1=vals.mean(), y0=0, y1=1,
                yref="paper", xref=xk, line=dict(color="#edf1fc", dash="dot", width=1.2))
            fig_n.add_annotation(x=vals.mean(), y=.95, yref="paper", xref=xk,
                text=f"Ort:{vals.mean():.3f}", showarrow=False,
                font=dict(color="#edf1fc", size=9, family="JetBrains Mono"), bgcolor="rgba(26,31,53,.85)")
        fig_n.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e2540",
            font=dict(family="Inter", color="#9aa5cc", size=10),
            height=300, margin=dict(l=8,r=8,t=36,b=8), showlegend=False)
        sub_stil(fig_n)
        st.plotly_chart(fig_n, width="stretch")

    # ── Tab 3: Anomaliya ───────────────────────────────────────────────────
    with ta:
        bb("İki metodlu anomaliya aşkarlanması")
        col_ai1, col_ai2 = st.columns(2, gap="medium")
        with col_ai1:
            st.markdown('<div class="izah"><b>Metod A — IQR</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">Q1−1.5·IQR ~ Q3+1.5·IQR<br>'
                'Tək sensor, statistik, sürətli</span>'
                '</div>', unsafe_allow_html=True)
        with col_ai2:
            st.markdown('<div class="izah"><b>Metod B — Isolation Forest</b><br>'
                '<span style="font-size:.79rem;color:#6878a8">ML əsaslı, <b>çoxölçülü</b><br>'
                'Bütün 9 sensoru birlikdə qiymətləndirir.<br>'
                'Tək sensorun dəyəri normal görünsə belə, sensor kombinasiyası nadir olarsa anomaliya kimi işarələnir.</span>'
                '</div>', unsafe_allow_html=True)
        st.markdown('<div class="xeb">Qrafik üzərindəki × işarəsinə hover etdikdə anomaliyanın tipi göstərilir: '
                    'IQR (statistik hüdud aşılıb) və ya IsolationForest (sensor kombinasiyası qeyri-adidir).</div>',
                    unsafe_allow_html=True)

        b1,b2,b3,b4 = st.columns(4)
        b1.metric("IQR anomaliya",    f"{n_iqr}")
        b2.metric("Isolation Forest", f"{n_iforest}")
        b3.metric("Yekun anomaliya",  f"{n_final}")
        b4.metric("Anomaliya faizi",  f"{a_rate:.2f}%")
        sp()

        ase = st.selectbox("Sensor", INPUT_FEATURES, format_func=lambda x: SENSOR_AD.get(x,x), key="ase")
        fig_a = go.Figure()

        if "anomaly_final" in adf.columns:
            nm_mask = ~adf["anomaly_final"]
            am_mask = adf["anomaly_final"]
        else:
            nm_mask = pd.Series(True, index=adf.index)
            am_mask = pd.Series(False, index=adf.index)

        ts_col = "timestamp" if "timestamp" in adf.columns else None

        fig_a.add_trace(go.Scatter(
            x=adf.loc[nm_mask, ts_col] if ts_col else adf.index[nm_mask],
            y=adf.loc[nm_mask, ase],
            name="Normal", line=dict(color=RENG["normal"], width=1.4),
            hovertemplate="Normal: %{y:.3f}<extra></extra>"))
        am_vals = adf.loc[am_mask, ase]
        if len(am_vals) > 0:
            fig_a.add_trace(go.Scatter(
                x=adf.loc[am_mask, ts_col] if ts_col else adf.index[am_mask],
                y=am_vals,
                name="Anomaliya", mode="markers",
                marker=dict(color=RENG["anomal"], size=7, symbol="x-thin",
                    line=dict(width=2, color=RENG["anomal"])),
                hovertemplate="Anomaliya: %{y:.3f}<extra></extra>"))
        fig_a.update_layout(**QL(f"{SENSOR_AD.get(ase,ase)} — Anomaliya aşkarlanması", hund=300))
        st.plotly_chart(fig_a, width="stretch")

        if "anomaly_score" in adf.columns:
            fsk = go.Figure(go.Histogram(x=adf["anomaly_score"], nbinsx=50,
                marker=dict(color=RENG["normal"], opacity=.82, line=dict(color="#1a1f35", width=.4))))
            fsk.add_vline(x=0, line_color=RENG["anomal"], line_dash="dash", line_width=2,
                annotation_text="Qərar həddi (0)", annotation_font_color=RENG["anomal"], annotation_font_size=10)
            fsk.update_layout(**QL("Isolation Forest anomaliya skoru", hund=220, legend=False))
            fsk.update_xaxes(title_text="Bal (aşağı = daha anomal)")
            st.plotly_chart(fsk, width="stretch")

    # ── Tab 4: Reqressiya (ARO) — xətti reqressiya ────────────────────────
    with tr:
        bb("Xətti Reqressiya modeli — ARO (Y = a₁T + a₂P + a₃F + ... + b)")
        st.markdown('<div class="izah">'
            '<b>Model:</b> Y = a₁·furnace_temp + a₂·pressure + a₃·flow_rate + … + b<br>'
            'Min-Maks normallaşdırılmış giriş xüsusiyyətləri üzərində öyrədilir.<br>'
            'Hər hədəf dəyişəni üçün ayrı LinearRegression — <b>CV R²</b> 3-qatlı cross-validation ilə ölçülür.</div>',
            unsafe_allow_html=True)
        sp()

        # ── Model metrikaları ──────────────────────────────────────────────
        raw_metrics = req_goster.copy()
        r2_col      = "R²" if "R²" in raw_metrics.columns else raw_metrics.columns[0]
        gdf         = raw_metrics.copy()
        gdf.index   = [HEDAF_AD.get(i, i) for i in gdf.index]

        rc1, rc2 = st.columns([1, 1.5], gap="medium")
        with rc1:
            st.markdown("**Model göstəriciləri**")
            fmt_dict = {}
            for c in gdf.columns:
                if "RMSE" in c: fmt_dict[c] = "{:.6f}"
                else:           fmt_dict[c] = "{:.4f}"
            st.dataframe(gdf.style.format(fmt_dict), width="stretch")
            st.markdown('<div class="izah" style="font-size:.76rem">'
                '<b>CV R²</b> = 3-qatlı cross-validation · <b>R²</b> = tam məlumat üzərində · <b>RMSE</b> = kök orta kvadrat xəta</div>',
                unsafe_allow_html=True)

        with rc2:
            r2v = gdf[r2_col].values
            r2e = list(gdf.index)
            fig_r2 = go.Figure(go.Bar(
                x=r2e, y=r2v,
                marker=dict(
                    color=["#d4941c" if v>=.95 else "#4a8fd9" if v>=.80 else "#d45060" for v in r2v],
                    opacity=.9, line=dict(color="#1a1f35", width=.5)),
                text=[f"{v:.4f}" for v in r2v], textposition="outside",
                textfont=dict(size=10, family="JetBrains Mono"),
                hovertemplate="%{x}: R²=%{y:.4f}<extra></extra>"))
            fig_r2.add_hline(y=.95, line_dash="dot", line_color=RENG["kukurd"], line_width=1.2,
                annotation_text="0.95 keyfiyyət həddi",
                annotation_font_color=RENG["kukurd"], annotation_font_size=9)
            fig_r2.update_yaxes(range=[0, 1.12], title_text="R²")
            fig_r2.update_xaxes(tickangle=-20)
            fig_r2.update_layout(**QL("Xətti Reqressiya — R² dəyərləri", hund=300, legend=False))
            st.plotly_chart(fig_r2, width="stretch")

        sp()

        # ── Feature əmsalları (Linear Regression coefficients) ─────────────
        bb("Feature əmsalları — Y = a₁·T + a₂·P + … + b")
        st.markdown(
            '<div class="izah">Normallaşdırılmış xüsusiyyətlər üzərindəki əmsallar. '
            '<b style="color:#d4941c">Müsbət</b> = həmin parametr artdıqca verim artır. '
            '<b style="color:#4a8fd9">Mənfi</b> = əks təsir.</div>',
            unsafe_allow_html=True)

        st.markdown('<div class="tenlik">Y = a₁·furnace_temp + a₂·pressure + a₃·flow + … + b<br>'
            'Əmsallar [0,1] normallaşdırılmış məlumat üzərindədir.</div>', unsafe_allow_html=True)

        if not lr_coeffs.empty:
            feat_cols_coeff = [c for c in lr_coeffs.columns if c != "intercept"]

            # Hədəf seçimi
            target_sel = st.selectbox("Hədəf dəyişəni", options=list(lr_coeffs.index),
                format_func=lambda x: HEDAF_AD.get(x, x), key="lr_target")

            row_vals = lr_coeffs.loc[target_sel, feat_cols_coeff].values.astype(float)
            feat_labels = [SENSOR_AD.get(c, c) for c in feat_cols_coeff]
            sort_idx    = np.argsort(np.abs(row_vals))[::-1]
            sv = row_vals[sort_idx]
            sl = [feat_labels[i] for i in sort_idx]

            fig_coeff = go.Figure(go.Bar(
                x=sv, y=sl, orientation="h",
                marker=dict(
                    color=["#d4941c" if v >= 0 else "#4a8fd9" for v in sv],
                    opacity=0.88, line=dict(color="#1a1f35", width=0.5)),
                text=[f"{v:+.4f}" for v in sv], textposition="outside",
                textfont=dict(size=9, family="JetBrains Mono"),
                hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>"))
            fig_coeff.add_vline(x=0, line_color="#6878a8", line_width=1.5)
            fig_coeff.update_layout(**QL(
                f"Feature Importance — {HEDAF_AD.get(target_sel, target_sel)}", hund=360, legend=False))
            fig_coeff.update_xaxes(title_text="Əmsal dəyəri")
            fig_coeff.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_coeff, width="stretch")

            # Tam əmsal matrisi — istilik xəritəsi
            st.markdown("**Bütün hədəflər — əmsal istilik xəritəsi**")
            z_mat = lr_coeffs[feat_cols_coeff].values.astype(float)
            target_labels = [HEDAF_AD.get(t, t) for t in lr_coeffs.index]
            fig_heat = go.Figure(go.Heatmap(
                z=z_mat, x=feat_labels, y=target_labels,
                colorscale=[[0,"#d45060"],[0.5,"#1e2540"],[1,"#d4941c"]],
                zmid=0, text=z_mat.round(3), texttemplate="%{text}",
                textfont=dict(size=8, family="JetBrains Mono"),
                hovertemplate="<b>%{y}</b> ← %{x}<br>Əmsal: %{z:.4f}<extra></extra>"))
            fig_heat.update_layout(**QL("Feature Importance matrisi — bütün hədəflər", hund=320, legend=False))
            fig_heat.update_xaxes(tickangle=-30, tickfont=dict(size=9, color="#9aa5cc"))
            st.plotly_chart(fig_heat, width="stretch")
        else:
            st.info("LR əmsalları mövcud deyil — pipeline yenidən işlədin.")

    # ── Tab 5: E2E · DT ───────────────────────────────────────────────────
    with tvm:
        bb("E2E, DT metodları")

        st.markdown("### Başdan-Başa Öyrənmə (E2E — End-to-End Learning)")
        st.markdown('<div class="izah">'
            'Sənəd bölmə 2.3, Şəkil 2.7. ARO-dan fərqli olaraq E2E proqnozlaşdırma '
            'və qərar qəbuletməni <b>vahid çərçivədə</b> birləşdirir.<br>'
            '• Hər iterasiyada optimallaşdırma nəticəsi geri yayılaraq modeli kalibrləyir<br>'
            '• Qərar mərkəzli itki funksiyası (decision-focused loss)<br>'
            '• ARO-dakı iki mərhələli ayrılmanın yaratdığı qərəzləri azaldır'
            '</div>', unsafe_allow_html=True)

        col_e1, col_e2 = st.columns(2, gap="medium")
        with col_e1:
            st.markdown("**ARO vs E2E müqayisəsi:**")
            comp_df = pd.DataFrame({
                "Xüsusiyyət": [
                    "Struktur",
                    "Öyrənmə",
                    "Sürət",
                    "Dəqiqlik",
                    "Tətbiq sahəsi",
                ],
                "ARO": [
                    "Ardıcıl (proqnoz → qərar), iki ayrı addım",
                    "Hər addım müstəqil öyrənilir",
                    "Sürətli — hər addım ayrıca optimallaşdırılır",
                    "Yüksək R² — ayrı-ayrı modellərdə",
                    "Sənaye standartı — real vaxt idarəetmə",
                ],
                "E2E": [
                    "Vahid çərçivə — proqnoz və qərar birlikdə",
                    "Birgə öyrənmə — son hədəfə doğru",
                    "Yavaş — böyük model qrafienti hesablanır",
                    "Daha optimal — qərar xətasını birbaşa minimumlaşdırır",
                    "Tədqiqat sahəsi — adaptiv idarəetmə sistemləri",
                ],
            })
            st.dataframe(
                comp_df.style.set_properties(**{
                    "color": "#c5cceb",
                    "background-color": "#1e2848",
                    "font-size": "0.82rem",
                }).set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", "#0e1524"),
                        ("color", "#edf1fc"),
                        ("font-weight", "700"),
                        ("font-size", "0.78rem"),
                        ("border-bottom", "2px solid #d4941c"),
                    ]},
                    {"selector": "td", "props": [
                        ("border-color", "#283058"),
                    ]},
                ]),
                width="stretch", hide_index=True)

        with col_e2:
            nesl    = list(range(1, 21))
            aro_r2  = [0.82+0.015*i-0.0003*i**2 for i in nesl]
            e2e_r2  = [0.78+0.019*i-0.0002*i**2 for i in nesl]
            fig_e2e = go.Figure()
            fig_e2e.add_trace(go.Scatter(x=nesl, y=aro_r2, name="ARO",
                line=dict(color=RENG["benzin"], width=2, dash="dot")))
            fig_e2e.add_trace(go.Scatter(x=nesl, y=e2e_r2, name="E2E",
                line=dict(color=RENG["dizel"], width=2)))
            fig_e2e.update_layout(**QL("ARO vs E2E — öyrənmə əyrisi (simulyasiya)", hund=300))
            fig_e2e.update_xaxes(title_text="İterasiya")
            fig_e2e.update_yaxes(title_text="R² dəyəri", range=[0.8, 1.0])
            st.plotly_chart(fig_e2e, width="stretch")
        sp()

        st.markdown("### Birbaşa Öyrənmə (DT — Decision-Tree / Direct Learning)")
        st.markdown('<div class="izah">'
            'Sənəd bölmə 2.3, Şəkil 2.8. DT məlumatları birbaşa qərarlara inteqrasiya edir.<br>'
            '• Açıq optimallaşdırma formulalarına ehtiyac yoxdur<br>'
            '• Ənənəvi optimallaşdırmanın mümkün olmadığı mühitlərdə üstünlük<br>'
            '• Son məqsəd performans metrikləri ilə birbaşa uyğunlaşdırma'
            '</div>', unsafe_allow_html=True)

        col_dt1, col_dt2 = st.columns([1, 1], gap="medium")
        with col_dt1:
            st.markdown("**DT qərar ağacı — sadələşdirilmiş nümunə:**")
            # Ağac strukturu — vizual HTML
            tree_html = """
<div style="font-family:'Inter',sans-serif;font-size:.82rem">
  <!-- Kök -->
  <div style="background:#212845;border:1px solid #334070;border-radius:8px;
       padding:.65rem .9rem;color:#c5cceb;text-align:center;font-weight:700;margin-bottom:4px">
    Soba temp. &gt; 370°C
  </div>
  <!-- Ox -->
  <div style="display:flex;justify-content:space-around;color:#6878a8;font-size:.8rem;margin:2px 0">
    <span>✓ Bəli</span><span>✗ Xeyr</span>
  </div>
  <!-- Birinci səviyyə -->
  <div style="display:flex;gap:8px;margin-bottom:4px">
    <div style="flex:1;background:#1a2038;border-left:3px solid #2ec98a;border-radius:6px;padding:.5rem .75rem;color:#9aa5cc">
      Benzin verimi &gt; 25%
    </div>
    <div style="flex:1;background:#1a2038;border-left:3px solid #d45060;border-radius:6px;padding:.5rem .75rem;color:#9aa5cc">
      Benzin verimi &lt; 20%
    </div>
  </div>
  <!-- Ox -->
  <div style="display:flex;gap:8px">
    <div style="flex:1;display:flex;justify-content:space-around;color:#6878a8;font-size:.75rem;margin:2px 0">
      <span>✓ Bəli</span><span>✗ Xeyr</span>
    </div>
    <div style="flex:1;display:flex;justify-content:space-around;color:#6878a8;font-size:.75rem;margin:2px 0">
      <span>✓ Bəli</span><span>✗ Xeyr</span>
    </div>
  </div>
  <!-- Yarpaqlar -->
  <div style="display:flex;gap:8px">
    <div style="flex:1;background:#0e1524;border-left:3px solid #2ec98a;border-radius:6px;padding:.4rem .6rem;font-size:.76rem;color:#2ec98a">
      Optimal rejim
    </div>
    <div style="flex:1;background:#0e1524;border-left:3px solid #d4941c;border-radius:6px;padding:.4rem .6rem;font-size:.76rem;color:#d4941c">
      Temperaturu artır
    </div>
    <div style="flex:1;background:#0e1524;border-left:3px solid #4a8fd9;border-radius:6px;padding:.4rem .6rem;font-size:.76rem;color:#4a8fd9">
      Refluks artır
    </div>
    <div style="flex:1;background:#0e1524;border-left:3px solid #6878a8;border-radius:6px;padding:.4rem .6rem;font-size:.76rem;color:#6878a8">
      Parametr düzəlt
    </div>
  </div>
</div>"""
            st.markdown(tree_html, unsafe_allow_html=True)

        with col_dt2:
            cats = ["Sürət","Dəqiqlik","Adaptivlik","Şəffaflıq","Tətbiq asanlığı"]
            fig_radar = go.Figure()
            for ad, vals, reng in [
                ("ARO",[85,82,60,90,95],RENG["benzin"]),
                ("E2E",[50,92,88,60,55],RENG["dizel"]),
                ("DT", [90,75,70,85,80],RENG["kukurd"]),
            ]:
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]], theta=cats+[cats[0]], name=ad,
                    fill="toself", line=dict(color=reng, width=2)))
            fig_radar.update_layout(**QL("ARO / E2E / DT müqayisəsi", hund=350))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#1e2540",
                    radialaxis=dict(visible=True, range=[0,100], gridcolor="#283058",
                        tickfont=dict(size=9, family="JetBrains Mono", color="#9aa5cc")),
                    angularaxis=dict(gridcolor="#283058", tickfont=dict(size=11, color="#c5cceb"))))
            st.plotly_chart(fig_radar, width="stretch")


# ══════════════════════════════════════════════════════════════════════════
# SƏHİFƏ 4 — OPTİMALLAŞDIRMA
# ══════════════════════════════════════════════════════════════════════════
elif sehife == " Optimallaşdırma":
    st.markdown("## NSGA-II Çoxhədəfli Optimallaşdırma")
    st.markdown(
        "NSGA-II — çoxhədəfli optimallaşdırma üçün genetik alqoritm. "
        "İki hədəf eyni vaxtda: F* maksimumlaşdır (məhsul gəliri), E minimumlaşdır (enerji).")
    st.markdown('<div class="izah"><b>Pareto frontu:</b> Bir hədəfi yaxşılaşdırmaq digərini pisləşdirirsə, '
        'bu həllər Pareto-bərabərdir. Front kompromis həllərin məcmusudur.</div>', unsafe_allow_html=True)
    sp()

    if st.button("NSGA-II İşlət", type="primary",
                 help="Yan paneldəki parametrlərlə optimallaşdırmanı yenidən başladır"):
        st.cache_data.clear()
        if "_opt_cache_key" in st.session_state:
            del st.session_state["_opt_cache_key"]
        if "_opt_result" in st.session_state:
            del st.session_state["_opt_result"]
        st.rerun()

    prog_container = st.empty()
    with st.spinner(f"NSGA-II: {pop_olcu} fərd × {nesl_sayi} nəsil hesablanır…"):
        bar = prog_container.progress(0, "NSGA-II başlanır…")
        for i in range(5):
            time.sleep(0.08)
            bar.progress((i+1)*15, f"Nəsil {(i+1)*nesl_sayi//5}/{nesl_sayi} işlənir…")
        bar.progress(100, "Tamamlandı!")
        time.sleep(0.4)
        prog_container.empty()
    opt = _OPT

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Pareto həll sayı", f"{opt.n_solutions}")
    k2.metric("Nəsil sayı",       f"{opt.n_generations}")
    k3.metric("Ən yaxşı F*",      f"{opt.best_compromise['f_star']:.4f}")
    k4.metric("Min. enerji",      f"{opt.best_compromise['energy_gj_h']:.3f} GS/s")
    sp()

    pc1, pc2 = st.columns([1.3, 1], gap="medium")
    with pc1:
        bb("Pareto frontu — enerji vs verim kompromisi")
        fig_pf = go.Figure()
        fig_pf.add_trace(go.Scatter(x=opt.pareto_F[:,1], y=-opt.pareto_F[:,0], mode="markers",
            name="Pareto həllər",
            marker=dict(color=-opt.pareto_F[:,0],
                colorscale=[[0,"#283058"],[0.5,"#4a8fd9"],[1,"#d4941c"]],
                size=10, opacity=.9, showscale=True,
                colorbar=dict(title="F*", thickness=12, tickfont=dict(size=9,family="JetBrains Mono",color="#9aa5cc"),
                              title_font=dict(color="#9aa5cc")),
                line=dict(color="#1a1f35", width=.5)),
            hovertemplate="Enerji: %{x:.3f} GS/s<br>F*: %{y:.4f}<extra></extra>"))
        ey = opt.best_compromise
        fig_pf.add_trace(go.Scatter(x=[ey["obj_energy"]], y=[ey["f_star"]],
            mode="markers", name="Ən yaxşı kompromis",
            marker=dict(color=RENG["enyaxsi"], size=18, symbol="star",
                line=dict(color="#edf1fc", width=2)),
            hovertemplate=f"Ən yaxşı<br>Enerji: {ey['obj_energy']:.3f}<br>F*: {ey['f_star']:.4f}<extra></extra>"))
        fig_pf.update_layout(**QL(f"Pareto frontu — {opt.n_solutions} dominant həll", hund=420))
        fig_pf.update_xaxes(title_text="Enerji sərfi (GS/s)")
        fig_pf.update_yaxes(title_text="F* — çəkili verim balı")
        st.plotly_chart(fig_pf, width="stretch")

    with pc2:
        bb("Konvergensiya — alqoritmin öyrənmə əyrisi")
        if opt.history:
            hdf = pd.DataFrame(opt.history)
            fig_cv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                subplot_titles=["F* konvergensiyası","Enerji konvergensiyası"], vertical_spacing=.14)
            fig_cv.add_trace(go.Scatter(x=hdf["generation"], y=hdf["mean_f_star"],
                name="Orta F*", line=dict(color=RENG["dizel"], width=1.8)), row=1, col=1)
            fig_cv.add_trace(go.Scatter(x=hdf["generation"], y=hdf["best_f_star"],
                name="Ən yaxşı F*", line=dict(color=RENG["benzin"], width=1.5, dash="dot")), row=1, col=1)
            fig_cv.add_trace(go.Scatter(x=hdf["generation"], y=hdf["mean_energy"],
                name="Orta enerji", line=dict(color=RENG["enerji"], width=1.8)), row=2, col=1)
            fig_cv.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1e2540",
                font=dict(family="Inter", color="#9aa5cc", size=10),
                height=420, margin=dict(l=8,r=8,t=36,b=8),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9,color="#9aa5cc"), orientation="h", y=-0.08))
            sub_stil(fig_cv)
            fig_cv.update_yaxes(title_text="F*", row=1, col=1)
            fig_cv.update_yaxes(title_text="GS/s", row=2, col=1)
            fig_cv.update_xaxes(title_text="Nəsil №", row=2, col=1)
            st.plotly_chart(fig_cv, width="stretch")
    sp()

    # ── Pareto ətraflı analiz ──────────────────────────────────────────────
    bb("Pareto frontu ətraflı analizi")
    pf_t1, pf_t2 = st.tabs([" Radar Müqayisəsi"," Pareto Cədvəli"])

    with pf_t1:
        st.markdown('<div class="izah">Pareto frontundan 3 seçilmiş həll profili: '
            '<b>Ən yaxşı F*</b> · <b>Ən az enerji</b> · <b>Kompromis</b>.</div>', unsafe_allow_html=True)
        cats_r = ["Benzin verimi","Dizel verimi","Kerosin verimi","Enerji (çevrilmiş)","Kükürd çıx.","F*"]
        def _norm_row(row):
            vals = [row["yield_gasoline"], row["yield_diesel"], row["yield_kerosene"],
                    1-(row["energy_gj_h"]-5)/17, row["sulfur_removal_pct"]/100, row["f_star"]]
            return [max(0,min(1,v))*100 for v in vals]
        idx_best_f = opt.pareto_df["f_star"].idxmax()
        idx_best_e = opt.pareto_df["energy_gj_h"].idxmin()
        idx_comp   = opt.best_compromise.name
        fig_rad = go.Figure()
        for label, idx_r, reng in [
            ("Ən yaxşı F*", idx_best_f, RENG["benzin"]),
            ("Ən az enerji", idx_best_e, RENG["dizel"]),
            ("Kompromis",   idx_comp,   RENG["kukurd"]),
        ]:
            row_r = opt.pareto_df.iloc[idx_r] if isinstance(idx_r, int) else opt.best_compromise
            rv = _norm_row(row_r)
            fig_rad.add_trace(go.Scatterpolar(r=rv+[rv[0]], theta=cats_r+[cats_r[0]], name=label,
                line=dict(color=reng, width=2),
                hovertemplate="%{theta}: %{r:.1f}<extra>"+label+"</extra>"))
        fig_rad.update_layout(**QL("Həll profili radar müqayisəsi", hund=400),
            polar=dict(bgcolor="#1e2540",
                radialaxis=dict(visible=True, range=[0,100], gridcolor="#283058",
                    tickfont=dict(size=9,family="JetBrains Mono",color="#9aa5cc")),
                angularaxis=dict(gridcolor="#283058", tickfont=dict(size=10,color="#c5cceb"))))
        st.plotly_chart(fig_rad, width="stretch")

    with pf_t2:
        st.markdown('<div class="izah">Pareto frontundakı bütün dominant həllər. Sütun başlıqlarına klikləyərək sıralayın.</div>', unsafe_allow_html=True)
        disp_cols_p = ["furnace_temp","total_yield_pct","yield_gasoline","yield_diesel",
                       "yield_kerosene","energy_gj_h","sulfur_removal_pct","f_star"]
        disp_cols_p = [c for c in disp_cols_p if c in opt.pareto_df.columns]
        rename_map  = {"furnace_temp":"Soba temp.","total_yield_pct":"Ümumi verim %",
                       "yield_gasoline":"Benzin %","yield_diesel":"Dizel %",
                       "yield_kerosene":"Kerosin %","energy_gj_h":"Enerji GS/s",
                       "sulfur_removal_pct":"Kükürd çıx %","f_star":"F*"}
        pshow = opt.pareto_df[disp_cols_p].copy()
        pshow.columns = [rename_map.get(c,c) for c in pshow.columns]
        pshow.index   = range(1, len(pshow)+1)
        fmt_p = {c:"{:.2f}" for c in pshow.columns}
        if "F*" in pshow.columns: fmt_p["F*"] = "{:.4f}"
        st.dataframe(pshow.style.format(fmt_p)
            .background_gradient(subset=["F*"] if "F*" in pshow.columns else [], cmap="YlOrRd")
            .background_gradient(subset=["Enerji GS/s"] if "Enerji GS/s" in pshow.columns else [], cmap="Blues_r"),
            width="stretch", height=380)
    sp()

    # ── Həssaslıq analizi — yalnız 3 əsas parametr ────────────────────────
    bb("Həssaslıq analizi — parametrin verim/enerji üzərindəki təsiri")
    st.markdown('<div class="izah">Bir parametr dəyişdirilir, digərləri optimal nöqtədə sabit qalır. '
        'Hansı parametrin ən çox təsirli olduğunu müəyyənləşdirir.</div>', unsafe_allow_html=True)

    # Yalnız 3 əsas parametr
    HASSAS_PARAMS = ["furnace_temp", "flow_rate", "reflux_ratio"]
    HASSAS_AD = {
        "furnace_temp":  "Soba temp. (°C)",
        "flow_rate":     "Axın sürəti (m³/s)",
        "reflux_ratio":  "Refluks nisbəti",
    }

    hs1, hs2 = st.columns([1, 3], gap="medium")
    with hs1:
        da = st.selectbox("Dəyişən parametr", HASSAS_PARAMS,
            format_func=lambda x: HASSAS_AD.get(x, x), key="hss")
        di = VAR_NAMES.index(da)
    with hs2:
        baza   = opt.pareto_X[np.argmin(np.linalg.norm(opt.pareto_F, axis=1))]
        hdf2   = sensitivity_analysis(baza, di)
        xc     = VAR_NAMES[di]
        fig_hs = go.Figure()
        for ad, sc, reng in [
            ("Benzin (%)","yield_gasoline",RENG["benzin"]),
            ("Dizel (%)", "yield_diesel",  RENG["dizel"]),
            ("Kerosin (%)","yield_kerosene",RENG["kerosin"]),
        ]:
            fig_hs.add_trace(go.Scatter(x=hdf2[xc], y=hdf2[sc], name=ad,
                line=dict(color=reng, width=2),
                hovertemplate=f"<b>{ad}</b>: %{{y:.2f}}%<extra></extra>"))
        fig_hs.add_trace(go.Scatter(x=hdf2[xc], y=hdf2["energy_gj_h"], name="Enerji (GS/s)",
            yaxis="y2", line=dict(color=RENG["enerji"], width=2, dash="dot"),
            hovertemplate="<b>Enerji</b>: %{y:.3f} GS/s<extra></extra>"))
        fig_hs.update_layout(**QL(f"Həssaslıq: {HASSAS_AD.get(da, da)}", hund=320),
            yaxis2=dict(overlaying="y", side="right", gridcolor="#283058",
                zeroline=False, tickfont=dict(size=9,color="#9aa5cc"), title_text="Enerji (GS/s)"))
        fig_hs.update_yaxes(title_text="Verim (%)")
        fig_hs.update_xaxes(title_text=HASSAS_AD.get(da, da))
        st.plotly_chart(fig_hs, width="stretch")


# ══════════════════════════════════════════════════════════════════════════
# SƏHİFƏ 5 — NƏTİCƏLƏR
# ══════════════════════════════════════════════════════════════════════════
elif sehife == " Nəticələr":
    st.markdown("## Optimallaşdırma Nəticələri")
    st.markdown("Pareto frontundan ən yaxşı kompromis həll — utopiya nöqtəsinə ən yaxın.")
    sp()

    opt = _OPT
    if opt is None:
        with st.spinner("Nəticələr hesablanır…"):
            opt = optimal_yukle(alfa, beta, qamma, delta, epsilon, pop_olcu, nesl_sayi, _ver="v3")
    ey = opt.best_compromise

    bb("Optimal iş nöqtəsi — KKT şərtlərini ödəyən ən yaxşı həll")
    n1,n2,n3,n4,n5,n6 = st.columns(6)
    n1.metric("Soba temp.",      f"{ey['furnace_temp']:.1f} °C")
    n2.metric("Benzin verimi",   f"{ey['yield_gasoline']:.1f} %")
    n3.metric("Dizel verimi",    f"{ey['yield_diesel']:.1f} %")
    n4.metric("Ümumi verim",     f"{ey['total_yield_pct']:.1f} %")
    n5.metric("Enerji",          f"{ey['energy_gj_h']:.3f} GS/s")
    n6.metric("F* balı",         f"{ey['f_star']:.4f}")
    sp()

    dv   = ey.get("total_yield_pct", 0) - xulase["avg_total_yield"]
    de   = ey.get("energy_gj_h", 0)    - xulase["avg_energy"]
    # Cari və optimal F* balları
    alpha_v, beta_v, gamma_v, delta_v, epsilon_v = alfa, beta, qamma, delta, epsilon
    E_MAX_V = 22.0
    f_cari = (alpha_v*(xulase["avg_yield_gasoline"]/100)
              + beta_v*(xulase["avg_yield_diesel"]/100)
              + gamma_v*(xulase["avg_yield_kerosene"]/100)
              - delta_v*(xulase["avg_energy"]/E_MAX_V)
              + epsilon_v*(xulase["avg_sulfur_removal"]/100))
    f_opt  = ey.get("f_star", 0)
    df_star = f_opt - f_cari

    # Enerji işarəsi: müsbət = optimal daha çox enerji işlədir (tradeoff)
    e_isare = "+" if de >= 0 else "-"
    e_rengi = "#d45060" if de > 0 else "#2ec98a"
    e_label = "ENERJİ ARTIMI" if de >= 0 else "ENERJİ AZALMASI"
    il_enerji = abs(de) * 8760

    st.markdown(
        f'<div class="optimal-card">'
        f'<h4 style="color:#f0ac30;margin:0 0 .5rem;font-size:1rem">Optimallaşdırma nəticəsi (cari → optimal)</h4>'
        f'<div style="display:flex;gap:2rem;flex-wrap:wrap">'
        f'<div><span style="color:#9aa5cc;font-size:.75rem">VERİM ARTIMI</span><br>'
        f'<span style="color:#2ec98a;font-size:1.3rem;font-family:JetBrains Mono;font-weight:700">+{dv:.2f}%</span></div>'
        f'<div><span style="color:#9aa5cc;font-size:.75rem">{e_label}</span><br>'
        f'<span style="color:{e_rengi};font-size:1.3rem;font-family:JetBrains Mono;font-weight:700">{e_isare}{abs(de):.3f} GS/s</span></div>'
        f'<div><span style="color:#9aa5cc;font-size:.75rem">F* ARTIMI</span><br>'
        f'<span style="color:#2ec98a;font-size:1.3rem;font-family:JetBrains Mono;font-weight:700">+{df_star:.4f}</span></div>'
        f'<div><span style="color:#9aa5cc;font-size:.75rem">İLLİK ENERJİ DƏYİŞİMİ</span><br>'
        f'<span style="color:{e_rengi};font-size:1.3rem;font-family:JetBrains Mono;font-weight:700">{e_isare}{il_enerji:,.0f} GJ/il</span></div>'
        f'<div><span style="color:#9aa5cc;font-size:.75rem">PARETO HƏLL</span><br>'
        f'<span style="color:#edf1fc;font-size:1.3rem;font-family:JetBrains Mono;font-weight:700">{opt.n_solutions} həll</span></div>'
        f'</div>'
        f'<div style="margin-top:.6rem;font-size:.78rem;color:#6878a8">'
        f'Cari F*={f_cari:.4f} → Optimal F*={f_opt:.4f} · '
        f'Cari enerji={xulase["avg_energy"]:.3f} → Optimal={ey.get("energy_gj_h",0):.3f} GS/s'
        f'</div></div>', unsafe_allow_html=True)
    sp()

    rc1, rc2 = st.columns([1.2, 1.8], gap="medium")
    with rc1:
        bb("Optimal vs Cari radar müqayisəsi")
        rk = ["Benzin %","Dizel %","Kerosin %","Kükürd çıx.%","Enerji sərfiy."]
        ov = [ey["yield_gasoline"], ey["yield_diesel"], ey["yield_kerosene"],
              ey["sulfur_removal_pct"], 100-(ey["energy_gj_h"]/22.0)*100]
        cv = [xulase["avg_yield_gasoline"], xulase["avg_yield_diesel"], xulase["avg_yield_kerosene"],
              xulase["avg_sulfur_removal"], 100-(xulase["avg_energy"]/22.0)*100]
        fig_rad = go.Figure()
        for ad, vals, reng, dolu in [
            ("Optimal", ov, RENG["benzin"], "rgba(212,148,28,.12)"),
            ("Cari",    cv, RENG["dizel"],  "rgba(74,143,217,.10)"),
        ]:
            fig_rad.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=rk+[rk[0]],
                name=ad, fill="toself", fillcolor=dolu, line=dict(color=reng, width=2.5),
                hovertemplate=f"<b>{ad}</b>: %{{r:.1f}}<extra></extra>"))
        fig_rad.update_layout(**QL("Optimal vs Cari profil", hund=430))
        fig_rad.update_layout(
            polar=dict(bgcolor="#1e2540",
                domain=dict(x=[0.05,0.95], y=[0.05,0.95]),
                radialaxis=dict(visible=True, range=[0,100], gridcolor="#283058",
                    tickfont=dict(size=10,family="JetBrains Mono",color="#9aa5cc"),
                    tickvals=[20,40,60,80,100], gridwidth=1, linewidth=0),
                angularaxis=dict(gridcolor="#283058", gridwidth=1,
                    tickfont=dict(size=13,color="#c5cceb",family="Inter"),
                    rotation=90, direction="clockwise")))
        fig_rad.update_layout(margin=dict(l=30,r=30,t=50,b=30))
        st.plotly_chart(fig_rad, width="stretch")

    with rc2:
        bb("Tam Pareto frontu — sıralı optimal həllər")
        gs  = ["furnace_temp","column_pressure","flow_rate","yield_gasoline","yield_diesel",
               "yield_kerosene","total_yield_pct","energy_gj_h","sulfur_removal_pct","f_star"]
        mv  = [s for s in gs if s in opt.pareto_df.columns]
        sc2 = "f_star" if "f_star" in opt.pareto_df.columns else opt.pareto_df.columns[-1]
        ct  = opt.pareto_df[mv].sort_values(sc2, ascending=False).reset_index(drop=True)
        st.dataframe(ct.style.format("{:.3f}").background_gradient(subset=[sc2], cmap="YlOrRd"),
                     width="stretch", height=390)
    sp()

    sp()
    bb("Vakuum Distilləsi — Sənəd bölmə 1.2")
    st.markdown('<div class="izah">Atmosfer distilləsindən sonrakı mərhələ. '
        'Mazutdan ağır fraksiyalar ayrılır. Vakuum altında qaynama nöqtəsi azalır → yüksək temp. parçalanma baş vermir.<br>'
        'Qazoylu (katalitik krekinq üçün xammal) ayrılır · Qalıq: qudron (yol örtüyü, izolyasiya materialı)</div>',
        unsafe_allow_html=True)

    # Cari verim məlumatlarına əsasən vakuum distilləsi məhsul paylanması
    _vd_toplam = ey.get("total_yield_pct", 65.0)          # atmosfer distilləsi çıxışı
    _vd_qaliq  = 100.0 - _vd_toplam                       # ağır fraksiya (mazut)
    # Vakuum distilləsi tipik paylanması — mazut qalığının faizi kimi
    _vgo_pct   = _vd_qaliq * 0.45   # Qazoylu (VGO)
    _mazut_pct = _vd_qaliq * 0.30   # Mazut qalığı
    _yag_pct   = _vd_qaliq * 0.15   # Yağ fraksiyaları
    _qudron    = _vd_qaliq * 0.10   # Qudron

    _vd_cols = ["Qazoylu (VGO)", "Mazut qalığı", "Yağ fraksiyaları", "Qudron"]
    _vd_vals = [_vgo_pct, _mazut_pct, _yag_pct, _qudron]
    _vd_rengler = [RENG["benzin"], RENG["dizel"], RENG["kerosin"], "#7c6a4a"]

    fig_vd = go.Figure()
    for ad, val, reng in zip(_vd_cols, _vd_vals, _vd_rengler):
        fig_vd.add_trace(go.Bar(
            name=ad, x=[ad], y=[round(val, 2)],
            marker=dict(color=reng, opacity=0.88, line=dict(color="#1a1f35", width=0.5)),
            text=[f"{val:.1f}%"], textposition="outside",
            textfont=dict(size=11, family="JetBrains Mono", color="#c5cceb"),
            hovertemplate=f"<b>{ad}</b><br>Faiz: %{{y:.1f}}%<br>(Atmosfer çıxışı: {_vd_toplam:.1f}% → Qalıq: {_vd_qaliq:.1f}%)<extra></extra>",
        ))
    fig_vd.add_annotation(
        text=f"Atmosfer distilləsi çıxışı: {_vd_toplam:.1f}% (benzin+dizel+kerosin)<br>"
             f"Vakuum distilləsi girişi (mazut qalığı): {_vd_qaliq:.1f}%",
        xref="paper", yref="paper", x=0.5, y=-0.22, showarrow=False,
        font=dict(size=10, color="#6878a8"), align="center")
    fig_vd.update_layout(**QL(f"Vakuum distilləsi — məhsul paylanması (tipik, optimal nöqtəyə əsasən)", hund=320, legend=True))
    fig_vd.update_yaxes(title_text="Faiz (%)", range=[0, max(_vd_vals)*1.3])
    fig_vd.update_xaxes(showticklabels=False)
    fig_vd.update_layout(barmode="group", showlegend=True,
        legend=dict(orientation="h", y=-0.35, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10, color="#9aa5cc")))
    st.plotly_chart(fig_vd, width="stretch")

    st.markdown(
        f'<div class="izah">Layihədə vakuum distilləsi mərhələsi qalıq fraksiya kimi sadələşdirilmişdir. Atmosfer distilləsinin çıxışı: benzin ({ey.get("yield_gasoline",0):.1f}%) + dizel ({ey.get("yield_diesel",0):.1f}%) + kerosin ({ey.get("yield_kerosene",0):.1f}%) + qalıq/mazut ({_vd_qaliq:.1f}%). Vakuum distilləsi bu qalığı yuxarıdakı fraksiyalara ayırır.</div>',
        unsafe_allow_html=True)
    sp()

    bb("Metodologiya xülasəsi")
    m1, m2 = st.columns(2, gap="medium")
    with m1:
        st.markdown('<div class="izah">'
            '<b>Riyazi model (sənəd II fəsil):</b><br>'
            '• F* = α·Ybenz + β·Ydiz + γ·Yker − δ·E (tənlik 1)<br>'
            '• KKT şərtləri: g(x)≤0, h(x)=0 (tənliklər 6–13)<br>'
            '• Balans tənlikləri: kütlə (3), enerji (4)<br>'
            '• Temperatur profili: T(z)=T_alt−k·z (5)<br>'
            '• Reqressiya: Y=a₁T+a₂P+a₃F+a₄+ε<br>'
            '• F*=α·Yb+β·Yd+γ·Yk−δ·(E/E<sub>max</sub>)+ε·S<sub>removal</sub></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="izah">'
            '<b>Alqoritm (sənəd III fəsil):</b><br>'
            '• NSGA-II — Non-dominated Sorting Genetic Algorithm II<br>'
            '• ARO — Ardıcıl Reqressiya Optimallaşdırma<br>'
            '• IQR + Isolation Forest anomaliya aşkarlanması<br>'
            '• Min-Maks normallaşdırma: [0,1] aralığı<br>'
            '• Pareto frontu — kompromis həllərin məcmusu</div>', unsafe_allow_html=True)
    sp()

    st.download_button("Pareto Frontunu Yüklə (CSV)",
        data=opt.pareto_df.to_csv(index=False),
        file_name="pareto_frontu_nsga2.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════
# AI KÖMƏKÇİ — Chat panel (default bağlı)
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.quick-action-btn > button {
    background: #111828 !important; color: #6878a8 !important;
    border: 1px solid #283058 !important; border-radius: 8px !important;
    padding: .3rem .65rem !important; font-size: .73rem !important;
    font-weight: 500 !important; transition: all .18s ease !important;
    letter-spacing: .01em !important; box-shadow: none !important;
}
.quick-action-btn > button:hover {
    background: rgba(212,148,28,.08) !important;
    border-color: rgba(212,148,28,.5) !important; color: #d4941c !important;
}
div[data-testid="stTextInput"] input {
    background: #151e34 !important; color: #edf1fc !important;
    border: 1px solid #283058 !important; border-radius: 10px !important;
    font-size: .84rem !important;
}
</style>""", unsafe_allow_html=True)

st.markdown("---")

# Session state başlatma
if "mesajlar"        not in st.session_state: st.session_state.mesajlar = []
if "chatbot"         not in st.session_state: st.session_state.chatbot  = chatbot_yarat()
if "tts_aktiv"       not in st.session_state: st.session_state.tts_aktiv = True
if "tts_ses"         not in st.session_state: st.session_state.tts_ses   = TTS_ELEVENLABS_VOICE
if "tts_surat"       not in st.session_state: st.session_state.tts_surat = 1.0
if "active_audio_key" not in st.session_state: st.session_state["active_audio_key"] = None

# Aktiv səhifəni təyin et
_aktiv_sehife_map = {
    " İcmal":            "İcmal",
    " Məlumat Analizi":  "Məlumat Analizi",
    " Emal Boru Kəməri": "Emal Boru Kəməri",
    " Optimallaşdırma":  "Optimallaşdırma",
    " Nəticələr":        "Nəticələr",
}
_aktiv = _aktiv_sehife_map.get(sehife, "İcmal")

# Optimallaşdırma nəticəsi — yalnız cache-dən al, hesablama yox
_opt_kontekst = _OPT

_kontekst = kontekst_yarat(
    xulase, _opt_kontekst, aktiv_sehife=_aktiv,
    çəkilər={"alpha": alfa, "beta": beta, "gamma": qamma,
              "delta": delta, "epsilon": epsilon},
)

# Chat panel başlığı
if "chat_open" not in st.session_state: st.session_state.chat_open = False

chat_col1, chat_col2, chat_col3 = st.columns([4, 1, 1])
with chat_col1:
    st.markdown('<div class="chat-header"><div class="chat-status"></div>'
        '<div><div class="chat-header-title">MeloSense</div>'
        '<div class="chat-header-sub">Proses analitik mühərriki · OpenAI GPT-4o</div></div></div>',
        unsafe_allow_html=True)
with chat_col2:
    if st.button('Aç' if not st.session_state.chat_open else 'Bağla', key='chat_toggle'):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()
with chat_col3:
    tts_checkbox = st.checkbox("TTS", value=st.session_state.tts_aktiv, key="tts_toggle")
    st.session_state.tts_aktiv = tts_checkbox

if st.session_state.chat_open:

    if st.session_state.tts_aktiv:
        st.session_state.tts_surat = st.slider("Sürət", 0.5, 2.0, 1.0, 0.1, label_visibility="collapsed")

    qa1, qa2, qa3, qa4, qa5 = st.columns(5)
    tez_sorgu = None
    with qa1:
        st.markdown('<div class="quick-action-btn">', unsafe_allow_html=True)
        if st.button("Pareto analizi", key="qa_pareto"): tez_sorgu = "pareto"
        st.markdown('</div>', unsafe_allow_html=True)
    with qa2:
        st.markdown('<div class="quick-action-btn">', unsafe_allow_html=True)
        if st.button("Anomaliya", key="qa_anom"): tez_sorgu = "anomaliya"
        st.markdown('</div>', unsafe_allow_html=True)
    with qa3:
        st.markdown('<div class="quick-action-btn">', unsafe_allow_html=True)
        if st.button("Tövsiyələr", key="qa_tov"): tez_sorgu = "tövsiyə"
        st.markdown('</div>', unsafe_allow_html=True)
    with qa4:
        st.markdown('<div class="quick-action-btn">', unsafe_allow_html=True)
        if st.button("F* tənliyi", key="qa_tenlik"): tez_sorgu = "tənlik"
        st.markdown('</div>', unsafe_allow_html=True)
    with qa5:
        st.markdown('<div class="quick-action-btn">', unsafe_allow_html=True)
        if st.button("Müqayisə", key="qa_muq"): tez_sorgu = "müqayisə"
        st.markdown('</div>', unsafe_allow_html=True)

    if tez_sorgu and st.session_state.chatbot:
        _TEZ_ADLAR = {
            "pareto":    "Pareto frontunu şərh et",
            "anomaliya": "Anomaliyaları analiz et",
            "tövsiyə":   "Operator tövsiyələri ver",
            "tənlik":    "F* tənliyini hesabla",
            "müqayisə":  "Cari vs optimal müqayisə",
        }
        parca_list = list(st.session_state.chatbot.tez_analiz(_kontekst, tez_sorgu))
        cavab_tez  = parca_list[-1] if parca_list else ""
        if cavab_tez:
            istifadeci_mesaj = _TEZ_ADLAR.get(tez_sorgu, tez_sorgu)
            st.session_state.mesajlar.append({"role": "user",      "content": istifadeci_mesaj})
            st.session_state.mesajlar.append({"role": "assistant", "content": cavab_tez})
            st.rerun()

    chat_box = st.container()
    with chat_box:
        _active_audio_key = st.session_state.get("active_audio_key")
        ai_idx = 0
        for i, msg in enumerate(st.session_state.mesajlar):
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='background:#d4941c;color:#141828;border-radius:14px 14px 4px 14px;"
                    f"padding:.6rem .9rem;font-size:.84rem;font-weight:500;margin:.3rem 0 .3rem 3rem;"
                    f"line-height:1.5'>{msg['content']}</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='background:#151e34;color:#c5cceb;border:1px solid #283058;"
                    f"border-radius:14px 14px 14px 4px;padding:.65rem .95rem;font-size:.84rem;"
                    f"margin:.3rem 3rem .2rem 0;line-height:1.6'>{msg['content']}</div>",
                    unsafe_allow_html=True)
                _btn_key   = f"play_{i}_{ai_idx}"
                _audio_key = f"audio_{i}_{ai_idx}"
                _is_active = (_active_audio_key == _audio_key)
                btn_col, _ = st.columns([1, 5])
                with btn_col:
                    if st.session_state.tts_aktiv and st.session_state.chatbot:
                        if st.button(
                            "Bağla" if _is_active else "Səsləndir",
                            key=_btn_key, width="stretch",
                            disabled=st.session_state.get("is_generating", False),
                        ):
                            if _is_active:
                                st.session_state["active_audio_key"] = None
                                st.rerun()
                            else:
                                if st.session_state.get(_audio_key) is None:
                                    with st.spinner("Səs hazırlanır…"):
                                        audio_bytes, motor = st.session_state.chatbot.ses_yarat(
                                            metn=msg["content"],
                                            ses=st.session_state.get("tts_ses", TTS_ELEVENLABS_VOICE),
                                            surət=st.session_state.get("tts_surat", 1.0),
                                        )
                                    st.session_state[_audio_key] = audio_bytes
                                    st.session_state[f"motor_{i}_{ai_idx}"] = motor
                                st.session_state["active_audio_key"] = _audio_key
                                st.rerun()
                if st.session_state.get(_audio_key) and _is_active:
                    st.audio(st.session_state[_audio_key], format="audio/mp3", autoplay=True)
                    st.markdown('<span class="info-badge" style="color:#d4941c">OpenAI TTS</span>',
                        unsafe_allow_html=True)
                ai_idx += 1

    _generating = st.session_state.get("is_generating", False)
    with st.form("chat_form", clear_on_submit=True):
        inp_col, btn_col = st.columns([9, 1])
        with inp_col:
            user_input = st.text_input(
                "msg",
                placeholder="Cavab gəlir, gözləyin…" if _generating else "Məsələn: Niyə benzin verimi aşağıdır?",
                label_visibility="collapsed",
                disabled=_generating,
            )
        with btn_col:
            gonder = st.form_submit_button("→", width="stretch", disabled=_generating)

    def stream_cavab(mesajlar, kontekst):
        tam = ""
        yazi_yeri = st.empty()
        bashlama  = time.time()
        st.session_state.is_generating = True
        yazi_yeri.markdown(
            "<div style='background:#151e34;color:#d4941c;border:1px solid #283058;"
            "border-radius:14px 14px 14px 4px;padding:.65rem .95rem;font-size:.84rem;"
            "margin:.3rem 3rem .3rem 0'>Yazır…</div>",
            unsafe_allow_html=True)
        try:
            for parca in st.session_state.chatbot.cavab_ver_stream(mesajlar, kontekst):
                tam += parca
                yazi_yeri.markdown(
                    f"<div style='background:#151e34;color:#c5cceb;border:1px solid #283058;"
                    f"border-radius:14px 14px 14px 4px;padding:.65rem .95rem;font-size:.84rem;"
                    f"margin:.3rem 3rem .3rem 0;line-height:1.6'>"
                    f"{tam}<span class='typing-cursor'></span></div>",
                    unsafe_allow_html=True)
        except Exception:
            tam = "Cavab alınarkən problem yarandı. Yenidən cəhd edin."
            yazi_yeri.markdown(
                f"<div style='background:#1a0e0e;color:#f87171;border:1px solid #5a2020;"
                f"border-radius:14px;padding:.65rem .95rem;font-size:.84rem;"
                f"margin:.3rem 3rem .3rem 0'>{tam}</div>",
                unsafe_allow_html=True)
        st.session_state.son_latency  = round(time.time() - bashlama, 2)
        st.session_state.is_generating = False
        yazi_yeri.empty()
        return tam

    if gonder and user_input.strip():
        st.session_state.mesajlar.append({"role": "user", "content": user_input.strip()})
        with chat_box:
            cavab = stream_cavab(st.session_state.mesajlar, _kontekst)
        st.session_state.mesajlar.append({"role": "assistant", "content": cavab})
        st.rerun()
    elif (st.session_state.mesajlar and
          st.session_state.mesajlar[-1]["role"] == "user"):
        with chat_box:
            cavab = stream_cavab(st.session_state.mesajlar, _kontekst)
        st.session_state.mesajlar.append({"role": "assistant", "content": cavab})
        st.rerun()

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        if st.button("Sil", key="clear_chat"):
            st.session_state.mesajlar = []
            audio_keys = [k for k in st.session_state if k.startswith("audio_") or k.startswith("motor_")]
            for k in audio_keys: del st.session_state[k]
            st.session_state["active_audio_key"] = None
            st.rerun()
    with f2:
        st.markdown(f'<span class="info-badge">{len(st.session_state.mesajlar)} mesaj</span>',
                    unsafe_allow_html=True)
    with f3:
        st.markdown('<span class="info-badge">MeloSense · GPT-4o</span>', unsafe_allow_html=True)
    with f4:
        lat = st.session_state.get("son_latency", 0)
        if lat > 0:
            st.markdown(f'<span class="info-badge">{lat}s</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="info-badge">Hazır</span>', unsafe_allow_html=True)
