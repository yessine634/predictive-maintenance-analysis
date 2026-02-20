"""
Predictive Maintenance - Machine Failure Classification App
Supports:
  • Binary Classification  → Machine failure (0/1) via XGBoost pipeline
  • Multi-class Classification → Failure type (None/OSF/HDF/PWF/TWF) via ANN (PyTorch)

Usage:
    streamlit run predictive_maintenance_app.py

Model files expected in the same directory:
    xgboost_pipeline_model.pkl          – sklearn Pipeline (StandardScaler + XGBClassifier)
    predictive_maintenance_model.pth    – ANN state-dict  (or best_model.pth)
    scaler_multiclass.pkl               – StandardScaler used for ANN input  (optional)
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── optional torch import ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – industrial/utilitarian dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #f0a500;
    --accent2:   #e05c00;
    --green:     #3fb950;
    --red:       #f85149;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --mono:      'Share Tech Mono', monospace;
    --sans:      'Barlow', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans);
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* headings */
h1 { font-family: var(--mono); color: var(--accent) !important; letter-spacing: 2px; }
h2 { font-family: var(--sans); color: var(--text) !important; font-weight: 600; }
h3 { font-family: var(--sans); color: var(--muted) !important; font-weight: 400; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 1px; }

/* cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-accent { border-left: 4px solid var(--accent); }

/* metric badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 1px;
}
.badge-green { background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid var(--green); }
.badge-red   { background: rgba(248,81,73,0.15);  color: var(--red);   border: 1px solid var(--red);   }
.badge-amber { background: rgba(240,165,0,0.15);  color: var(--accent); border: 1px solid var(--accent); }

/* inputs */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
    background: #21262d !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    border-radius: 4px !important;
}

/* buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.5rem 1.5rem !important;
    letter-spacing: 1px !important;
    transition: background 0.2s;
}
.stButton > button:hover { background: var(--accent2) !important; }

/* file uploader */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

/* progress / divider */
hr { border-color: var(--border) !important; }

/* tab */
[data-testid="stTabs"] [role="tablist"] { gap: 4px; }
button[role="tab"] {
    font-family: var(--mono) !important;
    color: var(--muted) !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    letter-spacing: 1px;
}
button[role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* hide default streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ANN architecture (must match notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class PredicMaint(nn.Module):
        def __init__(self, input_dim=5, output_dim=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(32, output_dim),
            )

        def forward(self, x):
            return self.net(x)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(".")

@st.cache_resource
def load_xgb_model(path):
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_ann_model(pth_path, load_mode="state_dict", scaler_path="C:\\Users\\DELL\\Desktop\\Predective maintainance\\multiclass_scaler.pkl"):
    if not TORCH_AVAILABLE:
        return None, None, "PyTorch not installed"
    try:
        if load_mode == "full_model":
            # torch.save(model, ...) — full model object
            model = torch.load(pth_path, map_location="cpu")
            model.eval()
        else:
            # torch.save(model.state_dict(), ...) — weights only
            model = PredicMaint(input_dim=5, output_dim=5)
            state = torch.load(pth_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
    except Exception as e:
        # fallback: try the other mode automatically
        try:
            if load_mode == "state_dict":
                model = torch.load(pth_path, map_location="cpu")
                model.eval()
            else:
                model = PredicMaint(input_dim=5, output_dim=5)
                state = torch.load(pth_path, map_location="cpu")
                model.load_state_dict(state)
                model.eval()
        except Exception as e2:
            return None, None, f"state_dict attempt: {e} | full_model attempt: {e2}"

    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            pass
    return model, scaler, None

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – model file selection
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ PRED-MAINT")
    st.markdown("---")
    st.markdown("### 📂 Model Files")

    XGB_PATH = r"C:\Users\DELL\Desktop\Predective maintainance\xgboost_pipeline_model.pkl"
    ANN_PATH = r"C:\Users\DELL\Desktop\Predective maintainance\predictive_maintenance_model_20260214_164943.pth"

    xgb_file = st.text_input("XGBoost pipeline (.pkl)", XGB_PATH)
    ann_file  = st.text_input("ANN state-dict (.pth)", ANN_PATH)
    ann_load_mode = "state_dict"
    scaler_file = st.text_input("ANN scaler (.pkl) – optional", "")

    st.markdown("---")
    st.markdown("### ℹ️ Feature Reference")
    st.markdown("""
| Feature | Range |
|---|---|
| Air Temp (K) | 290 – 320 |
| Process Temp (K) | 300 – 325 |
| Rot. Speed (rpm) | 1000 – 3000 |
| Torque (Nm) | 0 – 100 |
| Tool Wear (min) | 0 – 300 |
""")

    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────
xgb_model, xgb_err = load_xgb_model(xgb_file) if os.path.exists(xgb_file) else (None, f"{xgb_file} not found")
ann_model, ann_scaler, ann_err = load_ann_model(ann_file, ann_load_mode, scaler_file if os.path.exists(scaler_file) else None) if os.path.exists(ann_file) else (None, None, f"{ann_file} not found")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – model parameters (rendered after models are loaded)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔍 Model Parameters")

    # ── XGBoost parameters ───────────────────────────────────────────────────
    with st.expander("⚡ XGBoost Hyperparameters", expanded=False):
        if xgb_model:
            try:
                xgb_clf = xgb_model.named_steps.get("clf") or xgb_model.steps[-1][1]
                params = xgb_clf.get_params()
                display_keys = [
                    "n_estimators", "max_depth", "learning_rate",
                    "subsample", "colsample_bytree", "scale_pos_weight",
                    "eval_metric", "random_state",
                ]
                for k in display_keys:
                    v = params.get(k, "—")
                    if isinstance(v, float):
                        v = f"{v:.4f}"
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #30363d">'                        f'<span style="color:#8b949e;font-size:0.78rem;font-family:Share Tech Mono,monospace">{k}</span>'                        f'<span style="color:#f0a500;font-size:0.78rem;font-family:Share Tech Mono,monospace">{v}</span>'                        f'</div>',
                        unsafe_allow_html=True,
                    )
                # preprocessor info
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<span style="color:#3fb950;font-size:0.75rem;font-family:Share Tech Mono,monospace">'                    "Preprocessor: StandardScaler</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<span style="color:#3fb950;font-size:0.75rem;font-family:Share Tech Mono,monospace">'                    "Features: 5 numeric</span>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.warning(f"Could not read XGBoost params: {e}")
        else:
            st.info("Load XGBoost model to view parameters.")

    # ── ANN parameters ────────────────────────────────────────────────────────
    with st.expander("🧠 ANN Architecture", expanded=False):
        if ann_model:
            try:
                ann_params = [
                    ("Framework",    "PyTorch"),
                    ("Input dim",    "5"),
                    ("Output dim",   "5 classes"),
                    ("Optimizer",    "Adam"),
                    ("LR",           "0.001"),
                    ("Weight decay", "1e-4"),
                    ("Batch size",   "64"),
                    ("Loss fn",      "CrossEntropyLoss"),
                    ("Scheduler",    "ReduceLROnPlateau"),
                ]
                for k, v in ann_params:
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #30363d">'                        f'<span style="color:#8b949e;font-size:0.78rem;font-family:Share Tech Mono,monospace">{k}</span>'                        f'<span style="color:#f0a500;font-size:0.78rem;font-family:Share Tech Mono,monospace">{v}</span>'                        f'</div>',
                        unsafe_allow_html=True,
                    )
                # Layer breakdown
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<span style="color:#8b949e;font-size:0.75rem;font-family:Share Tech Mono,monospace">'                    "Layer stack:</span>",
                    unsafe_allow_html=True,
                )
                layers = [
                    ("Linear",  "5 → 256",  "BN + ReLU + Drop(0.30)"),
                    ("Linear",  "256 → 128", "BN + ReLU + Drop(0.25)"),
                    ("Linear",  "128 → 64",  "BN + ReLU + Drop(0.20)"),
                    ("Linear",  "64 → 32",   "BN + ReLU + Drop(0.15)"),
                    ("Linear",  "32 → 5",    "Output (logits)"),
                ]
                for ltype, dims, note in layers:
                    st.markdown(
                        f'<div style="padding:4px 0;border-bottom:1px solid #21262d">'                        f'<span style="color:#a371f7;font-size:0.75rem;font-family:Share Tech Mono,monospace">{ltype} {dims}</span>'                        f'<br><span style="color:#8b949e;font-size:0.7rem;font-family:Share Tech Mono,monospace">&nbsp;&nbsp;{note}</span>'                        f'</div>',
                        unsafe_allow_html=True,
                    )
                # total params
                total_params = sum(p.numel() for p in ann_model.parameters())
                trainable    = sum(p.numel() for p in ann_model.parameters() if p.requires_grad)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f'<span style="color:#3fb950;font-size:0.75rem;font-family:Share Tech Mono,monospace">'                    f"Total params: {total_params:,}<br>Trainable: {trainable:,}</span>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.warning(f"Could not read ANN params: {e}")
        else:
            st.info("Load ANN model to view architecture.")

    st.markdown("---")
    st.markdown("<p style='color:#8b949e;font-size:0.75rem;'>Predictive Maintenance v1.0<br>XGBoost + ANN Classifier</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1>⚙ PREDICTIVE MAINTENANCE</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e;font-family:Share Tech Mono,monospace;letter-spacing:1px;'>Machine Failure Detection &amp; Root-Cause Classification</p>", unsafe_allow_html=True)

# model status row
c1, c2 = st.columns(2)
with c1:
    if xgb_model:
        st.markdown('<span class="badge badge-green">✓ XGBoost LOADED</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge badge-red">✗ XGBoost: {xgb_err}</span>', unsafe_allow_html=True)
with c2:
    if ann_model:
        st.markdown('<span class="badge badge-green">✓ ANN LOADED</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge badge-red">✗ ANN: {ann_err}</span>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build donut chart
# ─────────────────────────────────────────────────────────────────────────────
FAILURE_COLORS = {
    "None": "#3fb950",
    "OSF":  "#f0a500",
    "HDF":  "#e05c00",
    "PWF":  "#f85149",
    "TWF":  "#a371f7",
}

def donut_chart(probabilities: dict, title: str):
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [FAILURE_COLORS.get(l, "#8b949e") for l in labels]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor="none")
    wedges, _ = ax.pie(
        values, labels=None, colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="#0d1117", linewidth=2),
        startangle=90,
    )
    ax.text(0, 0, f"{max(values):.1%}", ha="center", va="center",
            fontsize=14, color="#e6edf3", fontweight="bold",
            fontfamily="monospace")
    ax.set_title(title, color="#e6edf3", fontsize=10, pad=8,
                 fontfamily="monospace")
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, loc="lower center", ncol=3,
              fontsize=7, frameon=False,
              labelcolor="#8b949e", bbox_to_anchor=(0.5, -0.25))
    fig.patch.set_alpha(0)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────
TYPE_MAP = {"L": 0, "M": 1, "H": 2}
FAILURE_TYPES = ["None", "OSF", "HDF", "PWF", "TWF"]

def predict_binary_xgb(model, air_temp, proc_temp, rot_speed, torque, tool_wear):
    """XGBoost pipeline expects 5 numeric features only (Type was dropped before training):
       Air temperature, Process temperature, Rotational speed, Torque, Tool wear."""
    df = pd.DataFrame([{
        "Air temperature": air_temp,
        "Process temperature": proc_temp,
        "Rotational speed": rot_speed,
        "Torque": torque,
        "Tool wear": tool_wear,
    }])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    return int(pred), proba

def predict_multi_ann(model, scaler, air_temp, proc_temp, rot_speed, torque, tool_wear):
    """ANN expects scaled [Air temp, Process temp, Rot speed, Torque, Tool wear]."""
    x = np.array([[air_temp, proc_temp, rot_speed, torque, tool_wear]], dtype=np.float32)
    if scaler is not None:
        x = scaler.transform(x).astype(np.float32)
    tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    pred_idx = int(np.argmax(probs))
    return FAILURE_TYPES[pred_idx], probs

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["🔬  SINGLE PREDICTION", "📋  BATCH PREDICTION"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Single prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.markdown("<h2>Machine Parameters</h2>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 1.8], gap="large")

    with col_l:
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        air_temp    = st.number_input("Air Temperature (K)",     min_value=290.0, max_value=320.0, value=300.0, step=0.1, format="%.1f")
        proc_temp   = st.number_input("Process Temperature (K)", min_value=300.0, max_value=325.0, value=310.0, step=0.1, format="%.1f")
        rot_speed   = st.number_input("Rotational Speed (rpm)",  min_value=1000,  max_value=3000,  value=1500, step=10)
        torque      = st.number_input("Torque (Nm)",             min_value=0.0,   max_value=100.0, value=40.0, step=0.5, format="%.1f")
        tool_wear   = st.number_input("Tool Wear (min)",         min_value=0,     max_value=300,   value=100,  step=1)
        st.markdown('</div>', unsafe_allow_html=True)

        run_btn = st.button("⚡  RUN PREDICTION", use_container_width=True)

    with col_r:
        if run_btn:
            # ── Binary ──────────────────────────────────────────────────────
            st.markdown("<h3>BINARY CLASSIFICATION — XGBoost</h3>", unsafe_allow_html=True)
            if xgb_model:
                pred_bin, proba_bin = predict_binary_xgb(
                    xgb_model, air_temp, proc_temp, rot_speed, torque, tool_wear)

                b1, b2 = st.columns([1, 1.6])
                with b1:
                    if pred_bin == 1:
                        st.markdown('<div class="card" style="border:2px solid #f85149;text-align:center">'
                                    '<p style="font-size:2rem;margin:0">⚠️</p>'
                                    '<p style="font-family:Share Tech Mono,monospace;color:#f85149;font-size:1.1rem;margin:0">FAILURE</p>'
                                    '<p style="color:#8b949e;font-size:0.75rem">Machine likely to fail</p>'
                                    '</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="card" style="border:2px solid #3fb950;text-align:center">'
                                    '<p style="font-size:2rem;margin:0">✅</p>'
                                    '<p style="font-family:Share Tech Mono,monospace;color:#3fb950;font-size:1.1rem;margin:0">NORMAL</p>'
                                    '<p style="color:#8b949e;font-size:0.75rem">No failure expected</p>'
                                    '</div>', unsafe_allow_html=True)
                with b2:
                    fig = donut_chart({"No Failure": proba_bin[0], "Failure": proba_bin[1]},
                                      "Failure Probability")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                st.markdown(f"""
<div class="card">
  <span style="color:#8b949e;font-size:0.8rem;font-family:Share Tech Mono,monospace">
    CONFIDENCE — No Failure: <b style="color:#3fb950">{proba_bin[0]:.2%}</b>
    &nbsp;|&nbsp; Failure: <b style="color:#f85149">{proba_bin[1]:.2%}</b>
  </span>
</div>""", unsafe_allow_html=True)
            else:
                st.warning("XGBoost model not loaded. Check file path in sidebar.")

            st.markdown("---")

            # ── Multi-class ─────────────────────────────────────────────────
            st.markdown("<h3>MULTI-CLASS CLASSIFICATION — ANN</h3>", unsafe_allow_html=True)
            if ann_model:
                pred_type, probs_mc = predict_multi_ann(
                    ann_model, ann_scaler, air_temp, proc_temp, rot_speed, torque, tool_wear)

                m1, m2 = st.columns([1, 1.6])
                with m1:
                    color = FAILURE_COLORS.get(pred_type, "#8b949e")
                    label_desc = {
                        "None": "No failure detected",
                        "OSF":  "Overstrain Failure",
                        "HDF":  "Heat Dissipation Failure",
                        "PWF":  "Power Failure",
                        "TWF":  "Tool Wear Failure",
                    }
                    st.markdown(f'<div class="card" style="border:2px solid {color};text-align:center">'
                                f'<p style="font-size:2rem;margin:0">🔧</p>'
                                f'<p style="font-family:Share Tech Mono,monospace;color:{color};font-size:1.3rem;margin:0">{pred_type}</p>'
                                f'<p style="color:#8b949e;font-size:0.75rem">{label_desc[pred_type]}</p>'
                                f'</div>', unsafe_allow_html=True)
                with m2:
                    prob_dict = {ft: float(probs_mc[i]) for i, ft in enumerate(FAILURE_TYPES)}
                    fig2 = donut_chart(prob_dict, "Failure Type Probability")
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

                # prob bar
                prob_df = pd.DataFrame({
                    "Failure Type": FAILURE_TYPES,
                    "Probability":  probs_mc,
                }).sort_values("Probability", ascending=False)
                st.dataframe(
                    prob_df.style.format({"Probability": "{:.4f}"})
                    .background_gradient(subset=["Probability"], cmap="YlOrRd"),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("ANN model not loaded. Check file path in sidebar.")
        else:
            st.markdown("""
<div class="card" style="text-align:center;padding:3rem;">
  <p style="font-size:3rem;margin:0">⚙️</p>
  <p style="color:#8b949e;font-family:Share Tech Mono,monospace;letter-spacing:2px">
    AWAITING INPUT<br>
    <span style="font-size:0.75rem">Set parameters and click RUN PREDICTION</span>
  </p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Batch prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("<h2>Batch Prediction via CSV Upload</h2>", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
  <p style="color:#8b949e;font-size:0.85rem;margin:0">
  Upload a CSV with columns: <code style="color:#f0a500">Air temperature, Process temperature, Rotational speed, Torque, Tool wear</code><br>
  Both models will be applied and results appended as new columns.
  </p>
</div>""", unsafe_allow_html=True)

    # template download
    template = pd.DataFrame({
        "Air temperature":     [298.5, 300.1, 302.3],
        "Process temperature": [308.5, 310.2, 312.0],
        "Rotational speed":    [1500, 1800, 2100],
        "Torque":              [40.0, 35.5, 28.2],
        "Tool wear":           [50, 120, 200],
    })
    buf = io.BytesIO()
    template.to_csv(buf, index=False)
    st.download_button("⬇ Download CSV Template", buf.getvalue(),
                       "template.csv", "text/csv", use_container_width=False)

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_up)} rows loaded.**")
        st.dataframe(df_up.head(5), use_container_width=True)

        if st.button("⚡  RUN BATCH PREDICTION", use_container_width=False):
            results = df_up.copy()
            required_cols = ["Air temperature", "Process temperature",
                             "Rotational speed", "Torque", "Tool wear"]
            missing = [c for c in required_cols if c not in df_up.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                prog = st.progress(0, text="Processing…")

                xgb_preds, xgb_proba_fail = [], []
                ann_preds, ann_proba_top = [], []

                for i, row in df_up.iterrows():
                    at, pt, rs, tq, tw = (
                        row["Air temperature"], row["Process temperature"],
                        row["Rotational speed"], row["Torque"],
                        row["Tool wear"]
                    )

                    # XGBoost
                    if xgb_model:
                        p, pb = predict_binary_xgb(xgb_model, at, pt, rs, tq, tw)
                        xgb_preds.append("FAILURE" if p == 1 else "Normal")
                        xgb_proba_fail.append(round(float(pb[1]), 4))
                    else:
                        xgb_preds.append("N/A"); xgb_proba_fail.append(None)

                    # ANN
                    if ann_model:
                        ft, pb2 = predict_multi_ann(ann_model, ann_scaler, at, pt, rs, tq, tw)
                        ann_preds.append(ft)
                        ann_proba_top.append(round(float(max(pb2)), 4))
                    else:
                        ann_preds.append("N/A"); ann_proba_top.append(None)

                    prog.progress((i + 1) / len(df_up), text=f"Row {i+1}/{len(df_up)}")

                results["XGB_Prediction"]       = xgb_preds
                results["XGB_Failure_Prob"]     = xgb_proba_fail
                results["ANN_Failure_Type"]     = ann_preds
                results["ANN_Top_Probability"]  = ann_proba_top

                prog.empty()
                st.success("✅ Batch prediction complete!")
                st.dataframe(results, use_container_width=True)

                # Summary charts
                if xgb_model:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("<h3>XGBoost — Failure Distribution</h3>", unsafe_allow_html=True)
                        vc = results["XGB_Prediction"].value_counts()
                        fig3, ax3 = plt.subplots(facecolor="#161b22")
                        bars = ax3.bar(vc.index, vc.values,
                                       color=["#3fb950" if v == "Normal" else "#f85149" for v in vc.index])
                        ax3.set_facecolor("#161b22")
                        ax3.tick_params(colors="#e6edf3")
                        for sp in ax3.spines.values(): sp.set_color("#30363d")
                        ax3.bar_label(bars, color="#e6edf3", fontsize=11)
                        fig3.patch.set_alpha(0)
                        st.pyplot(fig3, use_container_width=True); plt.close(fig3)

                if ann_model:
                    with col_b:
                        st.markdown("<h3>ANN — Failure Type Distribution</h3>", unsafe_allow_html=True)
                        vc2 = results["ANN_Failure_Type"].value_counts()
                        colors2 = [FAILURE_COLORS.get(v, "#8b949e") for v in vc2.index]
                        fig4, ax4 = plt.subplots(facecolor="#161b22")
                        bars2 = ax4.bar(vc2.index, vc2.values, color=colors2)
                        ax4.set_facecolor("#161b22")
                        ax4.tick_params(colors="#e6edf3")
                        for sp in ax4.spines.values(): sp.set_color("#30363d")
                        ax4.bar_label(bars2, color="#e6edf3", fontsize=11)
                        fig4.patch.set_alpha(0)
                        st.pyplot(fig4, use_container_width=True); plt.close(fig4)

                # download
                out_buf = io.BytesIO()
                results.to_csv(out_buf, index=False)
                st.download_button("⬇ Download Results CSV", out_buf.getvalue(),
                                   "batch_predictions.csv", "text/csv",
                                   use_container_width=False)