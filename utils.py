
import streamlit as st
import pandas as pd
import joblib, json, os

MODEL_DIR = "model"

@st.cache_resource
def load_model(): return joblib.load(f"{MODEL_DIR}/loan_model.pkl")
@st.cache_resource
def load_encoders(): return joblib.load(f"{MODEL_DIR}/encoders.pkl")
@st.cache_resource
def load_feature_names(): return joblib.load(f"{MODEL_DIR}/feature_names.pkl")
@st.cache_resource
def load_shap_explainer(): return joblib.load(f"{MODEL_DIR}/shap_explainer.pkl")
@st.cache_data
def load_data(): return pd.read_csv(f"{MODEL_DIR}/clean_data.csv")
@st.cache_data
def load_metrics():
    with open(f"{MODEL_DIR}/metrics.json") as f: return json.load(f)
@st.cache_data
def load_label_map():
    with open(f"{MODEL_DIR}/label_map.json") as f: return json.load(f)
@st.cache_data
def load_approved_idx():
    with open(f"{MODEL_DIR}/approved_class_idx.json") as f:
        d = json.load(f)
    return d["approved_idx"], d["classes"], d.get("threshold", 0.5)

COLORS = {
    "primary":"#6366f1","success":"#10b981","danger":"#ef4444",
    "warning":"#f59e0b","secondary":"#0ea5e9",
    "cat":["#6366f1","#10b981","#f59e0b","#ef4444","#0ea5e9","#8b5cf6"],
}
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#475569"),
    margin=dict(t=40,b=40,l=40,r=20),
    xaxis=dict(gridcolor="#f1f5f9",showline=False,zeroline=False),
    yaxis=dict(gridcolor="#f1f5f9",showline=False,zeroline=False),
)
def apply_chart_style(fig, title="", height=380):
    fig.update_layout(**CHART_LAYOUT, height=height,
        title=dict(text=title, font=dict(size=14,color="#0f172a"), x=0, xanchor="left"))
    return fig
