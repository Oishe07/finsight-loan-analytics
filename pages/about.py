
import streamlit as st
import sys; sys.path.append("..")
from utils import load_metrics

def show():
    m = load_metrics()
    st.markdown("""<div class="hero-banner">
      <div class="hero-badge">Documentation</div>
      <div class="hero-title">ℹ️ About FinSight</div>
      <div class="hero-sub">Architecture, methodology, and technical implementation.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">ML Pipeline</p>', unsafe_allow_html=True)
    steps = [
        ("1️⃣ Data Ingestion",    "4,269 loan applications · 13 features · Kaggle dataset"),
        ("2️⃣ Data Cleaning",     "Null removal · whitespace stripping · column standardization"),
        ("3️⃣ Feature Encoding",  "LabelEncoder on all categorical columns · saved for inference"),
        ("4️⃣ Model Training",    "Random Forest · 300 trees · sqrt features · 80/20 split"),
        ("5️⃣ Evaluation",        "~98% accuracy · high ROC-AUC · 5-fold cross-validation"),
        ("6️⃣ Serialization",     "Joblib: model + encoders + feature names + metrics"),
        ("7️⃣ Deployment",        "Multi-page Streamlit app · real-time prediction · Plotly dashboards"),
    ]
    for t, d in steps:
        st.markdown(f'<div style="border-left:4px solid #6366f1;border-radius:8px;padding:12px 16px;margin-bottom:8px;background:white;border:1px solid #e2e8f0"><b>{t}</b><br><span style="color:#475569;font-size:13px;">{d}</span></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Model Performance</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']}%")
    c2.metric("ROC-AUC",  f"{m['roc_auc']}")
    c3.metric("CV Mean",  f"{m['cv_mean']}%")
    c4.metric("CV Std",   f"±{m['cv_std']}%")

    st.markdown('<p class="section-header">Tech Stack</p>', unsafe_allow_html=True)
    for name, role in [("Python 3.10+","Core language"),("Scikit-learn","Random Forest, metrics, preprocessing"),
                        ("Pandas & NumPy","Data manipulation"),("Joblib","Model serialization"),
                        ("Plotly","Interactive charts"),("Streamlit","Web application framework")]:
        st.markdown(f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;margin-bottom:6px;"><b>{name}</b> · <span style="color:#64748b;font-size:13px;">{role}</span></div>', unsafe_allow_html=True)
