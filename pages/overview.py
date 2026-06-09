
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys; sys.path.append("..")
from utils import load_data, load_metrics, COLORS, apply_chart_style

def show():
    m  = load_metrics()
    df = load_data()
    target = m["target"]

    st.markdown("""<div class="hero-banner">
      <div class="hero-badge">Live Analytics</div>
      <div class="hero-title">FinSight Overview Dashboard</div>
      <div class="hero-sub">ML-powered loan portfolio intelligence at a glance.</div>
    </div>""", unsafe_allow_html=True)

    total = m["n_total"]
    approved_count = int(df[target].value_counts().iloc[0])
    approval_rate  = round(approved_count/total*100,1)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Applications", f"{total:,}")
    c2.metric("Approval Rate",      f"{approval_rate}%")
    c3.metric("Model Accuracy",     f"{m['accuracy']}%")
    c4.metric("ROC-AUC Score",      f"{m['roc_auc']}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Approval Distribution</p>', unsafe_allow_html=True)
        counts = df[target].value_counts().reset_index()
        counts.columns = ["status","count"]
        fig = px.pie(counts, names="status", values="count", hole=0.55,
                     color_discrete_sequence=[COLORS["success"],COLORS["danger"]])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300,
                          margin=dict(t=20,b=20,l=20,r=20))
        fig.add_annotation(text=f"<b>{total:,}</b><br>Total",x=0.5,y=0.5,
                           showarrow=False,font=dict(size=14,color="#0f172a"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        income_col = next((c for c in df.columns if "income" in c and "co" not in c), None)
        if income_col:
            st.markdown('<p class="section-header">Income Distribution</p>', unsafe_allow_html=True)
            fig2 = px.histogram(df, x=income_col, color=target, nbins=50,
                                color_discrete_sequence=[COLORS["success"],COLORS["danger"]],
                                opacity=0.8, barmode="overlay")
            apply_chart_style(fig2, height=300)
            st.plotly_chart(fig2, use_container_width=True)

    cibil_col = next((c for c in df.columns if "cibil" in c), None)
    loan_col  = next((c for c in df.columns if "loan_amount" in c), None)
    c3b, c4b = st.columns(2)
    if cibil_col:
        with c3b:
            st.markdown('<p class="section-header">CIBIL Score by Status</p>', unsafe_allow_html=True)
            fig3 = px.box(df, x=target, y=cibil_col, color=target,
                          color_discrete_sequence=[COLORS["success"],COLORS["danger"]])
            apply_chart_style(fig3, height=320)
            st.plotly_chart(fig3, use_container_width=True)
    if cibil_col and loan_col:
        with c4b:
            st.markdown('<p class="section-header">Loan Amount vs CIBIL</p>', unsafe_allow_html=True)
            s = df.sample(min(600,len(df)), random_state=42)
            fig4 = px.scatter(s, x=cibil_col, y=loan_col, color=target,
                              color_discrete_sequence=[COLORS["success"],COLORS["danger"]], opacity=0.6)
            apply_chart_style(fig4, height=320)
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<p class="section-header">Feature Importances</p>', unsafe_allow_html=True)
    import pandas as pd
    fi = pd.DataFrame(m["top_features"]).sort_values("importance")
    fig5 = px.bar(fi, x="importance", y="feature", orientation="h",
                  color="importance",
                  color_continuous_scale=[[0,"#e0e7ff"],[1,COLORS["primary"]]])
    fig5.update_layout(coloraxis_showscale=False)
    apply_chart_style(fig5, height=300)
    st.plotly_chart(fig5, use_container_width=True)
