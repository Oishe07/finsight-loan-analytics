
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd, numpy as np
import sys; sys.path.append("..")
from utils import load_data, load_metrics, COLORS, apply_chart_style

def show():
    df = load_data(); m = load_metrics(); target = m["target"]
    st.markdown("""<div class="hero-banner">
      <div class="hero-badge">Risk Intelligence</div>
      <div class="hero-title">📊 Risk Analytics Dashboard</div>
      <div class="hero-sub">Deep-dive into credit risk segments and portfolio patterns.</div>
    </div>""", unsafe_allow_html=True)

    edu_col   = next((c for c in df.columns if "edu" in c), None)
    emp_col   = next((c for c in df.columns if "self_emp" in c or "employed" in c), None)
    cibil_col = next((c for c in df.columns if "cibil" in c), None)
    loan_col  = next((c for c in df.columns if "loan_amount" in c), None)
    term_col  = next((c for c in df.columns if "loan_term" in c or "term" in c), None)
    inc_col   = next((c for c in df.columns if "income" in c and "co" not in c), None)
    dep_col   = next((c for c in df.columns if "depend" in c), None)

    c1, c2 = st.columns(2)
    if edu_col:
        with c1:
            st.markdown('<p class="section-header">Approval by Education</p>', unsafe_allow_html=True)
            d = df.groupby([edu_col, target]).size().reset_index(name="count")
            fig = px.bar(d, x=edu_col, y="count", color=target, barmode="group",
                         color_discrete_sequence=[COLORS["success"],COLORS["danger"]])
            apply_chart_style(fig, height=300)
            st.plotly_chart(fig, use_container_width=True)
    if emp_col:
        with c2:
            st.markdown('<p class="section-header">Approval by Employment</p>', unsafe_allow_html=True)
            d2 = df.groupby([emp_col, target]).size().reset_index(name="count")
            fig2 = px.bar(d2, x=emp_col, y="count", color=target, barmode="group",
                          color_discrete_sequence=[COLORS["success"],COLORS["danger"]])
            apply_chart_style(fig2, height=300)
            st.plotly_chart(fig2, use_container_width=True)

    if cibil_col:
        st.markdown('<p class="section-header">CIBIL Score Risk Segments</p>', unsafe_allow_html=True)
        tmp = df.copy()
        bins=[300,550,650,750,900]; labels=["Poor","Fair","Good","Excellent"]
        tmp["segment"] = pd.cut(tmp[cibil_col],bins=bins,labels=labels,include_lowest=True)
        d3 = tmp.groupby(["segment",target],observed=True).size().reset_index(name="count")
        fig3 = px.bar(d3, x="segment", y="count", color=target, barmode="stack", text="count",
                      color_discrete_sequence=[COLORS["success"],COLORS["danger"]])
        apply_chart_style(fig3, height=320)
        st.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns(2)
    if loan_col:
        with c3:
            st.markdown('<p class="section-header">Loan Amount Distribution</p>', unsafe_allow_html=True)
            fig4 = px.violin(df, y=loan_col, x=target, color=target, box=True, points="outliers",
                             color_discrete_sequence=[COLORS["success"],COLORS["danger"]])
            apply_chart_style(fig4, height=320)
            st.plotly_chart(fig4, use_container_width=True)
    if term_col:
        with c4:
            st.markdown('<p class="section-header">Approval by Loan Term</p>', unsafe_allow_html=True)
            d4 = df.groupby([term_col, target]).size().reset_index(name="count")
            fig5 = px.bar(d4, x=term_col, y="count", color=target, barmode="group",
                          color_discrete_sequence=[COLORS["success"],COLORS["danger"]])
            apply_chart_style(fig5, height=320)
            st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<p class="section-header">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:10]
    corr = df[num_cols].corr()
    fig6 = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                colorscale=[[0,"#fef2f2"],[0.5,"#e0e7ff"],[1,"#4338ca"]],
                                zmid=0, text=np.round(corr.values,2), texttemplate="%{text}"))
    apply_chart_style(fig6, height=400)
    st.plotly_chart(fig6, use_container_width=True)
