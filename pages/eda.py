
import streamlit as st
import plotly.express as px
import pandas as pd, numpy as np
import sys; sys.path.append("..")
from utils import load_data, load_metrics, COLORS, apply_chart_style

def show():
    df = load_data(); m = load_metrics(); target = m["target"]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    st.markdown("""<div class="hero-banner">
      <div class="hero-badge">Data Science</div>
      <div class="hero-title">📈 EDA Explorer</div>
      <div class="hero-sub">Explore any variable interactively.</div>
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["📊 Distribution","🔗 Relationship","📦 Box Plot","🗂️ Raw Data"])

    with tabs[0]:
        col_d  = st.selectbox("Column", num_cols)
        col_c  = st.selectbox("Color by", ["None", target]+cat_cols)
        nbins  = st.slider("Bins", 10, 100, 40)
        fig = px.histogram(df, x=col_d, color=None if col_c=="None" else col_c,
                           nbins=nbins, color_discrete_sequence=COLORS["cat"],
                           opacity=0.82, barmode="overlay", marginal="box")
        apply_chart_style(fig, title=f"Distribution of {col_d}", height=400)
        st.plotly_chart(fig, use_container_width=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Mean",   f"{df[col_d].mean():,.0f}")
        c2.metric("Median", f"{df[col_d].median():,.0f}")
        c3.metric("Std",    f"{df[col_d].std():,.0f}")
        c4.metric("Min",    f"{df[col_d].min():,.0f}")
        c5.metric("Max",    f"{df[col_d].max():,.0f}")

    with tabs[1]:
        t1,t2,t3 = st.columns(3)
        x = t1.selectbox("X axis", num_cols, index=0)
        y = t2.selectbox("Y axis", num_cols, index=min(1,len(num_cols)-1))
        c = t3.selectbox("Color",  ["None",target]+cat_cols)
        trend = st.checkbox("OLS trendline", True)
        n = st.slider("Sample size", 100, len(df), min(600,len(df)), 50)
        samp = df.sample(n, random_state=42)
        fig2 = px.scatter(samp, x=x, y=y, color=None if c=="None" else c,
                          color_discrete_sequence=COLORS["cat"],
                          trendline="ols" if trend else None, opacity=0.7)
        apply_chart_style(fig2, title=f"{x} vs {y}", height=420)
        st.plotly_chart(fig2, use_container_width=True)
        cv = df[[x,y]].corr().iloc[0,1]
        st.info(f"Pearson correlation: **{cv:.4f}**")

    with tabs[2]:
        by = st.selectbox("Group by", [target]+cat_cols)
        yb = st.selectbox("Numeric (Y)", num_cols)
        fig3 = px.box(df, x=by, y=yb, color=by, color_discrete_sequence=COLORS["cat"],
                      points="outliers")
        apply_chart_style(fig3, height=380)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(df.groupby(by)[yb].agg(["mean","median","std"]).round(2),
                     use_container_width=True)

    with tabs[3]:
        search = st.text_input("Search rows")
        page_n = st.select_slider("Rows", [10,25,50,100], 25)
        disp = df[df.apply(lambda r: r.astype(str).str.contains(search,case=False).any(),axis=1)] if search else df
        st.dataframe(disp.head(page_n), use_container_width=True, height=400)
        st.download_button("⬇️ Download CSV", disp.to_csv(index=False),
                           "finsight_data.csv","text/csv")
