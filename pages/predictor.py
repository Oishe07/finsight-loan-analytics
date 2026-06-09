
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys; sys.path.append("..")
from utils import (load_model, load_encoders, load_feature_names,
                   load_label_map, load_shap_explainer,
                   load_approved_idx, COLORS)

FORM_LABELS = {
    "no_of_dependents":"Number of Dependents",
    "education":"Education Level",
    "self_employed":"Self Employed",
    "income_annum":"Annual Income",
    "loan_amount":"Loan Amount",
    "loan_term":"Loan Term (months)",
    "cibil_score":"CIBIL Score",
    "residential_assets_value":"Residential Asset Value",
    "commercial_assets_value":"Commercial Asset Value",
    "luxury_assets_value":"Luxury Asset Value",
    "bank_asset_value":"Bank Asset Value",
}

def show():
    model         = load_model()
    feature_names = load_feature_names()
    label_map     = load_label_map()
    explainer     = load_shap_explainer()
    approved_idx, target_classes, threshold = load_approved_idx()

    st.markdown(
        '<div class="hero-banner">'
        '<div class="hero-badge">AI + SHAP Explainability</div>'
        '<div class="hero-title">Loan Approval Predictor</div>'
        '<div class="hero-sub">Real-time prediction with full AI explanation.</div>'
        '</div>', unsafe_allow_html=True)

    st.info("💡 Tip: CIBIL score above 700 = high approval chance.")
    st.markdown('<p class="section-header">Applicant Details</p>', unsafe_allow_html=True)

    user_inputs = {}
    cols = st.columns(2)
    for i, feat in enumerate(feature_names):
        label = FORM_LABELS.get(feat, feat.replace("_"," ").title())
        with cols[i % 2]:
            if feat in label_map:
                user_inputs[feat] = st.selectbox(label, label_map[feat])
            elif "cibil" in feat:
                user_inputs[feat] = st.slider(label, 300, 900, 750,
                    help="300=very poor · 550=fair · 700=good · 800=excellent")
            elif "term" in feat:
                user_inputs[feat] = st.select_slider(label,
                    options=[24,48,72,96,120,144,168,192,216,240], value=120)
            elif "dependents" in feat:
                user_inputs[feat] = st.slider(label, 0, 10, 1)
            else:
                user_inputs[feat] = st.number_input(label, 0, 100_000_000, 500_000, 50_000)

    if st.button("Predict Loan Approval", type="primary", use_container_width=True):
        encoders = load_encoders()
        row = {}
        for feat in feature_names:
            val = user_inputs[feat]
            if feat in encoders:
                try: val = int(encoders[feat].transform([str(val)])[0])
                except: val = 0
            try: row[feat] = float(val)
            except: row[feat] = 0.0

        X_in  = pd.DataFrame([row])[feature_names]
        prob  = model.predict_proba(X_in)[0]

        # ── Use saved threshold — core fix ────────────────────────────────
        approve_prob = float(prob[approved_idx])
        reject_prob  = 1.0 - approve_prob
        is_ok        = approve_prob >= threshold
        conf         = approve_prob * 100 if is_ok else reject_prob * 100

        # SHAP
        try:
            shap_values = explainer.shap_values(X_in)
            raw = shap_values[approved_idx] if isinstance(shap_values, list) else shap_values
            sv  = np.array(raw).flatten()[:len(feature_names)]
            shap_ok = True
        except Exception as e:
            shap_ok = False
            shap_error = str(e)

        # Result banner
        st.markdown("---")
        if is_ok:
            st.markdown(
                f'<div class="result-approved">'
                f'<div style="font-size:26px;font-weight:700;color:#065f46;">✅ Loan Likely APPROVED</div>'
                f'<div style="color:#475569;margin-top:6px;">Approval probability: <b>{approve_prob*100:.1f}%</b> '
                f'(threshold: {threshold*100:.0f}%)</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="result-rejected">'
                f'<div style="font-size:26px;font-weight:700;color:#7f1d1d;">❌ Loan Likely REJECTED</div>'
                f'<div style="color:#475569;margin-top:6px;">Approval probability: <b>{approve_prob*100:.1f}%</b> '
                f'(threshold: {threshold*100:.0f}%)</div>'
                f'</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Gauge + Probabilities
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-header">Approval Gauge</p>', unsafe_allow_html=True)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=approve_prob*100,
                number={"suffix":"%","font":{"size":36}},
                gauge={
                    "axis":{"range":[0,100]},
                    "bar":{"color":COLORS["success"] if is_ok else COLORS["danger"],"thickness":0.3},
                    "threshold":{"line":{"color":"#0f172a","width":3},
                                 "thickness":0.85,"value":threshold*100},
                    "steps":[
                        {"range":[0, threshold*100],       "color":"#fee2e2"},
                        {"range":[threshold*100, 100],     "color":"#d1fae5"}
                    ]
                }
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=280,
                                margin=dict(t=20,b=10,l=20,r=20))
            st.plotly_chart(fig_g, use_container_width=True)

        with c2:
            st.markdown('<p class="section-header">Approval vs Rejection %</p>', unsafe_allow_html=True)
            fig_b = go.Figure(go.Bar(
                x=["Approved","Rejected"],
                y=[round(approve_prob*100,1), round(reject_prob*100,1)],
                marker_color=[COLORS["success"],COLORS["danger"]],
                text=[f"{round(approve_prob*100,1)}%",f"{round(reject_prob*100,1)}%"],
                textposition="outside", width=0.4
            ))
            fig_b.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=280,
                yaxis=dict(range=[0,120],title="Probability (%)",
                           gridcolor="#f1f5f9",showline=False,zeroline=False),
                xaxis=dict(showline=False,tickfont=dict(size=14)),
                margin=dict(t=30,b=20,l=30,r=30),showlegend=False)
            st.plotly_chart(fig_b, use_container_width=True)

        # SHAP section
        st.markdown("---")
        st.markdown('<p class="section-header">🔬 SHAP Explainability — Why did the model decide this?</p>',
                    unsafe_allow_html=True)

        if not shap_ok:
            st.error(f"SHAP failed: {shap_error}")
        else:
            shap_df = pd.DataFrame({
                "feature":   feature_names,
                "shap":      sv.tolist(),
                "label":     [FORM_LABELS.get(f,f.replace("_"," ").title()) for f in feature_names],
                "input_val": [user_inputs[f] for f in feature_names]
            }).sort_values("shap", ascending=True)

            st.markdown(
                "<div style='background:#f0f9ff;border-left:4px solid #0ea5e9;"
                "padding:14px 18px;border-radius:0 8px 8px 0;margin-bottom:20px;"
                "font-size:14px;color:#0c4a6e;'>"
                "🟢 Green = pushes toward <b>Approval</b>. "
                "🔴 Red = pushes toward <b>Rejection</b>. "
                "Longer bar = stronger influence.</div>",
                unsafe_allow_html=True)

            x_max = max(abs(shap_df["shap"].max()), abs(shap_df["shap"].min()))
            if x_max == 0: x_max = 0.01

            fig_shap = go.Figure(go.Bar(
                x=shap_df["shap"], y=shap_df["label"], orientation="h",
                marker_color=[COLORS["success"] if v>=0 else COLORS["danger"]
                              for v in shap_df["shap"]],
                text=[f"+{v:.3f}" if v>=0 else f"{v:.3f}" for v in shap_df["shap"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>"
            ))
            fig_shap.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=420,
                xaxis=dict(title="SHAP Value — positive=helps approval, negative=hurts approval",
                           range=[-(x_max*1.4),x_max*1.4],
                           gridcolor="#f1f5f9",showline=False,
                           zeroline=True,zerolinecolor="#94a3b8",zerolinewidth=2),
                yaxis=dict(showline=False),
                margin=dict(t=20,b=40,l=190,r=120)
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            positives = shap_df[shap_df["shap"]>0].sort_values("shap",ascending=False)
            negatives = shap_df[shap_df["shap"]<0].sort_values("shap")

            col_p, col_n = st.columns(2)
            with col_p:
                st.markdown(
                    "<div style='background:#f0fdf4;border:1px solid #10b981;"
                    "border-radius:12px;padding:16px 20px;'>"
                    "<div style='font-weight:700;color:#065f46;font-size:15px;"
                    "margin-bottom:10px;'>✅ Pushing toward APPROVAL</div>",
                    unsafe_allow_html=True)
                if len(positives)==0:
                    st.markdown("<p style='color:#475569;font-size:13px;'>None.</p>",
                                unsafe_allow_html=True)
                for _, r in positives.iterrows():
                    pct = min(100, int(abs(r["shap"])/x_max*100))
                    st.markdown(
                        f"<div style='margin-bottom:10px;'>"
                        f"<div style='font-size:13px;font-weight:600;color:#065f46;'>{r['label']}</div>"
                        f"<div style='font-size:12px;color:#475569;margin-bottom:4px;'>"
                        f"Value: {r['input_val']} &nbsp;|&nbsp; SHAP: +{r['shap']:.4f}</div>"
                        f"<div style='background:#d1fae5;border-radius:4px;height:8px;'>"
                        f"<div style='background:#10b981;width:{pct}%;height:8px;"
                        f"border-radius:4px;'></div></div></div>",
                        unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_n:
                st.markdown(
                    "<div style='background:#fef2f2;border:1px solid #ef4444;"
                    "border-radius:12px;padding:16px 20px;'>"
                    "<div style='font-weight:700;color:#7f1d1d;font-size:15px;"
                    "margin-bottom:10px;'>❌ Pushing toward REJECTION</div>",
                    unsafe_allow_html=True)
                if len(negatives)==0:
                    st.markdown("<p style='color:#475569;font-size:13px;'>None.</p>",
                                unsafe_allow_html=True)
                for _, r in negatives.iterrows():
                    pct = min(100, int(abs(r["shap"])/x_max*100))
                    st.markdown(
                        f"<div style='margin-bottom:10px;'>"
                        f"<div style='font-size:13px;font-weight:600;color:#7f1d1d;'>{r['label']}</div>"
                        f"<div style='font-size:12px;color:#475569;margin-bottom:4px;'>"
                        f"Value: {r['input_val']} &nbsp;|&nbsp; SHAP: {r['shap']:.4f}</div>"
                        f"<div style='background:#fee2e2;border-radius:4px;height:8px;'>"
                        f"<div style='background:#ef4444;width:{pct}%;height:8px;"
                        f"border-radius:4px;'></div></div></div>",
                        unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Plain English verdict
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            top_pos = positives.iloc[0] if len(positives)>0 else None
            top_neg = negatives.iloc[0] if len(negatives)>0 else None
            verdict = ("<div style='background:#f8fafc;border:1px solid #e2e8f0;"
                       "border-radius:12px;padding:18px 22px;font-size:14px;"
                       "color:#334155;line-height:1.8;'><b>📋 Plain English Verdict:</b><br>")
            if is_ok:
                verdict += f"This loan was <b style='color:#065f46;'>approved</b>"
                if top_pos is not None:
                    verdict += f" mainly because of <b>{top_pos['label']}</b> (SHAP: +{top_pos['shap']:.3f})"
                if top_neg is not None:
                    verdict += f", despite <b>{top_neg['label']}</b> working against it (SHAP: {top_neg['shap']:.3f})"
            else:
                verdict += f"This loan was <b style='color:#7f1d1d;'>rejected</b>"
                if top_neg is not None:
                    verdict += f" mainly because of <b>{top_neg['label']}</b> (SHAP: {top_neg['shap']:.3f})"
                if top_pos is not None:
                    verdict += f", even though <b>{top_pos['label']}</b> helped (SHAP: +{top_pos['shap']:.3f})"
            verdict += ".</div>"
            st.markdown(verdict, unsafe_allow_html=True)

            # Tips
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            cibil_val  = user_inputs.get("cibil_score", 0)
            income_val = user_inputs.get("income_annum", 0)
            loan_val   = user_inputs.get("loan_amount", 0)
            if cibil_val < 700:
                st.warning(f"CIBIL score is {cibil_val} — below 700. Try 750+ to see approval.")
            else:
                st.success(f"CIBIL score is {cibil_val} — good range.")
            if income_val > 0 and loan_val > 0:
                ratio = loan_val / income_val
                if ratio > 5:
                    st.warning(f"Loan-to-income ratio is {ratio:.1f}x — too high.")
                else:
                    st.success(f"Loan-to-income ratio is {ratio:.1f}x — acceptable.")

            # Summary table
            st.markdown('<p class="section-header">Full Applicant Summary</p>',
                        unsafe_allow_html=True)
            rows = []
            for i, feat in enumerate(feature_names):
                imp  = float(model.feature_importances_[i])
                sval = float(sv[i])
                rows.append({
                    "Feature":    FORM_LABELS.get(feat, feat.replace("_"," ").title()),
                    "Your Input": user_inputs[feat],
                    "Importance": f"{imp*100:.1f}%",
                    "SHAP Value": f"+{sval:.4f}" if sval>=0 else f"{sval:.4f}",
                    "Effect":     "✅ Helps Approval" if sval>=0 else "❌ Hurts Approval"
                })
            st.dataframe(
                pd.DataFrame(rows).sort_values("Importance", ascending=False),
                use_container_width=True, hide_index=True, height=340)
