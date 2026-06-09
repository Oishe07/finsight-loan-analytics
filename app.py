
import streamlit as st

st.set_page_config(page_title="FinSight | Loan Risk Platform", page_icon="💳",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f172a 0%,#1e293b 100%);
    min-width:260px !important;
    max-width:260px !important;
}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
[data-testid="collapsedControl"] {
    display:block !important;
    visibility:visible !important;
    background-color:#1e293b !important;
    border-radius:0 8px 8px 0 !important;
}
[data-testid="collapsedControl"] svg {
    fill:#e2e8f0 !important;
    stroke:#e2e8f0 !important;
}
button[kind="header"] {
    background-color:#1e293b !important;
    border-radius:0 8px 8px 0 !important;
}
button[kind="header"] svg {
    color:#e2e8f0 !important;
    fill:#e2e8f0 !important;
}
.hero-banner{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#1e293b 100%);
  border-radius:20px;padding:36px 40px;margin-bottom:28px;}
.hero-badge{display:inline-block;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);
  color:#818cf8;font-size:11px;font-weight:600;padding:4px 12px;border-radius:100px;
  margin-bottom:14px;letter-spacing:.08em;text-transform:uppercase;}
.hero-title{font-family:"Space Grotesk",sans-serif;font-size:32px;font-weight:700;color:white;margin:0 0 8px 0;}
.hero-sub{font-size:15px;color:#94a3b8;margin:0;}
.section-header{font-family:"Space Grotesk",sans-serif;font-size:18px;font-weight:600;
  color:#0f172a;margin:0 0 16px 0;padding-bottom:10px;border-bottom:2px solid #f1f5f9;}
.result-approved{background:linear-gradient(135deg,#ecfdf5,#d1fae5);border:2px solid #10b981;
  border-radius:16px;padding:28px;text-align:center;}
.result-rejected{background:linear-gradient(135deg,#fef2f2,#fee2e2);border:2px solid #ef4444;
  border-radius:16px;padding:28px;text-align:center;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px 0;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:white;">💳 FinSight</div>
      <div style="font-size:12px;color:#64748b;margin-top:4px;">Loan Risk & Approval Platform</div>
    </div>""", unsafe_allow_html=True)
    page = st.radio("Navigate",
        ["🏠 Overview","🔍 Predictor","📊 Risk Analytics",
         "📈 EDA Explorer","🤖 AI Advisor","ℹ️ About"],
        label_visibility="collapsed")
    st.markdown("<hr style='border-color:#334155;margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#475569;line-height:1.7;">
      <b style="color:#94a3b8;">Model</b><br>Random Forest · 300 trees<br><br>
      <b style="color:#94a3b8;">Dataset</b><br>Kaggle · 4,269 records<br><br>
      <b style="color:#94a3b8;">Accuracy</b><br>~98% test accuracy<br><br>
      <b style="color:#94a3b8;">AI Advisor</b><br>Powered by Groq LLM
    </div>""", unsafe_allow_html=True)

if   "Overview"   in page: from pages.overview      import show; show()
elif "Predictor"  in page: from pages.predictor      import show; show()
elif "Risk"       in page: from pages.risk_analytics import show; show()
elif "EDA"        in page: from pages.eda            import show; show()
elif "AI Advisor" in page: from pages.ai_advisor     import show; show()
elif "About"      in page: from pages.about          import show; show()
