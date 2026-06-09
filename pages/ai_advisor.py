
import streamlit as st
from groq import Groq

API_KEY = st.secrets["GROQ_API_KEY"]
MODEL   = "llama-3.3-70b-versatile"

def call_groq(messages):
    client = Groq(api_key=API_KEY)
    res = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=800,
        temperature=0.7
    )
    return res.choices[0].message.content

def show():
    st.markdown('<div class="hero-banner"><div class="hero-badge">AI Advisory</div><div class="hero-title">🤖 AI Loan Advisor</div><div class="hero-sub">Ask anything about your loan application.</div></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Applicant Details</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        cibil   = st.number_input("CIBIL Score", 300, 900, 650, key="ac1")
        income  = st.number_input("Annual Income", 0, 100000000, 500000, 50000, key="ac2")
        loan    = st.number_input("Loan Amount", 0, 100000000, 1000000, 50000, key="ac3")
    with c2:
        result  = st.selectbox("Prediction Result", ["Not predicted yet","Approved","Rejected"], key="ac4")
        edu     = st.selectbox("Education", ["Graduate","Not Graduate"], key="ac5")
        emp     = st.selectbox("Self Employed", ["Yes","No"], key="ac6")

    ratio = round(loan/income, 2) if income > 0 else 0
    ctx = (
        "CIBIL=" + str(cibil) +
        ", Income=" + str(income) +
        ", LoanAmount=" + str(loan) +
        ", LoanToIncomeRatio=" + str(ratio) +
        "x, Education=" + edu +
        ", SelfEmployed=" + emp +
        ", PredictionResult=" + result
    )

    system_msg = {
        "role": "system",
        "content": (
            "You are FinSight AI, an expert loan advisor for Bangladesh NBFIs. "
            "Help users understand their loan decisions and improve approval chances. "
            "Be specific, friendly, and use numbers. "
            "Applicant context: " + ctx
        )
    }

    st.markdown("---")
    st.markdown('<p class="section-header">💬 Chat</p>', unsafe_allow_html=True)

    quick = [
        "Why was my loan rejected?",
        "How to improve CIBIL score?",
        "Is my loan amount too high?",
        "What assets help approval?",
        "Explain SHAP values simply",
        "How long to improve credit?"
    ]
    cols = st.columns(3)
    clicked = None
    for i, q in enumerate(quick):
        if cols[i%3].button(q, key="q"+str(i), use_container_width=True):
            clicked = q

    if "adv_msgs" not in st.session_state:
        st.session_state.adv_msgs = [{"role":"assistant","content":"Hello! I am your FinSight AI Loan Advisor. Ask me anything about your loan application!"}]

    for m in st.session_state.adv_msgs:
        if m["role"] == "user":
            st.markdown("<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px 12px 4px 12px;padding:12px;margin:6px 0 6px 80px;font-size:14px;'><b style='color:#1e40af;'>You:</b> " + m["content"] + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px 12px 12px 4px;padding:12px;margin:6px 80px 6px 0;font-size:14px;'><b style='color:#065f46;'>🤖 AI:</b> " + m["content"] + "</div>", unsafe_allow_html=True)

    if clicked:
        st.session_state.adv_msgs.append({"role":"user","content":clicked})
        with st.spinner("Thinking..."):
            try:
                reply = call_groq([system_msg] + st.session_state.adv_msgs)
                st.session_state.adv_msgs.append({"role":"assistant","content":reply})
            except Exception as e:
                st.session_state.adv_msgs.append({"role":"assistant","content":"Error: "+str(e)})
        st.rerun()

    inp = st.chat_input("Ask your question...")
    if inp:
        st.session_state.adv_msgs.append({"role":"user","content":inp})
        with st.spinner("Thinking..."):
            try:
                reply = call_groq([system_msg] + st.session_state.adv_msgs)
                st.session_state.adv_msgs.append({"role":"assistant","content":reply})
            except Exception as e:
                st.session_state.adv_msgs.append({"role":"assistant","content":"Error: "+str(e)})
        st.rerun()

    if len(st.session_state.adv_msgs) > 1:
        if st.button("🗑️ Clear", type="secondary"):
            st.session_state.adv_msgs = [{"role":"assistant","content":"Hello! How can I help you today?"}]
            st.rerun()
