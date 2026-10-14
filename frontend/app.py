"""
frontend/app.py
Streamlit UI — Evidence-Verified RAG Diabetes Care System
Includes: XGBoost Risk Prediction (102) + SHAP Explanation (103)
          + Patient Query Generation (104) + RAG pipeline (105-115)
"""

import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Diabetes Care Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.grade-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 12px; font-weight: bold; font-size: 0.85rem;
}
.grade-A { background:#d4edda; color:#155724; }
.grade-B { background:#cce5ff; color:#004085; }
.grade-C { background:#fff3cd; color:#856404; }
.grade-D { background:#f8d7da; color:#721c24; }
.citation-box {
    background:#f8f9fa; border-left:4px solid #6c757d;
    padding:8px 12px; margin:6px 0;
    border-radius:4px; font-size:0.88rem;
}
.risk-card {
    padding: 16px 20px; border-radius: 10px;
    margin: 10px 0; border: 2px solid;
}
.shap-positive { color: #dc3545; font-weight: bold; }
.shap-negative { color: #28a745; font-weight: bold; }
.enriched-query {
    background: #e8f4fd; border-left: 4px solid #0056b3;
    padding: 10px 14px; border-radius: 4px;
    font-size: 0.88rem; font-style: italic;
}
.llm-badge {
    background:#e8f4fd; color:#0056b3;
    padding:3px 10px; border-radius:10px;
    font-size:0.8rem; font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history = []
if "last_response" not in st.session_state: st.session_state.last_response = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    top_k = st.slider("Evidence chunks", 3, 10, 5)
    st.divider()

    # LLM status
    st.subheader("LLM Status")
    st.markdown("""
<span class='llm-badge'>Groq — qwen/qwen3.8-27b</span>
<br><small>Free API · fast response</small>
""", unsafe_allow_html=True)
    st.divider()

    # System status
    st.subheader("System status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API online ✅")
        st.caption(f"Chunks: {health.get('chunks_loaded','?')}")
    except Exception:
        st.error("API offline ❌")
        st.caption("Run: uvicorn api.app:app --reload --port 8000")

    st.divider()
    st.subheader("Evidence grade legend")
    st.markdown("""
<span class='grade-badge grade-A'>A</span> RCT / Systematic review<br>
<span class='grade-badge grade-B'>B</span> Guidelines / Cohort<br>
<span class='grade-badge grade-C'>C</span> Case-control / Pilot<br>
<span class='grade-badge grade-D'>D</span> Expert opinion
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Care Assistant")
st.caption(
    "Evidence-verified responses from ADA guidelines and PubMed research  |  "
    "Powered by Groq (qwen3.8-27b)"
)
st.divider()

# ── Patient Parameters (Module 101) ───────────────────────────────────────────
with st.expander(
    "🧬 Enter Patient Clinical Parameters — activates XGBoost Risk Prediction (optional)",
    expanded=False
):
    st.caption(
        "Providing patient data activates **Module 102** (XGBoost risk prediction), "
        "**Module 103** (SHAP feature explanation), and **Module 104** "
        "(patient-specific query generation)."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        glucose = st.number_input("Glucose (mg/dL)",       0.0, 300.0, 0.0, step=1.0)
        bmi     = st.number_input("BMI (kg/m²)",           0.0,  70.0, 0.0, step=0.1)
    with col2:
        bp      = st.number_input("Blood Pressure (mmHg)", 0.0, 200.0, 0.0, step=1.0)
        age     = st.number_input("Age (years)",            0.0, 120.0, 0.0, step=1.0)
    with col3:
        insulin = st.number_input("Insulin (μU/mL)",       0.0, 900.0, 0.0, step=1.0)
        preg    = st.number_input("Pregnancies",            0.0,  20.0, 0.0, step=1.0)
    with col4:
        skin    = st.number_input("Skin Thickness (mm)",   0.0, 100.0, 0.0, step=1.0)
        dpf     = st.number_input("Diabetes Pedigree",     0.0,   3.0, 0.0, step=0.01)

    use_patient = st.checkbox("✅ Include patient parameters in this query", value=False)

# ── Query input ────────────────────────────────────────────────────────────────
col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "Clinical question",
        placeholder="e.g. What is the recommended HbA1c target for type 2 diabetes?",
        label_visibility="collapsed",
    )
with col_btn:
    submit = st.button("Ask 🔍", type="primary", use_container_width=True)

# Quick examples
st.markdown("**Quick examples:**")
ex_cols = st.columns(4)
examples = [
    "HbA1c target for T2DM?",
    "Metformin contraindications?",
    "SGLT2 inhibitors in heart failure?",
    "Diabetic neuropathy treatment?",
]
for i, ex in enumerate(examples):
    if ex_cols[i].button(ex, use_container_width=True):
        query  = ex
        submit = True

st.divider()

# ── Query execution ────────────────────────────────────────────────────────────
if submit and query:
    # Build patient params
    patient_params = None
    if use_patient:
        params = {}
        if glucose > 0: params["Glucose"]                   = glucose
        if bmi     > 0: params["BMI"]                       = bmi
        if bp      > 0: params["BloodPressure"]             = bp
        if age     > 0: params["Age"]                       = age
        if insulin > 0: params["Insulin"]                   = insulin
        if preg    > 0: params["Pregnancies"]               = preg
        if skin    > 0: params["SkinThickness"]             = skin
        if dpf     > 0: params["DiabetesPedigreeFunction"]  = dpf
        if params:      patient_params = params

    with st.spinner("🔍 Retrieving and verifying evidence…"):
        try:
            payload = {
                "question":       query,
                "use_local_llm":  False,
                "top_k":          top_k,
                "patient_params": patient_params,
            }
            resp = requests.post(f"{API_URL}/query", json=payload, timeout=120)
            if resp.status_code == 503:
                st.error("Knowledge base not ready.")
                st.stop()
            elif resp.status_code != 200:
                st.error(f"API error {resp.status_code}: {resp.text}")
                st.stop()

            data = resp.json()
            st.session_state.last_response = data
            st.session_state.history.insert(0, {
                "question":   query,
                "confidence": data["confidence"],
                "grade":      data["evidence_grade"],
            })
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Run: uvicorn api.app:app --reload --port 8000")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# ── Response display ───────────────────────────────────────────────────────────
if st.session_state.last_response:
    data = st.session_state.last_response

    # ── Module 102: Risk Prediction Panel ─────────────────────────────────────
    if data.get("risk_result"):
        risk = data["risk_result"]
        st.subheader("🧬 Module 102 — Diabetes Risk Prediction (XGBoost)")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Risk Score",    f"{risk['risk_score']:.0%}")
        r2.metric("Category",      risk["risk_category"])
        r3.metric("Prediction",    "Diabetic" if risk["prediction"] == 1 else "Non-Diabetic")
        r4.metric("Model",         "XGBoost")

        color = risk.get("risk_color", "#6c757d")
        st.markdown(
            f"<div class='risk-card' style='border-color:{color}; background:{color}18;'>"
            f"<strong style='color:{color}; font-size:1.1rem;'>{risk['risk_category']}</strong>"
            f" &nbsp;—&nbsp; Risk Score: <strong>{risk['risk_score']:.1%}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

    # ── Module 103: SHAP Feature Explanation ──────────────────────────────────
    if data.get("shap_result"):
        shap = data["shap_result"]
        st.subheader("📊 Module 103 — SHAP Feature Explanation")
        st.caption("Features contributing most to the diabetes risk prediction")

        # Bar chart
        shap_df = pd.DataFrame([
            {
                "Feature":    item["display"],
                "SHAP Value": item["shap_value"],
                "Direction":  item["direction"],
            }
            for item in shap["top_features"]
        ])
        st.bar_chart(shap_df.set_index("Feature")["SHAP Value"])

        # Feature table
        for item in shap["top_features"]:
            val  = item["shap_value"]
            sign = "🔴 increases" if val > 0 else "🟢 decreases"
            cls  = "shap-positive" if val > 0 else "shap-negative"
            st.markdown(
                f"**{item['display']}** = `{item['actual']}`  →  "
                f"<span class='{cls}'>{sign} risk</span>  "
                f"(SHAP: {val:+.4f})",
                unsafe_allow_html=True,
            )

        st.caption(f"💡 {shap['explanation']}")
        st.divider()

    # ── Module 104: Enriched Query ─────────────────────────────────────────────
    if data.get("patient_specific_query"):
        st.subheader("🔗 Module 104 — Patient-Specific Query")
        st.markdown(
            f"<div class='enriched-query'>"
            f"<strong>Enriched retrieval query:</strong><br>"
            f"{data['patient_specific_query']}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

    # ── Evidence metrics ───────────────────────────────────────────────────────
    conf = data["confidence"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence",      f"{conf:.0%}")
    m2.metric("Evidence Grade",  data["evidence_grade"])
    m3.metric("Faithfulness",    f"{data['faithfulness']:.0%}")
    m4.metric("Avg Credibility", f"{data['avg_credibility']:.1f}/5")

    # Confidence badge
    if conf >= 0.75:
        badge = f"🟢 {conf:.0%} — High confidence"
        badge_color = "#28a745"
    elif conf >= 0.5:
        badge = f"🟡 {conf:.0%} — Moderate confidence"
        badge_color = "#fd7e14"
    else:
        badge = f"🔴 {conf:.0%} — Low confidence"
        badge_color = "#dc3545"

    grade = data["evidence_grade"]
    llm   = data.get("llm_used", "Groq")
    st.markdown(
        f"<strong style='color:{badge_color}'>{badge}</strong>"
        f"  &nbsp;|&nbsp;  "
        f"Grade: <span class='grade-badge grade-{grade}'>{grade}</span>"
        f"  &nbsp;|&nbsp;  "
        f"<span class='llm-badge'>{llm}</span>",
        unsafe_allow_html=True,
    )

    # Warnings
    if data.get("contradictions_found"):
        st.warning("⚠️ Conflicting evidence detected across retrieved sources.")
    if data.get("hallucination_risk"):
        st.warning("🔶 Low faithfulness score — treat with additional caution.")

    st.divider()

    # ── Answer ─────────────────────────────────────────────────────────────────
    st.subheader("📋 Answer")
    st.write(data["answer"])
    st.divider()

    # ── Citations ──────────────────────────────────────────────────────────────
    with st.expander(f"📚 View {len(data['citations'])} cited sources", expanded=False):
        for cite in data["citations"]:
            g = cite["evidence_grade"]
            st.markdown(f"""
<div class='citation-box'>
<strong>[SOURCE {cite['source_num']}]</strong>
<span class='grade-badge grade-{g}'>{g}</span>
&nbsp;|&nbsp; <em>{cite['journal']}</em> ({cite['year']})
&nbsp;|&nbsp; Credibility: <strong>{cite['credibility_score']:.1f}/5</strong>
&nbsp;|&nbsp; Type: {cite['source_type']}<br>
<small>{cite['chunk_preview']}</small>
</div>""", unsafe_allow_html=True)

    # ── Feedback ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Rate this answer")
    fb1, fb2 = st.columns([2, 3])
    with fb1:
        rating = st.radio(
            "Rating", [1, 2, 3, 4, 5],
            format_func=lambda x: "⭐" * x,
            horizontal=True, index=3,
        )
    with fb2:
        comment = st.text_input("Comment (optional)")

    if st.button("Submit feedback"):
        try:
            requests.post(f"{API_URL}/feedback", json={
                "question": data["question"],
                "answer":   data["answer"],
                "rating":   rating,
                "comment":  comment,
            }, timeout=10)
            st.success("Feedback submitted!")
        except Exception:
            st.warning("Could not submit feedback.")

# ── Recent queries sidebar ─────────────────────────────────────────────────────
if st.session_state.history:
    with st.sidebar:
        st.divider()
        st.subheader("Recent queries")
        for item in st.session_state.history[:5]:
            g = item["grade"]
            st.markdown(
                f"<span class='grade-badge grade-{g}'>{g}</span> "
                f"{item['confidence']:.0%} — "
                f"<small>{item['question'][:45]}…</small>",
                unsafe_allow_html=True,
            )
