"""
frontend/app.py
Streamlit UI for the Evidence-Verified RAG Diabetes Care System.

LLM: Groq API (LLaMA-3.3 70B) — free, fast, no billing needed.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import requests

API_URL = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────────
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
.grade-A { background-color: #d4edda; color: #155724; }
.grade-B { background-color: #cce5ff; color: #004085; }
.grade-C { background-color: #fff3cd; color: #856404; }
.grade-D { background-color: #f8d7da; color: #721c24; }
.conf-high { color: #28a745; font-weight: bold; }
.conf-mid  { color: #fd7e14; font-weight: bold; }
.conf-low  { color: #dc3545; font-weight: bold; }
.citation-box {
    background: #f8f9fa; border-left: 4px solid #6c757d;
    padding: 8px 12px; margin: 6px 0;
    border-radius: 4px; font-size: 0.88rem;
}
.answer-box {
    background: #ffffff; border: 1px solid #dee2e6;
    border-radius: 8px; padding: 16px 20px; line-height: 1.7;
}
.warning-box {
    background: #fff3cd; border: 1px solid #ffc107;
    border-radius: 6px; padding: 10px 14px; font-size: 0.9rem;
}
.llm-badge {
    background: #e8f4fd; color: #0056b3;
    padding: 3px 10px; border-radius: 10px;
    font-size: 0.8rem; font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history = []
if "last_response" not in st.session_state: st.session_state.last_response = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/"
        "Diabetes_blue_circle.svg/200px-Diabetes_blue_circle.svg.png",
        width=60,
    )
    st.title("⚙️ Settings")

    top_k = st.slider(
        "Evidence chunks to retrieve", 3, 10, 5,
        help="More chunks = broader evidence but slightly slower"
    )

    st.divider()

    # LLM status
    st.subheader("LLM Status")
    st.markdown("""
<span class='llm-badge'>Groq — LLaMA-3.3 70B</span>
<br><small>Free API · ~1-2 sec response</small>
""", unsafe_allow_html=True)

    st.divider()

    # System status
    st.subheader("System status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API online ✅")
        st.caption(f"Chunks loaded: {health.get('chunks_loaded', '?')}")
    except Exception:
        st.error("API offline ❌")
        st.caption("Run: uvicorn api.app:app --reload --port 8000")

    st.divider()

    # Evidence grade legend
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
    "Powered by Groq (LLaMA-3.3 70B)"
)
st.divider()

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

# Quick example buttons
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
def confidence_label(conf: float) -> str:
    if conf >= 0.75:
        return f"<span class='conf-high'>🟢 {conf:.0%} — High confidence</span>"
    elif conf >= 0.5:
        return f"<span class='conf-mid'>🟡 {conf:.0%} — Moderate confidence</span>"
    else:
        return f"<span class='conf-low'>🔴 {conf:.0%} — Low confidence</span>"

def grade_badge(grade: str) -> str:
    return f"<span class='grade-badge grade-{grade}'>{grade}</span>"


if submit and query:
    with st.spinner("🔍 Retrieving and verifying evidence…"):
        try:
            payload = {
                "question":      query,
                "use_local_llm": False,   # always False — using Groq API
                "top_k":         top_k,
            }
            resp = requests.post(
                f"{API_URL}/query", json=payload, timeout=120
            )
            if resp.status_code == 503:
                st.error(
                    "Knowledge base not ready. "
                    "Complete Step 4 (download ChromaDB from Colab) first."
                )
                st.stop()
            elif resp.status_code != 200:
                st.error(f"API error {resp.status_code}: {resp.text}")
                st.stop()

            data = resp.json()
            st.session_state.last_response = data
            st.session_state.history.insert(0, {
                "question":   query,
                "timestamp":  datetime.now().strftime("%H:%M:%S"),
                "grade":      data["evidence_grade"],
                "confidence": data["confidence"],
            })

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to API. "
                "Run in Terminal 1: uvicorn api.app:app --reload --port 8000"
            )
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

# ── Response display ───────────────────────────────────────────────────────────
if st.session_state.last_response:
    data = st.session_state.last_response

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence",      f"{data['confidence']:.0%}")
    m2.metric("Evidence Grade",  data["evidence_grade"])
    m3.metric("Faithfulness",    f"{data['faithfulness']:.0%}")
    m4.metric("Avg Credibility", f"{data['avg_credibility']:.1f}/5")

    # Badges row
    llm_display = data.get("llm_used", "Groq — llama-3.3-70b-versatile")
    st.markdown(
        f"{confidence_label(data['confidence'])}  &nbsp;&nbsp;  "
        f"Grade: {grade_badge(data['evidence_grade'])}  &nbsp;&nbsp;  "
        f"<span class='llm-badge'>{llm_display}</span>",
        unsafe_allow_html=True,
    )

    # Warnings
    if data.get("contradictions_found"):
        st.markdown("""
<div class='warning-box'>
⚠️ <strong>Conflicting evidence detected</strong> — retrieved sources contain
contradictory information. Review all cited sources before clinical use.
</div>""", unsafe_allow_html=True)

    if data.get("hallucination_risk"):
        st.markdown("""
<div class='warning-box'>
🔶 <strong>Low faithfulness score</strong> — the answer may not be fully
supported by retrieved evidence. Treat with additional caution.
</div>""", unsafe_allow_html=True)

    st.divider()

    # Answer
    st.subheader("📋 Answer")
    st.markdown(
        f"<div class='answer-box'>{data['answer']}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Citations
    with st.expander(
        f"📚 View {len(data['citations'])} cited sources", expanded=False
    ):
        for cite in data["citations"]:
            gc = cite["evidence_grade"]
            st.markdown(f"""
<div class='citation-box'>
<strong>[SOURCE {cite['source_num']}]</strong>
<span class='grade-badge grade-{gc}'>{gc}</span>
&nbsp;|&nbsp; <em>{cite['journal']}</em> ({cite['year']})
&nbsp;|&nbsp; Credibility: <strong>{cite['credibility_score']:.1f}/5</strong>
&nbsp;|&nbsp; Type: {cite['source_type']}<br>
<small>{cite['chunk_preview']}</small>
</div>""", unsafe_allow_html=True)

    # Feedback
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
        comment = st.text_input(
            "Comment (optional)",
            placeholder="Any corrections or suggestions?",
        )

    if st.button("Submit feedback"):
        try:
            fb_resp = requests.post(
                f"{API_URL}/feedback",
                json={
                    "question": data["question"],
                    "answer":   data["answer"],
                    "rating":   rating,
                    "comment":  comment,
                },
                timeout=10,
            )
            if fb_resp.status_code == 200:
                st.success("Feedback submitted. Thank you!")
        except Exception:
            st.warning("Could not submit feedback — API offline.")

# ── Query history (sidebar) ────────────────────────────────────────────────────
if st.session_state.history:
    with st.sidebar:
        st.divider()
        st.subheader("Recent queries")
        for item in st.session_state.history[:5]:
            g = item["grade"]
            st.markdown(
                f"{grade_badge(g)} {item['confidence']:.0%} "
                f"<br><small>{item['question'][:50]}…</small>",
                unsafe_allow_html=True,
            )
