"""CampusHire Streamlit app.  Run:  streamlit run app.py"""
from datetime import date

import streamlit as st

from campushire import config
from campushire.eligibility import StudentProfile, evaluate_all
from campushire.generator import CampusHireQA
from campushire.ingest import KnowledgeBase
from campushire.retriever import HybridRetriever

st.set_page_config(page_title="CampusHire", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading knowledge base and models…")
def get_system():
    kb = KnowledgeBase.load()
    kb.sync()
    return kb, CampusHireQA(HybridRetriever(kb))


kb, qa = get_system()

# ---------------- sidebar: ingestion ----------------
with st.sidebar:
    st.title("🎓 CampusHire")
    st.caption("RAG-based placement intelligence")
    st.metric("Indexed chunks", len(kb.chunks))
    st.metric("Documents", len(kb.manifest))
    st.caption("LLM: " + (config.OPENAI_MODEL if config.OPENAI_API_KEY else "extractive mode (no key)"))

    st.subheader("Add documents")
    uploads = st.file_uploader("PDF / Markdown / Text", type=["pdf", "md", "txt"],
                               accept_multiple_files=True)
    if st.button("Ingest", disabled=not uploads):
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            (config.DOCS_DIR / f.name).write_bytes(f.getvalue())
        with st.spinner("Embedding new documents…"):
            st.success(kb.sync())
    if st.button("Re-sync docs folder"):
        st.info(kb.sync())

tab_chat, tab_elig, tab_kb = st.tabs(["💬 Ask", "✅ Eligibility", "📚 Knowledge base"])

# ---------------- chat ----------------
with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if q := st.chat_input("e.g. What is the eligibility for Kaboom AI and when is the deadline?"):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and reasoning…"):
                ans = qa.ask(q, st.session_state.history)
            st.markdown(ans.text)
            if ans.sources:
                with st.expander(f"Sources ({len(ans.sources)})"):
                    for i, h in enumerate(ans.sources, 1):
                        m = h.chunk.metadata
                        score = f"rerank={h.rerank:.2f}" if h.rerank is not None else f"rrf={h.rrf:.4f}"
                        st.markdown(f"**[{i}] {h.chunk.source}** · p.{h.chunk.page} · "
                                    f"`{m.get('doc_type')}` {m.get('company', '')} · {score}")
                        st.caption(h.chunk.text[:500])
            if ans.notes:
                st.caption("Guardrails: " + ", ".join(ans.notes))
        st.session_state.history += [{"role": "user", "content": q},
                                     {"role": "assistant", "content": ans.text}]

# ---------------- eligibility ----------------
with tab_elig:
    st.subheader("Student profile")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Name", "Student")
    degree = c1.selectbox("Degree", ["M.Tech", "B.Tech", "MCA"])
    branch = c1.selectbox("Branch", ["CSE", "IT", "AI&DS", "ECE", "EE", "ME", "Mathematics"])
    cgpa = c2.number_input("CGPA", 0.0, 10.0, 7.5, 0.1)
    tenth = c2.number_input("10th %", 0.0, 100.0, 80.0)
    twelfth = c2.number_input("12th %", 0.0, 100.0, 80.0)
    backlogs = c3.number_input("Active backlogs", 0, 20, 0)
    year = c3.number_input("Graduation year", 2024, 2030, 2026)
    has_offer = c3.checkbox("Already placed?")
    offer = c3.number_input("Current offer CTC (LPA)", 0.0, 100.0, 0.0, disabled=not has_offer)
    as_of = st.date_input("Evaluate as of", date.today())

    if st.button("Check eligibility", type="primary"):
        student = StudentProfile(name, branch, degree, cgpa, tenth, twelfth, int(backlogs),
                                 int(year), has_offer, offer)
        for r in evaluate_all(student, {"dream_offer_multiplier": 1.5}, as_of):
            icon = "✅" if r.eligible else "❌"
            with st.expander(f"{icon} {r.company} — {r.role}", expanded=not r.eligible):
                for ch in r.checks:
                    st.markdown(f"{'✔️' if ch.passed else '✖️'} **{ch.rule}** — {ch.detail}")

# ---------------- knowledge base ----------------
with tab_kb:
    rows = {}
    for c in kb.chunks:
        r = rows.setdefault(c.source, {"document": c.source, "type": c.metadata.get("doc_type"),
                                       "company": c.metadata.get("company"),
                                       "date": c.metadata.get("date", ""), "chunks": 0})
        r["chunks"] += 1
    st.dataframe(list(rows.values()), use_container_width=True)
