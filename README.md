# CampusHire — RAG-Based Placement Intelligence System

Ask questions about training-and-placement policies, company JDs, eligibility criteria, deadlines and notices — with cited answers — and check eligibility deterministically against a student profile.

**Tech stack:** Python · Streamlit · OpenAI GPT-4o-mini · FAISS · BM25 (rank-bm25) · Sentence-Transformers (MiniLM embeddings + ms-marco cross-encoder) · pypdf · pytest

## Features
- 💬 **Cited Q&A** over placement policies, JDs and notices — every claim links to a numbered source
- 🔎 **Hybrid retrieval** — BM25 + dense vectors fused with RRF, then cross-encoder reranking
- 🏷️ **Metadata-aware** — company/doc-type/date filters and boosts
- 🔄 **Incremental ingestion** — upload new PDFs; only changed files are re-embedded
- 🛡️ **Guardrails** — prompt-injection & off-topic blocking, PII redaction, citation validation
- ✅ **Eligibility checker** — deterministic rule engine with per-criterion explanations

## Architecture

```
data/docs/*.pdf|md|txt ──► ingest.py ──► metadata-tagged chunks
   (front matter: doc_type,   │  SHA-256 manifest → only new/changed files re-embedded
    company, date, deadline)  ├─► Dense: MiniLM embeddings → FAISS (cosine)
                              └─► Lexical: BM25

query ─► guardrails.check_input (injection / off-topic / PII redaction)
      ─► retriever.py: metadata filter (company detected in query)
                       BM25 top-N  +  FAISS top-N  ─► Reciprocal Rank Fusion
                       doc-type boost ─► cross-encoder rerank (ms-marco-MiniLM) ─► top-5
      ─► generator.py: GPT-4o-mini, numbered sources, cite-or-refuse prompt, temperature 0
      ─► guardrails.check_output (invalid citations stripped, uncited answers flagged)

eligibility.py: pure rule engine (degree, branch, CGPA, 10th/12th, backlogs, batch,
                deadline, "dream offer" policy) — no LLM, fully explainable.
```

## Run

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # add OPENAI_API_KEY (optional — extractive mode without it)
streamlit run app.py
```

Tests: `pip install pytest` then `py -m pytest -q`.
CLI ingestion: `py -m campushire.ingest`.

## Adding documents
Drop files in `data/docs/` (or upload in the sidebar) and click **Re-sync**. Optional front matter:

```
---
doc_type: jd
company: Acme
date: 2026-11-20
deadline: 2026-12-01
---
```
Structured eligibility criteria live in `data/companies.json`.

## Design choices
- **Hybrid retrieval**: BM25 catches exact terms (company names, "CGPA 7.0", dates); dense search catches paraphrases ("cutoff" ≈ "minimum CGPA"). RRF merges ranks without score calibration.
- **Cross-encoder rerank**: scores (query, chunk) jointly — more accurate than bi-encoder similarity; applied only to ~40 candidates to stay fast.
- **Eligibility is deterministic**: LLMs can mis-compare numbers; a rule engine gives auditable pass/fail per criterion.
- **Incremental ingestion**: hashes avoid re-embedding the whole corpus when new JDs arrive.

## Project structure
```
app.py                     Streamlit UI (Ask / Eligibility / Knowledge base)
campushire/config.py       Settings
campushire/ingest.py       Loading, chunking, embeddings, FAISS + BM25, incremental sync
campushire/retriever.py    Hybrid retrieval, RRF fusion, metadata filters, reranking
campushire/guardrails.py   Input/output guardrails
campushire/generator.py    Grounded generation with citations
campushire/eligibility.py  Rule-based eligibility engine
data/docs/                 Sample policy, JDs and notices (fictional sample data)
data/companies.json        Structured eligibility criteria
tests/                     Unit tests
```

## Note
Sample companies and documents are fictional demo data. Developed with the assistance of an AI coding assistant.
