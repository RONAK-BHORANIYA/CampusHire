"""Metadata-aware hybrid retrieval: BM25 + dense (FAISS) fused with RRF, then cross-encoder rerank."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from . import config
from .ingest import Chunk, KnowledgeBase, tokenize

DOC_TYPE_HINTS = {
    "policy": ["policy", "rule", "rules", "backlog", "one student one job", "dream", "penalty", "debar"],
    "jd": ["role", "jd", "job description", "responsibilit", "skills", "ctc", "package", "stipend"],
    "notice": ["deadline", "last date", "notice", "register", "registration", "schedule", "drive date"],
    "eligibility": ["eligible", "eligibility", "cgpa", "criteria", "cutoff"],
}


@dataclass
class Hit:
    chunk: Chunk
    rrf: float
    rerank: float | None = None


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(config.RERANK_MODEL)
        return self._reranker

    # ---------- metadata ----------
    def infer_filters(self, query: str) -> dict:
        """Detect company names / doc types mentioned in the query."""
        q = query.lower()
        companies = {c.metadata.get("company", "") for c in self.kb.chunks} - {""}
        filters = {}
        hit = [c for c in companies if re.search(rf"\b{re.escape(c.lower())}\b", q)]
        if hit:
            filters["company"] = hit
        return filters

    def _doc_type_boost(self, query: str, chunk: Chunk) -> float:
        q = query.lower()
        hints = DOC_TYPE_HINTS.get(chunk.metadata.get("doc_type", ""), [])
        return 1.15 if any(h in q for h in hints) else 1.0

    def _allowed(self, chunk: Chunk, filters: dict) -> bool:
        if "company" in filters:
            comp = chunk.metadata.get("company", "")
            # company-agnostic docs (policies) remain eligible
            return comp == "" or comp in filters["company"]
        if "doc_type" in filters:
            return chunk.metadata.get("doc_type") in filters["doc_type"]
        return True

    # ---------- retrieval ----------
    def _bm25(self, query: str, n: int) -> list[int]:
        scores = self.kb.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1]
        return [int(i) for i in order[:n] if scores[i] > 0]

    def _dense(self, query: str, n: int) -> list[int]:
        qv = self.kb.embed([query])
        _, idx = self.kb.faiss_index.search(qv, min(n, len(self.kb.chunks)))
        return [int(i) for i in idx[0] if i != -1]

    def retrieve(self, query: str, filters: dict | None = None,
                 top_k: int = config.TOP_K_FINAL, use_rerank: bool = True) -> list[Hit]:
        if not self.kb.chunks:
            return []
        filters = filters if filters is not None else self.infer_filters(query)
        n = config.CANDIDATES_PER_RETRIEVER

        # Reciprocal Rank Fusion of the two ranked lists
        fused: dict[int, float] = {}
        for ranked in (self._bm25(query, n * 2), self._dense(query, n * 2)):
            rank = 0
            for i in ranked:
                if not self._allowed(self.kb.chunks[i], filters):
                    continue
                fused[i] = fused.get(i, 0.0) + 1.0 / (config.RRF_K + rank + 1)
                rank += 1
                if rank >= n:
                    break

        hits = [Hit(self.kb.chunks[i], s * self._doc_type_boost(query, self.kb.chunks[i]))
                for i, s in fused.items()]
        hits.sort(key=lambda h: h.rrf, reverse=True)

        if use_rerank and hits:
            scores = self.reranker.predict([(query, h.chunk.text) for h in hits])
            for h, s in zip(hits, scores):
                h.rerank = float(s)
            hits = [h for h in hits if h.rerank >= config.MIN_RERANK_SCORE]
            hits.sort(key=lambda h: h.rerank, reverse=True)
        return hits[:top_k]
