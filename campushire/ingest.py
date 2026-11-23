"""Incremental ingestion: documents -> metadata-tagged chunks -> FAISS (dense) + BM25 (lexical).

Each document may start with a front-matter block to supply metadata:

    ---
    doc_type: jd            # policy | jd | notice | eligibility
    company: Kaboom AI
    date: 2025-10-02
    deadline: 2025-10-10
    ---

Files are tracked by SHA-256, so re-running ingestion only embeds new/changed files
and drops chunks for deleted files.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader

from . import config

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int
    metadata: dict = field(default_factory=dict)


# ---------- loading ----------
def _parse_front_matter(text: str) -> tuple[dict, str]:
    meta: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.strip()
            text = text[end + 4:]
    return meta, text


def _infer_doc_type(name: str) -> str:
    n = name.lower()
    for key, kind in [("policy", "policy"), ("jd", "jd"), ("notice", "notice"),
                      ("eligib", "eligibility"), ("job", "jd")]:
        if key in n:
            return kind
    return "general"


def load_file(path: Path) -> list[tuple[int, str, dict]]:
    """Return a list of (page_number, text, metadata)."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
        meta, first = _parse_front_matter(pages[0][1]) if pages else ({}, "")
        if pages:
            pages[0] = (1, first)
    else:
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        pages = [(1, body)]
    meta.setdefault("doc_type", _infer_doc_type(path.name))
    meta.setdefault("company", "")
    return [(pg, txt, meta) for pg, txt in pages if txt.strip()]


def split_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware splitter with character overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
            continue
        if cur:
            chunks.append(cur)
        while len(p) > size:                      # hard-split very long paragraphs
            chunks.append(p[:size])
            p = p[size - overlap:]
        cur = (chunks[-1][-overlap:] + "\n\n" + p) if chunks else p
    if cur:
        chunks.append(cur)
    return chunks


# ---------- index ----------
class KnowledgeBase:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.manifest: dict[str, str] = {}       # filename -> sha256
        self.embeddings: np.ndarray | None = None
        self.faiss_index = None
        self.bm25 = None
        self._embedder = None

    # lazy model load (keeps imports fast)
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(config.EMBED_MODEL)
        return self._embedder

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.embedder.encode(texts, normalize_embeddings=True, batch_size=32,
                                    show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def _rebuild_indexes(self):
        from rank_bm25 import BM25Okapi
        if not self.chunks:
            self.faiss_index, self.bm25 = None, None
            return
        self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])  # cosine (normalized)
        self.faiss_index.add(self.embeddings)
        self.bm25 = BM25Okapi([tokenize(c.text + " " + c.metadata.get("company", ""))
                               for c in self.chunks])

    # ---- incremental sync with DOCS_DIR ----
    def sync(self, docs_dir: Path = config.DOCS_DIR) -> dict:
        docs_dir.mkdir(parents=True, exist_ok=True)
        current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(docs_dir.iterdir())
                   if p.suffix.lower() in {".pdf", ".md", ".txt"}}

        changed = [n for n, h in current.items() if self.manifest.get(n) != h]
        removed = [n for n in self.manifest if n not in current]
        stale = set(changed) | set(removed)

        if stale and self.chunks:
            keep = [i for i, c in enumerate(self.chunks) if c.source not in stale]
            self.chunks = [self.chunks[i] for i in keep]
            self.embeddings = self.embeddings[keep] if keep else None

        new_chunks: list[Chunk] = []
        for name in changed:
            for page, text, meta in load_file(docs_dir / name):
                for j, piece in enumerate(split_text(text)):
                    new_chunks.append(Chunk(f"{name}#p{page}c{j}", piece, name, page, dict(meta)))

        if new_chunks:
            vecs = self.embed([c.text for c in new_chunks])
            self.embeddings = vecs if self.embeddings is None else np.vstack([self.embeddings, vecs])
            self.chunks.extend(new_chunks)

        for n in removed:
            self.manifest.pop(n, None)
        self.manifest.update({n: current[n] for n in changed})

        if stale:
            self._rebuild_indexes()
            self.save()
        return {"added_or_updated": changed, "removed": removed,
                "new_chunks": len(new_chunks), "total_chunks": len(self.chunks)}

    # ---- persistence ----
    def save(self, path: Path = config.INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "kb.pkl", "wb") as f:
            pickle.dump({"chunks": [asdict(c) for c in self.chunks],
                         "manifest": self.manifest, "embeddings": self.embeddings}, f)
        (path / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, path: Path = config.INDEX_DIR) -> "KnowledgeBase":
        kb = cls()
        f = path / "kb.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            kb.chunks = [Chunk(**c) for c in data["chunks"]]
            kb.manifest = data["manifest"]
            kb.embeddings = data["embeddings"]
            kb._rebuild_indexes()
        return kb


if __name__ == "__main__":
    kb = KnowledgeBase.load()
    print(kb.sync())
