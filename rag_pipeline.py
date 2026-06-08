"""
rag_pipeline.py
----------------
Core RAG logic: PDF -> chunk -> embed -> FAISS -> retrieve -> Groq LLM -> answer
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (free, local)
Vector store: FAISS (free, in-memory)
Generation: Groq API (free tier) — llama3, mixtral, gemma
"""

from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------- data structures ----------
@dataclass
class Chunk:
    text: str
    page: int
    index: int


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


# ---------- 1. LOAD ----------
def load_pdf_text(file) -> list[tuple[int, str]]:
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


# ---------- 2. CHUNK ----------
def chunk_pages(pages: list[tuple[int, str]], chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for page_num, text in pages:
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            piece = text[start:start + chunk_size]
            if piece.strip():
                chunks.append(Chunk(text=piece, page=page_num, index=idx))
                idx += 1
            start += chunk_size - overlap
    return chunks


# ---------- 3 & 4. EMBED + STORE ----------
class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(vecs)
        return vecs.astype("float32")

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        vecs = self._embed([c.text for c in chunks])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    # ---------- 5. RETRIEVE ----------
    def retrieve(self, query: str, k: int = 4) -> list[Retrieved]:
        if self.index is None:
            raise RuntimeError("Vector store is empty — call build() first.")
        qv = self._embed([query])
        scores, ids = self.index.search(qv, k)
        out = []
        for score, i in zip(scores[0], ids[0]):
            if i == -1:
                continue
            out.append(Retrieved(chunk=self.chunks[int(i)], score=float(score)))
        return out


# ---------- 6. GENERATE ----------
PROMPT_TEMPLATE = """You are a careful assistant answering questions about a document.
Use ONLY the context below to answer. If the answer is not in the context, say
"I couldn't find that in the document." Do not invent information.

Context:
{context}

Question: {question}

Answer (grounded in the context above):"""

FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

_client: Groq | None = None


def configure_llm(api_key: str | None = None) -> None:
    global _client
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("No Groq API key found.")
    _client = Groq(api_key=key)


def generate_answer(question: str, retrieved: list[Retrieved],
                    model_name: str = "llama-3.3-70b-versatile") -> str:
    if _client is None:
        raise RuntimeError("Call configure_llm() before generate_answer().")

    context = "\n\n".join(
        f"[page {r.chunk.page}] {r.chunk.text}" for r in retrieved
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    models_to_try = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
    last_error = None
    for m in models_to_try:
        try:
            resp = _client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"All models failed. Last error: {last_error}\n\n"
        "Please check your API key at https://console.groq.com"
    )


# ---------- convenience end-to-end ----------
def build_store_from_pdf(file, chunk_size: int = 800, overlap: int = 120) -> VectorStore:
    pages = load_pdf_text(file)
    if not pages:
        raise ValueError("No extractable text found in this PDF (it may be scanned images).")
    chunks = chunk_pages(pages, chunk_size, overlap)
    store = VectorStore()
    store.build(chunks)
    return store
