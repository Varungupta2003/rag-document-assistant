"""
app.py — Streamlit UI for RAG Document Q&A Assistant (Groq backend)
"""

import streamlit as st

from rag_pipeline import (
    build_store_from_pdf,
    configure_llm,
    generate_answer,
    FALLBACK_MODELS,
)

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")


def get_api_key() -> str | None:
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return st.session_state.get("api_key")


st.title("📄 RAG Document Q&A Assistant")
st.caption(
    "Upload a PDF and ask questions. Answers are grounded in the document using "
    "Retrieval-Augmented Generation — and every answer shows its source chunks."
)

with st.sidebar:
    st.header("Setup")
    if "GROQ_API_KEY" not in st.secrets:
        st.text_input(
            "Groq API key",
            type="password",
            key="api_key",
            help="Free key from https://console.groq.com",
            placeholder="gsk_...",
        )
    st.markdown("**Model**")
    model_choice = st.selectbox("Groq model", FALLBACK_MODELS, index=0)
    st.markdown("**Retrieval settings**")
    top_k = st.slider("Chunks to retrieve (k)", 1, 8, 4)
    chunk_size = st.slider("Chunk size (chars)", 300, 1500, 800, step=100)
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. PDF → split into chunks\n"
        "2. Chunks → embeddings (MiniLM)\n"
        "3. Stored in FAISS\n"
        "4. Your question → embedded\n"
        "5. Closest chunks retrieved (cosine)\n"
        "6. Chunks + question → Groq LLM → answer"
    )

# ---------- ingest ----------
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is not None:
    sig = (uploaded.name, uploaded.size, chunk_size)
    if st.session_state.get("sig") != sig:
        with st.spinner("Reading, chunking, and embedding the document..."):
            try:
                store = build_store_from_pdf(uploaded, chunk_size=chunk_size)
                st.session_state.store = store
                st.session_state.sig = sig
                st.success(f"Processed '{uploaded.name}' into {len(store.chunks)} chunks.")
            except Exception as e:
                st.error(f"Could not process this PDF: {e}")
                st.session_state.store = None

# ---------- ask ----------
question = st.text_input("Ask a question about the document")

if st.button("Ask", type="primary") and question:
    store = st.session_state.get("store")
    if store is None:
        st.warning("Upload a PDF first.")
        st.stop()

    key = get_api_key()
    if not key:
        st.warning("Add your Groq API key in the sidebar.")
        st.stop()

    try:
        configure_llm(key)
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            retrieved = store.retrieve(question, k=top_k)
            answer = generate_answer(question, retrieved, model_name=model_choice)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources used (retrieved chunks)")
        st.caption(
            "These are the exact passages retrieved and fed to the model. "
            "Higher similarity = more relevant."
        )
        for i, r in enumerate(retrieved, 1):
            with st.expander(f"Chunk {i} · page {r.chunk.page} · similarity {r.score:.3f}"):
                st.write(r.chunk.text)
    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.divider()
st.caption(
    "Built by Varun Gupta · LangChain-free RAG with FAISS + sentence-transformers + Groq · "
    "[GitHub](https://github.com/Varungupta2003)"
)
