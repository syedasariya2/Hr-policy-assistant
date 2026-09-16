import os
import re
import json
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer
import requests

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="HR Policy Assistant", page_icon="📄", layout="wide")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROK_MODEL = "grok-2-latest"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # characters overlap between chunks
TOP_K = 4               # number of chunks retrieved per question


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_grok_api_key():
    # Prefer Streamlit secrets (used on Streamlit Cloud), fall back to env var
    key = None
    try:
        key = st.secrets["GROK_API_KEY"]
    except Exception:
        key = os.environ.get("GROK_API_KEY")
    return key


# -----------------------------
# PDF -> text -> chunks
# -----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap
    return chunks


# -----------------------------
# FAISS index build / search
# -----------------------------
def build_faiss_index(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
    index.add(embeddings)
    return index, embeddings


def search_index(question, embedder, index, chunks, top_k=TOP_K):
    q_emb = embedder.encode([question], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((chunks[idx], float(score)))
    return results


# -----------------------------
# Grok call
# -----------------------------
def ask_grok(question, context_chunks, api_key):
    context_text = "\n\n---\n\n".join(
        [f"[Excerpt {i+1}]\n{chunk}" for i, (chunk, _score) in enumerate(context_chunks)]
    )

    system_prompt = (
        "You are an HR Policy Assistant. Answer the user's question using ONLY the "
        "provided policy excerpts. If the answer is not contained in the excerpts, "
        "say you could not find that information in the uploaded policy document. "
        "Be concise, accurate, and cite which excerpt(s) you used when helpful."
    )

    user_prompt = (
        f"Policy excerpts:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on the excerpts above."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    response = requests.post(GROK_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


# -----------------------------
# Session state
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------
# UI
# -----------------------------
st.title("📄 HR Policy Assistant")
st.caption("Upload an HR policy PDF and ask questions about it. Powered by RAG (FAISS + Sentence Transformers) and Grok.")

with st.sidebar:
    st.header("1. Upload Policy PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    api_key = get_grok_api_key()
    if not api_key:
        st.warning(
            "No Grok API key found. Add GROK_API_KEY to Streamlit secrets "
            "(Settings → Secrets) or as an environment variable."
        )

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Reading and indexing PDF..."):
                raw_text = extract_text_from_pdf(uploaded_file.read())
                cleaned = clean_text(raw_text)
                if not cleaned:
                    st.error("Could not extract any text from this PDF. It may be scanned/image-only.")
                else:
                    chunks = chunk_text(cleaned)
                    embedder = load_embedder()
                    index, _ = build_faiss_index(chunks, embedder)

                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
            st.success(f"Indexed '{uploaded_file.name}' into {len(st.session_state.chunks)} chunks.")

    if st.session_state.pdf_name:
        st.info(f"Active document: **{st.session_state.pdf_name}**")
        if st.button("Clear document"):
            st.session_state.chunks = None
            st.session_state.index = None
            st.session_state.pdf_name = None
            st.session_state.chat_history = []
            st.rerun()

st.header("2. Ask a Question")

if not st.session_state.index:
    st.info("Upload an HR policy PDF from the sidebar to get started.")
else:
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    question = st.chat_input("Ask something about the uploaded HR policy...")

    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not api_key:
                answer = "Grok API key is missing. Please add GROK_API_KEY in Streamlit secrets."
                st.markdown(answer)
            else:
                with st.spinner("Searching policy and generating answer..."):
                    embedder = load_embedder()
                    results = search_index(
                        question, embedder, st.session_state.index, st.session_state.chunks
                    )
                    try:
                        answer = ask_grok(question, results, api_key)
                    except Exception as e:
                        answer = f"Error calling Grok API: {e}"
                st.markdown(answer)

                with st.expander("View retrieved excerpts"):
                    for i, (chunk, score) in enumerate(results):
                        st.markdown(f"**Excerpt {i+1}** (similarity: {score:.3f})")
                        st.write(chunk)

        st.session_state.chat_history.append(("assistant", answer))
