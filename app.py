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
GROK_MODEL = "grok-4.3"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # characters overlap between chunks
TOP_K = 4               # number of chunks retrieved per question

EXAMPLE_QUESTIONS = [
    "What is the annual leave policy?",
    "How many sick leave days am I entitled to?",
    "What is the notice period for resignation?",
    "What are the standard working hours?",
    "What is the maternity/paternity leave policy?",
    "What happens if I violate the code of conduct?",
]


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
def extract_text_from_pdf(file_bytes: bytes):
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        num_pages = len(doc)
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts), num_pages


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
defaults = {
    "chunks": None,
    "index": None,
    "pdf_name": None,
    "num_pages": None,
    "chat_history": [],   # list of dicts: {role, content, sources (optional)}
    "pending_question": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def queue_question(q):
    """Called by example-question buttons and the chat input."""
    st.session_state.pending_question = q


# -----------------------------
# Sidebar
# -----------------------------
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
            progress = st.progress(0, text="Reading PDF...")
            raw_text, num_pages = extract_text_from_pdf(uploaded_file.read())
            progress.progress(30, text="Cleaning and chunking text...")
            cleaned = clean_text(raw_text)

            if not cleaned:
                progress.empty()
                st.error("Could not extract any text from this PDF. It may be scanned/image-only.")
            else:
                chunks = chunk_text(cleaned)
                progress.progress(60, text="Loading embedding model...")
                embedder = load_embedder()
                progress.progress(80, text="Building FAISS vector index...")
                index, _ = build_faiss_index(chunks, embedder)
                progress.progress(100, text="Done!")

                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.num_pages = num_pages
                st.session_state.chat_history = []
                progress.empty()
                st.success(f"Indexed '{uploaded_file.name}'")

    if st.session_state.pdf_name:
        st.markdown("---")
        st.subheader("📊 Document stats")
        c1, c2 = st.columns(2)
        c1.metric("Pages", st.session_state.num_pages)
        c2.metric("Chunks", len(st.session_state.chunks))
        st.caption(f"Active document: **{st.session_state.pdf_name}**")
        if st.button("🗑️ Clear document", use_container_width=True):
            for key, value in defaults.items():
                st.session_state[key] = value
            st.rerun()

        st.markdown("---")
        st.subheader("💡 Example questions")
        st.caption("Click one to ask it instantly.")
        for q in EXAMPLE_QUESTIONS:
            st.button(q, key=f"ex_{q}", use_container_width=True, on_click=queue_question, args=(q,))


# -----------------------------
# Main area
# -----------------------------
st.title("📄 HR Policy Assistant")
st.caption("Upload an HR policy PDF and ask questions about it. Powered by RAG (FAISS + Sentence Transformers) and Grok.")

with st.expander("ℹ️ How this works (RAG pipeline explained)"):
    st.markdown(
        """
This app answers your questions **only** using the content of the PDF you upload,
following a retrieval-augmented generation (RAG) pipeline:

1. **Extract** — Your PDF's text is pulled out page by page with PyMuPDF.
2. **Chunk** — The text is split into overlapping ~800-character passages so no
   context is lost at the edges.
3. **Embed** — Each passage is converted into a numeric vector using a Sentence
   Transformer model (`all-MiniLM-L6-v2`), capturing its meaning.
4. **Index** — All passage vectors are stored in a FAISS index for fast similarity search.
5. **Retrieve** — When you ask a question, it's embedded the same way, and FAISS
   finds the passages whose meaning is closest to your question.
6. **Generate** — The top matching passages are sent to Grok along with your
   question, with instructions to answer *only* from those passages.

Every answer includes a **"Sources & relevance"** panel underneath it so you can
see exactly which parts of the document were used, and how relevant each one was.
        """
    )

if not st.session_state.index:
    st.info("👈 Upload an HR policy PDF from the sidebar to get started, or try the sample document.")
else:
    # Render chat history
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 Sources & relevance for this answer"):
                    for j, (chunk, score) in enumerate(msg["sources"]):
                        pct = max(0.0, min(1.0, score))
                        st.markdown(f"**Excerpt {j+1}** — relevance {pct*100:.0f}%")
                        st.progress(pct)
                        st.caption(chunk[:400] + ("..." if len(chunk) > 400 else ""))

    # Chat input
    typed_question = st.chat_input("Ask something about the uploaded HR policy...")
    if typed_question:
        st.session_state.pending_question = typed_question

    # Process a pending question (from chat input OR an example-question button)
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not api_key:
                answer = "Grok API key is missing. Please add GROK_API_KEY in Streamlit secrets."
                sources = []
                st.markdown(answer)
            else:
                status = st.status("Thinking...", expanded=True)
                status.write("🔎 Searching the document for relevant passages...")
                embedder = load_embedder()
                results = search_index(
                    question, embedder, st.session_state.index, st.session_state.chunks
                )
                status.write(f"✅ Found {len(results)} relevant passage(s).")
                status.write("🧠 Asking Grok to generate an answer from those passages...")
                try:
                    answer = ask_grok(question, results, api_key)
                    status.update(label="Answer ready", state="complete", expanded=False)
                except Exception as e:
                    answer = f"Error calling Grok API: {e}"
                    status.update(label="Error", state="error", expanded=False)
                sources = results

            st.markdown(answer)
            if sources:
                with st.expander("🔍 Sources & relevance for this answer"):
                    for j, (chunk, score) in enumerate(sources):
                        pct = max(0.0, min(1.0, score))
                        st.markdown(f"**Excerpt {j+1}** — relevance {pct*100:.0f}%")
                        st.progress(pct)
                        st.caption(chunk[:400] + ("..." if len(chunk) > 400 else ""))

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
