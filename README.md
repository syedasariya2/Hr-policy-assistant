# 📄 HR Policy Assistant (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload an HR policy PDF
and ask natural-language questions about it.

**Stack**
- **Streamlit** – web UI
- **PyMuPDF (fitz)** – PDF text extraction
- **Sentence Transformers** (`all-MiniLM-L6-v2`) – text embeddings
- **FAISS** – vector similarity search
- **Grok (xAI API)** – answer generation

## How it works

1. You upload an HR policy PDF.
2. The app extracts the text (PyMuPDF) and splits it into overlapping chunks.
3. Each chunk is embedded with Sentence Transformers and stored in a FAISS index.
4. When you ask a question, the app embeds your question, retrieves the most
   relevant chunks from FAISS, and sends them along with your question to Grok.
5. Grok answers using only the retrieved excerpts, so answers stay grounded in
   your actual policy document.

## Interactive & explainable UI

- **In-app pipeline explainer** – an "How this works" expander on the main page
  walks through the extract → chunk → embed → index → retrieve → generate steps.
- **Example questions** – clickable buttons in the sidebar (e.g. "What is the
  annual leave policy?") ask a question instantly, no typing required.
- **Live status while answering** – a step-by-step status panel shows
  "Searching the document...", "Found N relevant passage(s)...", and
  "Asking Grok..." as each question is processed.
- **Sources & relevance panel** – every answer has an expandable panel showing
  exactly which document excerpts were retrieved, with a relevance percentage
  and progress bar for each, so you can see why the assistant answered the way
  it did.
- **Document stats** – once a PDF is indexed, the sidebar shows page count and
  chunk count for that document.

## Project files

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Files/folders excluded from git
```

## Getting a Grok API key

1. Go to the xAI developer console: https://console.x.ai
2. Create an account / sign in and generate an API key.
3. Keep this key secret — you'll add it as a Streamlit secret, never commit it to GitHub.

## Configuration

The app reads your key as `GROK_API_KEY`, either from:
- Streamlit secrets (`st.secrets["GROK_API_KEY"]`) — used automatically on Streamlit
  Community Cloud, **or**
- An environment variable named `GROK_API_KEY`.

## Deploying with no terminal, no VS Code, no local setup

Everything below is done in your web browser only, using the GitHub website and the
Streamlit Community Cloud website.

### Step 1 — Create a GitHub repository (browser only)

1. Go to https://github.com and log in (or sign up).
2. Click the **+** icon (top right) → **New repository**.
3. Name it, e.g. `hr-policy-assistant`.
4. Set it to **Public** (Streamlit Community Cloud free tier deploys public repos
   most easily; Private also works if your Streamlit account is linked to GitHub).
5. Click **Create repository**.

### Step 2 — Upload the project files (browser only)

1. On your new repo's page, click **Add file → Upload files**.
2. Drag and drop (or select) these files: `app.py`, `requirements.txt`,
   `README.md`, `.gitignore`.
3. Scroll down and click **Commit changes**.

### Step 3 — Deploy on Streamlit Community Cloud (browser only)

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **Create app** (or **New app**).
3. Choose **"Deploy a public app from GitHub"** (or select your repo directly).
4. Pick:
   - **Repository:** `<your-username>/hr-policy-assistant`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Before clicking Deploy, open **Advanced settings**.
6. Under **Secrets**, add:
   ```
   GROK_API_KEY = "your-real-grok-api-key-here"
   ```
7. Click **Save**, then click **Deploy**.
8. Streamlit Cloud will install everything from `requirements.txt` and start your
   app automatically. You'll get a public URL like:
   `https://your-app-name.streamlit.app`

### Step 4 — Updating the app later

Whenever you want to change the code:
1. Open the file on GitHub (e.g. `app.py`).
2. Click the pencil (✏️) **Edit** icon.
3. Make your changes in the browser editor.
4. Click **Commit changes**.
5. Streamlit Community Cloud auto-detects the update and redeploys the app
   within a minute or two — no manual redeploy step needed.

### Updating secrets later

1. Go to https://share.streamlit.io, open your app.
2. Click the **⋮** menu → **Settings** → **Secrets**.
3. Edit the `GROK_API_KEY` value, click **Save**. The app restarts automatically.

## Notes & limitations

- Scanned/image-only PDFs (no selectable text) won't extract text; this app does
  not include OCR.
- The FAISS index is rebuilt in memory each time you upload a new PDF and is not
  persisted between sessions — this is expected for a lightweight demo.
- `all-MiniLM-L6-v2` downloads automatically the first time the app runs on
  Streamlit Cloud (may take a little longer on first load).
