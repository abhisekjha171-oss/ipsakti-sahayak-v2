# app_streamlit.py - IP-SAKTI Sahayak (Premium Ayurveda Theme)
# Run with:  streamlit run app_streamlit.py
# IMPORTANT: put your Ayurveda image in the SAME folder as this file, named: bg.jpg
# NO API key lives in this file. Key comes from Streamlit secrets (cloud) or
# .streamlit/secrets.toml (local). See README.
import os
import base64
import time
import requests
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(
    page_title="IP-SAKTI Sahayak",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- premium Ayurvedic theme ----------------
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    bg_img = get_base64("bg.jpg")
except FileNotFoundError:
    st.error("bg.jpg not found. Put your Ayurveda background image (named bg.jpg) next to app_streamlit.py.")
    st.stop()

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

/* Background image with soft cream overlay so text stays readable */
.stApp {
    background:
        linear-gradient(rgba(248, 245, 236, 0.88), rgba(240, 238, 224, 0.92)),
        url("data:image/jpg;base64,BGIMG");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Hero header - deep green with gold, layered 3D shadow */
.hero {
    background: linear-gradient(120deg, #123524 0%, #1b4332 45%, #2d5a3d 80%, #6b4f1d 140%);
    padding: 1.8rem 2.2rem;
    border-radius: 20px;
    color: #f5f0e1;
    margin-bottom: 1.4rem;
    box-shadow:
        0 20px 50px rgba(18, 53, 36, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(201, 162, 39, 0.35);
}
.hero h1 {
    margin: 0;
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: .5px;
    text-shadow: 0 2px 8px rgba(0,0,0,.4);
}
.hero .gold {
    color: #e3c565;
}
.hero p {
    margin: .4rem 0 0 0;
    font-weight: 300;
    color: #e9e4d0;
    font-size: .95rem;
}
.hero hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, #c9a227, rgba(201,162,39,0));
    margin: .8rem 0 0 0;
}

/* Glassmorphism chat cards */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: .9rem 1.1rem;
    border: 1px solid rgba(255, 255, 255, 0.65);
    border-left: 4px solid #b07d2b;
    box-shadow: 0 8px 28px rgba(27, 67, 50, 0.14);
    margin-bottom: .8rem;
}

/* Sidebar - deep green glass with gold */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10281c 0%, #1b4332 60%, #234f38 100%);
}
[data-testid="stSidebar"] * { color: #eee8d5; }
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    font-family: 'Playfair Display', serif;
}
[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 255, 255, 0.07);
    color: #f3ecd8;
    border: 1px solid rgba(201, 162, 39, 0.4);
    border-radius: 12px;
    text-align: left;
    font-size: .85rem;
    transition: all .25s ease;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(201, 162, 39, 0.22);
    border-color: #e3c565;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,.3);
}

/* Buttons - gold accent, 3D hover lift */
.stButton button {
    border-radius: 12px;
    transition: all .25s ease;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(27,67,50,.25);
}

/* Chat input - glass pill */
.stChatInputContainer {
    border-radius: 14px;
    background: rgba(255,255,255,.75);
    backdrop-filter: blur(10px);
}

/* Expander headers in deep green */
details summary { font-weight: 600; color: #1b4332; }

/* Disclaimer footer */
.disclaimer {
    text-align: center;
    color: #6b5d3f;
    font-size: .8rem;
    padding: .8rem 0 .4rem 0;
}
"""

st.markdown("<style>" + CSS.replace("BGIMG", bg_img) + "</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>🌿 IP-SAKTI <span class="gold">Sahayak</span></h1>'
    "<p>Ayurveda Intellectual Property &amp; Regulatory Guidance · "
    "हिंदी + English · Every answer source-cited</p><hr></div>",
    unsafe_allow_html=True,
)

# ---------------- cached resources ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def _chunk_text(text):
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        c = text[i:i + CHUNK_SIZE].strip()
        if len(c) > 50:
            chunks.append(c)
    return chunks

@st.cache_resource
def load_collection():
    # No parameters - Streamlit cannot hash ML objects.
    emb_model = load_embedder()
    client = chromadb.PersistentClient(path="db")
    try:
        return client.get_collection("ipsakti")
    except Exception:
        pass  # not built yet -> build from data/ PDFs

    from pypdf import PdfReader

    docs, ids, metas = [], [], []
    doc_n = 0
    data_dir = "data"
    if os.path.isdir(data_dir):
        for fn in os.listdir(data_dir):
            if not fn.lower().endswith(".pdf"):
                continue
            reader = PdfReader(os.path.join(data_dir, fn))
            full = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    full += t + "\n"
            for j, c in enumerate(_chunk_text(full)):
                docs.append(c)
                ids.append(f"doc{doc_n}_chunk{j}")
                metas.append({"source": fn, "chunk": j})
            doc_n += 1

    if not docs:
        st.error(
            "Knowledge base not found. Upload either the db/ folder "
            "or your PDFs in a data/ folder to the repo, then Reboot."
        )
        st.stop()

    st.warning(
        "First run on this server: building the knowledge base from data/ PDFs. "
        "This can take 5-10 minutes - please don't close this tab."
    )
    emb = emb_model.encode(docs, show_progress_bar=False).tolist()
    col = client.get_or_create_collection(
        name="ipsakti", metadata={"hnsw:space": "cosine"}
    )
    col.add(documents=docs, ids=ids, metadatas=metas, embeddings=emb)
    return col

@st.cache_resource
def load_gemini():
    # Key is read from secrets/env ONLY - never hardcoded.
    key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not key:
        st.error("No Gemini API key found. Add GEMINI_API_KEY in Streamlit secrets.")
        st.stop()
    return genai.Client(api_key=key)

embedder = load_embedder()
collection = load_collection()
gemini = load_gemini()

# ---------------- LLM: Gemini with retries, then Ollama offline fallback ----------------
def ask_llm(prompt: str):
    for attempt in range(3):
        try:
            r = gemini.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return r.text, "Gemini (online)"
        except Exception:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    try:  # offline fallback - Ollama must be running: `ollama serve`
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=180,
        )
        return r.json()["response"], "Llama 3 (offline, Ollama)"
    except Exception:
        return ("Sorry, the AI service is temporarily unavailable and no offline "
                "model is running. Start Ollama or try again later."), "unavailable"

# ---------------- RAG ----------------
def answer(question: str):
    q_vec = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_vec, n_results=4)

    context = ""
    sources = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"[Source: {meta['source']}]\n{doc}\n\n"
        if meta["source"] not in sources:
            sources.append(meta["source"])

    prompt = f"""You are IP-SAKTI Sahayak, an assistant for Intellectual Property
and regulatory guidance in Ayurveda (Indian + international regimes).

Rules:
- Base your answer ONLY on the context below.
- Synthesize information from ALL provided chunks; combine them into one
  coherent answer. Do not mention "chunks" or "context" in your answer.
- If the context covers the topic but not the exact question asked, explain
  what the context DOES say about the topic.
- Only if the context is completely unrelated to the question, say:
  "I don't have enough information in my knowledge base."
- Always cite sources inline like [source filename].
- Answer in the SAME language as the question. Note: the context is in
  English - that's fine, translate/summarize it into the question's language.

CONTEXT:
{context}

QUESTION: {question}"""

    text, engine = ask_llm(prompt)
    return text, sources, engine

# ---------------- chat ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def render_history():
    for m in st.session_state.messages:
        avatar = "🧑‍💼" if m["role"] == "user" else "🌿"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("📚 Sources"):
                    for s in m["sources"]:
                        st.caption(f"• {s}")
                    st.caption(f"Engine: {m['engine']}")

def respond(q: str):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(q)
    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("🔍 Searching the knowledge base..."):
            text, sources, engine = answer(q)
        st.markdown(text)
        with st.expander("📚 Sources"):
            for s in sources:
                st.caption(f"• {s}")
            st.caption(f"Engine: {engine}")
    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources, "engine": engine}
    )

render_history()

# ---------------- sidebar: example questions + info ----------------
with st.sidebar:
    st.markdown("### ✨ Try an example")
    st.caption("Tap any question - perfect for demos.")
    examples = [
        "क्या आयुर्वेदिक उत्पाद को patent मिल सकता है?",
        "What is a Geographical Indication and how does it protect Ayurveda products?",
        "NDCT Rules 2019 में traditional knowledge के लिए क्या प्रावधान हैं?",
        "How can an Ayurveda startup register a trademark?",
        "Ayurveda में biopiracy से बचाव कैसे होता है?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            respond(ex)
            st.rerun()

    st.markdown("---")
    st.markdown("#### About")
    st.markdown(
        "Answers come **only** from official documents in the knowledge base — "
        "each claim cites its source. Works in Hindi & English."
    )
    st.markdown(
        "🛡️ *Offline mode:* if the API is unreachable, the app switches to a "
        "local Llama 3 model automatically."
    )

# ---------------- main input ----------------
if q := st.chat_input("Ask about Ayurveda patents, GI tags, trademarks..."):
    respond(q)

# ---------------- standing disclaimer (required by the PS) ----------------
st.markdown(
    '<div class="disclaimer">⚖️ IP-SAKTI Sahayak provides information for guidance, '
    "not legal advice. Always consult a qualified IP attorney or the relevant "
    "registry before acting.</div>",
    unsafe_allow_html=True,
)
