# app.py - IP-SAKTI Sahayak (terminal version)
# Run with:  python app.py
# NO API key in this file. Key is read from (in priority order):
#   1. Environment variable GEMINI_API_KEY
#   2. .streamlit/secrets.toml  (local file - it is gitignored, never uploaded)
import os
import time
import chromadb
import requests
from sentence_transformers import SentenceTransformer
from google import genai


def load_api_key():
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.isfile(secrets_path):
        with open(secrets_path) as f:
            for line in f:
                if line.strip().startswith("GEMINI_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


API_KEY = load_api_key()
if not API_KEY:
    raise SystemExit(
        "No API key found. Set GEMINI_API_KEY as an environment variable or "
        "create .streamlit/secrets.toml with: GEMINI_API_KEY = \"your-key\""
    )

gemini = genai.Client(api_key=API_KEY)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
collection = chromadb.PersistentClient(path="db").get_collection("ipsakti")


def ask_llm(prompt: str):
    for attempt in range(3):
        try:
            r = gemini.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return r.text, "Gemini (online)"
        except Exception:
            if attempt < 2:
                wait = 5 * (attempt + 1)
                print(f"(Google busy, retrying in {wait}s...)")
                time.sleep(wait)
    try:  # offline fallback - Ollama must be running: `ollama serve`
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=180,
        )
        return r.json()["response"], "Llama 3 (offline, Ollama)"
    except Exception:
        return ("Sorry, the AI service is temporarily unavailable and no offline "
                "model is running."), "unavailable"


def ask(question: str):
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


if __name__ == "__main__":
    print("=" * 60)
    print("  IP-SAKTI Sahayak - terminal version")
    print("  Type 'quit' to exit")
    print("=" * 60 + "\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        text, sources, engine = ask(q)
        print("\nSahayak:", text)
        print(f"\n[Engine: {engine}]")
        print("[Sources: " + ", ".join(sources) + "]\n")
