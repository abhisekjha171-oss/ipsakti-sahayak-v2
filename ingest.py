# ingest.py
# Runs ONCE. Builds the knowledge base from PDFs in /data.

import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
DATA_DIR = "data"
DB_DIR = "db"
COLLECTION_NAME = "ipsakti"
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap so sentences don't get cut mid-thought

# ---------- 1. Load the free local embedding model ----------
# This downloads ONCE (~90 MB) to your machine, then runs offline.
# It converts text -> a list of 384 numbers that capture meaning.
print("Loading embedding model (first run downloads ~90MB)...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ---------- 2. Helper: split one PDF into overlapping chunks ----------
def load_pdf_chunks(filepath):
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # simple character-based chunking (good enough for hackathon)
    chunks = []
    for i in range(0, len(full_text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = full_text[i : i + CHUNK_SIZE].strip()
        if len(chunk) > 50:      # skip tiny useless scraps
            chunks.append(chunk)
    return chunks

# ---------- 3. Create / open the local Chroma database ----------
# Chroma stores everything in the ./db folder. No server needed.
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}   # standard similarity measure
)

# ---------- 4. Loop over all PDFs, embed, store ----------
all_docs, all_ids, all_metas = [], [], []
doc_counter = 0

for filename in os.listdir(DATA_DIR):
    if not filename.lower().endswith(".pdf"):
        continue
    filepath = os.path.join(DATA_DIR, filename)
    print(f"Processing: {filename}")

    chunks = load_pdf_chunks(filepath)
    for j, chunk in enumerate(chunks):
        all_docs.append(chunk)
        all_ids.append(f"doc{doc_counter}_chunk{j}")
        all_metas.append({"source": filename, "chunk": j})
    doc_counter += 1

print(f"Total chunks to store: {len(all_docs)}")

# Batch embed (convert all chunks to vectors) and store in Chroma
embeddings = embedder.encode(all_docs, show_progress_bar=True).tolist()
collection.add(documents=all_docs, ids=all_ids, metadatas=all_metas, embeddings=embeddings)

print(f"Done! Knowledge base built with {collection.count()} chunks.")