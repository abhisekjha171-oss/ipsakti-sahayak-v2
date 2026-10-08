import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
collection = chromadb.PersistentClient(path="db").get_collection("ipsakti")

q = input("Q: ")
q_vec = embedder.encode([q]).tolist()
results = collection.query(query_embeddings=q_vec, n_results=8)

for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print("=" * 60)
    print("SOURCE:", meta["source"], "| distance:", round(dist, 3))
    print(doc[:400])
    print()