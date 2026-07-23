import os
import requests
import chromadb
from dotenv import load_dotenv
from knowledge_base import DOCUMENTS

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def embed_text(text: str) -> list[float]:
    url = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/gemini-embedding-2:embedContent?key={GEMINI_API_KEY}"
    )
    r = requests.post(
        url,
        json={"content": {"parts": [{"text": text}]}},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["embedding"]["values"]


def call_gemini(prompt: str) -> str:
    models = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]
    for model in models:
        url = (
            "https://generativelanguage.googleapis.com"
            f"/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        )
        try:
            r = requests.post(
                url,
                json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
                timeout=60,
            )
        except requests.exceptions.Timeout:
            continue
        if r.status_code in (503, 429):
            continue
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError("All Gemini models temporarily unavailable.")


def build_index() -> chromadb.Collection:
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="knowledge_base")

    for doc in DOCUMENTS:
        embedding = embed_text(doc["text"])
        collection.add(
            ids=[doc["id"]],
            embeddings=[embedding],
            documents=[doc["text"]],
            metadatas=[{"title": doc["title"], "source": doc["source"]}],
        )

    return collection


def retrieve(collection: chromadb.Collection, question: str, k: int = 3) -> list[dict]:
    embedding = embed_text(question)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas"],
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "title": meta["title"],
            "source": meta["source"],
        })
    return chunks


def generate_answer(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['title']}]\n{c['text']}" for c in chunks
    )
    prompt = f"""Answer the question using only the context below. Be concise and accurate.
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
    return call_gemini(prompt)
