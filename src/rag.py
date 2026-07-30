import os
import requests
import chromadb
from dotenv import load_dotenv
from knowledge_base import DOCUMENTS

from providers import EMBED_MODEL, call_llm, embed_text

load_dotenv()


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
    # A question, not a stored chunk. NVIDIA's retriever is trained
    # asymmetrically and quietly retrieves worse if told the wrong one.
    embedding = embed_text(question, kind="query")
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
    return call_llm(prompt)
