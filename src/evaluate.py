import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


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


def parse_score(raw: str) -> float:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    data = json.loads(raw[start:end])
    return float(data["score"])


def faithfulness(answer: str, chunks: list[dict]) -> float:
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = f"""You are evaluating whether an AI-generated answer is faithful to its source context.

Score 1.0 if every claim in the answer is supported by the context.
Score 0.0 if the answer contains claims not found in the context.
Use values in between for partial support.

Context:
{context}

Answer:
{answer}

Respond with ONLY this JSON: {{"score": <number between 0 and 1>}}"""
    return parse_score(call_gemini(prompt))


def answer_relevancy(question: str, answer: str) -> float:
    prompt = f"""You are evaluating whether an AI-generated answer actually addresses the question asked.

Score 1.0 if the answer directly and completely addresses the question.
Score 0.0 if the answer is off-topic or does not address the question at all.
Use values in between for partial relevance.

Question:
{question}

Answer:
{answer}

Respond with ONLY this JSON: {{"score": <number between 0 and 1>}}"""
    return parse_score(call_gemini(prompt))


def context_precision(question: str, chunks: list[dict]) -> float:
    scores = []
    for chunk in chunks:
        prompt = f"""You are evaluating whether a retrieved text chunk is relevant to a question.

Score 1.0 if the chunk contains information that directly helps answer the question.
Score 0.0 if the chunk is not useful for answering the question.

Question:
{question}

Chunk:
{chunk["text"]}

Respond with ONLY this JSON: {{"score": <number between 0 and 1>}}"""
        scores.append(parse_score(call_gemini(prompt)))
    return sum(scores) / len(scores) if scores else 0.0


def context_recall(question: str, ground_truth: str, chunks: list[dict]) -> float:
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = f"""You are evaluating whether retrieved context contains enough information to answer a question correctly.

The correct answer is given below as a reference. Score 1.0 if the context contains all the information needed to produce that answer. Score 0.0 if the context is missing critical information. Use values in between for partial coverage.

Question:
{question}

Correct answer (reference only):
{ground_truth}

Retrieved context:
{context}

Respond with ONLY this JSON: {{"score": <number between 0 and 1>}}"""
    return parse_score(call_gemini(prompt))


def evaluate_one(question: str, ground_truth: str, answer: str, chunks: list[dict]) -> dict:
    return {
        "faithfulness":      faithfulness(answer, chunks),
        "answer_relevancy":  answer_relevancy(question, answer),
        "context_precision": context_precision(question, chunks),
        "context_recall":    context_recall(question, ground_truth, chunks),
    }
