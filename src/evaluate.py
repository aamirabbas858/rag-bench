import os
import json
import requests
from dotenv import load_dotenv

from providers import EMBED_MODEL, call_llm, embed_text

load_dotenv()


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
    return parse_score(call_llm(prompt))


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
    return parse_score(call_llm(prompt))


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
        scores.append(parse_score(call_llm(prompt)))
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
    return parse_score(call_llm(prompt))


def evaluate_one(question: str, ground_truth: str, answer: str, chunks: list[dict]) -> dict:
    return {
        "faithfulness":      faithfulness(answer, chunks),
        "answer_relevancy":  answer_relevancy(question, answer),
        "context_precision": context_precision(question, chunks),
        "context_recall":    context_recall(question, ground_truth, chunks),
    }
