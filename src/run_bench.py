import sys
import json
import time
sys.path.insert(0, "src")

from rag import build_index, retrieve, generate_answer
from evaluate import evaluate_one
from test_set import TEST_CASES

print("Building index...")
collection = build_index()
print(f"Index ready with {collection.count()} documents.\n")

results = []

for i, case in enumerate(TEST_CASES):
    qid      = case["id"]
    question = case["question"]
    gt       = case["ground_truth"]

    chunks = retrieve(collection, question, k=3)
    answer = generate_answer(question, chunks)
    scores = evaluate_one(question, gt, answer, chunks)

    result = {
        "id":            qid,
        "question":      question,
        "answer":        answer,
        "ground_truth":  gt,
        "chunks":        chunks,
        "scores":        scores,
    }
    results.append(result)

    avg = sum(scores.values()) / len(scores)
    print(f"[{i+1:02d}/20] {qid}  avg={avg:.2f}  {scores}")

    if i < len(TEST_CASES) - 1:
        time.sleep(1)

with open("data/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to data/results.json")

overall = {}
for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    overall[metric] = sum(r["scores"][metric] for r in results) / len(results)

print("\nOverall scores:")
for k, v in overall.items():
    print(f"  {k}: {v:.3f}")
