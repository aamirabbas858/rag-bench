# RAG Bench

A RAG evaluation dashboard that measures retrieval and generation quality using four metrics inspired by the RAGAS research paper (Es et al., 2023).

Most RAG demos just show that the system can produce an answer. This one measures whether that answer is actually good — and explains where the pipeline is failing.

**Live demo:** [rag-bench-byabbasaamir1.streamlit.app](https://rag-bench-byabbasaamir1.streamlit.app)

---

## What it evaluates

| Metric | What it measures |
|---|---|
| Faithfulness | Every claim in the answer is supported by the retrieved context |
| Answer Relevancy | The answer directly addresses the question asked |
| Context Precision | The proportion of retrieved chunks that were actually useful |
| Context Recall | All the information needed to answer was present in the retrieved chunks |

Each metric is scored 0 to 1 by an LLM acting as an independent judge, using direct prompts rather than a library. This makes the scoring logic readable and easy to change — and makes it obvious that the judge is a parameter of the experiment, not neutral infrastructure.

## Results on the built-in test set

Evaluated on 20 questions across ML and AI fundamentals:

| Metric | Score |
|---|---|
| Faithfulness | 1.000 |
| Answer Relevancy | 0.975 |
| Context Precision | 0.367 |
| Context Recall | 0.980 |

**Judge: `gemini-3.5-flash`, July 2026.** That line is not a footnote — in
LLM-as-judge evaluation the judge *is* part of the measurement. These four
numbers are one model's opinion of another model's answers, and running the
same code against a different judge will produce different numbers on
identical data. A score reported without naming its judge is not reproducible,
and most published RAG numbers do not name one.

The pipeline now runs on Mistral by default, so re-running will not reproduce
the table above. That is the point rather than a defect.

Faithfulness and recall are near-perfect. Context precision is intentionally lower — retrieving k=3 chunks when only one is needed is a known retrieval trade-off, and surfacing that number is the point.

## How it works

1. Ten curated documents about ML and AI are embedded using `gemini-embedding-2` and stored in ChromaDB
2. For each test question, the top 3 most similar chunks are retrieved by cosine similarity
3. The model generates an answer using only those chunks as context
4. The judge is called four separate times, once per metric, so no score influences another
5. Results are saved and displayed on the dashboard

## Stack

- Python — RAG pipeline and evaluation logic
- Streamlit — dashboard UI
- ChromaDB — in-memory vector store
- Mistral (`mistral-small-latest` + `mistral-embed`), Gemini as fallback — embeddings, generation, and the evaluation judge
- Plotly — radar chart

## Run locally

```bash
git clone https://github.com/aamirabbas858/rag-bench
cd rag-bench
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add a `.env` file:

```
GEMINI_API_KEY=your_key_here
```

```bash
streamlit run app.py
```

The dashboard loads pre-computed results immediately. Use the Run Evaluation tab to re-run the full benchmark.

## Project structure

```
app.py              Streamlit dashboard
src/
  rag.py            RAG pipeline (embed, index, retrieve, generate)
  evaluate.py       Four evaluation metrics, LLM as judge
  knowledge_base.py Ten source documents
  test_set.py       Twenty questions with ground truth answers
data/
  results.json      Benchmark output (committed so the app loads instantly)
```

## Reference

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.
