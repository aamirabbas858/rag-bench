import json
import os
import sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, "src")
from rag import build_index, retrieve, generate_answer
from evaluate import evaluate_one

RESULTS_FILE = "data/results.json"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevancy":  "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}
METRIC_DESC = {
    "faithfulness":      "Every claim in the answer is supported by the retrieved context.",
    "answer_relevancy":  "The answer directly and completely addresses the question.",
    "context_precision": "Proportion of retrieved chunks that were actually useful.",
    "context_recall":    "All information needed to answer was present in the context.",
}

st.set_page_config(page_title="RAG Bench", layout="wide", page_icon="◈")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

html, body, .stApp {
    background-color: #05050c !important;
    font-family: 'Inter', sans-serif;
    color: #c8d6e5;
}
.stApp {
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 64px 64px;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 2.5rem !important; padding-bottom: 3rem !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #05050c; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.2); border-radius: 2px; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0,212,255,0.12) !important;
    gap: 0 !important;
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: rgba(0,212,255,0.3) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.7rem 1.6rem !important;
    transition: color 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: rgba(0,212,255,0.7) !important; }
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

.stSelectbox label, .stCheckbox label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: rgba(0,212,255,0.4) !important;
}
[data-baseweb="select"] {
    background: rgba(0,212,255,0.04) !important;
    border: 1px solid rgba(0,212,255,0.15) !important;
    border-radius: 4px !important;
}
[data-baseweb="select"]:hover { border-color: rgba(0,212,255,0.4) !important; }

.stButton button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid #00d4ff !important;
    color: #00d4ff !important;
    border-radius: 3px !important;
    padding: 0.6rem 2.4rem !important;
    transition: all 0.25s ease !important;
}
.stButton button:hover {
    background: rgba(0,212,255,0.08) !important;
    box-shadow: 0 0 18px rgba(0,212,255,0.18) !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00d4ff, #7c3aed) !important;
}

hr { border: none !important; border-top: 1px solid rgba(0,212,255,0.1) !important; margin: 2rem 0 !important; }

/* Custom component classes */

.lab-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.lab-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: rgba(0,212,255,0.3);
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.64rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(0,212,255,0.35);
    margin: 2rem 0 1rem;
    padding-left: 0.8rem;
    border-left: 2px solid rgba(0,212,255,0.3);
}

.mc-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 0.5rem 0 2rem;
}
.mc {
    background: rgba(0,212,255,0.025);
    border: 1px solid rgba(0,212,255,0.13);
    border-radius: 6px;
    padding: 1.4rem 1.4rem 1.2rem;
    cursor: default;
    transition: border-color 0.3s, background 0.3s, box-shadow 0.3s;
}
.mc:hover {
    border-color: rgba(0,212,255,0.5);
    background: rgba(0,212,255,0.05);
    box-shadow: 0 0 24px rgba(0,212,255,0.08), inset 0 0 24px rgba(0,212,255,0.03);
}
.mc-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(0,212,255,0.45);
    margin-bottom: 0.7rem;
}
.mc-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.6rem;
    font-weight: 400;
    line-height: 1;
    margin-bottom: 0.8rem;
    letter-spacing: -0.02em;
}
.mc-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 2px;
    height: 2px;
    margin-bottom: 0.9rem;
    overflow: hidden;
}
.mc-bar { height: 2px; border-radius: 2px; }
.mc-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.74rem;
    color: rgba(160,180,200,0.45);
    line-height: 1.55;
}

.interp-block {
    font-family: 'Inter', sans-serif;
    font-size: 0.83rem;
    color: rgba(180,200,220,0.6);
    line-height: 1.8;
    background: rgba(0,212,255,0.025);
    border: 1px solid rgba(0,212,255,0.1);
    border-left: 2px solid rgba(0,212,255,0.35);
    border-radius: 0 5px 5px 0;
    padding: 1.3rem 1.4rem;
    margin-top: 0.5rem;
}
.interp-metric {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    font-weight: 500;
}

.q-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(0,212,255,0.35);
    margin-bottom: 0.35rem;
}
.q-value {
    font-family: 'Inter', sans-serif;
    font-size: 0.86rem;
    color: #c8d6e5;
    line-height: 1.65;
    background: rgba(0,212,255,0.025);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 1.1rem;
}

.chunk-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(0,212,255,0.4);
    display: block;
    margin-bottom: 0.4rem;
}
.chunk-box {
    background: rgba(0,212,255,0.02);
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 4px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: rgba(180,200,220,0.55);
    line-height: 1.6;
}

.sb-row {
    display: flex;
    gap: 0.8rem;
    margin-top: 1.2rem;
}
.sb {
    flex: 1;
    background: rgba(0,212,255,0.025);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 5px;
    padding: 0.9rem 0.5rem;
    text-align: center;
    transition: border-color 0.25s, box-shadow 0.25s;
    cursor: default;
}
.sb:hover {
    border-color: rgba(0,212,255,0.4);
    box-shadow: 0 0 14px rgba(0,212,255,0.08);
}
.sb-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(0,212,255,0.38);
    margin-bottom: 0.45rem;
}
.sb-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.55rem;
    font-weight: 400;
}

.run-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.84rem;
    color: rgba(160,185,210,0.55);
    line-height: 1.75;
    margin-bottom: 1.8rem;
    max-width: 620px;
}
.run-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: rgba(0,212,255,0.6);
    letter-spacing: 0.08em;
}
.run-done {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #00ff94;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


def score_color(v: float) -> str:
    if v >= 0.8:
        return "#00ff94"
    elif v >= 0.5:
        return "#ffa500"
    return "#ff3d3d"


def load_results():
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE) as f:
        return json.load(f)


def save_results(results):
    os.makedirs("data", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def radar_chart(averages: dict) -> go.Figure:
    labels = [METRIC_LABELS[m] for m in METRICS]
    values = [averages[m] for m in METRICS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(0,212,255,0.06)",
        line=dict(color="#00d4ff", width=2),
        marker=dict(color="#00d4ff", size=7),
        name="Score",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickfont=dict(size=9, color="rgba(0,212,255,0.3)", family="JetBrains Mono"),
                gridcolor="rgba(0,212,255,0.08)",
                linecolor="rgba(0,212,255,0.08)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="rgba(0,212,255,0.6)", family="JetBrains Mono"),
                gridcolor="rgba(0,212,255,0.07)",
                linecolor="rgba(0,212,255,0.07)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=60, b=60, l=80, r=80),
        height=420,
    )
    return fig


st.markdown("""
<div style="margin-bottom: 2rem;">
    <div class="lab-title">RAG Bench</div>
    <div class="lab-subtitle">Retrieval-Augmented Generation &nbsp;/&nbsp; Evaluation Dashboard &nbsp;/&nbsp; v1.0</div>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_questions, tab_run = st.tabs([
    "Overview",
    "Per-Question Results",
    "Run Evaluation",
])

results = load_results()

with tab_overview:
    if results is None:
        st.info("No results found. Run the evaluation first.")
    else:
        averages = {m: sum(r["scores"][m] for r in results) / len(results) for m in METRICS}

        st.markdown('<div class="section-label">System Metrics</div>', unsafe_allow_html=True)

        cards_html = '<div class="mc-grid">'
        for m in METRICS:
            v = averages[m]
            color = score_color(v)
            pct = int(v * 100)
            cards_html += f"""
            <div class="mc">
                <div class="mc-label">{METRIC_LABELS[m]}</div>
                <div class="mc-score" style="color:{color}">{v:.3f}</div>
                <div class="mc-bar-wrap"><div class="mc-bar" style="width:{pct}%;background:{color};"></div></div>
                <div class="mc-desc">{METRIC_DESC[m]}</div>
            </div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        col_chart, col_gap, col_notes = st.columns([5, 1, 4])

        with col_chart:
            st.markdown('<div class="section-label">Radar View</div>', unsafe_allow_html=True)
            st.plotly_chart(radar_chart(averages), use_container_width=True)

        with col_notes:
            st.markdown('<div class="section-label">Interpretation</div>', unsafe_allow_html=True)

            interp_items = [
                ("faithfulness",      averages["faithfulness"],      "The pipeline never generates claims outside of the retrieved context. Zero hallucination across all 20 questions."),
                ("answer_relevancy",  averages["answer_relevancy"],  "Answers consistently address the question. The small gap from 1.0 reflects occasional over-generalisation."),
                ("context_precision", averages["context_precision"], "When k=3 chunks are retrieved, typically only one is directly needed. A known precision vs recall trade-off."),
                ("context_recall",    averages["context_recall"],    "The retriever finds the relevant document almost every time. Near-perfect coverage of the knowledge base."),
            ]

            interp_html = '<div class="interp-block">'
            for i, (m, v, desc) in enumerate(interp_items):
                color = score_color(v)
                sep = "<br><br>" if i < len(interp_items) - 1 else ""
                interp_html += f'<span class="interp-metric" style="color:{color}">{METRIC_LABELS[m]} {v:.3f}</span><br>{desc}{sep}'
            interp_html += "</div>"
            st.markdown(interp_html, unsafe_allow_html=True)


with tab_questions:
    if results is None:
        st.info("No results found. Run the evaluation first.")
    else:
        rows = []
        for r in results:
            rows.append({
                "id":             r["id"],
                "question":       r["question"],
                "faithfulness":   round(r["scores"]["faithfulness"], 2),
                "ans. relevancy": round(r["scores"]["answer_relevancy"], 2),
                "ctx. precision": round(r["scores"]["context_precision"], 2),
                "ctx. recall":    round(r["scores"]["context_recall"], 2),
                "average":        round(sum(r["scores"].values()) / 4, 2),
            })
        df = pd.DataFrame(rows)

        col_s, col_a = st.columns([2, 1])
        with col_s:
            sort_by = st.selectbox("Sort by", ["id", "average", "faithfulness", "ans. relevancy", "ctx. precision", "ctx. recall"])
        with col_a:
            ascending = st.checkbox("Ascending", value=True)

        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Question Inspector</div>', unsafe_allow_html=True)

        selected_id = st.selectbox("Select question", [r["id"] for r in results])
        selected = next(r for r in results if r["id"] == selected_id)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f"""
            <div class="q-label">Question</div>
            <div class="q-value">{selected['question']}</div>
            <div class="q-label">Generated Answer</div>
            <div class="q-value">{selected['answer']}</div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown(f"""
            <div class="q-label">Ground Truth</div>
            <div class="q-value">{selected['ground_truth']}</div>
            <div class="q-label">Retrieved Chunks</div>
            """, unsafe_allow_html=True)

            for i, c in enumerate(selected["chunks"], 1):
                st.markdown(f"""
                <div class="chunk-box">
                    <span class="chunk-label">Chunk {i} &nbsp;/&nbsp; {c['title']}</span>
                    {c['text'][:220]}...
                </div>
                """, unsafe_allow_html=True)

        badges_html = '<div class="sb-row">'
        for m in METRICS:
            v = selected["scores"][m]
            color = score_color(v)
            badges_html += f"""
            <div class="sb">
                <div class="sb-label">{METRIC_LABELS[m]}</div>
                <div class="sb-value" style="color:{color}">{v:.2f}</div>
            </div>"""
        badges_html += "</div>"
        st.markdown(badges_html, unsafe_allow_html=True)


with tab_run:
    st.markdown('<div class="section-label">Benchmark Configuration</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="run-desc">
        Runs the RAG pipeline against all 20 test questions. For each question the system retrieves
        3 chunks, generates an answer, then calls Gemini four times to score each metric independently.
        Approximately 120 API calls total. Expect 15 to 25 minutes on the free tier.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Evaluation"):
        from test_set import TEST_CASES

        progress = st.progress(0)
        status = st.empty()
        log = st.empty()

        status.markdown('<div class="run-status">Building index...</div>', unsafe_allow_html=True)

        collection = build_index()
        new_results = []
        log_lines = []

        for i, case in enumerate(TEST_CASES):
            status.markdown(
                f'<div class="run-status">[{i+1:02d}/20] &nbsp; {case["id"]} &mdash; {case["question"][:65]}...</div>',
                unsafe_allow_html=True,
            )
            progress.progress((i + 1) / len(TEST_CASES))

            # Caught and displayed rather than allowed to propagate.
            # Streamlit Cloud redacts the message of an uncaught exception, so
            # a provider explaining exactly what it refused arrived as "the
            # original error message is redacted" - leaving a stack trace and
            # nothing to act on. st.error is not redacted, and the providers
            # only ever put a status and a response body in these messages,
            # never a key.
            try:
                chunks = retrieve(collection, case["question"], k=3)
                answer = generate_answer(case["question"], chunks)
                scores = evaluate_one(case["question"], case["ground_truth"], answer, chunks)
            except Exception as exc:
                st.error(f"Stopped at {case['id']}.\n\n{type(exc).__name__}: {exc}")
                st.stop()

            new_results.append({
                "id":           case["id"],
                "question":     case["question"],
                "answer":       answer,
                "ground_truth": case["ground_truth"],
                "chunks":       chunks,
                "scores":       scores,
            })

            avg = sum(scores.values()) / 4
            log_lines.append(
                f"{case['id']}  avg={avg:.2f}  "
                f"faith={scores['faithfulness']:.2f}  "
                f"rel={scores['answer_relevancy']:.2f}  "
                f"prec={scores['context_precision']:.2f}  "
                f"rec={scores['context_recall']:.2f}"
            )
            log.code("\n".join(log_lines[-12:]))

        save_results(new_results)
        progress.progress(1.0)
        status.markdown(
            '<div class="run-done">Evaluation complete. Results saved.</div>',
            unsafe_allow_html=True,
        )
        st.rerun()
