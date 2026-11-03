# LLM-EVALS
# RAG Reliability Lab — No API / No Model Download Edition

A beginner-friendly RAG evaluation workbench. It does **not** download an LLM, use Ollama, or require an API key.

## What makes this useful?
You can build a small document retrieval pipeline, paste answers produced by any LLM, and inspect transparent evaluation indicators:
- Groundedness: how much answer vocabulary appears in retrieved evidence.
- Answer relevance: overlap between answer vocabulary and question vocabulary (a rough signal).
- Retrieval precision: share of retrieved chunks that overlap with query terms.
- Reference coverage: overlap with an optional reference answer.
- Unsupported-word risk: inverse of the lexical groundedness estimate.
- Batch evaluation from CSV, charts, and downloadable reports.

## Important honesty about metrics
This is a lightweight educational evaluation project. The metrics are deterministic lexical indicators, **not** RAGAS scores, semantic correctness, or proof that an answer is true. They can penalize valid paraphrases and reward copied words. Use them to screen and inspect examples, not to make production safety claims. A future upgrade can add RAGAS/DeepEval or an LLM judge.

## Install and run
Requires Python 3.10+.

```bash
python -m venv venv
```
Windows:
```powershell
venv\Scripts\activate
```
macOS/Linux:
```bash
source venv/bin/activate
```
Then:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local address printed in the terminal (usually http://localhost:8501).

## First run
1. Leave “Include sample company handbook” selected.
2. Click **Build knowledge base**.
3. Paste a response into the answer box. Example: “Employees receive 20 days of paid annual leave per calendar year.”
4. Click **Retrieve evidence and evaluate**.
5. Inspect the evidence chunks and metrics.
6. Try an unsupported question such as “What is the office address?” and inspect what retrieval returns.
7. Use the sample CSV in Batch evaluation to run multiple test cases.

## CSV format
Required columns: `question`, `answer`
Optional: `reference_answer`

## Stack
Python, Streamlit, TF-IDF retrieval, pandas, pypdf.

## Next engineering upgrades
- Create a human-labelled golden dataset with expected source chunk IDs.
- Add semantic embeddings for retrieval.
- Integrate RAGAS/DeepEval and compare their scores against lexical indicators.
- Add regression thresholds and CI checks.
- Add adversarial prompt-injection and data-leakage test cases.
