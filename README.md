# Data Science Portfolio — Pushkar Singh

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Notebooks](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Four projects covering LLM evaluation, exploratory data analysis, end-to-end ML pipelines, and semantic clustering with embeddings.

---

## Projects

### 1. LLM Benchmark Framework [`llm-evaluation/`](./llm-evaluation/)
Benchmarks Claude Sonnet, Claude Haiku, and GPT-4o-mini across reasoning, instruction-following, and factual accuracy tasks. Measures accuracy, latency, and cost per correct answer.

**Stack:** Anthropic SDK, OpenAI SDK, pandas, matplotlib  
**Notebook:** [`llm_benchmark.ipynb`](./llm-evaluation/llm_benchmark.ipynb)

| Model | Accuracy | Avg Latency | Cost/Correct |
|---|---|---|---|
| claude-sonnet-4-6 | **0.873** | 1241ms | $0.00044 |
| gpt-4o-mini | 0.751 | 887ms | $0.00002 |
| claude-haiku-4-5 | 0.714 | 668ms | $0.00010 |

---

### 2. Tech Layoffs EDA [`eda-analysis/`](./eda-analysis/)
Exploratory analysis of 3,500+ tech layoff events (2022–2025). Time-series decomposition, industry/funding-stage breakdown, and correlation analysis with statistical testing.

**Stack:** pandas, seaborn, matplotlib, scipy  
**Notebook:** [`tech_layoffs_eda.ipynb`](./eda-analysis/tech_layoffs_eda.ipynb)

Key findings: Jan 2023 peak at 89k layoffs · Series B worst % cut among growth-stage · Seed-stage = 71% avg headcount (shutdowns) · AI/ML companies underrepresented throughout

---

### 3. Customer Churn Prediction [`ml-pipeline/`](./ml-pipeline/)
End-to-end pipeline: feature engineering → Logistic Regression / Random Forest / XGBoost comparison → SHAP explainability. Identifies key churn drivers and a high-risk customer segment.

**Stack:** scikit-learn, XGBoost, SHAP, pandas  
**Notebook:** [`churn_prediction.ipynb`](./ml-pipeline/churn_prediction.ipynb)

| Model | Test AUC | CV AUC |
|---|---|---|
| XGBoost | **0.891** | 0.885 ± 0.011 |
| Random Forest | 0.856 | 0.849 ± 0.013 |
| Logistic Regression | 0.812 | 0.808 ± 0.015 |

---

### 4. Semantic Clustering with Embeddings [`embeddings-clustering/`](./embeddings-clustering/)
Embeds 3,000 documents with `sentence-transformers`, reduces to 10D with UMAP, clusters with HDBSCAN (no labels required). Evaluates against ground truth and includes a semantic search demo.

**Stack:** sentence-transformers, UMAP, HDBSCAN, sklearn  
**Notebook:** [`embeddings_clustering.ipynb`](./embeddings-clustering/embeddings_clustering.ipynb)

ARI: 0.683 · NMI: 0.741 · 6/7 clusters match ground-truth categories · Applications: RAG deduplication, intent detection, ticket routing

---

## Setup

```bash
git clone https://github.com/pushkarsingh32/ds-portfolio
cd ds-portfolio
pip install -r requirements.txt

# For embeddings project
pip install sentence-transformers umap-learn hdbscan
```

Each project folder has its own README with dataset download links and run instructions.
