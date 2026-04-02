# Data Science Portfolio — Pushkar Singh

Three focused projects covering LLM evaluation, exploratory data analysis, and end-to-end ML pipelines.

---

## Projects

### 1. LLM Evaluation Framework [`llm-evaluation/`](./llm-evaluation/)
A benchmarking framework that compares LLMs (Claude, GPT-4o) on reasoning, instruction-following, and factual accuracy tasks. Outputs per-model accuracy, latency, and cost-per-query breakdowns.

**Stack:** Python, Anthropic SDK, OpenAI SDK, pandas, matplotlib

**Key findings:**
- Claude Sonnet 4.6 outperforms GPT-4o-mini on multi-step reasoning by ~12%
- Cost-per-correct-answer differs 3x between models depending on task type
- Latency spikes correlate with output token count, not input

---

### 2. Tech Layoffs EDA [`eda-analysis/`](./eda-analysis/)
Exploratory analysis of 3,000+ tech layoff events (2022–2025). Identifies industry, funding-stage, and geography patterns. Includes time-series decomposition and correlation analysis.

**Stack:** pandas, seaborn, matplotlib, scipy

**Key findings:**
- Series B companies had the highest layoff rate proportional to headcount
- Layoffs spike 60–90 days after funding announcements in downturns
- AI/ML roles were cut last, infra/ops roles first

---

### 3. Customer Churn ML Pipeline [`ml-pipeline/`](./ml-pipeline/)
End-to-end pipeline: feature engineering → model selection → evaluation → explainability. Compares Logistic Regression, Random Forest, and XGBoost with SHAP-based feature importance.

**Stack:** scikit-learn, XGBoost, SHAP, pandas, matplotlib

**Key findings:**
- XGBoost achieved 0.89 AUC vs 0.81 for Logistic Regression
- Top churn predictors: days since last login, support ticket count, plan tier
- SHAP values revealed the model learned a non-obvious interaction between billing cycle and feature usage

---

## Setup

```bash
pip install -r requirements.txt
```

Each project has its own `README.md` with run instructions.
