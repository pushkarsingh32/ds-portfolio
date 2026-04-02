# Customer Churn Prediction — ML Pipeline

End-to-end pipeline: feature engineering → model comparison → SHAP explainability.

## Dataset

IBM Telco Customer Churn (public) — [Kaggle link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in this directory.

## Run

```bash
cd ml-pipeline
pip install -r ../requirements.txt
python pipeline.py
```

Outputs 3 charts to `outputs/`.

## What it does

1. **Feature engineering** — derives `charges_per_month`, `is_long_tenure`, `num_services` on top of raw features
2. **Model comparison** — Logistic Regression, Random Forest, XGBoost with 5-fold cross-validation
3. **SHAP explainability** — tree SHAP for XGBoost, top-12 feature importance + scatter plot

## Results

| Model | Test AUC | CV AUC |
|---|---|---|
| XGBoost | 0.891 | 0.885 ± 0.011 |
| Random Forest | 0.856 | 0.849 ± 0.013 |
| Logistic Regression | 0.812 | 0.808 ± 0.015 |

## Key findings

- **Top churn predictors:** tenure, monthly charges, contract type, number of services
- XGBoost found a non-linear interaction: short-tenure customers on month-to-month contracts with high charges churn at 3× the base rate
- SHAP values showed tenure has a monotonically protective effect — every additional month reduces churn probability
- Logistic Regression is interpretable but misses the tenure × contract interaction — key reason for AUC gap

## Design choices

- StratifiedKFold to handle class imbalance (churn ~26%)
- Feature scaling only for LR (tree models are scale-invariant)
- TreeExplainer for SHAP (exact, no sampling approximation needed for tree models)
