"""
Customer Churn Prediction — End-to-End ML Pipeline
Uses the IBM Telco Customer Churn dataset (public domain).

Download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Place WA_Fn-UseC_-Telco-Customer-Churn.csv in this directory.

Usage:
    python pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")
os.makedirs("outputs", exist_ok=True)
RANDOM_STATE = 42


# ----- Data loading & feature engineering -----

def load_data(path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Target
    df["churn"] = (df["churn"].str.strip().str.lower() == "yes").astype(int)

    # Fix TotalCharges (sometimes loaded as string)
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["totalcharges"].fillna(df["monthlycharges"] * df["tenure"], inplace=True)

    # Drop customerid — not predictive
    df.drop(columns=["customerid"], errors="ignore", inplace=True)

    print(f"Loaded {len(df)} customers. Churn rate: {df['churn'].mean():.1%}")
    return df, df.pop("churn")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derived features
    df["charges_per_month"] = df["totalcharges"] / (df["tenure"] + 1)
    df["is_long_tenure"] = (df["tenure"] > 24).astype(int)
    df["num_services"] = (
        df.get("phoneservice", pd.Series(["No"] * len(df))).eq("Yes").astype(int)
        + df.get("internetservice", pd.Series(["No"] * len(df))).ne("No").astype(int)
        + df.get("onlinesecurity", pd.Series(["No"] * len(df))).eq("Yes").astype(int)
        + df.get("onlinebackup", pd.Series(["No"] * len(df))).eq("Yes").astype(int)
        + df.get("techsupport", pd.Series(["No"] * len(df))).eq("Yes").astype(int)
        + df.get("streamingtv", pd.Series(["No"] * len(df))).eq("Yes").astype(int)
        + df.get("streamingmovies", pd.Series(["No"] * len(df))).eq("Yes").astype(int)
    )

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


# ----- Models -----

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "XGBoost":             XGBClassifier(
                               n_estimators=200, learning_rate=0.05, max_depth=5,
                               use_label_encoder=False, eval_metric="logloss",
                               random_state=RANDOM_STATE, verbosity=0,
                           ),
}


def evaluate_models(X_train, X_test, y_train, y_test) -> dict:
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5))

    for ax, (name, model) in zip(axes, MODELS.items()):
        X_tr = X_train_sc if name == "Logistic Regression" else X_train
        X_te = X_test_sc  if name == "Logistic Regression" else X_test

        model.fit(X_tr, y_train)
        proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_test, proba)

        cv_scores = cross_val_score(model, X_tr, y_train, cv=StratifiedKFold(5),
                                    scoring="roc_auc", n_jobs=-1)

        results[name] = {
            "model": model,
            "scaler": scaler if name == "Logistic Regression" else None,
            "auc": auc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "proba": proba,
        }
        print(f"{name:22s}  AUC={auc:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

        RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=name)
        ax.set_title(f"{name}\nAUC={auc:.3f}", fontsize=10)

    plt.suptitle("ROC Curves — Model Comparison", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/01_roc_curves.png", dpi=150)
    plt.close()
    print("Saved outputs/01_roc_curves.png")
    return results


def plot_confusion_matrix(results: dict, X_test, y_test):
    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]
    X_te = best["scaler"].transform(X_test) if best["scaler"] else X_test
    y_pred = best["model"].predict(X_te)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("outputs/02_confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\nClassification Report ({best_name}):")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))


def shap_analysis(results: dict, X_train, X_test, feature_names):
    # Use XGBoost (tree SHAP = fast + exact)
    if "XGBoost" not in results:
        return
    model = results["XGBoost"]["model"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot — mean |SHAP|
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[-12:]
    axes[0].barh(
        [feature_names[i] for i in top_idx],
        mean_shap[top_idx],
        color="#4C9BE8",
    )
    axes[0].set_title("Top 12 Features by Mean |SHAP|", fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP value|")

    # Beeswarm-style scatter for top feature
    top_feat = feature_names[top_idx[-1]]
    top_i = top_idx[-1]
    scatter = axes[1].scatter(
        shap_values[:, top_i],
        X_test.iloc[:, top_i] if hasattr(X_test, "iloc") else X_test[:, top_i],
        c=shap_values[:, top_i], cmap="coolwarm", alpha=0.4, s=10,
    )
    axes[1].set_xlabel("SHAP value")
    axes[1].set_ylabel(f"Feature value: {top_feat}")
    axes[1].set_title(f"SHAP vs Feature Value\n({top_feat})", fontweight="bold")
    plt.colorbar(scatter, ax=axes[1], label="SHAP value")

    plt.suptitle("SHAP Explainability — XGBoost", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/03_shap.png", dpi=150)
    plt.close()
    print("Saved outputs/03_shap.png")


def summarise_results(results: dict):
    print("\n" + "=" * 50)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 50)
    rows = [
        {"Model": k, "Test AUC": v["auc"], "CV AUC (mean)": v["cv_mean"], "CV AUC (std)": v["cv_std"]}
        for k, v in results.items()
    ]
    df = pd.DataFrame(rows).sort_values("Test AUC", ascending=False)
    print(df.to_string(index=False))


def main():
    df, y = load_data()
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("\nTraining and evaluating models...")
    results = evaluate_models(X_train, X_test, y_train, y_test)
    plot_confusion_matrix(results, X_test, y_test)
    shap_analysis(results, X_train, X_test, list(df.columns))
    summarise_results(results)
    print("\nAll outputs saved to outputs/")


if __name__ == "__main__":
    main()
