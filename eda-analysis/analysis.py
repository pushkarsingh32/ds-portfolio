"""
Tech Layoffs EDA (2022–2025)
Dataset: https://www.kaggle.com/datasets/swaptr/layoffs-2022
Download layoffs.csv from Kaggle and place in this directory.

Usage:
    python analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import os

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("outputs", exist_ok=True)


def load_and_clean(path: str = "layoffs.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Numeric coercion
    for col in ["total_laid_off", "percentage_laid_off", "funds_raised_millions"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where we can't measure anything
    df = df.dropna(subset=["total_laid_off", "percentage_laid_off"], how="all")

    df["year_month"] = df["date"].dt.to_period("M")
    df["year"] = df["date"].dt.year

    print(f"Loaded {len(df)} records from {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def time_series_analysis(df: pd.DataFrame):
    monthly = (
        df.groupby("year_month")["total_laid_off"]
        .sum()
        .reset_index()
    )
    monthly["year_month_dt"] = monthly["year_month"].dt.to_timestamp()
    monthly["rolling_3m"] = monthly["total_laid_off"].rolling(3).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(monthly["year_month_dt"], monthly["total_laid_off"], width=20, alpha=0.5, label="Monthly")
    ax.plot(monthly["year_month_dt"], monthly["rolling_3m"], color="crimson", lw=2, label="3-month rolling avg")
    ax.set_title("Tech Layoffs Over Time (2022–2025)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Laid Off")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/01_time_series.png", dpi=150)
    plt.close()
    print("Saved outputs/01_time_series.png")
    return monthly


def industry_breakdown(df: pd.DataFrame):
    if "industry" not in df.columns:
        return

    industry = (
        df.groupby("industry")["total_laid_off"]
        .sum()
        .dropna()
        .sort_values(ascending=False)
        .head(12)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    industry.plot(kind="barh", ax=ax, color=sns.color_palette("muted", len(industry)))
    ax.set_title("Total Layoffs by Industry (Top 12)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Laid Off")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/02_industry.png", dpi=150)
    plt.close()
    print("Saved outputs/02_industry.png")


def funding_stage_analysis(df: pd.DataFrame):
    stage_col = next((c for c in df.columns if "stage" in c), None)
    if not stage_col:
        return

    stage = (
        df.groupby(stage_col)
        .agg(
            total_laid_off=("total_laid_off", "sum"),
            avg_pct=("percentage_laid_off", "mean"),
            count=("total_laid_off", "count"),
        )
        .dropna()
        .sort_values("total_laid_off", ascending=False)
        .head(10)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    stage["total_laid_off"].plot(kind="bar", ax=axes[0], color="#4C9BE8")
    axes[0].set_title("Total Layoffs by Funding Stage", fontweight="bold")
    axes[0].set_ylabel("Total Laid Off")
    axes[0].set_xticklabels(stage.index, rotation=30, ha="right")

    stage["avg_pct"].plot(kind="bar", ax=axes[1], color="#E8844C")
    axes[1].set_title("Avg % Laid Off by Funding Stage", fontweight="bold")
    axes[1].set_ylabel("Average %")
    axes[1].set_xticklabels(stage.index, rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig("outputs/03_funding_stage.png", dpi=150)
    plt.close()
    print("Saved outputs/03_funding_stage.png")


def correlation_analysis(df: pd.DataFrame):
    num_cols = ["total_laid_off", "percentage_laid_off", "funds_raised_millions"]
    num_cols = [c for c in num_cols if c in df.columns]
    sub = df[num_cols].dropna()

    if sub.empty or len(num_cols) < 2:
        return

    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
    ax.set_title("Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/04_correlation.png", dpi=150)
    plt.close()
    print("Saved outputs/04_correlation.png")

    # Pearson test: funds raised vs layoffs
    r, p = stats.pearsonr(sub["funds_raised_millions"], sub["total_laid_off"])
    print(f"\nPearson r (funds_raised vs layoffs): {r:.3f}, p={p:.4f}")
    if p < 0.05:
        print("→ Statistically significant correlation.")
    else:
        print("→ Not statistically significant.")


def print_key_stats(df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("KEY STATISTICS")
    print("=" * 50)
    print(f"Total layoff events:     {len(df):,}")
    print(f"Total people laid off:   {df['total_laid_off'].sum():,.0f}")
    print(f"Median layoff size:      {df['total_laid_off'].median():,.0f}")

    if "company" in df.columns:
        top = df.groupby("company")["total_laid_off"].sum().nlargest(5)
        print("\nTop 5 companies by total layoffs:")
        for co, n in top.items():
            print(f"  {co}: {n:,.0f}")


def main():
    df = load_and_clean()
    print_key_stats(df)
    time_series_analysis(df)
    industry_breakdown(df)
    funding_stage_analysis(df)
    correlation_analysis(df)
    print("\nAll outputs saved to outputs/")


if __name__ == "__main__":
    main()
