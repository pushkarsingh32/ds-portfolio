"""
LLM Evaluation Framework
Benchmarks multiple models on reasoning, instruction-following, and factual accuracy.
Outputs accuracy, latency, and estimated cost per model.

Usage:
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    python eval_framework.py
"""

import json
import time
import os
from dataclasses import dataclass, field
from typing import Optional
import anthropic
import openai
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv

load_dotenv()

# ----- Cost table (per 1M tokens, USD) -----
COST_TABLE = {
    "claude-sonnet-4-6":      {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":       {"input": 0.80,  "output": 4.00},
    "gpt-4o":                 {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini":            {"input": 0.15,  "output": 0.60},
}


@dataclass
class EvalResult:
    model: str
    task_id: str
    category: str
    difficulty: str
    response: str
    expected: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    score: float = 0.0  # filled in after grading


@dataclass
class ModelStats:
    model: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else 0

    @property
    def avg_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results) / len(self.results) if self.results else 0

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def cost_per_correct(self) -> float:
        correct = sum(r.score for r in self.results)
        return self.total_cost_usd / correct if correct > 0 else float("inf")

    def by_category(self) -> dict:
        cats = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r.score)
        return {cat: sum(scores) / len(scores) for cat, scores in cats.items()}


# ----- Model clients -----

def call_claude(model: str, prompt: str) -> tuple[str, int, int, float]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.time()
    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.time() - t0) * 1000
    response = message.content[0].text
    input_tok = message.usage.input_tokens
    output_tok = message.usage.output_tokens
    cost = (
        input_tok * COST_TABLE[model]["input"] / 1_000_000
        + output_tok * COST_TABLE[model]["output"] / 1_000_000
    )
    return response, input_tok, output_tok, latency_ms, cost


def call_openai(model: str, prompt: str) -> tuple[str, int, int, float]:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    t0 = time.time()
    completion = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.time() - t0) * 1000
    response = completion.choices[0].message.content
    input_tok = completion.usage.prompt_tokens
    output_tok = completion.usage.completion_tokens
    cost = (
        input_tok * COST_TABLE[model]["input"] / 1_000_000
        + output_tok * COST_TABLE[model]["output"] / 1_000_000
    )
    return response, input_tok, output_tok, latency_ms, cost


# ----- Grading -----

def grade_response(response: str, expected: str, category: str) -> float:
    """
    Simple keyword-overlap grader. Returns 0.0–1.0.
    In production you'd use an LLM-as-judge here.
    """
    resp_lower = response.lower()
    key_terms = [t.strip().lower() for t in expected.split() if len(t) > 3]
    if not key_terms:
        return 0.0
    matched = sum(1 for t in key_terms if t in resp_lower)
    score = matched / len(key_terms)
    # Instruction-following: penalize if response is too long when task says "exactly"
    if category == "instruction_following" and "exactly" in expected.lower():
        word_count = len(response.split())
        if word_count > 50:
            score *= 0.7
    return round(min(score, 1.0), 3)


# ----- Runner -----

MODELS = {
    "claude-sonnet-4-6": call_claude,
    "claude-haiku-4-5":  call_claude,
    "gpt-4o-mini":       call_openai,
}


def run_benchmark(tasks: list[dict], models: dict = MODELS) -> dict[str, ModelStats]:
    stats = {m: ModelStats(model=m) for m in models}

    for task in tasks:
        print(f"\n  Task {task['id']} [{task['category']}] — {task['difficulty']}")
        for model_name, caller in models.items():
            try:
                response, in_tok, out_tok, latency, cost = caller(model_name, task["prompt"])
                score = grade_response(response, task["expected_answer"], task["category"])
                result = EvalResult(
                    model=model_name,
                    task_id=task["id"],
                    category=task["category"],
                    difficulty=task["difficulty"],
                    response=response,
                    expected=task["expected_answer"],
                    latency_ms=latency,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=cost,
                    score=score,
                )
                stats[model_name].results.append(result)
                print(f"    {model_name:30s} score={score:.2f}  latency={latency:.0f}ms  cost=${cost:.5f}")
            except Exception as e:
                print(f"    {model_name:30s} ERROR: {e}")

    return stats


# ----- Reporting -----

def build_summary_df(stats: dict[str, ModelStats]) -> pd.DataFrame:
    rows = []
    for model, s in stats.items():
        row = {
            "model": model,
            "accuracy": round(s.accuracy, 3),
            "avg_latency_ms": round(s.avg_latency_ms, 1),
            "total_cost_usd": round(s.total_cost_usd, 5),
            "cost_per_correct": round(s.cost_per_correct, 5),
        }
        for cat, acc in s.by_category().items():
            row[f"acc_{cat}"] = round(acc, 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def plot_results(df: pd.DataFrame, stats: dict[str, ModelStats], save_path: str = "results/benchmark.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    models = df["model"].tolist()
    colors = ["#4C9BE8", "#E8844C", "#4CE87A"][:len(models)]

    # 1. Overall accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(models, df["accuracy"], color=colors)
    ax1.set_title("Overall Accuracy", fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Score (0–1)")
    for bar, val in zip(bars, df["accuracy"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", fontsize=9)
    ax1.set_xticklabels(models, rotation=15, ha="right", fontsize=8)

    # 2. Avg latency
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(models, df["avg_latency_ms"], color=colors)
    ax2.set_title("Avg Latency (ms)", fontweight="bold")
    ax2.set_ylabel("Milliseconds")
    ax2.set_xticklabels(models, rotation=15, ha="right", fontsize=8)

    # 3. Cost per correct answer
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(models, df["cost_per_correct"] * 100, color=colors)  # in cents
    ax3.set_title("Cost per Correct Answer (¢)", fontweight="bold")
    ax3.set_ylabel("US Cents")
    ax3.set_xticklabels(models, rotation=15, ha="right", fontsize=8)

    # 4. Category breakdown (first model as example)
    ax4 = fig.add_subplot(gs[1, 1])
    cat_cols = [c for c in df.columns if c.startswith("acc_")]
    cat_labels = [c.replace("acc_", "").replace("_", "\n") for c in cat_cols]
    for i, (model, s) in enumerate(stats.items()):
        cat_accs = [df[df["model"] == model][c].values[0] for c in cat_cols]
        x = range(len(cat_labels))
        ax4.plot(x, cat_accs, marker="o", label=model, color=colors[i])
    ax4.set_xticks(range(len(cat_labels)))
    ax4.set_xticklabels(cat_labels, fontsize=7)
    ax4.set_title("Accuracy by Category", fontweight="bold")
    ax4.set_ylim(0, 1.1)
    ax4.legend(fontsize=7)

    plt.suptitle("LLM Benchmark Results", fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {save_path}")


def save_results(stats: dict[str, ModelStats], df: pd.DataFrame):
    os.makedirs("results", exist_ok=True)
    # Full results CSV
    rows = []
    for s in stats.values():
        for r in s.results:
            rows.append({
                "model": r.model, "task_id": r.task_id,
                "category": r.category, "difficulty": r.difficulty,
                "score": r.score, "latency_ms": r.latency_ms,
                "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
            })
    pd.DataFrame(rows).to_csv("results/full_results.csv", index=False)
    df.to_csv("results/summary.csv", index=False)
    print("\nResults saved to results/")


def main():
    with open("benchmarks/reasoning_tasks.json") as f:
        tasks = json.load(f)

    print(f"Running benchmark on {len(tasks)} tasks across {len(MODELS)} models...")
    stats = run_benchmark(tasks)

    df = build_summary_df(stats)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))

    plot_results(df, stats)
    save_results(stats, df)


if __name__ == "__main__":
    main()
