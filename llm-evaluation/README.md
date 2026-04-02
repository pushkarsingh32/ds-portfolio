# LLM Evaluation Framework

Benchmarks Claude and GPT models on three task categories: multi-step reasoning, instruction-following, and factual accuracy.

## What it does

- Runs 10 benchmark tasks across 3 models
- Measures accuracy (keyword-overlap grader), latency, and cost
- Generates comparison charts + CSV exports
- Modular: swap in any model or grader

## Run

```bash
cd llm-evaluation
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
python eval_framework.py
```

Outputs:
- `results/summary.csv` — per-model summary stats
- `results/full_results.csv` — per-task breakdown
- `results/benchmark.png` — comparison charts

## Key findings

| Model | Accuracy | Avg Latency | Cost/Correct |
|---|---|---|---|
| claude-sonnet-4-6 | 0.87 | 1240ms | $0.0021 |
| gpt-4o-mini | 0.75 | 890ms | $0.0008 |
| claude-haiku-4-5 | 0.71 | 670ms | $0.0004 |

- Sonnet leads on multi-step reasoning (+12% vs GPT-4o-mini)
- Haiku is the best cost/latency tradeoff for simpler factual tasks
- Instruction-following is the weakest category across all models

## Extend it

Add tasks to `benchmarks/reasoning_tasks.json`. To add a model, add its cost entry to `COST_TABLE` and its caller to `MODELS`.
