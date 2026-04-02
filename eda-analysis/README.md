# Tech Layoffs EDA (2022–2025)

Exploratory analysis of 3,000+ tech layoff events to identify patterns by industry, funding stage, and time.

## Dataset

Download from Kaggle: [Tech Layoffs 2022–2025](https://www.kaggle.com/datasets/swaptr/layoffs-2022)  
Place `layoffs.csv` in this directory.

## Run

```bash
cd eda-analysis
pip install -r ../requirements.txt
python analysis.py
```

Outputs 4 charts to `outputs/`.

## Key Findings

**Time series:**
- Layoffs peaked in Jan 2023 (post-pandemic over-hiring correction)
- Second spike in Jan 2024 after Fed held rates higher for longer
- 3-month rolling average reveals two distinct "waves"

**By industry:**
- Consumer tech and retail tech had the highest absolute numbers
- Fintech had the highest layoffs as % of total headcount

**By funding stage:**
- Series B companies showed the highest average % laid off
- Post-IPO companies had the largest absolute numbers (larger headcounts)
- Seed-stage layoffs were rare but near-total when they happened (100% = shutdown)

**Correlation:**
- Funds raised is weakly positively correlated with total laid off (larger companies raise more and have more to cut)
- Statistically significant but low practical effect size (r ≈ 0.3)

## Interesting observation

Layoffs among AI/ML-specific companies were disproportionately low throughout the dataset, even during the 2023 peak — consistent with the AI investment boom acting as a buffer.
