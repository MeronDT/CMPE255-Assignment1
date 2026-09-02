# Time Series Forecasting Engine

Daily revenue forecasting for the Online Retail store (reused from Projects 03/04/10)
via Seasonal Naive, Holt-Winters, SARIMA, and Prophet, following the full CRISP-DM
lifecycle with an AutoResearch hyperparameter search and a data-scientist-facing admin
dashboard.

## Result

**Tuned Prophet wins at 19.58% MAPE** — but the honest story is in getting there:
default Prophet (25.78% MAPE) actually underperformed the naive seasonal baseline
(26.23%). AutoResearch's hill-climb found that switching to **multiplicative**
seasonality (matching this retailer's seasonal swings scaling with revenue level) was
the decisive fix, not a marginal tweak. See `RESEARCH_REPORT.md §4` for the full story.

| Model | MAPE |
|---|---:|
| Seasonal Naive | 26.23% |
| SARIMA | 31.99% |
| Prophet (default) | 25.78% |
| Holt-Winters | 22.86% |
| **Prophet (tuned)** | **19.58%** |

## Running it

```bash
cd scripts
python 01_eda_and_prepare.py && python 02_train_models.py
python 03_autoresearch.py && python 04_evaluate.py

cd ../backend && python -m uvicorn app.main:app --port 8012
cd ../frontend && npm install && npm run dev -- --port 5183
```

Full design decisions in [DESIGN_DOC.md](./DESIGN_DOC.md); the complete research
writeup is in [RESEARCH_REPORT.md](./RESEARCH_REPORT.md).
