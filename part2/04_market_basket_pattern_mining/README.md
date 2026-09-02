# Market Basket Pattern Mining

Association rule mining on the UCI/Kaggle Online Retail dataset (15,122 multi-item
baskets after cleaning) via Apriori/FP-Growth, following the full CRISP-DM lifecycle
with an AutoResearch hill-climbing search and a data-scientist-facing admin dashboard.

## Result

**27 business-actionable rules** (support ≥ 3%, confidence ≥ 60%, avg lift 9.31x, max
lift 14.13x) — chosen by AutoResearch specifically to stay in a merchandiser-reviewable
range (5–50 rules), after an unconstrained first pass returned 7,799 rules and had to be
corrected (see `RESEARCH_REPORT.md §4`).

Example rules, all matching real product-collection logic:
- `PINK REGENCY TEACUP AND SAUCER` → `GREEN REGENCY TEACUP AND SAUCER` (lift 12.5x)
- `GARDENERS KNEELING PAD CUP OF TEA` → `GARDENERS KNEELING PAD KEEP CALM` (lift 12.1x)
- `ALARM CLOCK BAKELIKE RED` → `ALARM CLOCK BAKELIKE GREEN` (lift 9.5x)

## Running it

```bash
cd scripts
python 01_eda.py && python 02_prepare_data.py && python 03_train_models.py
python 04_autoresearch.py && python 05_evaluate.py

cd ../backend && python -m uvicorn app.main:app --port 8004
cd ../frontend && npm install && npm run dev -- --port 5177
```

Full design decisions in [DESIGN_DOC.md](./DESIGN_DOC.md); the complete research
writeup is in [RESEARCH_REPORT.md](./RESEARCH_REPORT.md).
