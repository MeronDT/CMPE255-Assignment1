# Data Science Skills Mastery Lab

Installs and demonstrates 46 skills from two vetted third-party GitHub skill
libraries — `param087/agent-ml-skills` and `nimrodfisher/data-analytics-skills` —
mapped to CRISP-DM phases, with 13 executed live against real data.

## Result

**46 skills installed and cataloged** (real descriptions parsed from their own
`SKILL.md` frontmatter, browsable/searchable on the dashboard). **13 executed live**
with genuine computation and custom interactive visualizations — not the raw JSON
dump the follow-up request flagged as unfriendly. Datasets: a fresh Titanic
download plus reused processed data from Projects 03/06/`12_time_series_forecasting_engine`.

Full safety vetting (install-script inspection, license check, cross-reference
against the professor's own skill-name index) in
[docs/00_skill_vetting.md](./docs/00_skill_vetting.md).

## Running it

```bash
cd scripts
python 01_build_catalog.py && python 02_execute_skills.py

cd ../backend && python -m uvicorn app.main:app --port 8005
cd ../frontend && npm install && npm run dev -- --port 5185
```

Full design decisions, including a real CORS/NaN bug caught and fixed during this
build, in [DESIGN_DOC.md](./DESIGN_DOC.md).
