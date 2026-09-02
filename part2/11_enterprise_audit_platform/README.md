# Enterprise Data Science Audit Platform

An evidence-backed audit of every completed project in this repository (Projects
00–10, `12_time_series_forecasting_engine`) — every score derived from a real filesystem/git check, not
a subjective read of each project. Updated after Projects 05 and 09 (initially
deferred, completed later) were added.

## Result

**12 projects audited, average overall score 4.8/5, zero flagged repo-hygiene or
verification-evidence issues.** Real issues were caught and fixed while building
and later updating the audit tool itself — type-blind CRISP-DM scoring, a narrow
doc-file scan, and an honest-language marker list too narrow to catch two newly
added projects' genuine disclosures — all disclosed in `DESIGN_DOC.md §3-4` rather
than hidden behind the final clean scorecard. One apparent bug (a stale-looking
score in a screenshot) turned out, on stronger verification, not to be a bug at
all — also disclosed rather than written up as a false "catch," in `DESIGN_DOC.md
§4`.

## Running it

```bash
cd scripts
python 01_audit.py

cd ../backend && python -m uvicorn app.main:app --port 8011
cd ../frontend && npm install && npm run dev -- --port 5184
```

Full design decisions, including the audit methodology's own known limitations, in
[DESIGN_DOC.md](./DESIGN_DOC.md).
