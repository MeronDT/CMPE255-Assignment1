# CRISP-DM Phase 1: Business Understanding

## Project

**Data Science Skills Mastery Lab** — installs and demonstrates two vetted
third-party Claude Code skill libraries (46 skills total; see
[00_skill_vetting.md](./00_skill_vetting.md)) against real, popular datasets, mapped
explicitly onto CRISP-DM phases.

## Business Objective

A data science team adopting AI-agent skill libraries needs confidence that (a) the
skills are safe to install, (b) they actually work as documented, and (c) their
output is reviewable, not an opaque JSON blob. This lab demonstrates all three.

## Scope

All 46 installed skills are cataloged (name, description, source, CRISP-DM phase
mapping) on the dashboard's Catalog tab — genuinely browsable, not a placeholder list.
A representative cross-section of 13 skills, spanning both source repos and every
CRISP-DM phase, is **actually executed** against real data (Titanic, plus reused
datasets from Projects 03/06/`12_time_series_forecasting_engine` — the same efficient-reuse pattern
used throughout this repo), with results rendered as an interactive dashboard rather
than raw JSON — directly addressing the follow-up request ("execute skill live...
make that visual interactive").

Demonstrating all 46 skills with full live execution was scoped down to this
representative 13 deliberately, stated here rather than silently: many of the
remaining 34 skills are communication/documentation-workflow skills (e.g.
`executive-summary-generator`, `stakeholder-requirements-gathering`) whose "output"
is prose guidance for a human conversation, not a numeric result a dashboard chart
can meaningfully render — cataloging them accurately (what they do, when to use them)
is the honest way to represent them, not forcing a fake chart.

## CRISP-DM Phase Mapping (of the 12 demonstrated skills)

| Phase | Skills Demonstrated |
|---|---|
| Data Understanding | `programmatic-eda`, `exploratory-data-analysis` |
| Data Preparation | `data-cleaning`, `feature-engineering` |
| Modeling | `sklearn-pipelines`, `hyperparameter-tuning` |
| Evaluation | `model-evaluation`, `imbalanced-data` |
| Business Analysis | `segmentation-analysis`, `cohort-analysis`, `business-metrics-calculator`, `ab-test-analysis` |
| Temporal Analysis | `time-series-analysis` |
