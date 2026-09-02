# Scope: Enterprise Data Science Audit Platform

## What this is

**Project 11**, built last by design — it audits Projects 00–10 and `12_time_series_forecasting_engine`
(12 total; Projects 05 and 09 were added in a later update once they were completed,
having been deferred until the user was present to review the third-party skill
installs) so it cannot exist before them. Not a
CRISP-DM data-mining project itself; a governance/QA tool applied to this repo's own
prior work, following the same "detailed report on a website" instruction as every
other project's admin dashboard requirement.

## Audit Methodology

The audit script (`scripts/01_audit.py`) inspects each project directory
**programmatically** — checking for actual file existence, actual JSON result
contents, actual `.gitignore` correctness — rather than relying on memory of having
built each project. Every score is backed by a concrete, machine-checked signal listed
alongside it, not a subjective impression.

### Dimensions Scored (0–5 each)

1. **CRISP-DM Completeness** — presence of a business-understanding doc, EDA/prep
   scripts, modeling scripts, an evaluation/results artifact, and a deployed
   backend+frontend.
2. **Documentation Quality** — presence and non-triviality (byte size floor, not just
   existence) of `DESIGN_DOC.md`, `README.md`, and (where applicable)
   `RESEARCH_REPORT.md`.
3. **Verification Evidence** — presence of `docs/screenshots/` with actual PNG files,
   the concrete proof each dashboard was checked, not just built.
4. **Honest-Reporting Discipline** — a programmatic scan of each project's
   `RESEARCH_REPORT.md`/`README.md` for language patterns indicating a genuine
   finding, limitation, or correction was surfaced (e.g., "honest", "reported
   honestly", "limitation", a "caught and fixed" bug narrative) — this repo's
   consistent, deliberate practice across every project, checked for its actual
   presence rather than assumed.
5. **Repository Hygiene** — `.gitignore` correctness (large raw datasets and
   `node_modules`/`dist` actually excluded, checked via `git check-ignore`, not
   assumed), and no anomalously large tracked files.

## Constraints

- This audit was run **after** the projects it covers were already committed and
  pushed — it cannot retroactively fix anything it finds wrong without a further,
  explicit follow-up action; issues found are reported, not silently patched.
- Scores are relative to this repo's own established conventions (documented across
  every audited project's own DESIGN_DOCs), not an external industry rubric — stated
  explicitly so a reader doesn't mistake a 5/5 here for an absolute claim of
  perfection.
