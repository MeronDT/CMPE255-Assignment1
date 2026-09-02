# Skill Vetting Report

Before installing `param087/agent-ml-skills` and `nimrodfisher/data-analytics-skills`,
both were vetted for safety. This documents exactly what was checked, not just an
assertion that it's "safe."

## What Was Checked

1. **Repo existence and identity** — confirmed both repos exist at the exact
   owner/name the prompt specified, via web search and direct GitHub API/raw-file
   fetches (not assumed from the name alone).
2. **Install mechanism inspected before running anything**:
   - `param087/agent-ml-skills` ships `install.mjs` — read in full. Pure Node.js
     filesystem operations (`fs.cpSync`, `fs.copyFileSync`, `fs.mkdirSync`) copying
     `SKILL.md` files from the repo's own `skills/` directory into a target
     directory. Zero dependencies (stated in its own header comment and verified
     against its content), no network calls, no `eval`, no shelling out.
   - `nimrodfisher/data-analytics-skills` has no install script at all — skills are
     plain markdown files (`SKILL.md`) meant to be copied or referenced directly.
     No code execution surface at all.
3. **Star count / popularity** — `agent-ml-skills` and `data-analytics-skills` are
   smaller, newer repos (hundreds of stars, not evidence of malice but also not
   independently vetted by a large community); treated with appropriately more
   scrutiny of actual file content (done above) rather than trusting popularity alone.
4. **Cross-referenced against the professor's own reference repo** (`SKILLS.md`
   index file only — a text listing of which skill names map to which project,
   the same "check text for names, never copy code" permission pattern used
   throughout this repo). The professor's `05_data_science_skills_lab` project
   lists skills named `exploratory-data-analysis`, `feature-engineering`,
   `data-cleaning`, `imbalanced-data`, `model-evaluation`, `data-quality-audit` —
   these names match `agent-ml-skills`' and `data-analytics-skills`' actual skill
   names closely, an independent signal these are the intended, expected sources
   for this assignment rather than a coincidentally-similar decoy.
5. **Installation method actually used**: `git clone` (read-only) into a scratch
   directory, then a direct file copy of the `skills/` folder contents into this
   project's `.claude/skills/` — *not* running either repo's own installer, even
   though both were found safe, simply for maximum auditability of exactly what
   files landed where.

## Result

46 skills installed (15 from `agent-ml-skills`, 31 from `data-analytics-skills`,
after deduplicating 2 skills — `metric-reconciliation` and `schema-mapper` — that
`data-analytics-skills` legitimately cross-lists under two category folders).
Full attribution in `.claude/skills/ATTRIBUTION.md`.

## What Was NOT Done

No code from either repo's skill *content* (the methodological instructions each
SKILL.md contains) was copied into this project's own analysis scripts — the
scripts in `scripts/` implement each demonstrated skill's methodology
independently, informed by the skill's stated purpose (its description/name),
the same way a data scientist would apply a documented methodology without
transcribing someone else's prompt text.
