# Skill Source Attribution

The 46 skills installed in this directory are sourced from two third-party
open-source repositories, vetted for safety before installation (see
`docs/00_skill_vetting.md`). No code from either repo's *implementation* was
copied into this project's own scripts/backend/frontend — only their SKILL.md
instruction files were installed as-is, per their own intended distribution
mechanism.

- **15 skills** from [param087/agent-ml-skills](https://github.com/param087/agent-ml-skills)
  (Param Bhavsar) — MIT License.
- **31 skills** from [nimrodfisher/data-analytics-skills](https://github.com/nimrodfisher/data-analytics-skills)
  (Nimrod Fisher) — no explicit license file found in the source repo at time
  of installation; the repo is publicly published specifically for this kind
  of installation into a user's own Claude Code setup, and is used here
  accordingly with source attribution preserved.

Installed via `git clone` (read access only) + direct file copy — no install
script from either repo was executed against this machine; both repos' own
installers were inspected first (see the vetting doc) and found safe, but a
plain copy was used instead for simplicity and full auditability of exactly
what was placed where.
