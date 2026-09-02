# Skill Vetting Report — mattpocock/skills

Before installing `mattpocock/skills`, it was vetted:

1. **Identity confirmed** — the exact repo the prompt named (`mattpocock/skills`,
   "Skills for Real Engineers") via web search, not assumed from the name alone.
2. **Popularity/legitimacy**: tens of thousands of GitHub stars, officially listed
   in Claude Code's own plugin marketplace (`claude plugins install
   mattpocock-skills`) — a much stronger trust signal than an unknown repo, since
   marketplace listing implies Anthropic-side review.
3. **Install mechanisms inspected before running anything**: `scripts/link-skills.sh`
   (explicitly labeled "dev-only... not a supported installer" in its own header)
   does pure filesystem symlinking (`ln -sfn`) from the repo into
   `~/.claude/skills` — no network calls, no eval, no `curl | sh` pattern.
   `scripts/list-skills.sh` is a one-line `find`. Both read in full.
4. **License**: MIT.
5. **Cross-referenced against the professor's own reference repo** (`SKILLS.md`
   text index only, names never code): the professor's `09_flowforge_dag_engine`
   equivalent lists `matt-pocock-typescript-patterns`, `matt-pocock-to-spec`,
   `matt-pocock-to-tickets`, `matt-pocock-grill-me` — `to-spec`, `to-tickets`, and
   `grill-me` (renamed `grilling`/`grill-with-docs` variants) all exist verbatim in
   the actual repo's `skills/engineering/` and `skills/productivity/` directories,
   an independent confirmation this is the intended, expected source.

## Installation Actually Used

`git clone` (read-only) into scratch space, then a direct file copy of
`skills/engineering/` and `skills/productivity/` (25 skills; `deprecated/` and
`in-progress/` subdirectories excluded as not release-ready) into this project's
`.claude/skills/` — not either of the repo's own scripts, for maximum
auditability of exactly what landed where. Attribution preserved in
`.claude/skills/ATTRIBUTION.md`-equivalent note below.

**Attribution**: 25 skills from [mattpocock/skills](https://github.com/mattpocock/skills)
(Matt Pocock) — MIT License. No skill *content* was copied into this project's own
TypeScript source (`backend/src/`) — the DAG engine's design patterns (branded
types, discriminated unions, exhaustiveness checking via `assertNever`) are applied
independently, informed by the general engineering literature these skills point
toward, not transcribed from any skill's prose.
