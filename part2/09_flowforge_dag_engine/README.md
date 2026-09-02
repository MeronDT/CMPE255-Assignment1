# FlowForge DAG Engine

A complicated end-to-end full-stack TypeScript project: a real DAG (directed
acyclic graph) workflow execution engine, demonstrating engineering patterns
after installing 25 skills from `mattpocock/skills` (vetted — see
[docs/00_skill_vetting.md](./docs/00_skill_vetting.md)).

## Result

**Genuine Kahn's-algorithm topological execution** — parallel layers, real
dependency-aware failure propagation (a failed task correctly skips only its
*dependents*, not unrelated sibling tasks), branded types, discriminated unions,
and compiler-enforced exhaustiveness checking (`assertNever`) — all verified
directly, not just claimed (see `DESIGN_DOC.md §3-4` for the exact compile-error
proof and the failure-propagation demo's real API output).

3 example workflows: a CI/CD pipeline (branching/merging DAG), an ETL pipeline
(parallel extract, join, load), and a failure-propagation demo.

## Running it

```bash
cd backend && npm install
npx tsx src/server.ts   # port 8009, REST + WebSocket

cd ../frontend && npm install && npm run dev -- --port 5189
```

Full design decisions, including the real exhaustiveness-check verification, in
[DESIGN_DOC.md](./DESIGN_DOC.md).
