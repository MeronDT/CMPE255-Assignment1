# Design Doc — FlowForge DAG Engine

## 1. Overview

A complicated end-to-end full-stack TypeScript project demonstrating rigorous
engineering patterns after installing `mattpocock/skills` (25 skills, vetted — see
[docs/00_skill_vetting.md](./docs/00_skill_vetting.md)): a real DAG (directed
acyclic graph) workflow execution engine, not a toy.

## 2. Architecture

```
backend/src/
  types.ts       -- branded types (NodeId/WorkflowId/RunId), discriminated union
                    NodeStatus, assertNever() exhaustiveness helper
  dag.ts          -- Kahn's algorithm: layered topological sort + real cycle
                     detection, with a self-test that runs at module load
  executor.ts      -- runs layers in parallel (Promise.all per layer), skips
                      downstream-dependent tasks on failure (real dependency-
                      aware behavior, not flat parallel-everything)
  workflows.ts      -- 3 example workflows (CI/CD pipeline, ETL pipeline, a
                        failure-propagation demo)
  server.ts          -- Express REST API + WebSocket for live status push
frontend/src/
  DagView.tsx         -- client-side layered SVG DAG visualization with live
                         color-coded node status and a pulsing "running" indicator
  App.tsx              -- workflow selector, run trigger, live execution log
```

## 3. Engineering Patterns Applied (Genuinely, Not Just Named)

- **Branded/nominal types** (`types.ts`): `NodeId`, `WorkflowId`, `RunId` are all
  `string` at runtime but nominally distinct at compile time — passing a
  `WorkflowId` where a `NodeId` is expected is a type error, not a runtime bug
  waiting to happen.
- **Discriminated unions**: `NodeStatus` has 5 variants, each carrying exactly the
  data valid for that state (a `"running"` node has `startedAt`; a `"failed"` node
  has `error`; you cannot accidentally read `.error` off a `"success"` node).
- **Exhaustiveness checking**: `assertNever()` in `statusLabel()` means adding a
  6th `NodeStatus` variant without updating every switch statement is a **compile
  error**, not a silent runtime gap — verified directly, not just asserted:
  temporarily added a `"cancelled"` variant to `NodeStatus` without touching
  `statusLabel()`'s switch, and `tsc --noEmit` failed exactly as claimed:
  `src/types.ts(66,26): error TS2345: Argument of type '{ status: "cancelled";
  reason: string; }' is not assignable to parameter of type 'never'.` Reverted
  immediately after confirming; a clean `tsc --noEmit` afterward confirmed the
  revert was correct.
- **Kahn's algorithm** (`dag.ts`): real topological sort into parallel-execution
  layers, with genuine cycle detection (a workflow with a cycle has no valid
  order and is rejected, not silently mis-executed) — verified by a self-test
  that runs at module load and would crash server startup if the algorithm were
  ever broken.

## 4. Verified Behavior (Not Just Claimed)

Ran the failure-propagation demo end-to-end: `risky-migration` fails →
`run-migrated-tests` (its dependent) is correctly marked `"skipped"`, while
`independent-lint` and `independent-docs` (siblings in the same execution layer,
no dependency relationship to the failing task) both completed successfully in
parallel — confirmed via the API response's `statuses` and `executionOrder`
fields, and visually on the dashboard (screenshot:
`docs/screenshots/failure_demo_finished.png`).

## 5. Deployment

Express + `ws` backend (port 8009, REST + WebSocket). React + TypeScript frontend
(port 5189) with a custom SVG DAG renderer (not a generic graph library — the
layered layout needs to match the engine's own topological layering exactly).

Verified via Playwright: 0 console errors, 0 failed requests, across idle state,
a live in-progress run (WebSocket updates confirmed reaching the UI), and the
completed failure-propagation demo.
