/**
 * Core domain types for the FlowForge DAG engine.
 *
 * Applies engineering patterns independently derived from the general
 * TypeScript engineering literature this project's installed skills point
 * toward (branded/nominal types, discriminated unions, exhaustive matching)
 * -- NOT copied from any skill's prose content. See ../../.claude/skills for
 * the installed skill instructions themselves; this file is original code
 * applying the *methodology*, not a transcription of the skill text.
 */

// ---------- Branded types ----------
// Plain `string` node/workflow IDs are structurally interchangeable in
// TypeScript's type system -- a NodeId could be passed where a WorkflowId is
// expected and the compiler would never catch it. Branding closes that hole:
// the two types are nominally distinct even though both are strings at runtime.
declare const brand: unique symbol;
type Brand<T, B> = T & { readonly [brand]: B };

export type NodeId = Brand<string, "NodeId">;
export type WorkflowId = Brand<string, "WorkflowId">;
export type RunId = Brand<string, "RunId">;

export const NodeId = (id: string): NodeId => id as NodeId;
export const WorkflowId = (id: string): WorkflowId => id as WorkflowId;
export const RunId = (id: string): RunId => id as RunId;

// ---------- Discriminated union: node execution status ----------
// Each variant carries exactly the data that's valid for that state -- a
// "running" node has a startedAt, a "failed" node has an error, and the
// compiler enforces you can't accidentally read `.error` off a "success"
// node. This is the core discriminated-union idiom: the `status` field is
// the discriminant every consumer switches on.
export type NodeStatus =
  | { status: "pending" }
  | { status: "running"; startedAt: number }
  | { status: "success"; startedAt: number; finishedAt: number; output: unknown }
  | { status: "failed"; startedAt: number; finishedAt: number; error: string }
  | { status: "skipped"; reason: string };

/**
 * Exhaustiveness helper. If a new NodeStatus variant is ever added and a
 * switch statement doesn't handle it, this function's parameter type
 * (`never`) makes the call site a COMPILE ERROR, not a runtime surprise --
 * the standard `assertNever` pattern for making discriminated-union
 * handling provably complete.
 */
export function assertNever(x: never): never {
  throw new Error(`Unhandled discriminated union member: ${JSON.stringify(x)}`);
}

export function statusLabel(s: NodeStatus): string {
  switch (s.status) {
    case "pending":
      return "Pending";
    case "running":
      return "Running";
    case "success":
      return "Success";
    case "failed":
      return "Failed";
    case "skipped":
      return `Skipped (${s.reason})`;
    default:
      return assertNever(s);
  }
}

// ---------- Workflow definition ----------
export interface TaskDef {
  id: NodeId;
  label: string;
  dependsOn: NodeId[];
  /** Simulated work duration in ms -- stands in for a real task's runtime. */
  durationMs: number;
  /** If true, this task is scripted to fail, to demonstrate downstream skipping. */
  simulateFailure?: boolean;
}

export interface WorkflowDef {
  id: WorkflowId;
  name: string;
  description: string;
  tasks: TaskDef[];
}

export interface RunState {
  runId: RunId;
  workflowId: WorkflowId;
  startedAt: number;
  finishedAt: number | null;
  statuses: Record<string, NodeStatus>;
  executionOrder: NodeId[][]; // layers of nodes that ran in parallel, in order
}
